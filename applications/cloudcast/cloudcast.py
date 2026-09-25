"""Cloudcast broadcast routing: the region graph and jobs, the seed and reference programs, the sandbox, and the grader.

The link profiles, the five training jobs, and the cost model follow ADRS commit 2139fd8. A
candidate runs only inside the sandbox and reports the route each partition takes; this process
checks every hop against the real region graph and prices the routes itself, so a candidate
cannot fake a score.
"""
import csv
import io
import json
import math
import os
import random
import shutil
import subprocess
import textwrap
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import networkx as nx
import meta_evolve as meta
from meta_evolve.adapters import DevelopmentSubprocessSandbox, DirectoryWorkspaces
from meta_evolve.domain import EvaluatorFailure, InvalidOutput
from meta_evolve.ports import ProcessSpec

ADRS = ("https://raw.githubusercontent.com/UCB-ADRS/ADRS/2139fd86c0be3fb41c75768033fa8af5e4872c1a/"
        "openevolve/examples/ADRS/cloudcast/")
TRAIN_JOBS = ("intra_aws", "intra_azure", "intra_gcp", "inter_agz", "inter_gaz2")  # the search sees these
HELD_OUT_SEED = 20260922  # 8 generated jobs, used once after the search
RANDOM_SEED = 20260922    # reseeded before every job, so a sampling candidate scores the same each time

# The released cost model: 2 VMs per region at $0.54 an hour, and per-provider rate limits (Gbps).
NUM_VMS = 2
COST_PER_INSTANCE_HR = 0.54
DEFAULT_INGRESS = {"aws": 10, "gcp": 16, "azure": 16}
DEFAULT_EGRESS = {"aws": 5, "gcp": 7, "azure": 16}


def adrs_file(name):
    """A file from the released ADRS example, downloaded once next to this module."""
    path = Path(__file__).resolve().parent / "adrs_cloudcast" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(ADRS + name, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text()


def _links():
    """`[src, dst, cost ($/GB), throughput (Gbps)]` per link, in the order the released `make_nx_graph` adds them."""
    links = {}
    for row in csv.DictReader(io.StringIO(adrs_file("profiles/throughput.csv"))):
        if row["src_region"] != row["dst_region"]:
            links[row["src_region"], row["dst_region"]] = [None, NUM_VMS * float(row["throughput_sent"]) / 1e9]
    for row in csv.DictReader(io.StringIO(adrs_file("profiles/cost.csv"))):
        if (row["src"], row["dest"]) in links:
            links[row["src"], row["dest"]][0] = float(row["cost"])
    return [[u, v, cost, throughput] for (u, v), (cost, throughput) in links.items()]


LINKS = _links()
GRAPH = nx.DiGraph()
for _u, _v, _cost, _throughput in LINKS:
    GRAPH.add_edge(_u, _v, cost=_cost, throughput=_throughput)


def train_jobs():
    return [{**json.loads(adrs_file(f"examples/config/{name}.json")), "name": name} for name in TRAIN_JOBS]


def held_out_jobs(seed=HELD_OUT_SEED, num_cases=8):
    """Unseen jobs of the released shape (300 GB in 10 partitions): only the source and destinations vary."""
    regions = sorted(GRAPH.nodes)
    rng = random.Random(seed)
    jobs = []
    for index in range(num_cases):
        source = rng.choice(regions)
        dests = rng.sample([region for region in regions if region != source], rng.choice([6, 7]))
        jobs.append({"name": f"held_out_{index:02d}", "source_node": source, "dest_nodes": dests,
                     "data_vol": 300, "num_partitions": 10,
                     "ingress_limit": dict(DEFAULT_INGRESS), "egress_limit": dict(DEFAULT_EGRESS)})
    return jobs


SEED_SOURCE = textwrap.dedent('''
    def search_algorithm(src, dsts, G, num_partitions):
        """Direct replication: the source sends a full copy to every destination.

        This is the ADRS paper's Cloudcast initial program (appendix C.3); the
        reported 31.1% cost reduction is measured against it.
        """
        topology = BroadCastTopology(src, dsts, num_partitions)
        for dst in dsts:
            edge = G[src][dst]
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, [[src, dst, edge]])
        return topology
''').strip() + "\n"

# The two released baselines: each destination's cheapest path, and one minimum-cost
# tree over the source and destinations (MDST), which relays only through destinations.
DIJKSTRA_SOURCE = textwrap.dedent('''
    import networkx as nx

    def search_algorithm(src, dsts, G, num_partitions):
        h = G.copy()
        h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
        topology = BroadCastTopology(src, dsts, num_partitions)
        for dst in dsts:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, [[s, t, G[s][t]] for s, t in zip(path, path[1:])])
        return topology
''').strip() + "\n"

MDST_SOURCE = textwrap.dedent('''
    import networkx as nx

    def search_algorithm(src, dsts, G, num_partitions):
        h = G.copy()
        h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
        tree = nx.minimum_spanning_arborescence(h.subgraph([src] + list(dsts)), attr="cost", preserve_attrs=True)
        topology = BroadCastTopology(src, dsts, num_partitions)
        for dst in dsts:
            path = nx.shortest_path(tree, src, dst)
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, [[s, t, G[s][t]] for s, t in zip(path, path[1:])])
        return topology
''').strip() + "\n"

REFERENCES = {"direct_replication_seed": SEED_SOURCE, "released_dijkstra": DIJKSTRA_SOURCE,
              "released_mdst": MDST_SOURCE}

# Runs inside the container: it calls the candidate on each job and prints the routes. It
# does no pricing, and the grader ignores the edge data it copies, so nothing it prints can
# lower a cost.
RUNNER = '''
import json, random, sys
sys.path.insert(0, "/opt/networkx")  # the host's networkx, mounted read-only
import networkx as nx


class BroadCastTopology:
    """paths[dst][str(partition)] is the list of hops [u, v, G[u][v]] that partition takes."""

    def __init__(self, src, dsts, num_partitions):
        self.src, self.dsts, self.num_partitions = src, list(dsts), num_partitions
        self.paths = {dst: {str(p): None for p in range(num_partitions)} for dst in self.dsts}

    def set_dst_partition_paths(self, dst, partition, hops):
        self.paths[dst][str(partition)] = hops

    def append_dst_partition_path(self, dst, partition, hop):
        self.paths[dst][str(partition)] = (self.paths[dst].get(str(partition)) or []) + [hop]


graph = nx.DiGraph()
for u, v, cost, throughput in json.load(open("links.json")):
    graph.add_edge(u, v, cost=cost, throughput=throughput)
out, sys.stdout = sys.stdout, sys.stderr  # the candidate's own prints go to stderr
namespace = {"BroadCastTopology": BroadCastTopology, "nx": nx, "networkx": nx}
exec(compile(open("candidate.py").read(), "candidate.py", "exec"), namespace)
routes = []
for job in json.load(open("jobs.json")):
    random.seed(int(sys.argv[1]))
    try:
        topology = namespace["search_algorithm"](
            job["source_node"], list(job["dest_nodes"]), graph.copy(), job["num_partitions"])
        routes.append({"src": topology.src, "dsts": list(topology.dsts), "paths": topology.paths})
    except Exception as error:
        routes.append(f"{type(error).__name__}: {error}")
out.write(json.dumps(routes, default=repr))
'''


# DockerSandbox, a Meta-Evolve SandboxProvider: each candidate runs in a throwaway
# container with no network, no privileges, a read-only workspace, capped memory,
# CPU and processes, and only the environment the spec names. Meta-Evolve's
# DevelopmentSubprocessSandbox still enforces the deadline and output limits.
class DockerSandbox:
    def __init__(self, image="python:3.12-slim"):
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("DockerSandbox needs the docker CLI on PATH")
        self.docker = os.path.realpath(docker)
        self.image = image
        self._enforcer = DevelopmentSubprocessSandbox()

    def run(self, spec, workspace):
        root = workspace.root.resolve()
        name = f"meta-evolve-{uuid.uuid4().hex[:12]}"
        argv = spec.argv
        if argv[0] in {"python", "python3"}:  # `env -i` clears PATH inside the container
            argv = ("/usr/local/bin/python3", *argv[1:])
        container = (
            self.docker, "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m",
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--volume", f"{root}:{root}:ro",
            "--volume", f"{os.path.dirname(nx.__file__)}:/opt/networkx/networkx:ro",
            "--workdir", str(root.joinpath(*spec.cwd.split("/"))),
            self.image,
            # The container ends itself even if the docker client is killed.
            "timeout", "-s", "KILL", str(math.ceil(spec.timeout_seconds) + 1),
            "env", "-i", *(f"{key}={value}" for key, value in spec.environment.items()),
            *argv,
        )
        # The docker client only needs enough environment to find docker.
        outer = replace(spec, argv=container, cwd=".", environment={
            "HOME": os.environ.get("HOME", "/"),
            "PATH": os.pathsep.join((os.path.dirname(self.docker), os.defpath)),
        })
        result = self._enforcer.run(outer, workspace)
        if result.failure is not None:
            subprocess.run((self.docker, "rm", "--force", name), capture_output=True)
        return result


class InvalidRoutes(ValueError):
    """The candidate's routes do not deliver the data over real links."""


def price(routes, job):
    """Check one job's routes against the real graph; return `(transfer_time, cost)` as the released `BCSimulator` does."""
    if isinstance(routes, str):
        raise InvalidRoutes(f"search_algorithm raised {routes}")
    source, dests, partitions = job["source_node"], job["dest_nodes"], job["num_partitions"]
    if routes["src"] != source or set(routes["dsts"]) != set(dests):
        raise InvalidRoutes("the topology's source or destinations do not match the job")

    # Every (destination, partition) needs real hops that reach it from the source.
    overlay = nx.DiGraph()  # each link used, once, with the partitions that cross it
    for dst in dests:
        for partition in range(partitions):
            hops = (routes["paths"].get(dst) or {}).get(str(partition))
            if not isinstance(hops, list) or not 0 < len(hops) <= 200:
                raise InvalidRoutes(f"partition {partition} of {dst} needs 1 to 200 hops")
            for hop in hops:
                u, v = hop[:2] if isinstance(hop, list) and len(hop) >= 2 else (None, None)
                if not (isinstance(u, str) and isinstance(v, str) and GRAPH.has_edge(u, v)) or v == source:
                    raise InvalidRoutes(f"hop {str(hop)[:80]} is not a graph link away from the source")
                if not overlay.has_edge(u, v):  # priced from the graph; the candidate's edge data is ignored
                    overlay.add_edge(u, v, **GRAPH[u][v], flow=GRAPH[u][v]["throughput"], partitions=set())
                overlay[u][v]["partitions"].add(partition)
            reached = nx.DiGraph([hop[:2] for hop in hops])
            if source not in reached or dst not in nx.descendants(reached, source):
                raise InvalidRoutes(f"partition {partition} of {dst} never reaches it from {source}")
    if overlay.number_of_edges() > 400:
        raise InvalidRoutes(f"the routes use {overlay.number_of_edges()} distinct links, limit is 400")

    # A region's ingress and egress capacity is split equally across its links when demand exceeds it.
    ingress = {**DEFAULT_INGRESS, **{p: NUM_VMS * limit for p, limit in job["ingress_limit"].items()}}
    egress = {**DEFAULT_EGRESS, **{p: NUM_VMS * limit for p, limit in job["egress_limit"].items()}}
    for node in overlay.nodes:
        provider = node.split(":")[0]
        for links, limit in ((list(overlay.in_edges(node)), ingress[provider]),
                             (list(overlay.out_edges(node)), egress[provider])):
            if links and sum(overlay[u][v]["flow"] for u, v in links) > limit:
                share = 1 / len(links)
                for u, v in links:
                    overlay[u][v]["flow"] = min(overlay[u][v]["flow"], limit * share)

    # Transfer time is the slowest link; each link's egress is paid once per partition crossing it.
    gb_per_partition = job["data_vol"] / partitions
    links = [data for _, _, data in overlay.edges.data()]
    transfer_time = max(len(data["partitions"]) * gb_per_partition * 8 / data["flow"] for data in links)
    egress_cost = sum(len(data["partitions"]) * gb_per_partition * data["cost"] for data in links)
    instance_cost = overlay.number_of_nodes() * NUM_VMS * (COST_PER_INSTANCE_HR / 3600) * round(transfer_time, 2)
    return transfer_time, egress_cost + instance_cost


def evaluate(source, *, sandbox, jobs=None):
    """Meta-Evolve evaluator: run the candidate in the sandbox, price its routes here."""
    jobs = train_jobs() if jobs is None else jobs
    files = {"candidate.py": str(source), "runner.py": RUNNER,
             "links.json": json.dumps(LINKS), "jobs.json": json.dumps(jobs)}
    workspaces = DirectoryWorkspaces()
    workspace = workspaces.materialize(meta.SourceTree(files))
    try:
        # PYTHONHASHSEED pins networkx's tie-breaking between equal-cost links.
        process = sandbox.run(ProcessSpec(argv=("python3", "runner.py", str(RANDOM_SEED)),
                                          environment={"PYTHONHASHSEED": "0"},
                                          timeout_seconds=90, stdout_limit=16_000_000), workspace)
    finally:
        workspaces.dispose(workspace)

    if process.failure is not None:  # timeout or output flood
        return meta.EvaluationResult(failure=process.failure)
    if process.exit_code != 0:
        lines = process.stderr.strip().splitlines() or [f"candidate killed (exit {process.exit_code})"]
        return meta.EvaluationResult(failure=EvaluatorFailure(message=lines[-1][:500]))

    per_config, total_cost, total_time = [], 0.0, 0.0
    for job, job_routes in zip(jobs, json.loads(process.stdout)):
        try:
            transfer_time, cost = price(job_routes, job)
        except Exception as error:
            reason = error if isinstance(error, InvalidRoutes) else f"{type(error).__name__}: {error}"
            return meta.EvaluationResult(failure=InvalidOutput(message=f"invalid routes for {job['name']}: {reason}"[:500]))
        per_config.append({"config": job["name"], "cost": cost, "transfer_time": transfer_time})
        total_cost += cost  # added in order, as the released evaluator does
        total_time += transfer_time
    # combined_score is the released ADRS score, 1 / (1 + total cost).
    return meta.EvaluationResult(
        metrics={"total_cost": total_cost, "combined_score": 1 / (1 + total_cost), "total_transfer_time": total_time},
        evidence=(meta.EvidenceDraft(kind="cloudcast-evaluation", data={"per_config": per_config}),))

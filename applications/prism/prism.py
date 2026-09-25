"""PRISM model placement: the workloads, the seed and reference programs, the sandbox, and the grader.

The workload generator and the released score follow ADRS commit 2139fd8. A candidate runs only
inside the sandbox and reports which model went to which GPU; this process checks
that placement and computes the pressure itself, so a candidate cannot fake a score.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import textwrap
import uuid
from dataclasses import astuple, dataclass, replace

import numpy as np
import meta_evolve as meta
from meta_evolve.adapters import DevelopmentSubprocessSandbox, DirectoryWorkspaces
from meta_evolve.domain import EvaluatorFailure, InvalidOutput
from meta_evolve.ports import ProcessSpec

GPU_MEM_SIZE = 80
TRAIN_SEED = 42            # the 50 cases the search sees
HELD_OUT_SEED = 20260921   # 100 cases used once, after the search
TRAIN_CASES = 50
HELD_OUT_CASES = 100


@dataclass(frozen=True)
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def generate_workloads(seed: int = TRAIN_SEED, num_tests: int = TRAIN_CASES):
    """The released ADRS workload generator, with a local RNG."""
    rng = np.random.RandomState(seed)
    cases = []
    for case_id in range(num_tests):
        gpu_num = int(rng.randint(5, 10))
        models = [
            Model(
                model_name=f"case_{case_id}_model_{model_id}",
                model_size=int(rng.randint(10, 30)),
                req_rate=int(rng.randint(1, 10)),
                slo=int(rng.randint(5, 10)),
                cur_gpu_id=model_id,
            )
            for model_id in range(gpu_num * 2)
        ]
        cases.append((gpu_num, models))
    return tuple(cases)


def max_kvpr(gpus: list[list[Model]]) -> float:
    """Worst GPU pressure: sum(req_rate / slo) / remaining memory."""
    pressures = []
    for models in gpus:
        remaining = GPU_MEM_SIZE - sum(model.model_size for model in models)
        load = sum(model.req_rate / model.slo for model in models)
        pressures.append(load / remaining if remaining > 0 else 1_000_000.0)
    return max(pressures, default=0.0)


SEED_SOURCE = textwrap.dedent("""
def compute_model_placement(gpu_num, models):
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    for model in models:
        for gpu_id in range(gpu_num):
            used = sum(item.model_size for item in placement[gpu_id])
            if model.model_size <= 80 - used:
                placement[gpu_id].append(model)
                break
    return placement
""").strip() + "\n"


PRISM_GREEDY_SOURCE = textwrap.dedent("""
def compute_model_placement(gpu_num, models):
    sorted_models = sorted(
        models, key=lambda model: model.req_rate / model.slo, reverse=True
    )
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [80 for _ in range(gpu_num)]
    weighted_rate = [0.0 for _ in range(gpu_num)]

    for model in sorted_models:
        best_gpu = None
        best_ratio = float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= remaining[gpu_id] and remaining[gpu_id] > 0:
                ratio = weighted_rate[gpu_id] / remaining[gpu_id]
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_gpu = gpu_id
        if best_gpu is None:
            raise ValueError("model does not fit")
        placement[best_gpu].append(model)
        weighted_rate[best_gpu] += model.req_rate / model.slo
        remaining[best_gpu] -= model.model_size
    return placement
""").strip() + "\n"


ADRS_BEST_SOURCE = textwrap.dedent("""
def compute_model_placement(gpu_num, models):
    def pressure(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    def maximum(weights, memories):
        return max(pressure(weights[i], memories[i]) for i in range(gpu_num))

    ordered = sorted(
        models,
        key=lambda model: (model.req_rate / model.slo, model.model_size),
        reverse=True,
    )
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [80] * gpu_num
    weights = [0.0] * gpu_num

    for model in ordered:
        contribution = model.req_rate / model.slo
        best_gpu = None
        best_future = float("inf")
        best_remaining = -1
        for gpu_id in range(gpu_num):
            if model.model_size > remaining[gpu_id]:
                continue
            future_weights = weights.copy()
            future_memory = remaining.copy()
            future_weights[gpu_id] += contribution
            future_memory[gpu_id] -= model.model_size
            future_max = maximum(future_weights, future_memory)
            if (
                future_max < best_future - 1e-9
                or (
                    abs(future_max - best_future) <= 1e-9
                    and future_memory[gpu_id] > best_remaining
                )
            ):
                best_gpu = gpu_id
                best_future = future_max
                best_remaining = future_memory[gpu_id]
        if best_gpu is None:
            raise ValueError("model does not fit")
        placement[best_gpu].append(model)
        weights[best_gpu] += contribution
        remaining[best_gpu] -= model.model_size

    improved = True
    while improved:
        improved = False
        current = maximum(weights, remaining)
        gpu_order = sorted(
            range(gpu_num),
            key=lambda gpu: pressure(weights[gpu], remaining[gpu]),
            reverse=True,
        )
        for source in gpu_order:
            for model in list(placement[source]):
                contribution = model.req_rate / model.slo
                for target in range(gpu_num):
                    if target == source or model.model_size > remaining[target]:
                        continue
                    candidate_weights = weights.copy()
                    candidate_memory = remaining.copy()
                    candidate_weights[source] -= contribution
                    candidate_memory[source] += model.model_size
                    candidate_weights[target] += contribution
                    candidate_memory[target] -= model.model_size
                    if maximum(candidate_weights, candidate_memory) < current:
                        placement[source].remove(model)
                        placement[target].append(model)
                        weights = candidate_weights
                        remaining = candidate_memory
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        for left in range(gpu_num):
            for right in range(left + 1, gpu_num):
                for first in list(placement[left]):
                    first_weight = first.req_rate / first.slo
                    for second in list(placement[right]):
                        second_weight = second.req_rate / second.slo
                        left_memory = (
                            remaining[left] + first.model_size - second.model_size
                        )
                        right_memory = (
                            remaining[right] + second.model_size - first.model_size
                        )
                        if left_memory < 0 or right_memory < 0:
                            continue
                        candidate_weights = weights.copy()
                        candidate_memory = remaining.copy()
                        candidate_weights[left] += second_weight - first_weight
                        candidate_weights[right] += first_weight - second_weight
                        candidate_memory[left] = left_memory
                        candidate_memory[right] = right_memory
                        if maximum(candidate_weights, candidate_memory) < current:
                            placement[left].remove(first)
                            placement[right].remove(second)
                            placement[left].append(second)
                            placement[right].append(first)
                            weights = candidate_weights
                            remaining = candidate_memory
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return placement
""").strip() + "\n"


REFERENCES = {
    "first_fit_seed": SEED_SOURCE,
    "prism_greedy": PRISM_GREEDY_SOURCE,
    "adrs_released_best": ADRS_BEST_SOURCE,
}

# Runs inside the container. It calls the candidate and reports placements as
# model indices; it does no scoring, so nothing it prints can raise a score.
RUNNER = """
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int

namespace = {}
exec(compile(open("candidate.py").read(), "candidate.py", "exec"), namespace)
place = namespace["compute_model_placement"]
results = []
for gpu_num, rows in json.load(open("cases.json")):
    models = [Model(*row) for row in rows]
    index = {id(model): i for i, model in enumerate(models)}
    try:
        placement = place(gpu_num, models)
        results.append([[gpu, [index.get(id(model), -1) for model in placed]]
                        for gpu, placed in placement.items()])
    except Exception as error:
        results.append(f"{type(error).__name__}: {error}")
print(json.dumps(results))
"""


# DockerSandbox, a Meta-Evolve SandboxProvider: each candidate runs in a throwaway
# container with no network, no privileges, a read-only workspace, capped memory,
# CPU and processes, and an empty environment. Meta-Evolve's
# DevelopmentSubprocessSandbox still enforces the deadline and output limits.
class DockerSandbox:
    def __init__(self, image: str = "python:3.12-slim") -> None:
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


def grade(result, gpu_num, models):
    """Check one case's placement strictly; return (max_kvpr, reason)."""
    if isinstance(result, str):
        return None, result
    try:
        gpus = {gpu: list(indices) for gpu, indices in result}
    except (TypeError, ValueError):
        return None, "placement must map GPU ids to lists of models"
    if any(type(gpu) is not int for gpu in gpus) or set(gpus) != set(range(gpu_num)):
        return None, f"GPU ids must be exactly the integers 0..{gpu_num - 1}"
    placed = [i for indices in gpus.values() for i in indices]
    if any(type(i) is not int for i in placed) or sorted(placed) != list(range(len(models))):
        return None, "every model must be placed exactly once (no copies, no omissions)"
    layout = [[models[i] for i in gpus[gpu]] for gpu in range(gpu_num)]
    for gpu, assigned in enumerate(layout):
        used = sum(model.model_size for model in assigned)
        if used > GPU_MEM_SIZE:
            return None, f"GPU {gpu} uses {used} GB, above {GPU_MEM_SIZE} GB"
    return max_kvpr(layout), "valid"


def evaluate(source, *, sandbox, seed=TRAIN_SEED, num_tests=TRAIN_CASES):
    """Meta-Evolve evaluator: run the candidate in the sandbox, grade it here."""
    workloads = generate_workloads(seed, num_tests)
    cases = [[gpu_num, [astuple(model) for model in models]] for gpu_num, models in workloads]
    files = {"candidate.py": str(source), "runner.py": RUNNER, "cases.json": json.dumps(cases)}
    workspaces = DirectoryWorkspaces()
    workspace = workspaces.materialize(meta.SourceTree(files))
    try:
        process = sandbox.run(ProcessSpec(argv=("python3", "-I", "runner.py"), timeout_seconds=60),
                              workspace)
    finally:
        workspaces.dispose(workspace)

    if process.failure is not None:  # timeout or output flood
        return meta.EvaluationResult(failure=process.failure)
    if process.exit_code != 0:
        lines = process.stderr.strip().splitlines() or [f"candidate killed (exit {process.exit_code})"]
        return meta.EvaluationResult(failure=EvaluatorFailure(message=lines[-1][:500]))
    try:
        results = json.loads(process.stdout)
        assert isinstance(results, list) and len(results) == len(workloads)
    except (ValueError, AssertionError):
        return meta.EvaluationResult(failure=InvalidOutput(message="malformed runner output"))

    pressures = []
    for case, (result, (gpu_num, models)) in enumerate(zip(results, workloads)):
        value, reason = grade(result, gpu_num, models)
        if value is None:
            return meta.EvaluationResult(
                failure=InvalidOutput(message=f"invalid placement in case {case}: {reason}"))
        pressures.append(value)
    mean = statistics.fmean(pressures)
    # combined_score is the released ADRS score: success rate + 1 / mean(max KVPR).
    return meta.EvaluationResult(metrics={"mean_max_kvpr": mean, "combined_score": 1.0 + 1.0 / mean})

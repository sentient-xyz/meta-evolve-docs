"""OpenCodeAgent: a Meta-Evolve proposer backed by an OpenCode coding agent.

Meta-Evolve calls ``agent(parent, context=...)`` once per trial. Each trial runs one fresh
``opencode run`` session in a private workspace holding the parent as ``candidate.py``. The
prompt carries the task, the parent's scores, and a digest of what the search remembers,
including the best other trial's code. The agent edits the file and checks it with
``python agent_check.py``. If its last edit leaves the file broken, the last version the
checker accepted is proposed instead, and Meta-Evolve scores the proposal itself.
"""
import ast
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import meta_evolve as meta
from meta_evolve.domain import InfrastructureFailure, InvalidOutput, TimeoutFailure

from cloudcast import REFERENCES

MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
TASK_DIR = Path(__file__).resolve().parent
HYPOTHESIS = re.compile(r"<hypothesis>\s*(.*?)\s*</hypothesis>", re.DOTALL | re.IGNORECASE)

# Adapted from the released example's system message: it names the cost structure and the
# levers, and deliberately no algorithm.
TASK = """# Cloudcast broadcast-overlay mutation

Improve `search_algorithm(src, dsts, G, num_partitions)` in `candidate.py`.

You are an expert in cloud infrastructure optimization. The task is to broadcast
one object from a source region to every destination region across AWS, GCP and
Azure at the lowest total cost, by leveraging parallel paths and overlapping
transfers instead of sending a redundant copy per destination.

## Inputs

- `G` is a `networkx.DiGraph` over about 70 cloud regions named `provider:region`.
  It is effectively complete. Every edge carries `cost` (US$ per GB of egress)
  and `throughput` (Gbps).
- `dsts` is the list of destination regions; `src` is not among them.
- `num_partitions` is how many pieces the object is split into.

## Output

Return a `BroadCastTopology` (already in scope, do not redefine it) built as
`BroadCastTopology(src, dsts, num_partitions)`, where
`topology.paths[dst][str(p)]` is a non-empty list of hops `[u, v, G[u][v]]`.
Use `set_dst_partition_paths(dst, p, hops)` or `append_dst_partition_path(dst,
p, hop)`.

Requirements, all enforced:

- Every destination and every partition `0 .. num_partitions-1` needs a hop list
  whose edges connect `src` to that destination.
- Every hop must be a real edge of `G`. No self-loops, and no edge back into
  `src`.
- Edge payloads are re-resolved from `G` before scoring, so editing a hop's
  `cost` or `throughput` changes nothing.
- At most 200 hops per partition and 400 distinct edges per configuration.

## Objective

Minimize `total_cost` summed over the five training configurations. Lower is
better. For each configuration:

- egress cost is, over every distinct edge used, `(number of distinct partitions
  crossing that edge) x (data_vol / num_partitions) x edge cost`; so an edge
  shared by several destinations is paid for once, not once per destination;
- instance cost is `nodes_in_overlay x 2 x ($0.54/3600) x transfer_time`, so
  each extra relay region and each extra second is also billed;
- transfer time is the slowest hop on any destination's route, where a node's
  ingress and egress capacity is shared equally across its incident edges, so
  fanning out widely from one region slows every branch.

Any region in `G` may be used as an intermediate waypoint, not only the
destinations.

## Rules

- Only `networkx`, `math`, `heapq`, `itertools`, `collections` and `functools`
  may be imported.
- Be deterministic: no wall-clock, no unseeded randomness.
- All five configurations must be solved within 90 seconds total.
- Run `python agent_check.py` after editing. Held-out configurations are used to
  check generalization afterwards and are not available during evolution.
"""

# The agent: 10 steps, may edit only candidate.py and run only `python agent_check.py`.
DENY = ("task", "external_directory", "todowrite", "webfetch", "websearch", "lsp", "skill", "question", "doom_loop")
OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "agent": {"cloudcast-evolver": {
        "description": "Improve only the Cloudcast candidate in a bounded workspace",
        "mode": "primary", "steps": 10, "temperature": 0.35, "prompt": "{file:./agent_prompt.md}",
        "permission": {
            "read": "allow", "glob": "allow", "grep": "allow", "list": "allow",
            "edit": {"*": "deny", "candidate.py": "allow", "**/candidate.py": "allow"},
            "bash": {"*": "deny", "python agent_check.py": "allow"},
            **dict.fromkeys(DENY, "deny"),
        },
    }},
    # Reasoning capped at 6,000 tokens a turn: uncapped, the model spent its whole output
    # budget planning and never called a tool.
    "provider": {"openrouter": {"models": {"deepseek/deepseek-v4-flash-0731": {
        "options": {"reasoning": {"max_tokens": 6000}}}}}},
}

AGENT_PROMPT = """You are the tool-using mutation operator inside a Meta-Evolve run.

Your entire job is to improve `candidate.py` for the Cloudcast multi-cloud
broadcast task in this workspace.

Workflow:

1. The user message contains the complete authoritative task, Meta-Evolve
   context, and current candidate. Do not spend a turn rereading those files.
   Do not inspect `agent_check.py`, configuration, environment, or unrelated
   files; the only permitted command is already given below.
2. Form one focused algorithmic hypothesis. Use lineage memory and experiment
   history, but do not repeat an idea that already failed.
3. Make your first tool action an edit to `candidate.py`.
4. Run exactly `python agent_check.py` to validate and measure the candidate.
5. If the checker fails or the cost regresses, repair or revert the change.

When editing, write Python source exactly as it should appear in the file.
Never escape quotes: a docstring is `\"\"\"text\"\"\"`, not `\\"\\"\\"text\\"\\"\\"`. Escaped
quotes outside a string are a `SyntaxError` and cost a turn to repair.

Prioritize completing an edit and checker run over explaining your plan. Keep
the implementation bounded, deterministic, and easy to understand. Do not use
the network, install packages, create subagents, change the evaluator, or
inspect files outside this workspace. End with one short hypothesis wrapped in
`<hypothesis>...</hypothesis>`.
"""

AGENT_CHECK = '''# agent_check.py: the agent's only command. It scores candidate.py exactly as the
# evaluator will, and saves each version that passes next to (not inside) the workspace.
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["CLOUDCAST_TASK_ROOT"])
from cloudcast import DockerSandbox, evaluate  # noqa: E402
from opencode_agent import repair_escaped_quotes  # noqa: E402

path = Path("candidate.py")
source = path.read_text()
if (repaired := repair_escaped_quotes(source)) is not None:
    path.write_text(source := repaired)
    print('note: replaced \\\\" with " in candidate.py; write plain quotes in edits', file=sys.stderr)
result = evaluate(source, sandbox=DockerSandbox())
if result.failure is not None:
    print(f"FAILED ({result.failure.kind}): {result.failure.message}")
    raise SystemExit(1)
print(json.dumps({**result.metrics, "per_config": [dict(item) for item in result.evidence[0].data["per_config"]]},
                 indent=2))
Path.cwd().with_name(f"{Path.cwd().name}.last_ok.py").write_text(source)
'''


def repair_escaped_quotes(source):
    """Undo the `\\"` an edit tool sometimes writes for `"`, but only if that makes the file parse."""
    if '\\"' not in source or _parses(source):
        return None
    repaired = source.replace('\\"', '"')
    return repaired if _parses(repaired) else None


def _parses(source):
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


class OpenCodeAgent:
    """Pass it the evaluator it should show the agent: ``OpenCodeAgent(workdir, score)``."""

    def __init__(self, workdir, score, *, model=MODEL, timeout_seconds=900):
        self.workdir = Path(workdir).resolve()
        self.score = score
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.references = None
        self.first_step = {}  # source -> the trial that first produced it
        self.proposals = {}   # trial -> the source it proposed

    def __call__(self, parent, *, context):
        parent = str(parent)
        step = int(context.step)
        workspace = self._workspace(parent, step)
        prompt = self._prompt(parent, context)
        report, text, usage = self._run_opencode(workspace, prompt, step)

        path = workspace / "candidate.py"
        if (repaired := repair_escaped_quotes(path.read_text())) is not None:
            path.write_text(repaired)
        # Agents often edit once more after their last passing check and end on a broken file.
        last_ok = workspace.with_name(f"{workspace.name}.last_ok.py")
        report["fell_back_to_last_ok"] = (
            last_ok.exists() and last_ok.read_text().strip() != parent.strip()
            and path.read_text().strip() != last_ok.read_text().strip()
            and self.score(path.read_text()).failure is not None)
        if report["fell_back_to_last_ok"]:
            path.write_text(last_ok.read_text())

        evidence = (meta.EvidenceDraft(kind="opencode-session", data=report),)
        candidate = path.read_text().strip() + "\n"
        if candidate == parent:
            failure = (TimeoutFailure(message=report["error"]) if report["timed_out"] else
                       InfrastructureFailure(message=report["error"]) if report["error"] else
                       InvalidOutput(message="OpenCode left candidate.py unchanged"))
            return meta.ProposalResult(failure=failure, usage=usage, evidence=evidence)
        self.first_step.setdefault(candidate, step)
        self.proposals[step] = candidate
        found = HYPOTHESIS.findall(text)
        hypothesis = (" ".join(found[-1].split()) if found else
                      next((line.strip() for line in reversed(text.splitlines()) if line.strip()), ""))
        return meta.ProposalResult(
            candidate=meta.Text(candidate), usage=usage, evidence=evidence,
            hypothesis=hypothesis[:500].strip() or f"OpenCode harness mutation at step {step}")

    def _workspace(self, parent, step):
        workspace = self.workdir / "trials" / f"trial_{step:02d}"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.with_name(f"{workspace.name}.last_ok.py").unlink(missing_ok=True)
        workspace.mkdir(parents=True)
        for name, text in (("opencode.json", json.dumps(OPENCODE_CONFIG, indent=2)), ("agent_prompt.md", AGENT_PROMPT),
                           ("agent_check.py", AGENT_CHECK)):
            (workspace / name).write_text(text)
            (workspace / name).chmod(0o444)
        (workspace / "candidate.py").write_text(parent)
        return workspace

    def _prompt(self, parent, context):
        if self.references is None:
            costs = {name: self.score(source).metrics["total_cost"] for name, source in REFERENCES.items()}
            seed = costs["direct_replication_seed"]
            self.references = {name: {"total_cost": cost, "cost_reduction_vs_seed_percent":
                                      100.0 * (seed - cost) / seed} for name, cost in costs.items()}
        result = self.score(parent)
        metrics = ({**result.metrics, "per_config": [dict(item) for item in result.evidence[0].data["per_config"]]}
                   if result.failure is None else {"failure": result.failure.message})
        records = list(context.bundle.records)
        if context.experience is not None:
            records += context.experience.search(
                record_kinds=("evaluation", "evidence", "proposal", "outcome"), limit=40).records
        memory = self._remember(records, self.first_step.get(parent, 0))
        return (
            "Improve candidate.py now. The authoritative inputs follow; do not reread them with tools. "
            "Make your first tool action the edit, then run exactly `python agent_check.py`.\n\n"
            f"<task>\n{TASK}\n</task>\n\n<meta_evolve_context>\n# Meta-Evolve context\n\n"
            f"## Parent training metrics\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
            f"## Fixed reference metrics\n```json\n{json.dumps(self.references, indent=2)}\n```\n\n"
            f"{memory}\n</meta_evolve_context>\n\n<current_candidate>\n{parent}\n</current_candidate>"
        )

    def _remember(self, records, parent_step):
        """Group what Meta-Evolve supplied by trial: idea, cost or failure, per-job costs; add the best other code."""
        trials = {}
        for record in records:
            value, entry = record.value, trials.setdefault(int(record.logical_step), {})
            if record.kind == "proposal" and value.hypothesis:
                entry["hypothesis"] = " ".join(str(value.hypothesis).split())[:400]
            elif record.kind in ("evaluation", "outcome"):
                if value.failure is not None:
                    entry.setdefault("failure", str(value.failure.message)[:240])
                elif record.kind == "evaluation" and "total_cost" in value.metrics:
                    entry["cost"] = float(value.metrics["total_cost"])
            elif record.kind == "evidence" and value.kind == "cloudcast-evaluation":
                entry["per_config"] = {str(item["config"]): round(float(item["cost"]), 2)
                                       for item in value.data["per_config"]}
        trials.pop(parent_step, None)
        if not trials:
            return "## Earlier trials\n\nNo other trials are recorded yet.\n"

        lines = [
            "## Earlier trials", "",
            "These are OTHER trials in this run, not your file. Your `candidate.py` is valid and "
            "scores what \"Parent training metrics\" shows; do not rewrite it to fix errors listed "
            "here. Scores are training `total_cost` (lower is better). Learn from what worked and "
            "do not repeat ideas that scored worse.", "",
        ]
        for step, entry in sorted(trials.items()):
            if "cost" in entry:
                outcome = f"total_cost={entry['cost']:.2f}"
            elif "failure" in entry:
                editing = any(marker in entry["failure"] for marker in (
                    "SyntaxError", "IndentationError", "line continuation", "unchanged", "exceeded"))
                outcome = (f"no result (editing or session problem, not an idea result): {entry['failure']}"
                           if editing else f"FAILED: {entry['failure']}")
            else:
                outcome = "no result"
            lines.append(f"- step {step}: {outcome}")
            if entry.get("per_config"):
                lines.append(f"  per-config: {entry['per_config']}")
            if entry.get("hypothesis"):
                lines.append(f"  hypothesis: {entry['hypothesis']}")
        scored = [(entry["cost"], step) for step, entry in trials.items() if "cost" in entry and step != 0]
        if scored and (best := min(scored))[1] in self.proposals:
            code = self.proposals[best[1]]
            code = code if len(code) <= 12_000 else code[:12_000] + "\n# ... truncated ...\n"
            lines += ["", f"## Best other trial's code (step {best[1]}, total_cost={best[0]:.2f})", "",
                      "```python", code.rstrip(), "```"]
        return "\n".join(lines) + "\n"

    def _run_opencode(self, workspace, prompt, step):
        """One fresh `opencode run` session. Forking the parent's session hung every time in OpenCode 1.18."""
        bin_dir = self.workdir / "bin"  # `python` for the agent's check is this kernel's interpreter
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        (bin_dir / "python").chmod(0o755)
        env = {
            **os.environ,
            "OPENCODE_CONFIG": str(workspace / "opencode.json"),
            "OPENCODE_CONFIG_DIR": str(workspace),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE": "true",
            "CLOUDCAST_TASK_ROOT": str(TASK_DIR),
            "PATH": os.pathsep.join((str(bin_dir), os.environ["PATH"])),
        }
        opencode = shutil.which("opencode") or str(Path.home() / ".opencode" / "bin" / "opencode")
        command = (opencode, "run", "--pure", "--format", "json", "--model", self.model,
                   "--agent", "cloudcast-evolver", "--dir", str(workspace),
                   "--title", f"Cloudcast Meta-Evolve trial {step}", prompt)
        started = time.monotonic()
        # `opencode run` waits for EOF on a piped stdin, so give it none.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
        wall = time.monotonic() - started
        workspace.with_name(f"{workspace.name}.jsonl").write_text(stdout)

        tokens, cost, steps, texts, errors = 0, 0.0, 0, [], []
        for line in stdout.splitlines():  # OpenCode's newline-delimited event stream
            try:
                event = json.loads(line)
            except ValueError:
                continue
            part = event.get("part") or {}
            if part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
            elif part.get("type") == "step-finish":
                steps += 1
                tokens += int(part.get("tokens", {}).get("total") or sum(
                    part.get("tokens", {}).get(key, 0) for key in ("input", "output", "reasoning")))
                cost += float(part.get("cost") or 0.0)
            if event.get("type") == "error":
                error = event.get("error", {})
                errors.append(str(error.get("data", {}).get("message") or error)[:500])
        if stderr.strip():
            errors.append(stderr.strip()[-500:])
        if timed_out:
            errors.append(f"OpenCode exceeded {self.timeout_seconds:.0f}s")
        elif process.returncode != 0:
            errors.append(f"OpenCode exited with status {process.returncode}")
        elif not steps:
            errors.append("OpenCode produced no completed agent step")
        report = {"completed_steps": steps, "tokens": tokens, "cost_usd": cost, "timed_out": timed_out,
                  "error": " | ".join(errors) or None}
        usage = meta.Usage(tokens=tokens, spend_micros=math.ceil(cost * 1_000_000), wall_seconds=wall)
        return report, "\n".join(texts).strip(), usage

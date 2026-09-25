"""OpenCodeAgent: a Meta-Evolve proposer backed by an OpenCode coding agent.

Meta-Evolve calls ``agent(parent, context=...)`` once per trial. The agent gets a private
workspace holding the parent as ``candidate.py``, plus what the search remembers. It
edits the file and checks it with ``prism-check``. The best version that passed (or,
if none passed, the file as left) becomes the proposal, and Meta-Evolve scores it.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import meta_evolve as meta
from meta_evolve.domain import InfrastructureFailure, InvalidOutput
from opencode_ai import APIStatusError, APITimeoutError, Opencode

MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
TASK_DIR = Path(__file__).resolve().parent
HYPOTHESIS = re.compile(r"<hypothesis>\s*(.*?)\s*</hypothesis>", re.DOTALL | re.IGNORECASE)

TASK = """\
# Task: PRISM model placement

`compute_model_placement(gpu_num, models)` in `candidate.py` must return
`{gpu_id: [model, ...]}` for GPU ids `0..gpu_num-1`, placing every input model
object exactly once, with at most 80 GB of `model_size` per GPU.

A GPU's KV-cache pressure is `sum(req_rate / slo) / (80 - used_memory)`. A GPU
filled to exactly 80 GB has no free memory: the evaluator scores that case
1,000,000, and any division by its free memory raises ZeroDivisionError, so
handle full GPUs explicitly. The score is the worst GPU's pressure, averaged
over 50 fixed training cases:
**lower `mean_max_kvpr` is better.** For scale: PRISM greedy scores 0.04787 and
the best released ADRS program scores 0.04046.
"""


# The agent: 15 steps, may edit only candidate.py and run only `prism-check`.
OPENCODE_CONFIG = """{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "prism-evolver": {
      "description": "Improve the PRISM placement function in candidate.py",
      "mode": "primary",
      "steps": 15,
      "temperature": 0,
      "top_p": 1,
      "options": { "seed": 20260921, "reasoning": { "max_tokens": 8000 } },
      "prompt": "{file:./agent_prompt.md}",
      "permission": {
        "read": "allow",
        "edit": { "*": "deny", "**/candidate.py": "allow" },
        "bash": { "*": "deny", "prism-check": "allow" },
        "glob": "deny",
        "grep": "deny",
        "list": "deny",
        "task": "deny",
        "todowrite": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "skill": "deny",
        "question": "deny",
        "external_directory": "deny"
      }
    }
  }
}
"""

AGENT_PROMPT = """You are the mutation step of a Meta-Evolve search over PRISM placement programs.

Everything you need is in the message: the task, the current `candidate.py`,
and what earlier trials in this search tried and scored. The evaluator's code is
deliberately out of reach; the `prism-check` command is its exact stand-in. Do
not explore files, the environment, or directories: the workspace holds only
`candidate.py`, and every other command is refused.

1. Pick one concrete algorithmic idea. Build on what scored well; do not repeat
   an idea the memory shows already failed or tied.
2. Edit `candidate.py` (your first tool call).
3. Run exactly `prism-check`. It runs `candidate.py` in the same sandbox and
   on the same cases as the real evaluator, and prints its score or the error.
4. If it fails or scores worse than the current program, fix it or try again.
   Stop once you have a valid improvement or have used your steps.

Only the Python standard library is available. Keep it deterministic and fast
(well under a second per case).

Finish with one sentence: <hypothesis>what you changed and why it should lower
mean_max_kvpr</hypothesis>
"""

PRISM_CHECK = '''# prism-check: the agent's only command. It scores candidate.py exactly as the
# evaluator will, and keeps the best version that passed next to (not inside) the
# workspace, so an untested last edit cannot throw away a checked improvement.
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["PRISM_TASK_ROOT"])
from prism import DockerSandbox, evaluate  # noqa: E402

source = Path("candidate.py").read_text()
result = evaluate(source, sandbox=DockerSandbox())
if result.failure is not None:
    print(f"FAILED ({result.failure.kind}): {result.failure.message}")
    raise SystemExit(1)
print(json.dumps(dict(result.metrics), indent=2))

best = Path.cwd().parent / f"{Path.cwd().name}.best.json"
score = result.metrics["mean_max_kvpr"]
if not best.exists() or score < json.loads(best.read_text())["mean_max_kvpr"]:
    best.write_text(json.dumps({"mean_max_kvpr": score, "source": source}))
'''

class OpenCodeAgent:
    """Use as ``with OpenCodeAgent(workdir) as agent:`` and pass ``agent`` as the proposer."""

    def __init__(self, workdir, *, model=MODEL, timeout_seconds=900):
        self.workdir = Path(workdir).resolve()
        self.provider, self.model = model.split("/", 1)
        self.timeout_seconds = timeout_seconds
        self.server = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.server is not None:
            self.server.terminate()
            self.server.wait(timeout=10)

    def _start_server(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        # OpenCode finds its config by walking up from each trial's workspace.
        (self.workdir / "opencode.json").write_text(OPENCODE_CONFIG)
        (self.workdir / "agent_prompt.md").write_text(AGENT_PROMPT)
        # `prism-check` lives outside every workspace, so nothing the agent can read
        # points at the evaluator.
        checker = self.workdir / "bin" / "prism-check"
        checker.parent.mkdir(exist_ok=True)
        checker.write_text(f"#!{sys.executable}\n" + PRISM_CHECK)
        checker.chmod(0o755)
        env = {
            **os.environ,
            # A private data/config home keeps the user's own OpenCode settings out.
            "XDG_DATA_HOME": str(self.workdir / "opencode" / "data"),
            "XDG_CONFIG_HOME": str(self.workdir / "opencode" / "config"),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "PRISM_TASK_ROOT": str(TASK_DIR),
            "PATH": os.pathsep.join((str(checker.parent), os.environ["PATH"])),
        }
        opencode = shutil.which("opencode") or str(Path.home() / ".opencode" / "bin" / "opencode")
        self.server = subprocess.Popen((opencode, "serve", "--pure", "--port", "0"), cwd=self.workdir,
                                       env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in self.server.stdout:  # "opencode server listening on http://..."
            if "listening on" in line:
                url = line.rsplit(" ", 1)[-1].strip()
                break
        else:
            raise RuntimeError("opencode serve exited before it started listening")
        self.client = Opencode(base_url=url, timeout=self.timeout_seconds, max_retries=0)

    def __call__(self, parent, *, context):
        parent = str(parent)
        prompt = (
            f"{TASK}\n## Current program (candidate.py)\n```python\n{parent}```\n\n"
            f"## What this Meta-Evolve search remembers\n{remember(context)}\n"
        )
        if self.server is None:
            self._start_server()
        candidate, report, wall = self._ask_agent(parent, prompt, context.step)
        usage = meta.Usage(tokens=report["tokens"], wall_seconds=wall,
                           spend_micros=math.ceil(report["cost_usd"] * 1_000_000))
        evidence = (meta.EvidenceDraft(kind="opencode-session", data=report),)
        if candidate.strip() == parent.strip():
            failure = (InfrastructureFailure(message=report["error"]) if report["error"]
                       else InvalidOutput(message="the agent left candidate.py unchanged"))
            return meta.ProposalResult(failure=failure, usage=usage, evidence=evidence)
        return meta.ProposalResult(candidate=meta.Text(candidate), hypothesis=report["hypothesis"],
                                   usage=usage, evidence=evidence)

    def _ask_agent(self, parent, prompt, step):
        workspace = self.workdir / "trials" / f"trial_{step:02d}"
        best = workspace.with_name(f"{workspace.name}.best.json")  # written by prism-check
        shutil.rmtree(workspace, ignore_errors=True)
        best.unlink(missing_ok=True)
        workspace.mkdir(parents=True)
        (workspace / "candidate.py").write_text(parent)

        scope = {"directory": str(workspace)}
        started = time.monotonic()
        session = self.client.session.create(extra_query=scope)
        error = None
        try:
            # The SDK's typed `session.chat` predates the server's message schema,
            # so post the current shape through the same client.
            self.client.post(
                f"/session/{session.id}/message", cast_to=httpx.Response,
                body={"agent": "prism-evolver",
                      "model": {"providerID": self.provider, "modelID": self.model},
                      "parts": [{"type": "text", "text": prompt}]},
                options={"params": scope},
            )
        except APITimeoutError:
            error = f"agent exceeded {self.timeout_seconds:.0f}s"
            self.client.session.abort(session.id, extra_query=scope)
        except APIStatusError as failure:
            error = f"OpenCode server error {failure.status_code}: {failure.message}"

        transcript = self.client.get(f"/session/{session.id}/message", cast_to=httpx.Response,
                                     options={"params": scope}).json()
        assistant = [message for message in transcript if message["info"]["role"] == "assistant"]
        texts = [part["text"] for message in assistant for part in message["parts"]
                 if part["type"] == "text" and part.get("text")]
        found = HYPOTHESIS.findall("\n".join(texts))
        report = {
            "tokens": sum(message["info"].get("tokens", {}).get("total", 0) for message in assistant),
            "cost_usd": sum(message["info"].get("cost", 0.0) for message in assistant),
            "hypothesis": " ".join(found[-1].split())[:500] if found else None,
            "error": error,
        }
        candidate = (json.loads(best.read_text())["source"] if best.exists()
                     else (workspace / "candidate.py").read_text())
        return candidate, report, time.monotonic() - started


def remember(context) -> str:
    """Render what Meta-Evolve supplied: the parent's lineage, then one audited pull."""
    lineage = {record.logical_step for record in context.bundle.records}
    records = list(context.bundle.records)
    if context.experience is not None:
        records += context.experience.search(record_kinds=("proposal", "evaluation"), limit=12).records
    trials: dict[int, dict[str, str]] = {}
    for record in records:
        trial = trials.setdefault(record.logical_step, {})
        if record.kind == "proposal" and record.value.hypothesis:
            trial["idea"] = record.value.hypothesis
        elif record.kind == "evaluation":
            failure = record.value.failure
            trial["result"] = (f"failed ({failure.kind}): {failure.message[:200]}" if failure else
                               f"mean_max_kvpr={record.value.metrics['mean_max_kvpr']:.6f}")

    def line(step: int) -> str:
        trial = trials[step]
        idea = "the seed program" if step == 0 else trial.get("idea", "no hypothesis recorded")
        return f"- trial {step}: {idea} -> {trial.get('result', 'not evaluated')}"

    ours = [line(step) for step in sorted(trials) if step in lineage]
    others = [line(step) for step in sorted(trials) if step not in lineage]
    return "\n".join([
        "Lineage of the current program (its own score is the last line):",
        *(ours or ["- (none)"]),
        "Other trials elsewhere in the search:",
        *(others or ["- (none yet)"]),
    ])

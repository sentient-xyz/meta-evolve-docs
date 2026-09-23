"""Score one Evolved program on ADRS llm_sql. Prints one JSON object to stdout."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

PROGRAM_PATH = os.path.abspath(sys.argv[1])
TASK_DIR = os.environ["ADRS_TASK_DIR"]
sys.path.insert(0, TASK_DIR)
os.chdir(TASK_DIR)

import pandas as pd  # noqa: E402
from utils import evaluate_df_prefix_hit_cnt  # noqa: E402

TEST_FILES = [
    "datasets/movies.csv",
    "datasets/beer.csv",
    "datasets/BIRD.csv",
    "datasets/PDMX.csv",
    "datasets/products.csv",
]
COL_MERGES = [
    [["movieinfo", "movietitle", "rottentomatoeslink"]],
    [["beer/beerId", "beer/name"]],
    [["PostId", "Body"]],
    [
        ["path", "metadata"],
        ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"],
    ],
    [["product_title", "parent_asin"]],
]
REORDER_KWARGS = dict(
    early_stop=100000, distinct_value_threshold=0.7, row_stop=4, col_stop=2
)
MAX_CHAR_SHRINK = 0.01


def total_chars(frame):
    return int(frame.astype(str).map(len).to_numpy().sum())


def load_evolved(path):
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evolved


def main() -> int:
    try:
        evolved = load_evolved(PROGRAM_PATH)
    except Exception as error:
        print(json.dumps({"ok": False, "stage": "import", "error": f"{type(error).__name__}: {error}"}))
        return 0

    per_dataset = []
    for filename, col_merge in zip(TEST_FILES, COL_MERGES):
        frame = pd.read_csv(filename)
        before_chars = total_chars(frame)
        before_rows = len(frame)
        try:
            start = time.time()
            reordered, _ = evolved().reorder(frame, col_merge=col_merge, **REORDER_KWARGS)
            runtime = time.time() - start
        except Exception as error:
            print(json.dumps({
                "ok": False, "stage": "run", "dataset": filename,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }))
            return 0
        after_chars = total_chars(reordered)
        if len(reordered) != before_rows:
            print(json.dumps({"ok": False, "stage": "invalid", "error": "row count changed"}))
            return 0
        shrink = (before_chars - after_chars) / before_chars if before_chars else 0.0
        if shrink > MAX_CHAR_SHRINK:
            print(json.dumps({
                "ok": False, "stage": "invalid",
                "error": f"serialized text shrank by {shrink * 100:.2f}%",
            }))
            return 0
        hit_count, hit_pct = evaluate_df_prefix_hit_cnt(reordered)
        per_dataset.append({
            "dataset": os.path.basename(filename),
            "hit_rate": hit_pct / 100.0,
            "runtime": runtime,
            "hit_count": hit_count,
        })

    hit_rates = [row["hit_rate"] for row in per_dataset]
    runtimes = [row["runtime"] for row in per_dataset]
    average_hit_rate = sum(hit_rates) / len(hit_rates)
    average_runtime = sum(runtimes) / len(runtimes)
    combined = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12
    print(json.dumps({
        "ok": True,
        "combined_score": combined,
        "average_hit_rate": average_hit_rate,
        "average_runtime": average_runtime,
        "per_dataset": per_dataset,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

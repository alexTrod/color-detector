#!/usr/bin/env python3
"""Run N color train_labram experiments with different configs; log to results.tsv."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_TSV = ROOT / "results.tsv"
INDEX_COLOR = ROOT / "data/manifests/epoch_index_color.csv"
METRICS_OUT = ROOT / "runs/labram_metrics_color_combined.json"
PY = ROOT / ".venv/bin/python"
TRAIN_SCRIPT = ROOT / "scripts/train_labram.py"


def get_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def run_one(label_grouping: str, feature_set: str, probe_c: float, n_splits: int) -> float | None:
    cmd = [
        str(PY),
        str(TRAIN_SCRIPT),
        "--epoch-index",
        str(INDEX_COLOR),
        "--out-metrics",
        str(METRICS_OUT),
        "--epochs",
        "5",
        "--batch-size",
        "16",
        "--label-grouping",
        label_grouping,
        "--feature-set",
        feature_set,
        "--probe-c",
        str(probe_c),
        "--n-splits",
        str(n_splits),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    # Parse macro_f1 from last line like "Macro F1 (mean +/- std): 0.3180 +/- 0.0067"
    for line in result.stdout.splitlines():
        if "Macro F1" in line and "+/-" in line:
            try:
                return float(line.split(":")[1].split("+/-")[0].strip())
            except (IndexError, ValueError):
                pass
    # Fallback: read from written JSON
    if METRICS_OUT.exists():
        import json
        data = json.loads(METRICS_OUT.read_text())
        return float(data.get("macro_f1", 0))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100)
    args = parser.parse_args()

    if not INDEX_COLOR.exists():
        print(f"Missing {INDEX_COLOR}", file=sys.stderr)
        return 1

    # Config space
    labels = ["basic_4", "hue_8", "hue_6", "hue_12"]
    features = ["erp", "labram_plus_erp", "labram"]
    probe_cs = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2]
    n_splits_list = [5, 10]

    import itertools
    configs = list(itertools.product(labels, features, probe_cs, n_splits_list))
    # Shuffle and take first N so we get variety
    import random
    random.seed(42)
    random.shuffle(configs)
    configs = configs[: args.num]

    best_f1 = 0.0
    commit = get_commit()
    for i, (lg, fs, pc, ns) in enumerate(configs):
        desc = f"color {lg} {fs} C={pc} n_splits={ns}"
        print(f"[{i+1}/{args.num}] {desc} ...", flush=True)
        f1 = run_one(lg, fs, pc, ns)
        if f1 is None:
            status = "crash"
            f1 = 0.0
            mem = 0.0
        else:
            mem = 0.0
            status = "keep" if f1 > best_f1 else "discard"
            if f1 > best_f1:
                best_f1 = f1
        line = f"{commit}\t{f1:.4f}\t{mem:.1f}\t{status}\t{desc}\n"
        if not RESULTS_TSV.exists():
            RESULTS_TSV.write_text("commit\tmacro_f1\tmemory_gb\tstatus\tdescription\n")
        with RESULTS_TSV.open("a") as f:
            f.write(line)
        print(f"  macro_f1={f1:.4f} {status} (best={best_f1:.4f})", flush=True)
    print(f"Done. Best macro_f1={best_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

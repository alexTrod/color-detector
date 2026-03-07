#!/usr/bin/env python3
"""Load four Labram metrics JSONs and print comparison table. Optional: bar chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

DEFAULT_RUNS = {
    "raw": "labram_metrics_raw.json",
    "raw_zuna": "labram_metrics_raw_zuna.json",
    "derivatives": "labram_metrics_derivatives.json",
    "derivatives_zuna": "labram_metrics_derivatives_zuna.json",
}


def compare_labram_runs(
    runs_dir: str | Path = RUNS_DIR,
    names: list[str] | None = None,
    files: list[str] | None = None,
    plot: bool = False,
) -> list[dict]:
    runs_dir = Path(runs_dir)
    names = names or list(DEFAULT_RUNS.keys())
    files = files or list(DEFAULT_RUNS.values())
    rows = []
    for name, fname in zip(names, files):
        path = runs_dir / fname
        if not path.exists():
            rows.append({"pipeline": name, "accuracy": None, "n_train": None, "n_test": None})
            continue
        data = json.loads(path.read_text())
        rows.append({
            "pipeline": name,
            "accuracy": data.get("accuracy"),
            "n_train": data.get("n_train"),
            "n_test": data.get("n_test"),
            "classes": data.get("label_classes"),
        })
    print("Pipeline\tAccuracy\tN_train\tN_test")
    for r in rows:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        print(f"{r['pipeline']}\t{acc}\t{r.get('n_train', '')}\t{r.get('n_test', '')}")
    if plot and rows:
        try:
            import matplotlib.pyplot as plt
            names = [r["pipeline"] for r in rows]
            accs = [r["accuracy"] if r.get("accuracy") is not None else 0 for r in rows]
            plt.bar(names, accs)
            plt.ylabel("Accuracy")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(runs_dir / "labram_comparison.png")
            print(f"Saved {runs_dir / 'labram_comparison.png'}")
        except ImportError:
            print("matplotlib not available; skip --plot")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    parser.add_argument("--names", type=str, nargs="+", default=list(DEFAULT_RUNS.keys()))
    parser.add_argument("--files", type=str, nargs="+", default=list(DEFAULT_RUNS.values()))
    parser.add_argument("--plot", action="store_true", help="Show bar chart (requires matplotlib)")
    args = parser.parse_args()
    compare_labram_runs(
        runs_dir=args.runs_dir,
        names=args.names,
        files=args.files,
        plot=args.plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

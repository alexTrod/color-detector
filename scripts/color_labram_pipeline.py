#!/usr/bin/env python3
"""Run color EEG pipeline: download -> preprocess -> train Labram per-dataset and combined -> compare."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGE_ORDER = [
    "download",
    "manifest",
    "preprocess",
    "train_per_dataset",
    "train_combined",
    "compare",
]


def run_cmd(cmd: list[str], root: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def resolve_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (root / "configs").exists():
        return root
    return Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(description="Color EEG -> Labram pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", *STAGE_ORDER],
        help="Stages to run",
    )
    parser.add_argument("--color-max-subjects", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--label-grouping",
        type=str,
        default="basic_4",
        choices=("none", "hue_12", "hue_8", "hue_6", "basic_4"),
        help="Coarse label grouping (basic_4=best for combined color; none=raw 969 classes)",
    )
    parser.add_argument("--feature-set", type=str, default="labram_plus_erp", choices=("erp", "labram", "labram_plus_erp"))
    parser.add_argument("--probe-c", type=float, default=0.005)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    root = resolve_root()
    steps = STAGE_ORDER if "all" in args.steps else [s for s in STAGE_ORDER if s in args.steps]
    manifests = root / "data" / "manifests"
    runs = root / "runs"
    epoch_index_color = manifests / "epoch_index_color.csv"

    for stage in steps:
        if stage == "download":
            run_cmd(
                [
                    sys.executable,
                    "scripts/download_sample_subjects.py",
                    "--color",
                    "--color-max-subjects",
                    str(args.color_max_subjects),
                ],
                root,
            )
        elif stage == "manifest":
            run_cmd([sys.executable, "scripts/build_unified_manifest.py"], root)
        elif stage == "preprocess":
            run_cmd([sys.executable, "scripts/preprocess_color.py"], root)
        elif stage == "train_per_dataset":
            if not epoch_index_color.exists():
                print(f"Skip train_per_dataset: {epoch_index_color} not found")
            else:
                import pandas as pd
                df = pd.read_csv(epoch_index_color)
                # Build per-dataset indices from current full index
                for dataset_id in df["dataset_id"].unique():
                    index_one = manifests / f"epoch_index_color_{dataset_id}.csv"
                    df[df["dataset_id"] == dataset_id].to_csv(index_one, index=False)
                for dataset_id in df["dataset_id"].unique():
                    index_one = manifests / f"epoch_index_color_{dataset_id}.csv"
                    out_metrics = runs / f"labram_metrics_color_{dataset_id}.json"
                    run_cmd(
                        [
                            sys.executable,
                            "scripts/train_labram.py",
                            "--epoch-index",
                            str(index_one),
                            "--out-metrics",
                            str(out_metrics),
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                        ],
                        root,
                    )
        elif stage == "train_combined":
            if not epoch_index_color.exists():
                print(f"Skip train_combined: {epoch_index_color} not found")
            else:
                run_cmd(
                    [
                        sys.executable,
                        "scripts/train_labram.py",
                        "--epoch-index",
                        str(epoch_index_color),
                        "--out-metrics",
                        str(runs / "labram_metrics_color_combined.json"),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--label-grouping",
                        args.label_grouping,
                        "--feature-set",
                        args.feature_set,
                        "--probe-c",
                        str(args.probe_c),
                        "--n-splits",
                        str(args.n_splits),
                    ],
                    root,
                )
        elif stage == "compare":
            names, files = [], []
            for name, fname in [
                ("color_hajonides", "labram_metrics_color_hajonides_j289e.json"),
                ("color_baeluck", "labram_metrics_color_baeluck_jnwut.json"),
                ("color_chauhan", "labram_metrics_color_chauhan_v9ewj.json"),
                ("color_combined", "labram_metrics_color_combined.json"),
            ]:
                if (runs / fname).exists():
                    names.append(name)
                    files.append(fname)
            if names:
                run_cmd(
                    [sys.executable, "scripts/compare_labram_runs.py", "--runs-dir", str(runs), "--names", *names, "--files", *files, "--plot"],
                    root,
                )
            else:
                print("No color metrics JSONs found; skip compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

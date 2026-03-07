#!/usr/bin/env python3
"""Run YOTO -> Zuna -> Labram pipeline stages from CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGE_ORDER = [
    "download",
    "manifest",
    "preprocess",
    "export_fif",
    "zuna",
    "build_zuna_index",
    "train",
    "compare",
]


def run_cmd(cmd: list[str], root: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def resolve_root() -> Path:
    root = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    return root


def resolve_stages(requested: list[str]) -> list[str]:
    if "all" in requested:
        return STAGE_ORDER
    wanted = set(requested)
    return [stage for stage in STAGE_ORDER if stage in wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configurable stages from yoto_labram_pipeline notebook."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["all", *STAGE_ORDER],
        help="Stages to run (default: all). Example: --steps preprocess export_fif zuna",
    )
    parser.add_argument(
        "--subject-id",
        default="",
        help='Optional subject filter for Zuna stage, e.g. "sub-01".',
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=8,
        help="Diffusion sampling steps for Zuna inference.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for Labram stage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Labram stage.",
    )
    parser.add_argument(
        "--run-asr",
        action="store_true",
        help="Enable ASR in preprocessing/export stages (default notebook behavior skips it).",
    )
    parser.add_argument(
        "--run-ica",
        action="store_true",
        help="Enable ICA in preprocessing/export stages (default notebook behavior skips it).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_root()

    data_root = root / "data"
    manifests = data_root / "manifests"
    runs = root / "runs"
    zuna_work = runs / "zuna_yoto"

    stages = resolve_stages(args.steps)
    print("Selected stages:", ", ".join(stages))

    for stage in stages:
        if stage == "download":
            run_cmd(
                [
                    sys.executable,
                    "scripts/download_sample_subjects.py",
                    "--yoto-five",
                    "--skip-other",
                ],
                root,
            )

        elif stage == "manifest":
            run_cmd([sys.executable, "scripts/build_unified_manifest.py"], root)

        elif stage == "preprocess":
            cmd = [sys.executable, "scripts/preprocess_yoto.py"]
            if not args.run_asr:
                cmd.append("--skip-asr")
            if not args.run_ica:
                cmd.append("--skip-ica")
            run_cmd(cmd, root)

        elif stage == "export_fif":
            cmd = [sys.executable, "scripts/export_yoto_to_fif.py"]
            if not args.run_asr:
                cmd.append("--skip-asr")
            if not args.run_ica:
                cmd.append("--skip-ica")
            run_cmd(cmd, root)

        elif stage == "zuna":
            zuna_work.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/run_zuna_augmentation.py",
                "--input-fif-dir",
                str(root / "data/processed/fif_for_zuna"),
                "--work-dir",
                str(zuna_work),
                "--diffusion-steps",
                str(args.diffusion_steps),
            ]
            if args.subject_id:
                cmd += ["--subject-id", args.subject_id]
            run_cmd(cmd, root)

        elif stage == "build_zuna_index":
            run_cmd(
                [
                    sys.executable,
                    "scripts/build_zuna_augmented_index_yoto.py",
                    "--zuna-fif-dir",
                    str(zuna_work / "4_fif_output"),
                ],
                root,
            )

        elif stage == "train":
            index_raw = manifests / "epoch_index_yoto_tones.csv"
            index_zuna = manifests / "epoch_index_yoto_tones_zuna.csv"
            for name, index, out in [
                ("raw", index_raw, runs / "labram_metrics_raw.json"),
                ("raw_zuna", index_zuna, runs / "labram_metrics_raw_zuna.json"),
            ]:
                if not index.exists():
                    print(f"Skip {name}: {index} not found")
                    continue
                run_cmd(
                    [
                        sys.executable,
                        "scripts/train_labram.py",
                        "--epoch-index",
                        str(index),
                        "--out-metrics",
                        str(out),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                    ],
                    root,
                )

        elif stage == "compare":
            run_cmd([sys.executable, "scripts/compare_labram_runs.py", "--plot"], root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

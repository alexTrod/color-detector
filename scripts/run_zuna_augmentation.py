#!/usr/bin/env python3
"""Run ZUNA augmentation/inference on FIF exports."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import zuna


def run_zuna_augmentation(
    input_fif_dir: Path,
    work_dir: Path,
    subject_id: str = "",
    gpu_device: str = "0",
    tokens_per_batch: int = 2048,
    diffusion_steps: int = 8,
) -> dict:
    work = Path(work_dir)
    d1 = work / "1_fif_filter"
    d2 = work / "2_pt_input"
    d3 = work / "3_pt_output"
    d4 = work / "4_fif_output"
    fig = work / "FIGURES"
    for d in [d1, d2, d3, d4, fig]:
        d.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_fif_dir)
    with tempfile.TemporaryDirectory(prefix="zuna_input_") as tmp_input_dir:
        zuna_input_dir = input_dir
        selected = None
        if subject_id:
            matches = sorted(input_dir.glob(f"*{subject_id}*.fif"))
            if not matches:
                raise SystemExit(
                    f"No FIF files found in {input_dir} matching subject filter: {subject_id}"
                )
            zuna_input_dir = Path(tmp_input_dir)
            for src in matches:
                (zuna_input_dir / src.name).symlink_to(src.resolve())
            selected = len(matches)
            print(
                f"Subject filter {subject_id}: selected {len(matches)} file(s) for ZUNA input."
            )

        zuna.preprocessing(
            input_dir=str(zuna_input_dir),
            output_dir=str(d2),
            apply_notch_filter=False,
            apply_highpass_filter=True,
            apply_average_reference=True,
            target_channel_count=["AF3", "AF4", "F1", "F2", "C3", "C4", "P3", "P4"],
            bad_channels=[],
            preprocessed_fif_dir=str(d1),
        )
    zuna.inference(
        input_dir=str(d2),
        output_dir=str(d3),
        gpu_device=gpu_device,
        tokens_per_batch=tokens_per_batch,
        data_norm=10.0,
        diffusion_cfg=1.0,
        diffusion_sample_steps=diffusion_steps,
        plot_eeg_signal_samples=False,
        inference_figures_dir=str(fig),
    )
    zuna.pt_to_fif(input_dir=str(d3), output_dir=str(d4))
    print(f"ZUNA outputs written to {d4}")
    return {
        "input_dir": str(input_dir),
        "work_dir": str(work_dir),
        "subject_filter": subject_id or None,
        "selected_files": selected,
        "out_dir": str(d4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ZUNA preprocessing+inference+reconstruction pipeline.")
    parser.add_argument("--input-fif-dir", required=True, help="Directory with input .fif files")
    parser.add_argument("--work-dir", required=True, help="ZUNA working directory")
    parser.add_argument(
        "--subject-id",
        default="",
        help="Optional subject filter (e.g. sub-01). Only matching FIF files are processed.",
    )
    parser.add_argument("--gpu-device", default="0", help='GPU index, or "" for CPU')
    parser.add_argument("--tokens-per-batch", type=int, default=2048, help="Lower this on memory-constrained machines")
    parser.add_argument("--diffusion-steps", type=int, default=8, help="Fewer steps for smoke tests")
    args = parser.parse_args()

    run_zuna_augmentation(
        input_fif_dir=Path(args.input_fif_dir),
        work_dir=Path(args.work_dir),
        subject_id=args.subject_id,
        gpu_device=args.gpu_device,
        tokens_per_batch=args.tokens_per_batch,
        diffusion_steps=args.diffusion_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

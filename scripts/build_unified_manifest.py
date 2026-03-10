#!/usr/bin/env python3
"""Build a harmonized manifest from downloaded sample datasets."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw_samples"
OUT_JSON = ROOT / "data/manifests/unified_manifest.json"
OUT_CSV = ROOT / "data/manifests/unified_manifest.csv"

EEG_EXTS = {".vhdr", ".eeg", ".vmrk", ".edf", ".bdf", ".set", ".fif", ".mat"}


def infer_subject(path: Path) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(path))
    return m.group(1) if m else "unknown_subject"


def infer_task_entity(path: Path) -> str:
    m = re.search(r"task-([A-Za-z0-9]+)", path.name, flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""


def infer_modality(path: Path) -> str:
    name = path.name.lower()
    task = infer_task_entity(path)

    # Explicit BIDS task mapping for ERP CORE-like naming.
    if task in {"mmn", "auditory"}:
        return "auditory"
    if task in {"n170", "n2pc", "p3", "p3b", "p300", "n400", "ern", "flanker", "visual"}:
        return "visual"

    if "audiovisual" in name or re.search(r"(^|[_-])av([_-]|$)", name):
        return "audiovisual"
    if any(k in name for k in ["audio", "auditory", "oddball", "tone", "vowel"]):
        return "auditory"
    if any(k in name for k in ["visual", "color", "colour", "rgb", "gabor", "face"]):
        return "visual"
    return "unknown"


def infer_task_label(path: Path) -> str:
    name = path.name.lower()
    task = infer_task_entity(path)
    if task:
        return task
    if "rest" in name:
        return "rest"
    if "oddball" in name:
        return "oddball"
    if "task" in name:
        return "task"
    if "rgb" in name:
        return "rgb"
    if "color" in name or "colour" in name:
        return "color"
    return "unknown"


def build_manifest() -> list[dict]:
    rows = []
    for p in RAW_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EEG_EXTS:
            continue
        dataset = p.relative_to(RAW_ROOT).parts[0]
        rows.append(
            {
                "dataset_id": dataset,
                "subject_id": infer_subject(p),
                "file_path": str(p),
                "file_ext": p.suffix.lower(),
                "modality": infer_modality(p),
                "task_label": infer_task_label(p),
                "has_events_sidecar": (
                    p.with_suffix(".tsv").exists()
                    or p.with_suffix(".json").exists()
                    or p.with_name(p.name.replace("_eeg", "_events")).with_suffix(".tsv").exists()
                ),
            }
        )
    return rows


def write_manifest(
    rows: list[dict],
    out_json: Path = OUT_JSON,
    out_csv: Path = OUT_CSV,
) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} records to {out_json} and {out_csv}")
    return {"n_rows": len(rows), "out_json": str(out_json), "out_csv": str(out_csv)}


def main() -> int:
    rows = build_manifest()
    write_manifest(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

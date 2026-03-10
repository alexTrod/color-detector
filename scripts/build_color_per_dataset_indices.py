#!/usr/bin/env python3
"""Build per-dataset epoch index CSVs from epoch_index_color.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"
INDEX = MANIFESTS / "epoch_index_color.csv"


def main() -> int:
    if not INDEX.exists():
        print(f"Missing {INDEX}")
        return 1
    df = pd.read_csv(INDEX)
    for dataset_id in df["dataset_id"].unique():
        out = MANIFESTS / f"epoch_index_color_{dataset_id}.csv"
        sub = df[df["dataset_id"] == dataset_id]
        sub.to_csv(out, index=False)
        print(f"Wrote {out.name} ({len(sub)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

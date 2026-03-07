#!/usr/bin/env python3
"""Shared helpers for YOTO preprocessing and Zuna indexing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def get_event_mapping(config_events: dict[str, Any]) -> dict[int | str, str]:
    out: dict[int | str, str] = {}
    for k, v in config_events.get("event_value_to_stimulus", {}).items():
        out[int(k)] = str(v)
    for k, v in config_events.get("trial_type_to_stimulus", {}).items():
        out[str(k)] = str(v)
    return out


def load_events_tsv(vhdr_path: Path) -> pd.DataFrame | None:
    stem = vhdr_path.stem.replace("_eeg", "")
    tsv = vhdr_path.parent / (stem + "_events.tsv")
    if not tsv.exists():
        tsv = vhdr_path.with_suffix(".tsv")
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep="\t")


def extract_tone_onsets(
    events_df: pd.DataFrame,
    event_mapping: dict[int | str, str],
    onset_col: str = "onset",
    value_col: str = "value",
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for _, row in events_df.iterrows():
        stim_id = None
        if value_col in events_df.columns and pd.notna(row.get(value_col)):
            try:
                v = int(float(row[value_col]))
                stim_id = event_mapping.get(v)
            except (ValueError, TypeError):
                pass
        if stim_id not in ("tone_C", "tone_D", "tone_E"):
            continue
        out.append((float(row[onset_col]), stim_id))
    return out

#!/usr/bin/env python3
"""Shared helpers for YOTO-style preprocessing and epoching."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
import numpy as np


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def get_event_mapping(config_events: dict) -> dict[int | str, str]:
    mapping: dict[int | str, str] = {}
    for k, v in config_events.get("event_value_to_stimulus", {}).items():
        mapping[int(k)] = str(v)
    for k, v in config_events.get("trial_type_to_stimulus", {}).items():
        mapping[str(k)] = str(v)
    return mapping


def load_events_tsv(vhdr_path: Path) -> pd.DataFrame | None:
    stem = vhdr_path.stem.replace("_eeg", "")
    tsv = vhdr_path.parent / (stem + "_events.tsv")
    if not tsv.exists():
        tsv = vhdr_path.with_suffix(".tsv")
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep="\t", encoding="utf-8", encoding_errors="replace")


def extract_tone_onsets(
    events_df: pd.DataFrame,
    event_mapping: dict,
    onset_col: str = "onset",
    value_col: str = "value",
    trial_type_col: str = "trial_type",
    onset_offset_seconds: float = 0.0,
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
        if stim_id is None and trial_type_col in events_df.columns:
            stim_id = event_mapping.get(str(row[trial_type_col]).strip())
        if stim_id not in ("tone_C", "tone_D", "tone_E"):
            continue
        out.append((float(row[onset_col]) + float(onset_offset_seconds), stim_id))
    out.sort(key=lambda item: item[0])
    return out


def _parse_subject_session(vhdr_path: Path) -> tuple[str | None, int | None]:
    subject_id = next((part for part in vhdr_path.parts if part.startswith("sub-")), None)
    session = next((part for part in vhdr_path.parts if part.startswith("ses-")), None)
    if session is None:
        return subject_id, None
    digits = "".join(ch for ch in session if ch.isdigit())
    if not digits:
        return subject_id, None
    return subject_id, int(digits)


def load_derivative_tone_onsets(
    vhdr_path: Path,
    event_mapping: dict,
    onset_offset_seconds: float = 0.0,
) -> list[tuple[float, str]]:
    """Load tone onsets from derivatives/*/task_event.mat if present."""
    try:
        from scipy.io import loadmat
    except Exception:  # noqa: BLE001
        return []

    subject_id, session_num = _parse_subject_session(vhdr_path)
    if subject_id is None or session_num is None:
        return []

    # .../ds005815/sub-XX/ses-YY/eeg/file.vhdr -> .../ds005815
    try:
        dataset_root = vhdr_path.parents[3]
    except IndexError:
        return []

    derivatives_root = dataset_root / "derivatives" / subject_id
    candidates = [
        derivatives_root / f"ses-{session_num}" / "task_event.mat",
        derivatives_root / f"ses-{session_num:02d}" / "task_event.mat",
    ]
    mat_path = next((path for path in candidates if path.exists()), None)
    if mat_path is None:
        return []

    try:
        payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        eprime = payload.get("Eprime_event")
        if eprime is None:
            return []
        event_types = np.array(eprime.type).ravel()
        event_latencies = np.array(eprime.latency).ravel().astype(float)
    except Exception:  # noqa: BLE001
        return []

    out: list[tuple[float, str]] = []
    for event_type, latency_ms in zip(event_types, event_latencies):
        stim_id = None
        try:
            stim_id = event_mapping.get(int(event_type))
        except Exception:  # noqa: BLE001
            stim_id = event_mapping.get(str(event_type).strip())
        if stim_id not in ("tone_C", "tone_D", "tone_E"):
            continue
        onset_sec = (float(latency_ms) / 1000.0) + float(onset_offset_seconds)
        out.append((onset_sec, stim_id))
    out.sort(key=lambda item: item[0])
    return out


def estimate_onset_shift_seconds(
    raw_onsets: list[tuple[float, str]],
    derivative_onsets: list[tuple[float, str]],
) -> float | None:
    """Estimate global time shift as median derivative_raw onset delta."""
    if not raw_onsets or not derivative_onsets:
        return None

    # Keep labels aligned in-order; YOTO files have balanced fixed class counts.
    deltas: list[float] = []
    n_pairs = min(len(raw_onsets), len(derivative_onsets))
    for idx in range(n_pairs):
        raw_t, raw_label = raw_onsets[idx]
        der_t, der_label = derivative_onsets[idx]
        if raw_label != der_label:
            continue
        deltas.append(float(der_t - raw_t))
    if not deltas:
        return None
    return float(np.median(np.array(deltas, dtype=np.float64)))

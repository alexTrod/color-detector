#!/usr/bin/env python3
"""Compare processed epoch labels to derivative task_event.mat tone order for YOTO.
When derivatives exist and prefer_derivatives is used, epoch i should match derivative tone list position i.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from scripts.yoto_utils import (
        get_event_mapping,
        load_config,
        load_derivative_tone_onsets,
        load_events_tsv,
        extract_tone_onsets,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "scripts"))
    from yoto_utils import (
        get_event_mapping,
        load_config,
        load_derivative_tone_onsets,
        load_events_tsv,
        extract_tone_onsets,
    )


def main() -> int:
    config_events = load_config(ROOT / "configs" / "yoto_events.yaml")
    mapping = get_event_mapping(config_events)
    index_path = ROOT / "data/manifests/epoch_index_yoto_tones.csv"
    df = pd.read_csv(index_path)

    # One vhdr that has derivatives: sub-01 ses-1
    vhdr = ROOT / "data/raw_samples/ds005815/sub-01/ses-1/eeg/sub-01_ses-1_task-task_eeg.vhdr"
    if not vhdr.exists():
        print("Sample vhdr not found:", vhdr)
        return 1

    derivative_onsets = load_derivative_tone_onsets(vhdr, mapping, 0.0)
    raw_events = load_events_tsv(vhdr)
    raw_onsets = extract_tone_onsets(raw_events, mapping, onset_offset_seconds=0.0) if raw_events is not None else []

    rows = df[df["source_file"].astype(str).str.endswith(vhdr.name)]
    if rows.empty:
        print("No epoch index rows for", vhdr.name)
        return 1

    rows = rows.sort_values("epoch_idx")
    index_labels = rows["stimulus_id"].tolist()
    n_epochs = len(index_labels)

    print(f"Source: {vhdr.name}")
    print(f"Derivative tone count: {len(derivative_onsets)}")
    print(f"Raw tone count: {len(raw_onsets)}")
    print(f"Epoch index rows: {n_epochs}")

    if derivative_onsets and n_epochs <= len(derivative_onsets):
        der_labels = [lab for _, lab in derivative_onsets[:n_epochs]]
        match_der = sum(1 for a, b in zip(index_labels, der_labels) if a == b)
        print(f"Match index vs derivative (first {n_epochs}): {match_der}/{n_epochs}")
        if match_der < n_epochs:
            for i, (a, b) in enumerate(zip(index_labels, der_labels)):
                if a != b:
                    print(f"  epoch_idx {i}: index={a} derivative={b}")
    if raw_onsets and n_epochs <= len(raw_onsets):
        raw_labels = [lab for _, lab in raw_onsets[:n_epochs]]
        match_raw = sum(1 for a, b in zip(index_labels, raw_labels) if a == b)
        print(f"Match index vs raw (first {n_epochs}): {match_raw}/{n_epochs}")

    # When epochs are fewer than onsets (some dropped at boundaries), we can't align by position
    # without knowing which onsets were dropped. So the strict check is only when n_epochs == len(derivative_onsets).
    if derivative_onsets and n_epochs == len(derivative_onsets):
        der_labels = [lab for _, lab in derivative_onsets]
        if index_labels == der_labels:
            print("OK: Processed labels match derivative order exactly.")
        else:
            print("MISMATCH: Processed labels do not match derivative order.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

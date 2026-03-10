#!/usr/bin/env python3
"""YOTO paper preprocessing: causal FIR, ASR, ICA, ICLabel; event-based epoching for tones C/D/E."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import mne
import numpy as np
import pandas as pd

try:
    from scripts.yoto_utils import (
        estimate_onset_shift_seconds,
        extract_tone_onsets,
        get_event_mapping,
        load_config,
        load_derivative_tone_onsets,
        load_events_tsv,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yoto_utils import (  # type: ignore
        estimate_onset_shift_seconds,
        extract_tone_onsets,
        get_event_mapping,
        load_config,
        load_derivative_tone_onsets,
        load_events_tsv,
    )

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PREPROC = ROOT / "configs/yoto_preprocessing.yaml"
CONFIG_EVENTS = ROOT / "configs/yoto_events.yaml"
RAW_ROOT = ROOT / "data/raw_samples"
OUT_EPOCH_DIR = ROOT / "data/processed/epochs_yoto_tones"
OUT_INDEX = ROOT / "data/manifests/epoch_index_yoto_tones.csv"
MANIFEST_CSV = ROOT / "data/manifests/unified_manifest.csv"


def _resolve_preproc_config(
    config_preproc_path: Path,
    skip_asr: bool,
    skip_ica: bool,
) -> dict[str, Any]:
    cfg_preproc = load_config(config_preproc_path)
    cfg_preproc = dict(cfg_preproc)
    if "preprocessing" not in cfg_preproc:
        cfg_preproc["preprocessing"] = {}
    pre = cfg_preproc["preprocessing"] = dict(cfg_preproc.get("preprocessing", {}))
    if skip_ica:
        pre["ica_enabled"] = False
    if skip_asr:
        pre["asr_enabled"] = False
    return cfg_preproc


def load_raw(path: Path) -> mne.io.Raw | None:
    if path.suffix.lower() != ".vhdr":
        return None
    try:
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    except Exception:  # noqa: BLE001
        return None


def apply_causal_fir(raw: mne.io.Raw, cfg: dict) -> None:
    pre = cfg.get("preprocessing", cfg)
    l_freq, h_freq = pre.get("bandpass_hz", [1, 50])
    fir_order = pre.get("fir_order", 500)
    buf_sec = pre.get("filter_buffer_seconds", 30)
    sfreq = raw.info["sfreq"]
    filter_length = max(int(buf_sec * sfreq), 3500)
    filtered = mne.filter.filter_data(
        raw.get_data(),
        sfreq,
        l_freq,
        h_freq,
        filter_length=filter_length,
        phase="minimum",
        verbose=False,
    )
    raw._data[:] = filtered


def apply_asr(raw: mne.io.Raw, cfg: dict) -> None:
    pre = cfg.get("preprocessing", cfg)
    if not pre.get("asr_enabled", True):
        return
    try:
        import asrpy
    except ImportError:
        print("Warning: asrpy not installed; skipping ASR")
        return
    thresh = pre.get("asr_threshold", 20)
    data = raw.get_data()
    asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=thresh)
    asr.fit(raw.get_data())
    clean = asr.transform(data)
    raw._data[:] = clean


def apply_ica_iclabel(raw: mne.io.Raw, cfg: dict) -> None:
    pre = cfg.get("preprocessing", cfg)
    if not pre.get("ica_enabled", True):
        return
    n_comp = pre.get("ica_max_pca_components")
    if n_comp is None:
        n_comp = min(raw.info["nchan"], 64) - 1
    ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42, method="fastica")
    ica.fit(raw, verbose=False)
    thresh = pre.get("iclabel_reject_threshold", 0.8)
    exclude = []
    try:
        from mne_icalabel import label_components
        ic_labels = label_components(raw, ica, method="iclabel")
        for idx, probs in enumerate(ic_labels.get("y_pred_proba", [])):
            if isinstance(probs, (list, np.ndarray)) and len(probs) >= 2:
                if probs[1] > thresh or (len(probs) > 2 and probs[2] > thresh):
                    exclude.append(idx)
    except ImportError:
        pass
    ica.exclude = exclude
    raw_clean = ica.apply(raw.copy(), verbose=False)
    raw._data[:] = raw_clean._data


def preprocess_chain(raw: mne.io.Raw, cfg: dict) -> None:
    pre = cfg.get("preprocessing", cfg)
    apply_causal_fir(raw, cfg)
    sfreq = pre.get("resample_hz", 250)
    raw.resample(sfreq, verbose=False)
    apply_asr(raw, cfg)
    apply_ica_iclabel(raw, cfg)
    if pre.get("rereference") == "average":
        raw.set_eeg_reference("average", verbose=False)


def events_to_epochs(
    raw: mne.io.Raw,
    tone_onsets: list[tuple[float, str]],
    tmin: float,
    tmax: float,
    normalize_to_recording: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
    """Return (epochs array [N,C,T], event_indices, stimulus_ids)."""

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_samp_pre = int(round(-tmin * sfreq))
    n_samp_post = int(round(tmax * sfreq))
    n_samp = n_samp_pre + n_samp_post
    n_ch = data.shape[0]
    duration_sec = float(data.shape[1] / sfreq)
    stimulus_ids = []
    valid_starts = []
    onset_seconds = []
    for onset_sec, stim_id in tone_onsets:
        onset_val = float(onset_sec)
        if normalize_to_recording and duration_sec > 0.0:
            onset_val = onset_val % duration_sec
        onset_samp = int(round(onset_val * sfreq))
        start = onset_samp - n_samp_pre
        end = onset_samp + n_samp_post
        if start < 0 or end > data.shape[1]:
            continue
        stimulus_ids.append(stim_id)
        valid_starts.append(start)
        onset_seconds.append(onset_val)
    if not valid_starts:
        return np.empty((0, n_ch, n_samp), dtype=np.float32), np.array([], dtype=int), [], []
    epochs = np.zeros((len(valid_starts), n_ch, n_samp), dtype=np.float32)
    for i, start in enumerate(valid_starts):
        epochs[i] = data[:, start : start + n_samp]
    return epochs, np.array(valid_starts), stimulus_ids, onset_seconds


def apply_epoch_baseline(epochs: np.ndarray, sfreq: float, tmin: float, baseline: tuple[float, float] | None) -> None:
    """In-place baseline correction: subtract mean of baseline window from each epoch."""
    if baseline is None or len(baseline) != 2:
        return
    tmin_ep, tmax_ep = float(tmin), float(epochs.shape[2] / sfreq + tmin)
    b_start, b_end = baseline[0], baseline[1]
    if b_start >= b_end or b_start < tmin_ep or b_end > tmax_ep:
        return
    i0 = int(round((b_start - tmin_ep) * sfreq))
    i1 = int(round((b_end - tmin_ep) * sfreq))
    if i0 < 0 or i1 > epochs.shape[2]:
        return
    baseline_mean = epochs[:, :, i0:i1].mean(axis=2, keepdims=True)
    epochs -= baseline_mean


def _estimate_global_onset_shift(
    vhdr_paths: list[Path],
    event_mapping: dict[int | str, str],
) -> float | None:
    shifts: list[float] = []
    for vhdr in vhdr_paths:
        events_df = load_events_tsv(vhdr)
        if events_df is None:
            continue
        raw_onsets = extract_tone_onsets(events_df, event_mapping, onset_offset_seconds=0.0)
        derivative_onsets = load_derivative_tone_onsets(vhdr, event_mapping, onset_offset_seconds=0.0)
        shift = estimate_onset_shift_seconds(raw_onsets, derivative_onsets)
        if shift is not None:
            shifts.append(float(shift))
    if not shifts:
        return None
    return float(np.median(np.array(shifts, dtype=np.float64)))


def run_preprocess_yoto(
    manifest_path: Path = MANIFEST_CSV,
    dataset_id: str = "ds005815",
    task_contains: str = "task-task",
    out_index: Path = OUT_INDEX,
    out_dir: Path = OUT_EPOCH_DIR,
    skip_asr: bool = False,
    skip_ica: bool = False,
    config_preproc_path: Path = CONFIG_PREPROC,
    config_events_path: Path = CONFIG_EVENTS,
    raw_root: Path = RAW_ROOT,
    tmin_override: float | None = None,
    tmax_override: float | None = None,
) -> dict[str, Any]:

    cfg_preproc = _resolve_preproc_config(config_preproc_path, skip_asr, skip_ica)
    cfg_events = load_config(config_events_path)
    event_mapping = get_event_mapping(cfg_events)
    if not event_mapping:
        print("Warning: no event_value_to_stimulus in yoto_events.yaml; no tone epochs will be produced.")

    prefer_derivatives = bool(cfg_events.get("prefer_derivatives_task_event", True))
    raw_onset_offset = float(cfg_events.get("onset_offset_seconds", 0.0))
    derivative_onset_offset = float(cfg_events.get("derivative_onset_offset_seconds", 0.0))
    normalize_onsets_to_recording = bool(cfg_events.get("normalize_onsets_to_recording", True))

    epoch_cfg = cfg_preproc.get("epoch", {})
    tmin = float(tmin_override if tmin_override is not None else epoch_cfg.get("tmin", -0.2))
    tmax = float(tmax_override if tmax_override is not None else epoch_cfg.get("tmax", 0.8))
    baseline = epoch_cfg.get("baseline")
    if isinstance(baseline, list) and len(baseline) == 2:
        baseline = (float(baseline[0]), float(baseline[1]))
    else:
        baseline = None

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        df = df[df["file_ext"] == ".vhdr"]
        df = df[df["dataset_id"] == dataset_id]
        df = df[df["file_path"].str.contains(task_contains, case=False, na=False)]
        vhdr_paths = [Path(p) for p in df["file_path"].unique()]
    else:
        vhdr_paths = list((raw_root / dataset_id).rglob(f"*{task_contains}*eeg.vhdr"))
    vhdr_paths = sorted(vhdr_paths)

    if bool(cfg_events.get("auto_estimate_onset_offset_from_derivatives", True)):
        estimated_shift = _estimate_global_onset_shift(vhdr_paths=vhdr_paths, event_mapping=event_mapping)
        if estimated_shift is not None:
            raw_onset_offset = estimated_shift
            print(f"Estimated raw->derivative onset shift: {raw_onset_offset:.3f}s (median)")

    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []

    for vhdr in vhdr_paths:
        if not vhdr.is_file():
            continue
        raw = load_raw(vhdr)
        if raw is None:
            continue
        preprocess_chain(raw, cfg_preproc)
        events_df = load_events_tsv(vhdr)
        if events_df is None:
            continue

        raw_onsets = extract_tone_onsets(
            events_df,
            event_mapping,
            onset_offset_seconds=raw_onset_offset,
        )
        derivative_onsets = load_derivative_tone_onsets(
            vhdr,
            event_mapping,
            onset_offset_seconds=derivative_onset_offset,
        )
        if prefer_derivatives and derivative_onsets:
            tone_onsets = derivative_onsets
            onset_source = "derivatives_task_event"
            offset_applied = derivative_onset_offset
        else:
            tone_onsets = raw_onsets
            onset_source = "events_tsv"
            offset_applied = raw_onset_offset

        epochs, _, stimulus_ids, onset_seconds = events_to_epochs(
            raw,
            tone_onsets,
            tmin,
            tmax,
            normalize_to_recording=normalize_onsets_to_recording,
        )
        if epochs.size == 0:
            continue
        apply_epoch_baseline(epochs, raw.info["sfreq"], tmin, baseline)
        parts = vhdr.parts
        subject_id = next((p for p in parts if p.startswith("sub-")), "unknown")
        stem = f"{dataset_id}__{subject_id}__{vhdr.stem}"
        out_path = out_dir / f"{stem}.npy"
        np.save(out_path, epochs)
        print(f"{subject_id} {vhdr.name}: {len(stimulus_ids)} epochs via {onset_source}")
        for epoch_idx, (stim_id, onset_sec) in enumerate(zip(stimulus_ids, onset_seconds)):
            index_rows.append({
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "source_file": str(vhdr),
                "epochs_file": str(out_path),
                "epoch_idx": epoch_idx,
                "stimulus_id": stim_id,
                "onset_sec": float(onset_sec),
                "onset_source": onset_source,
                "onset_offset_sec_applied": float(offset_applied),
                "onset_wrapped_to_recording": normalize_onsets_to_recording,
                "n_channels": int(epochs.shape[1]),
                "n_samples": int(epochs.shape[2]),
            })

    out_index.parent.mkdir(parents=True, exist_ok=True)
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_index, index=False)
        out_index.with_suffix(".json").write_text(
            json.dumps(index_rows, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(index_rows)} epoch rows to {out_index}")
    else:

        print("No tone epochs produced; check event mapping and task files.")
    return {
        "dataset_id": dataset_id,
        "n_vhdr": len(vhdr_paths),
        "n_epochs": len(index_rows),
        "out_index": str(out_index),
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV))
    parser.add_argument("--dataset", type=str, default="ds005815")
    parser.add_argument("--task-contains", type=str, default="task-task")
    parser.add_argument("--out-index", type=str, default=str(OUT_INDEX))
    parser.add_argument("--out-dir", type=str, default=str(OUT_EPOCH_DIR))
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ica", action="store_true")
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    args = parser.parse_args()

    run_preprocess_yoto(
        manifest_path=Path(args.manifest),
        dataset_id=args.dataset,
        task_contains=args.task_contains,
        out_index=Path(args.out_index),
        out_dir=Path(args.out_dir),
        skip_asr=args.skip_asr,
        skip_ica=args.skip_ica,
        tmin_override=args.tmin,
        tmax_override=args.tmax,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Lightweight pipeline verifier for this repo.

Checks (best-effort):
- epoch_index presence and basic sanity
- sample epochs file load and shape (n_windows, n_channels, n_samples)
- amplitude (per-channel mean/std) and dead/noisy channels
- PSD power-in-band estimate (0.5-45 Hz)
- simple outlier window detection
- labels and split basic checks
- optional: augmented outputs integrity

Exit code: 0 on pass (no critical failures), 1 on any critical failure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path as PPath

import importlib.util
_ROOT = Path(__file__).resolve().parents[1]
# Try importing yoto_utils from package first, otherwise load directly from scripts/yoto_utils.py
load_events_tsv = None
load_yoto_config = None
get_event_mapping = None
try:
    from scripts.yoto_utils import load_events_tsv, load_config as load_yoto_config, get_event_mapping  # type: ignore
except Exception:
    yoto_path = _ROOT / "scripts" / "yoto_utils.py"
    if yoto_path.exists():
        spec = importlib.util.spec_from_file_location("yoto_utils_local", str(yoto_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            load_events_tsv = getattr(module, "load_events_tsv", None)
            load_yoto_config = getattr(module, "load_config", None)
            get_event_mapping = getattr(module, "get_event_mapping", None)




try:
    import mne
    from mne.time_frequency import psd_array_welch
except Exception:
    mne = None
    psd_array_welch = None

def fail(msg: str):
    print("FAIL:", msg)
    return False


def ok(msg: str):
    print("OK:", msg)
    return True


def load_epoch_index(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        fail(f"epoch_index not found: {path}")
        return None
    df = pd.read_csv(path)
    ok(f"Loaded epoch_index ({len(df)} rows)")
    return df


def try_load_epochs(path: Path):
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        return arr
    except Exception as e:
        print("Could not load epochs npy:", e)
        return None


def check_shape(arr: np.ndarray, expected_samples: int | None):
    if arr is None:
        return fail("No epoch array to inspect")
    if arr.ndim == 3:
        n_windows, n_channels, n_samples = arr.shape
    elif arr.ndim == 2:
        # maybe single epoch saved as (n_channels, n_samples)
        n_windows = 1
        n_channels, n_samples = arr.shape
    else:
        return fail(f"Unexpected epoch array ndim={arr.ndim}")
    ok(f"Epoch array shape: windows={n_windows}, channels={n_channels}, samples={n_samples}")
    if expected_samples is not None and n_samples != expected_samples:
        print("WARN: n_samples mismatch:", n_samples, "!= expected", expected_samples)
    return True


def check_amplitude(arr: np.ndarray, dead_thresh: float = 0.1, noisy_factor: float = 3.0):
    if arr is None:
        return False
    # collapse windows if many
    if arr.ndim == 3:
        data = arr.reshape(-1, arr.shape[2])  # (windows*channels, samples)
        # Better per-channel:
        per_chan = arr.mean(axis=(0, 2)), arr.std(axis=(0, 2))
        means = per_chan[0]
        stds = per_chan[1]
    elif arr.ndim == 2:
        means = arr.mean(axis=1)
        stds = arr.std(axis=1)
    else:
        return False
    global_std = float(np.nanmean(stds))
    dead = np.where(stds < dead_thresh)[0]
    noisy = np.where(stds > noisy_factor * global_std)[0]
    ok(f"Per-channel mean (first 5): {np.round(means[:5],3).tolist()}")
    ok(f"Per-channel std (first 5): {np.round(stds[:5],3).tolist()}")
    if len(dead) > 0:
        print(f"WARN: dead channels (std < {dead_thresh}): {dead.tolist()}")
    if len(noisy) > 0:
        print(f"WARN: noisy channels (> {noisy_factor}×global std): {noisy.tolist()}")
    return True


def compute_band_power(arr: np.ndarray, sfreq: float = 256.0, low: float = 0.5, high: float = 45.0):
    if psd_array_welch is None:
        print("mne not available: skipping PSD checks")
        return None
    # use a single example epoch (first)
    if arr.ndim == 3:
        ep = arr[0]  # (n_channels, n_samples)
    elif arr.ndim == 2:
        ep = arr
    else:
        return None
    # Ensure segment length isn't larger than the signal (avoids n_fft > n_times error)
    n_times = ep.shape[1]
    n_per_seg = min(256, n_times)
    psd, freqs = psd_array_welch(
        ep, sfreq=sfreq, fmin=0.0, fmax=sfreq / 2.0, n_per_seg=n_per_seg, verbose=False
    )
    # psd shape (n_channels, n_freqs)
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = psd[:, band_mask].sum(axis=1)
    total_power = psd.sum(axis=1)
    frac = band_power.sum() / (total_power.sum() + 1e-12)
    ok(f"Band {low}-{high} Hz power fraction (all channels): {frac:.3f}")
    if frac < 0.9:
        print("WARN: <90% power in target band")
    return frac


def outlier_windows(arr: np.ndarray):
    if arr is None or arr.ndim != 3:
        return None
    # compute std per window (across channels and samples)
    wstd = arr.std(axis=(1, 2))
    mean = wstd.mean()
    sd = wstd.std()
    flags = np.where(wstd > mean + 4 * sd)[0]
    ok(f"Outlier windows: {len(flags)} / {len(wstd)}")
    return flags.tolist()


def check_augmented_dir(path: Path):
    if not path.exists():
        return fail(f"augmented dir not found: {path}")
    npys = list(path.rglob("*.npy"))
    if not npys:
        return fail("no .npy files found in augmented dir")
    sample = try_load_epochs(npys[0])
    if sample is None:
        return fail("could not load augmented sample")
    # check NaN/Inf
    if np.isnan(sample).any():
        print("WARN: NaN in augmented data")
    if np.isinf(sample).any():
        print("WARN: Inf in augmented data")
    ok(f"Augmented sample loaded ({npys[0].name})")
    return True


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="verify_pipeline.py")
    p.add_argument("--epoch-index", type=Path, default=Path("data/manifests/epoch_index_yoto_tones.csv"))
    p.add_argument("--expected-samples", type=int, default=None)
    p.add_argument("--sfreq", type=float, default=256.0)
    p.add_argument("--augmented-dir", type=Path, default=None)
    p.add_argument("--check-onsets", action="store_true", help="Verify stimulus onsets for event-based epoch indexes (YOTO)")
    args = p.parse_args(argv)

    ok("Starting pipeline verification")
    df = load_epoch_index(args.epoch_index)
    if df is None:
        sys.exit(1)

    critical_fail = False

    # basic label checks
    if "stimulus_id" not in df.columns:
        print("WARN: 'stimulus_id' not in epoch_index")
    else:
        ok("Labels column present")

    # attempt to find an epochs file and load it
    sample_files = df.get("epochs_file", pd.Series(dtype=str)).dropna().unique().tolist()
    sample_path = None
    for fp in sample_files:
        pfp = Path(fp)
        if not pfp.exists():
            # try relative to repo root
            pfp2 = Path.cwd() / fp
            if pfp2.exists():
                pfp = pfp2
            else:
                continue
        sample_path = pfp
        break

    if sample_path is None:
        print("WARN: no epochs_file found or files missing in epoch_index")
    else:
        ok(f"Found sample epochs file: {sample_path}")
        arr = try_load_epochs(sample_path)
        if arr is None:
            critical_fail = True
        else:
            # auto-detect units: if max abs < 1e-2 assume values are in volts and scale to µV
            max_abs = float(np.max(np.abs(arr)))
            if max_abs < 1e-2:
                print(f"INFO: detected small amplitudes (max abs={max_abs:.3g}) — assuming volts. Scaling checks by 1e6 to µV.")
                arr_for_checks = arr * 1e6
            else:
                arr_for_checks = arr

            check_shape(arr_for_checks, args.expected_samples)
            check_amplitude(arr_for_checks)
            if mne is not None:
                compute_band_power(arr_for_checks, sfreq=args.sfreq)
            else:
                print("Skipping PSD check (mne unavailable)")
            outlier_windows(arr_for_checks)

            # optional: verify stimulus onsets for YOTO-style index
            if args.check_onsets and "yoto" in str(args.epoch_index).lower():
                if load_events_tsv is None:
                    print("WARN: yoto_utils not importable; skipping onset checks")
                else:
                    # group epoch_index by source_file and check event counts and indices
                    try:
                        e_df = pd.read_csv(args.epoch_index)
                        for src, group in e_df.groupby("source_file"):
                            ev = load_events_tsv(PPath(src))
                            if ev is None:
                                print(f"WARN: no events file for {src}")
                                continue
                            cfg = load_yoto_config(PPath.cwd() / "configs" / "yoto_events.yaml") if load_yoto_config else {}
                            mapping = get_event_mapping(cfg) if get_event_mapping else {}
                            # extract tone onsets via same logic as preprocess
                            tone_onsets = []
                            for _, row in ev.iterrows():
                                stim_id = None
                                if "value" in ev.columns and pd.notna(row.get("value")):
                                    try:
                                        v = int(float(row["value"]))
                                        stim_id = mapping.get(v)
                                    except Exception:
                                        pass
                                if stim_id is None and "trial_type" in ev.columns:
                                    stim_id = mapping.get(str(row["trial_type"]).strip())
                                if stim_id in ("tone_C", "tone_D", "tone_E"):
                                    tone_onsets.append((float(row.get("onset", 0.0)), stim_id))
                            n_onsets = len(tone_onsets)
                            n_epochs = len(group)
                            if n_onsets == 0:
                                print(f"WARN: no tone onsets found in events for {src}")
                            elif n_onsets < n_epochs:
                                print(f"WARN: fewer onsets ({n_onsets}) than epochs ({n_epochs}) for {src}")
                            else:
                                print(f"OK: {src} has {n_onsets} tone onsets and {n_epochs} epochs")
                            # check epoch_idx bounds
                            bad_idx = []
                            for idx in group["epoch_idx"].astype(int):
                                if idx >= n_onsets:
                                    bad_idx.append(int(idx))
                            if bad_idx:
                                print(f"WARN: epoch_idx out of bounds for {src}: {bad_idx}")
                    except Exception as e:
                        print("Error during onset checks:", e)

    # augmented dir checks
    if args.augmented_dir:
        if not check_augmented_dir(args.augmented_dir):
            critical_fail = True

    if critical_fail:
        print("One or more critical checks failed")
        sys.exit(1)
    else:
        print("Verification finished (no critical failures detected). Review WARN messages above.")
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])


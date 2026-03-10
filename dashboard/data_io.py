from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_registry_df(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    path = root / "eeg_datasets_for_labram.csv"
    df = read_csv_if_exists(path)
    return df.fillna("")


def load_dataset_size_report(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return read_json_if_exists(root / "data" / "manifests" / "dataset_size_report.json")


def load_unified_manifest(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    path = root / "data" / "manifests" / "unified_manifest.csv"
    df = read_csv_if_exists(path)
    return df.fillna("")


def load_epoch_index(with_zuna: bool = False, root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    filename = "epoch_index_with_zuna.csv" if with_zuna else "epoch_index.csv"
    path = root / "data" / "manifests" / filename
    df = read_csv_if_exists(path)
    return df.fillna("")


def load_epoch_index_options(root: Path | None = None) -> list[tuple[str, Path]]:
    """Return list of (label, path) for available epoch indices including YOTO long-format."""
    root = root or project_root()
    manifests = root / "data" / "manifests"
    out = [
        ("Base (epoch_index.csv)", manifests / "epoch_index.csv"),
        ("Base + Zuna (epoch_index_with_zuna.csv)", manifests / "epoch_index_with_zuna.csv"),
    ]
    for name, fname in [
        ("YOTO tones (long-format)", "epoch_index_yoto_tones.csv"),
        ("YOTO tones + Zuna (long-format)", "epoch_index_yoto_tones_zuna.csv"),
        ("Color (all datasets)", "epoch_index_color.csv"),
    ]:
        if (manifests / fname).exists():
            out.append((name, manifests / fname))
    for path in sorted(manifests.glob("epoch_index_color_*.csv")):
        if path.name != "epoch_index_color.csv":
            out.append((f"Color ({path.stem.replace('epoch_index_color_', '')})", path))
    return out


def load_epoch_array(epochs_file: str) -> np.ndarray:
    return np.load(epochs_file, allow_pickle=False)


def load_channel_names_from_source(source_file: str, root: Path | None = None) -> list[str]:
    path = Path(source_file)
    if not path.is_absolute() and root:
        path = root / path
    ext = path.suffix.lower()
    try:
        if ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(path, preload=False, verbose=False)
            return list(raw.ch_names)
        if ext == ".edf":
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            return list(raw.ch_names)
        if ext == ".bdf":
            raw = mne.io.read_raw_bdf(path, preload=False, verbose=False)
            return list(raw.ch_names)
        if ext == ".set":
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
            return list(raw.ch_names)
        if ext == ".fif":
            raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
            return list(raw.ch_names)
        if ext == ".mat":
            root = root or project_root()
            config_path = root / "configs" / "color_channel_layouts.yaml"
            if config_path.exists():
                import yaml
                layouts = yaml.safe_load(config_path.read_text()) or {}
                for key in ("hajonides_j289e", "baeluck_jnwut", "chauhan_v9ewj"):
                    if key in str(path) and key in layouts:
                        return list(layouts[key])
            return []
        return []
    except Exception:  # noqa: BLE001
        return []


def load_runs_json(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = root or project_root()
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return {}

    output: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        payload = read_json_if_exists(path)
        if isinstance(payload, dict):
            output[path.stem] = payload
    return output


def normalize_run_rows(runs_payload: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, payload in runs_payload.items():
        row: dict[str, Any] = {"run_name": run_name}
        for key in ("accuracy", "macro_f1", "n_train", "n_test", "model", "checkpoint", "note"):
            row[key] = payload.get(key)

        if run_name == "baseline_metrics":
            row["run_type"] = "baseline"
        elif run_name == "baseline_metrics_zuna_aug":
            row["run_type"] = "baseline_zuna_aug"
        elif "labram" in run_name:
            row["run_type"] = "labram"
        elif run_name == "zuna_ablation_summary":
            row["run_type"] = "zuna_ablation"
        elif run_name == "labram_vs_baseline":
            row["run_type"] = "comparison"
        else:
            row["run_type"] = "other"

        rows.append(row)
    return pd.DataFrame(rows)

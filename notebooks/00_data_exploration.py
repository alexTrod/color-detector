
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
DATA_ROOT = ROOT / "data"
MANIFEST_DIR = DATA_ROOT / "manifests"
EPOCH_INDEX_PATH = MANIFEST_DIR / "epoch_index_yoto_tones.csv"
CONFIG_EVENTS = ROOT / "configs/yoto_events.yaml"
TARGET_DATASET = "ds005815"

sns.set_theme(style="whitegrid")



def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}

_event_config = _load_yaml(CONFIG_EVENTS)
stimuli_meta = _event_config.get("stimuli", {})
event_mapping = {
    int(k) if str(k).isdigit() else str(k): v
    for k, v in _event_config.get("event_value_to_stimulus", {}).items()
}
event_mapping.update(_event_config.get("trial_type_to_stimulus", {}))

def _format_label(stimulus: str | None) -> str:
    if stimulus is None:
        return "unknown"
    meta = stimuli_meta.get(stimulus, {})
    freq = meta.get("frequency_hz")
    if freq:
        return f"{stimulus} ({freq} Hz)"
    return stimulus

stimulus_labels = {
    sid: _format_label(sid)
    for sid in set(event_mapping.values()) | set(stimuli_meta.keys())
}


epoch_index = pd.read_csv(EPOCH_INDEX_PATH)
epoch_index = epoch_index[epoch_index["dataset_id"] == TARGET_DATASET]
epoch_index = epoch_index.assign(
    subject_id=epoch_index["subject_id"].fillna("unknown"),
    stimulus_label=epoch_index["stimulus_id"].map(
        lambda sid: stimulus_labels.get(sid, sid if pd.notna(sid) else "unknown")
    ),
)

epoch_index.head()


subject_overview = (
    epoch_index.groupby("subject_id")
    .agg(
        epochs=("epoch_idx", "count"),
        unique_runs=("source_file", "nunique"),
        tones=("stimulus_label", "nunique"),
    )
    .sort_values("epochs", ascending=False)
)
subject_overview


subject_tone_matrix = (
    epoch_index
    .pivot_table(
        index="subject_id",
        columns="stimulus_label",
        values="epoch_idx",
        aggfunc="size",
        fill_value=0,
    )
)
subject_tone_matrix


def load_epoch(row: pd.Series) -> np.ndarray:
    epochs_path = ROOT / row.epochs_file if not Path(row.epochs_file).is_absolute() else Path(row.epochs_file)
    arr = np.load(epochs_path)
    idx = int(row.epoch_idx)
    return arr[idx]


def plot_epoch(epoch: np.ndarray, channel_slice: slice = slice(0, 6), title: str = "", scale: float = 1000) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for channel_idx in range(*channel_slice.indices(epoch.shape[0])):
        offset = channel_idx * 0.5
        # Scale amplitude for visibility (EEG data is in V, scale to mV or uV)
        scaled = epoch[channel_idx] * scale + offset
        ax.plot(scaled, label=f"chan_{channel_idx + 1}")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel(f"Amplitude x{scale} + offset")
    ax.legend(ncol=2, fontsize="x-small")
    plt.tight_layout()

sample_row = epoch_index.iloc[0]
sample_epoch = load_epoch(sample_row)
plot_epoch(sample_epoch, channel_slice=slice(0, min(6, sample_epoch.shape[0])), title="First epoch (subject {}, stimulus {})".format(
    sample_row.subject_id, sample_row.stimulus_label
))



class_counts = epoch_index["stimulus_label"].value_counts()
class_counts


tone_averages = {}
for tone, group in epoch_index.groupby("stimulus_label"):
    samples = []
    for _, row in group.head(3).iterrows():
        samples.append(load_epoch(row))
    if samples:
        tone_averages[tone] = np.mean(samples, axis=0)

if tone_averages:
    fig, axes = plt.subplots(len(tone_averages), 1, figsize=(10, 2 * len(tone_averages)), sharex=True)
    if len(tone_averages) == 1:
        axes = [axes]
    for ax, (tone, avg) in zip(axes, tone_averages.items()):
        scale = 1000  # Scale EEG data for visibility
        ax.plot(avg[0] * scale, color="tab:blue")
        ax.set_title(f"Average first channel for {tone}")
        ax.set_ylabel(f"Amplitude x{scale}")
    plt.tight_layout()


base_epoch = load_epoch(epoch_index.iloc[0])
channel_stats = pd.DataFrame(
    {
        "mean": base_epoch.mean(axis=1),
        "std": base_epoch.std(axis=1),
    }
)
channel_stats.index = [f"chan_{i + 1}" for i in channel_stats.index]
channel_stats.head()


amplitude_scale = 1000  # Multiply signal amplitude (try 1, 10, 100, 1000, etc.)
offset_scale = 0.5      # Vertical spacing between channels

fig, ax = plt.subplots(figsize=(10, 4))
for i in range(min(5, base_epoch.shape[0])):
    ax.plot(base_epoch[i] * amplitude_scale + i * offset_scale, label=f"chan_{i + 1}")
ax.set_title("Channel overview for first epoch")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude + offset")
ax.legend(ncol=2, fontsize="x-small")
plt.tight_layout()


import mne
import matplotlib.pyplot as plt

sample_vhdr = DATA_ROOT / "raw_samples/ds005815/sub-01/ses-1/eeg/sub-01_ses-1_task-task_eeg.vhdr"
raw = mne.io.read_raw_brainvision(sample_vhdr, preload=False, verbose=False)

# Attach a standard 10-20 montage (the file lacks digitized positions)
montage = mne.channels.make_standard_montage("standard_1020")
raw_mont = raw.copy().set_montage(montage, match_case=False, on_missing="ignore")

# Create numbered channel labels (1, 2, 3, ...)
ch_names = raw_mont.ch_names
num_labels = [f"{i+1} {n}" for i, n in enumerate(ch_names)]
print(f"Channels ({len(ch_names)}): {ch_names[:10]}...")
print(f"Numbered labels (first 10): {num_labels[:10]}")

# Rename channels on a plotting copy so original `raw` stays unchanged
rename_map = {orig: new for orig, new in zip(ch_names, num_labels)}
plot_raw = raw_mont.copy()
plot_raw.rename_channels(rename_map)

fig = plot_raw.plot_sensors(show_names=True, show=False, sphere=(0, 0, 0, 0.095))
fig.set_size_inches(8, 8)
plt.title("EEG Channel Montage (10-20 System) — numbered labels")
plt.tight_layout()
plt.show()


# Alternative: Create a 2D topomap layout plot
montage = raw.get_montage()
if montage is not None:
    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_montage(montage, show_names=True, axes=ax, sphere=(0, 0, 0, 0.095))
    ax.set_title("EEG Channel Layout (Top View)")
    plt.tight_layout()
else:
    print("No montage information available - using standard 10-20 layout")
    # Create standard 10-20 montage for 30 channels
    std_montage = mne.channels.make_standard_montage("standard_1020")
    # Select only the channels that match our data
    ch_selection = [ch for ch in std_montage.ch_names if ch in ch_names or ch.upper() in [c.upper() for c in ch_names]]
    if len(ch_selection) > 0:
        std_montage.pick_channels(ch_selection)
        fig, ax = plt.subplots(figsize=(8, 8))
        mne.viz.plot_montage(std_montage, show_names=True, axes=ax)
        ax.set_title("Standard 10-20 Layout (Approximate)")
        plt.tight_layout()



def search_epochs(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df.iloc[:0]
    mask = (
        df["subject_id"].str.contains(query, case=False, na=False)
        | df["stimulus_label"].str.contains(query, case=False, na=False)
        | df["source_file"].str.contains(query, case=False, na=False)
    )
    return df[mask]

search_epochs(epoch_index, "tone_D").head()
#
# %% npy reader and visualizer
from typing import Iterable


def list_npy_files(root: Path | str = DATA_ROOT / "processed/epochs", pattern: str = "**/*.npy") -> list[Path]:
    """
    Return a list of .npy files under `root` matching `pattern`.
    """
    root_p = Path(root)
    return sorted(root_p.glob(pattern))


def read_npy(path: Path | str) -> np.ndarray:
    """
    Load an .npy file and return the array.
    """
    return np.load(Path(path))


def visualize_npy(
    path: Path | str | None = None,
    arr: np.ndarray | None = None,
    epoch_idx: int = 0,
    channels: Iterable[int] | None = None,
    start: int = 0,
    end: int | None = None,
    scale: float = 1000.0,
    figsize: tuple = (10, 4),
) -> None:
    """
    Visualize a .npy array (1D, 2D channels x samples, or 3D epochs x channels x samples).
    - If `path` is provided it will be loaded; otherwise pass `arr`.
    - For 3D arrays, `epoch_idx` selects which epoch to plot.
    - `channels` can be an iterable of channel indices to plot (default: first 6 or all).
    """
    if arr is None:
        if path is None:
            raise ValueError("Either path or arr must be provided")
        arr = read_npy(path)

    # Handle epochs dimension
    if arr.ndim == 3:
        if epoch_idx < 0 or epoch_idx >= arr.shape[0]:
            raise IndexError("epoch_idx out of range")
        arr = arr[epoch_idx]

    # If 1D, plot single series
    if arr.ndim == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(arr[start:end] * scale)
        ax.set_title(f"{Path(path).name if path is not None else 'array'} (1D)")
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"Amplitude x{scale}")
        plt.tight_layout()
        return

    # If 2D assume (channels, samples) or (samples, channels)
    ch_axis, samp_axis = (0, 1) if arr.shape[0] < arr.shape[1] else (0, 1)
    n_channels = arr.shape[0]
    sel_channels = list(channels) if channels is not None else list(range(min(6, n_channels)))

    fig, ax = plt.subplots(figsize=figsize)
    for i, ch in enumerate(sel_channels):
        if ch < 0 or ch >= n_channels:
            continue
        offset = i * 0.5
        ax.plot(arr[ch, start:end] * scale + offset, label=f"chan_{ch + 1}")
    ax.set_title(f"{Path(path).name if path is not None else 'array'} (channels plotted: {sel_channels})")
    ax.set_xlabel("Sample")
    ax.set_ylabel(f"Amplitude x{scale} + offset")
    ax.legend(ncol=2, fontsize="x-small")
    plt.tight_layout()


def imshow_npy(path: Path | str | None = None, arr: np.ndarray | None = None, cmap: str = "viridis", aspect: str = "auto"):
    """
    Show a 2D array as an image (useful for time-frequency arrays or channels x samples heatmaps).
    """
    if arr is None:
        if path is None:
            raise ValueError("Either path or arr must be provided")
        arr = read_npy(path)

    if arr.ndim == 3:
        # collapse epochs by mean
        arr = arr.mean(axis=0)

    if arr.ndim != 2:
        raise ValueError("imshow_npy expects a 2D array (channels x samples) or 3D (epochs x channels x samples)")

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(arr, aspect=aspect, cmap=cmap, origin="lower")
    ax.set_title(f"{Path(path).name if path is not None else 'array'} (imshow)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02)
    plt.tight_layout()


# Optional interactive browser (works in Jupyter/IPython)
try:
    import ipywidgets as widgets
    from IPython.display import display

    def interactive_npy_browser(root: Path | str = DATA_ROOT / "processed/epochs"):
        files = list_npy_files(root)
        if not files:
            print("No .npy files found under", root)
            return

        file_dd = widgets.Dropdown(options=[str(p) for p in files], description="file")
        epoch_slider = widgets.IntSlider(value=0, min=0, max=0, description="epoch")
        channel_select = widgets.SelectMultiple(options=list(range(16)), description="channels")
        view_type = widgets.ToggleButtons(options=["plot", "imshow"], description="view")

        def _on_file_change(change):
            arr = read_npy(change["new"])
            if arr.ndim == 3:
                epoch_slider.max = arr.shape[0] - 1
            else:
                epoch_slider.max = 0
            channel_select.options = list(range(arr.shape[0])) if arr.ndim >= 2 else [0]

        file_dd.observe(_on_file_change, names="value")

        out = widgets.Output()

        def _update(*_):
            out.clear_output(wait=True)
            with out:
                arr = read_npy(file_dd.value)
                if view_type.value == "plot":
                    visualize_npy(arr=arr, epoch_idx=epoch_slider.value, channels=channel_select.value)
                else:
                    imshow_npy(arr=arr)

        controls = widgets.VBox([file_dd, epoch_slider, channel_select, view_type])
        display(controls, out)
        file_dd.observe(lambda *_: _update(), names="value")
        epoch_slider.observe(lambda *_: _update(), names="value")
        channel_select.observe(lambda *_: _update(), names="value")
        view_type.observe(lambda *_: _update(), names="value")

except Exception:
    # ipywidgets not available — skip interactive helper
    pass

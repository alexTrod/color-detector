from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from dashboard.data_io import (
        load_dataset_size_report,
        load_channel_names_from_source,
        load_epoch_array,
        load_epoch_index,
        load_epoch_index_options,
        load_registry_df,
        load_runs_json,
        load_unified_manifest,
        normalize_run_rows,
    )
    from dashboard.plots import (
        bar_modality_distribution,
        bar_stimulus_distribution,
        bar_task_distribution,
        bar_total_epochs_by_dataset,
        grouped_run_metrics,
        hist_channel_counts,
        line_eeg_traces,
        montage_scatter,
    )
    from dashboard.stats import compare_tensor_stats, compute_psd, dataset_breakdown, describe_epoch_tensor, summarise_epoch_index
except ModuleNotFoundError:
    from data_io import (
        load_dataset_size_report,
        load_channel_names_from_source,
        load_epoch_array,
        load_epoch_index,
        load_epoch_index_options,
        load_registry_df,
        load_runs_json,
        load_unified_manifest,
        normalize_run_rows,
    )
    from plots import (
        bar_modality_distribution,
        bar_stimulus_distribution,
        bar_task_distribution,
        bar_total_epochs_by_dataset,
        grouped_run_metrics,
        hist_channel_counts,
        line_eeg_traces,
        montage_scatter,
    )
    from stats import compare_tensor_stats, compute_psd, dataset_breakdown, describe_epoch_tensor, summarise_epoch_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def cached_registry() -> pd.DataFrame:
    return load_registry_df(PROJECT_ROOT)


@st.cache_data
def cached_size_report() -> dict[str, Any]:
    return load_dataset_size_report(PROJECT_ROOT)


@st.cache_data
def cached_unified_manifest() -> pd.DataFrame:
    return load_unified_manifest(PROJECT_ROOT)


@st.cache_data
def cached_epoch_index(with_zuna: bool) -> pd.DataFrame:
    return load_epoch_index(with_zuna=with_zuna, root=PROJECT_ROOT)


@st.cache_data
def cached_epoch_index_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    return df


@st.cache_data
def cached_runs() -> dict[str, dict[str, Any]]:
    return load_runs_json(PROJECT_ROOT)


@st.cache_data
def cached_epoch_array(path: str) -> np.ndarray:
    return load_epoch_array(path)


@st.cache_data
def cached_channel_names(path: str) -> list[str]:
    return load_channel_names_from_source(path)


def normalize_task_label(label: str) -> str:
    return label.replace("_zuna_aug", "")


def render_dataset_explorer(epoch_df: pd.DataFrame, unified_df: pd.DataFrame, registry_df: pd.DataFrame):
    st.subheader("Dataset Explorer")
    if epoch_df.empty:
        st.warning("No epoch index data found.")
        return

    has_stimulus = "stimulus_id" in epoch_df.columns
    left, right, third = st.columns(3)
    datasets = sorted(epoch_df["dataset_id"].astype(str).unique().tolist())
    subjects = sorted(epoch_df["subject_id"].astype(str).unique().tolist())
    tasks = sorted(epoch_df["task_label"].astype(str).unique().tolist()) if "task_label" in epoch_df.columns else []
    stimuli = sorted(epoch_df["stimulus_id"].astype(str).unique().tolist()) if has_stimulus else []

    selected_dataset = left.selectbox("Dataset", ["All"] + datasets)
    selected_subject = right.selectbox("Subject", ["All"] + subjects)
    if has_stimulus:
        selected_stimulus = st.multiselect("Stimulus (condition)", stimuli, default=stimuli)
        selected_task = "All"
    else:
        selected_stimulus = []
        selected_task = third.selectbox("Task", ["All"] + tasks) if tasks else "All"

    view = epoch_df.copy()
    if selected_dataset != "All":
        view = view[view["dataset_id"] == selected_dataset]
    if selected_subject != "All":
        view = view[view["subject_id"] == selected_subject]
    if has_stimulus and selected_stimulus:
        view = view[view["stimulus_id"].astype(str).isin(selected_stimulus)]
    elif not has_stimulus and tasks and selected_task != "All":
        view = view[view["task_label"] == selected_task]

    summary = summarise_epoch_index(view)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", summary["datasets"])
    c2.metric("Subjects", summary["subjects"])
    c3.metric("Recordings", summary["recordings"])
    c4.metric("Total epochs", summary["total_epochs"])

    c5, c6, c7 = st.columns(3)
    c5.metric("Min channels", summary["channels_min"])
    c6.metric("Max channels", summary["channels_max"])
    c7.metric("Samples/epoch", summary["samples_per_epoch"])

    st.plotly_chart(bar_total_epochs_by_dataset(view), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        if has_stimulus:
            st.plotly_chart(bar_stimulus_distribution(view), use_container_width=True)
        else:
            st.plotly_chart(bar_modality_distribution(view), use_container_width=True)
    with col2:
        st.plotly_chart(bar_task_distribution(view), use_container_width=True) if "task_label" in view.columns else st.empty()
    st.plotly_chart(hist_channel_counts(view), use_container_width=True)

    st.markdown("#### Dataset-level Aggregates")
    st.dataframe(dataset_breakdown(view), use_container_width=True)

    st.markdown("#### Epoch Index")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.markdown("#### Unified Manifest")
    manifest_view = unified_df.copy()
    if selected_dataset != "All" and "dataset_id" in manifest_view.columns:
        manifest_view = manifest_view[manifest_view["dataset_id"] == selected_dataset]
    st.dataframe(manifest_view, use_container_width=True, hide_index=True)

    st.markdown("#### Dataset Registry")
    st.dataframe(registry_df, use_container_width=True, hide_index=True)


def render_eeg_visual_inspector(epoch_df: pd.DataFrame):
    st.subheader("EEG Visual Inspector")
    if epoch_df.empty:
        st.warning("No epoch index data found.")
        return

    is_long_format = "stimulus_id" in epoch_df.columns and "epoch_idx" in epoch_df.columns
    if is_long_format:
        row_idx = st.selectbox(
            "Select epoch",
            options=list(range(len(epoch_df))),
            format_func=lambda i: f"{epoch_df.iloc[i]['dataset_id']} | {epoch_df.iloc[i]['subject_id']} | {Path(str(epoch_df.iloc[i]['epochs_file'])).name} | idx={epoch_df.iloc[i]['epoch_idx']} | {epoch_df.iloc[i]['stimulus_id']}",
        )
        row = epoch_df.iloc[row_idx]
        arr = cached_epoch_array(str(row["epochs_file"]))
        if arr.ndim != 3:
            st.error(f"Expected 3D tensor [epochs, channels, samples], got shape={arr.shape}")
            return
        epoch_idx = int(row["epoch_idx"])
        st.caption(f"Stimulus: {row['stimulus_id']}")
    else:
        row_idx = st.selectbox(
            "Select recording",
            options=list(range(len(epoch_df))),
            format_func=lambda i: f"{epoch_df.iloc[i]['dataset_id']} | {epoch_df.iloc[i]['subject_id']} | {Path(str(epoch_df.iloc[i]['epochs_file'])).name}",
        )
        row = epoch_df.iloc[row_idx]
        arr = cached_epoch_array(str(row["epochs_file"]))
        if arr.ndim != 3:
            st.error(f"Expected 3D tensor [epochs, channels, samples], got shape={arr.shape}")
            return
        epoch_idx = st.slider("Epoch", 0, int(arr.shape[0] - 1), 0)
    default_channels = list(range(min(6, arr.shape[1])))
    channels = st.multiselect("Channels", options=list(range(arr.shape[1])), default=default_channels)
    normalize = st.checkbox("Normalize selected channels (z-score)", value=False)

    epoch_signal = arr[epoch_idx].copy()
    if normalize and channels:
        for ch in channels:
            x = epoch_signal[ch]
            std = float(x.std()) or 1.0
            epoch_signal[ch] = (x - x.mean()) / std

    if not channels:
        st.info("Select at least one channel.")
    else:
        st.plotly_chart(line_eeg_traces(epoch_signal, channels), use_container_width=True)

    st.markdown("#### Channel Statistics")
    channel_stats = describe_epoch_tensor(arr)
    st.dataframe(channel_stats, use_container_width=True, hide_index=True)

    st.markdown("#### Montage Visualization")
    source_file = str(row["source_file"])
    channel_names = cached_channel_names(source_file)
    if not channel_names:
        channel_names = [f"CH{idx}" for idx in range(arr.shape[1])]

    channel_value_map: dict[str, float] = {}
    for _, stat_row in channel_stats.iterrows():
        ch_idx = int(stat_row["channel"])
        if ch_idx < len(channel_names):
            channel_value_map[channel_names[ch_idx]] = float(stat_row["std"])

    st.plotly_chart(montage_scatter(channel_names[: arr.shape[1]], channel_value_map), use_container_width=True)

    st.markdown("#### PSD (First Selected Channel)")
    if channels:
        freqs, power = compute_psd(epoch_signal[channels[0]])
        psd_df = pd.DataFrame({"freq_hz": freqs, "power": power})
        st.line_chart(psd_df.set_index("freq_hz"))
    else:
        st.info("Select a channel to show PSD.")

    st.markdown("#### Original vs ZUNA Compare")
    zuna_rows = epoch_df[epoch_df["dataset_id"].astype(str).str.contains("zuna", case=False, na=False)]
    if "task_label" in epoch_df.columns:
        base_task = normalize_task_label(str(row.get("task_label", "")))
        candidates = zuna_rows[
            (zuna_rows["subject_id"] == row["subject_id"])
            & (zuna_rows["task_label"].astype(str).map(normalize_task_label) == base_task)
        ]
    elif is_long_format and "pipeline_source" in epoch_df.columns:
        candidates = zuna_rows[(zuna_rows["subject_id"] == row["subject_id"]) & (zuna_rows["epoch_idx"] == row["epoch_idx"])]
        candidates = candidates[candidates["epochs_file"].str.replace("zuna_", "").str.contains(Path(row["epochs_file"]).stem, na=False)]
    else:
        candidates = pd.DataFrame()
    if candidates.empty:
        st.info("No matching ZUNA-augmented recording found for this selection.")
    else:
        zuna_row = candidates.iloc[0]
        zuna_arr = cached_epoch_array(str(zuna_row["epochs_file"]))
        if is_long_format and "epoch_idx" in zuna_row:
            zuna_ep = zuna_arr[int(zuna_row["epoch_idx"]) : int(zuna_row["epoch_idx"]) + 1]
            stats_df = compare_tensor_stats(arr[epoch_idx : epoch_idx + 1], zuna_ep)
        else:
            stats_df = compare_tensor_stats(arr, zuna_arr)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


def render_data_characteristics(epoch_df_base: pd.DataFrame, epoch_df_with_zuna: pd.DataFrame, size_report: dict[str, Any]):
    st.subheader("Data Characteristics & Statistics")
    summary_base = summarise_epoch_index(epoch_df_base)
    summary_zuna = summarise_epoch_index(epoch_df_with_zuna)

    c1, c2, c3 = st.columns(3)
    c1.metric("Base total epochs", summary_base["total_epochs"])
    c2.metric("With ZUNA total epochs", summary_zuna["total_epochs"])
    c3.metric("Delta epochs", summary_zuna["total_epochs"] - summary_base["total_epochs"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Base recordings", summary_base["recordings"])
    c5.metric("With ZUNA recordings", summary_zuna["recordings"])
    c6.metric("Delta recordings", summary_zuna["recordings"] - summary_base["recordings"])

    st.markdown("#### Dataset Breakdown (Base)")
    st.dataframe(dataset_breakdown(epoch_df_base), use_container_width=True, hide_index=True)

    st.markdown("#### Dataset Breakdown (With ZUNA)")
    st.dataframe(dataset_breakdown(epoch_df_with_zuna), use_container_width=True, hide_index=True)

    st.markdown("#### Unknown Label Counts (Base)")
    if epoch_df_base.empty:
        st.info("No base epoch data.")
    else:
        unknown_modality = int((epoch_df_base["modality"] == "unknown").sum())
        unknown_task = int((epoch_df_base["task_label"] == "unknown").sum())
        st.write({"unknown_modality_rows": unknown_modality, "unknown_task_rows": unknown_task})

    st.markdown("#### Dataset Size Report")
    st.json(size_report)


def render_run_inspector():
    st.subheader("Run Inspector")
    runs_payload = cached_runs()
    if not runs_payload:
        st.warning("No run JSON artifacts found in `runs/`.")
        return

    run_df = normalize_run_rows(runs_payload)
    st.dataframe(run_df, use_container_width=True, hide_index=True)
    st.plotly_chart(grouped_run_metrics(run_df), use_container_width=True)

    st.markdown("#### Detailed Run Payload")
    selected_run = st.selectbox("Run JSON", options=sorted(runs_payload.keys()))
    st.json(runs_payload[selected_run])

    if "zuna_ablation_summary" in runs_payload:
        st.markdown("#### ZUNA Ablation Deltas")
        st.write(runs_payload["zuna_ablation_summary"])


def render_pipeline_visual():
    st.subheader("Pipeline Visual + Explanation")
    st.markdown(
        """
This pipeline ingests dataset registry entries, downloads sample EEG recordings, harmonizes files into
a unified manifest, preprocesses them into fixed-size epoch tensors, then trains baseline and LaBraM models.
An optional ZUNA branch augments EEG signals and builds a separate index for augmentation-aware analysis.
        """
    )

    mermaid_code = """
flowchart TD
registryCsv[eeg_datasets_for_labram.csv] --> sizeInventory[scripts/size_inventory.py]
registryCsv --> downloadSamples[scripts/download_sample_subjects.py]
sizeInventory --> datasetSizeReport[data/manifests/dataset_size_report.json]
downloadSamples --> rawSamples[data/raw_samples]
rawSamples --> buildManifest[scripts/build_unified_manifest.py]
buildManifest --> unifiedManifest[data/manifests/unified_manifest.csv]
unifiedManifest --> preprocess[scripts/preprocess_eeg.py]
preprocess --> epochIndex[data/manifests/epoch_index.csv]
preprocess --> epochsNpy[data/processed/epochs]
epochsNpy --> trainBaseline[scripts/train_baseline.py]
trainBaseline --> baselineRuns[runs/baseline_metrics.json]
rawSamples --> zunaAug[scripts/run_zuna_augmentation.py]
zunaAug --> epochsAug[data/processed/epochs_aug]
epochsAug --> zunaIndex[scripts/build_zuna_augmented_index.py]
zunaIndex --> epochIndexZuna[data/manifests/epoch_index_with_zuna.csv]
epochsNpy --> trainLabram[scripts/train_labram.py]
trainLabram --> labramRuns[runs/labram_metrics.json]
"""
    html = f"""
<div class="mermaid">{mermaid_code}</div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true }});
</script>
"""
    components.html(html, height=650, scrolling=True)

    st.markdown("#### Stage-by-stage")
    st.markdown(
        """
- `scripts/size_inventory.py`: computes dataset footprint estimates.
- `scripts/download_sample_subjects.py`: acquires sample subject recordings into `data/raw_samples`.
- `scripts/build_unified_manifest.py`: normalizes raw file inventory into `data/manifests/unified_manifest.csv`.
- `scripts/preprocess_eeg.py`: filters/resamples/epochs recordings and writes tensor artifacts + epoch index.
- `scripts/train_baseline.py`: trains a classical baseline and writes baseline metrics JSON.
- `scripts/run_zuna_augmentation.py`: runs ZUNA augmentation branch.
- `scripts/build_zuna_augmented_index.py`: adds augmented rows into the ZUNA-inclusive epoch index.
- `scripts/train_labram.py`: runs LaBraM fine-tuning scaffold and writes `runs/labram_metrics.json`.
        """
    )


def main():
    st.set_page_config(page_title="EEG Exploration Dashboard", layout="wide")
    st.title("EEG Exploration Dashboard")

    registry_df = cached_registry()
    size_report = cached_size_report()
    unified_df = cached_unified_manifest()
    epoch_df_base = cached_epoch_index(with_zuna=False)
    epoch_df_with_zuna = cached_epoch_index(with_zuna=True)

    index_options = load_epoch_index_options(PROJECT_ROOT)
    index_labels = [opt[0] for opt in index_options]
    index_paths = [str(opt[1]) for opt in index_options]
    selected_index_label = st.sidebar.selectbox("Epoch index", index_labels, index=0)
    selected_index_path = index_paths[index_labels.index(selected_index_label)]
    if Path(selected_index_path).exists():
        active_epoch_df = cached_epoch_index_from_path(selected_index_path)
    else:
        include_zuna = st.sidebar.toggle("Use ZUNA-inclusive epoch index", value=True)
        active_epoch_df = epoch_df_with_zuna if include_zuna else epoch_df_base

    tabs = st.tabs(
        [
            "Dataset Explorer",
            "EEG Visual Inspector",
            "Data Characteristics",
            "Run Inspector",
            "Pipeline Visual",
        ]
    )

    with tabs[0]:
        render_dataset_explorer(active_epoch_df, unified_df, registry_df)
    with tabs[1]:
        render_eeg_visual_inspector(active_epoch_df)
    with tabs[2]:
        render_data_characteristics(epoch_df_base, epoch_df_with_zuna, size_report)
    with tabs[3]:
        render_run_inspector()
    with tabs[4]:
        render_pipeline_visual()

    st.caption(f"Project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()

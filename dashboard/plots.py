from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bar_total_epochs_by_dataset(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    if "stimulus_id" in df.columns:
        view = df.groupby("dataset_id", as_index=False).size().rename(columns={"size": "n_epochs"}).sort_values("n_epochs", ascending=False)
    else:
        view = (
            df.assign(n_epochs=pd.to_numeric(df["n_epochs"], errors="coerce").fillna(0))
            .groupby("dataset_id", as_index=False)["n_epochs"]
            .sum()
            .sort_values("n_epochs", ascending=False)
        )
    return px.bar(view, x="dataset_id", y="n_epochs", title="Epoch Count by Dataset")


def bar_modality_distribution(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    view = df.groupby("modality", as_index=False).size().rename(columns={"size": "count"})
    return px.bar(view, x="modality", y="count", title="Recording Count by Modality")


def bar_task_distribution(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    if "task_label" not in df.columns:
        return go.Figure()
    view = df.groupby("task_label", as_index=False).size().rename(columns={"size": "count"})
    return px.bar(view, x="task_label", y="count", title="Recording Count by Task")


def bar_stimulus_distribution(df: pd.DataFrame):
    if df.empty or "stimulus_id" not in df.columns:
        return go.Figure()
    view = df.groupby("stimulus_id", as_index=False).size().rename(columns={"size": "count"})
    return px.bar(view, x="stimulus_id", y="count", title="Epoch Count by Stimulus")


def hist_channel_counts(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    view = df.assign(n_channels=pd.to_numeric(df["n_channels"], errors="coerce").fillna(0))
    return px.histogram(view, x="n_channels", nbins=15, title="Channel Count Distribution")


def line_eeg_traces(epoch_signal, selected_channels):
    fig = go.Figure()
    for ch in selected_channels:
        fig.add_trace(go.Scatter(y=epoch_signal[ch], mode="lines", name=f"ch_{ch}"))
    fig.update_layout(
        title="Epoch Time-Series (Selected Channels)",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=450,
    )
    return fig


def grouped_run_metrics(run_df: pd.DataFrame):
    if run_df.empty:
        return go.Figure()
    usable = run_df[run_df["run_type"].isin(["baseline", "baseline_zuna_aug", "labram"])].copy()
    if usable.empty:
        return go.Figure()
    melted = usable.melt(
        id_vars=["run_name", "run_type"],
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )
    return px.bar(
        melted,
        x="run_type",
        y="value",
        color="metric",
        barmode="group",
        title="Run Metrics (Accuracy / Macro-F1)",
    )


def montage_scatter(channel_names: list[str], values: dict[str, float] | None = None):
    fig = go.Figure()
    if not channel_names:
        fig.update_layout(title="Montage (no channel names available)")
        return fig

    values = values or {}
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_map = montage.get_positions()["ch_pos"]
    xs, ys, labels, colors = [], [], [], []

    for idx, ch in enumerate(channel_names):
        key = ch.upper()
        pos = pos_map.get(ch) or pos_map.get(key)
        if pos is None:
            # Place unknown channels on a ring so they are still visible.
            angle = 2 * np.pi * (idx / max(1, len(channel_names)))
            x, y = float(np.cos(angle)), float(np.sin(angle))
        else:
            x, y = float(pos[0]), float(pos[1])
        xs.append(x)
        ys.append(y)
        labels.append(ch)
        colors.append(float(values.get(ch, values.get(key, 0.0))))

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 10, "color": colors, "colorscale": "Viridis", "showscale": True},
            hovertemplate="Channel: %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="EEG Montage View (2D top projection)",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        height=520,
    )
    return fig

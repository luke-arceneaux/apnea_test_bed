"""Plotly charts for the Streamlit test bed."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_signal_preview(
    df: pd.DataFrame,
    flatline_times: Sequence[Tuple[float, float]] | None = None,
    onset_times: Sequence[Tuple[float, float]] | None = None,
    max_points: int = 8000,
) -> go.Figure:
    """Audio + SpO2 preview with optional event overlays."""
    plot_df = df
    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        plot_df = df.iloc[::step]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=("Conditioned audio", "SpO2"),
    )

    fig.add_trace(
        go.Scatter(x=plot_df["time"], y=plot_df["value"], name="Audio", line=dict(width=1, color="#3f51b5")),
        row=1,
        col=1,
    )

    if "oxygen" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df["time"], y=plot_df["oxygen"], name="SpO2", line=dict(width=1, color="#e53935")),
            row=2,
            col=1,
        )

    shapes = []
    if flatline_times:
        for start, end in flatline_times:
            shapes.append(dict(type="rect", x0=start, x1=end, y0=0, y1=1, yref="paper", fillcolor="rgba(255,152,0,0.15)", line_width=0))
    if onset_times:
        for start, end in onset_times:
            shapes.append(dict(type="rect", x0=start, x1=end, y0=0, y1=1, yref="paper", fillcolor="rgba(76,175,80,0.12)", line_width=0))

    fig.update_layout(height=520, hovermode="x unified", shapes=shapes, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    return fig


def plot_confusion_matrix(cm: List[List[int]], labels: List[str]) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"True {l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(title="Validation window confusion matrix", height=380)
    return fig


def plot_label_distribution(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    name_map = {"0": "Non-apnea", "1": "Onset apnea", "2": "Flatline"}
    display = [name_map.get(str(k), str(k)) for k in labels]
    fig = go.Figure(data=[go.Bar(x=display, y=values, marker_color=["#4caf50", "#ff9800", "#f44336"][: len(display)])])
    fig.update_layout(title="Validation window labels", yaxis_title="Count", height=320)
    return fig

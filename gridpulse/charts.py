"""Plotly figure builders for the dashboard's mosaic panels.

Kept separate from the callback so ``app.py`` stays about *what* the app is (one
typed function), not *how* each chart is drawn.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .config import ACCENT, FUEL_TYPES

_FUEL_COLOR = {label: color for _code, (label, color) in FUEL_TYPES.items()}

# No Plotly title — the mosaic panel header already labels each chart, and a
# figure title collided with the top horizontal legend.
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=56, r=20, t=34, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="x unified",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
)


def demand_figure(
    region_label: str,
    demand: pd.DataFrame,
    fc: pd.DataFrame | None,
    metric_col: str,
    metric_label: str,
    show_forecast: bool,
) -> go.Figure:
    """History of the selected metric with an optional forward forecast + band."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=demand.index, y=demand[metric_col], name=metric_label,
            line=dict(color=ACCENT, width=2.2),
        )
    )
    if metric_col == "demand" and "forecast" in demand.columns:
        fig.add_trace(
            go.Scatter(
                x=demand.index, y=demand["forecast"], name="Day-ahead forecast",
                line=dict(color="#868e96", width=1.2, dash="dot"),
            )
        )
    if show_forecast and fc is not None and not fc.empty:
        band_x = list(fc.index) + list(fc.index[::-1])
        band_y = list(fc["upper"]) + list(fc["lower"][::-1])
        fig.add_trace(
            go.Scatter(
                x=band_x, y=band_y, fill="toself",
                fillcolor="rgba(47,158,68,0.14)", line=dict(width=0),
                name="95% interval", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc.index, y=fc["forecast"], name=f"{len(fc)}h forecast",
                line=dict(color=ACCENT, width=2.2, dash="dash"),
            )
        )
        fig.add_vline(
            x=demand.index[-1], line=dict(color="#adb5bd", width=1, dash="dot")
        )
    fig.update_layout(yaxis_title="MW", **_LAYOUT)
    return fig


def mix_figure(region_label: str, fuel: pd.DataFrame) -> go.Figure:
    """Stacked-area generation mix over the window, coloured by fuel."""
    fig = go.Figure()
    for label in fuel.columns:
        color = _FUEL_COLOR.get(label, "#adb5bd")
        fig.add_trace(
            go.Scatter(
                x=fuel.index, y=fuel[label], name=label, stackgroup="mix",
                mode="none", fillcolor=color, hovertemplate="%{y:,.0f} MW",
            )
        )
    fig.update_layout(yaxis_title="MW", **_LAYOUT)
    return fig

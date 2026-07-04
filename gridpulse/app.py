"""GridPulse — the whole dashboard is the one typed function below.

``explore(...)`` takes typed inputs (Fast Dash infers a region dropdown, metric
dropdown, two sliders, and a switch from the annotations) and returns four
outputs arranged in a mosaic. A LangGraph + OpenRouter copilot is mounted as a
sidecar that reads and drives those same inputs.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pandas as pd

from fast_dash import FastDash, Graph, Markdown, Table

from . import charts
from .agent import make_copilot
from .config import ACCENT, REGIONS
from .data import data_source, load_demand, load_fuel_mix
from .forecast import forecast_demand
from .insights import build_insights

# --- typed input aliases (drive Fast Dash component inference) ------------- #
RegionLit = Literal[tuple(REGIONS)]            # region dropdown (friendly labels)
MetricLit = Literal["Demand", "Net generation"]  # which series to plot + forecast
Lookback = Annotated[int, range(1, 31)]        # days of history -> Slider
Horizon = Annotated[int, range(6, 73)]         # forecast hours -> Slider

_METRIC_COLUMN = {"Demand": "demand", "Net generation": "net_generation"}


def explore(
    region: RegionLit = "Texas (ERCOT)",
    metric: MetricLit = "Demand",
    lookback_days: Lookback = 7,
    forecast_horizon: Horizon = 48,
    show_forecast: bool = True,
):
    """Explore a US grid region's demand, generation mix, and short-term forecast.

    GridPulse pulls hourly electricity data (US EIA Form 930) for a chosen grid
    operator, charts demand and the generation mix, and forecasts demand a day or
    two ahead with a confidence band. Ask the Grid Copilot to reconfigure it for
    you, or drive the controls yourself. Built on Fast Dash; the entire app is
    this one typed function.

    Args:
        region: Balancing authority (grid operator) to inspect.
        metric: Series to chart and forecast (demand or net generation).
        lookback_days: Days of hourly history to show.
        forecast_horizon: Hours to forecast ahead.
        show_forecast: Overlay the forward forecast and its confidence band.

    Returns:
        A demand-and-forecast chart, a generation-mix chart, a key-stats table,
        and an auto-generated grid-insights summary.
    """
    demand = load_demand(region, lookback_days)
    fuel = load_fuel_mix(region, lookback_days)
    source = data_source()

    metric_col = _METRIC_COLUMN[metric]
    fc = forecast_demand(demand[metric_col], horizon=forecast_horizon) if show_forecast else None

    demand_fig = charts.demand_figure(
        region, demand, fc, metric_col, metric, show_forecast
    )
    mix_fig = charts.mix_figure(region, fuel)
    stats = _stats_table(demand, fuel, fc, metric_col, source)
    insights = build_insights(region, demand, fuel, fc if fc is not None else pd.DataFrame(), source)
    return demand_fig, mix_fig, stats, insights


def _stats_table(demand, fuel, fc, metric_col, source) -> pd.DataFrame:
    """Compact key/value table of the current grid state."""
    cur = float(demand[metric_col].iloc[-1])
    peak = float(demand[metric_col].tail(24).max())
    trough = float(demand[metric_col].tail(24).min())
    rows = [
        ("Current", f"{cur:,.0f} MW"),
        ("24h peak", f"{peak:,.0f} MW"),
        ("24h low", f"{trough:,.0f} MW"),
    ]
    if not fuel.empty:
        last = fuel.iloc[-1]
        total = float(last.sum())
        renew = float(sum(last.get(f, 0.0) for f in ("Solar", "Wind", "Hydro")))
        rows.append(("Renewable share", f"{renew / total * 100:.0f}%" if total else "n/a"))
    if fc is not None and not fc.empty:
        rows.append(("Forecast peak", f"{float(fc['forecast'].max()):,.0f} MW"))
    rows.append(("Data source", "Live EIA" if source == "live" else "Sample"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


_PAGE_TITLE = "GridPulse — live US grid explorer with an AI copilot"
_PAGE_DESC = (
    "Explore live US electricity demand, generation mix, and a short-term "
    "forecast for any grid operator. An AI copilot reconfigures the dashboard "
    "for you. Built on Fast Dash."
)
# Lightning-bolt favicon as an inline SVG data URI (no asset file needed).
_FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
    "%3Ctext y='.9em' font-size='90'%3E%E2%9A%A1%3C/text%3E%3C/svg%3E"
)


def _index_html() -> str:
    """Dash index template with social meta + a favicon (all placeholders kept)."""
    return f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        <meta name="description" content="{_PAGE_DESC}">
        <meta property="og:title" content="{_PAGE_TITLE}">
        <meta property="og:description" content="{_PAGE_DESC}">
        <meta property="og:type" content="website">
        <meta name="twitter:card" content="summary">
        <meta name="twitter:title" content="{_PAGE_TITLE}">
        <meta name="twitter:description" content="{_PAGE_DESC}">
        <link rel="icon" href="{_FAVICON}">
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>"""


def build_app() -> FastDash:
    """Construct the Fast Dash app (kept a function so tests can build in isolation)."""
    fd = FastDash(
        callback_fn=explore,
        outputs=[Graph, Graph, Table, Markdown],
        output_labels=[
            "Demand & forecast",
            "Generation mix",
            "Key stats",
            "Grid insights",
        ],
        title="GridPulse",
        subheader="Live US electricity grid explorer with an AI copilot",
        accent=ACCENT,
        github_url="https://github.com/dkedar7/fast_dash",
        chat_agent=make_copilot(),
        chat_agent_title="Grid Copilot",
        chat_agent_drive=True,
        about=True,
    )
    fd.app.title = _PAGE_TITLE
    fd.app.index_string = _index_html()
    return fd


app = build_app()
server = app.app.server  # WSGI entry point for gunicorn / Fly

if __name__ == "__main__":
    app.run()

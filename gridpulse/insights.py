"""Auto-generated grid narrative — the 'beyond basic analysis' layer.

Turns the numbers behind the charts into a few plain-English sentences: where
demand sits now, the renewable share, the recent peak, and what the forecast
implies for headroom. Deterministic, no LLM — this is the dashboard's own
readout, distinct from the chat copilot.
"""

from __future__ import annotations

import pandas as pd

_RENEWABLE = {"Solar", "Wind", "Hydro"}


def build_insights(
    region_label: str,
    demand: pd.DataFrame,
    fuel: pd.DataFrame,
    fc: pd.DataFrame,
    source: str,
) -> str:
    """Markdown summary of the current grid state and forecast."""
    lines: list[str] = [f"### {region_label} — grid readout"]

    cur = float(demand["demand"].iloc[-1])
    day_ago = (
        float(demand["demand"].iloc[-25])
        if len(demand) >= 25
        else float(demand["demand"].iloc[0])
    )
    delta = (cur - day_ago) / day_ago * 100 if day_ago else 0.0
    arrow = "up" if delta >= 0 else "down"
    lines.append(
        f"- **Demand now:** {cur:,.0f} MW "
        f"({arrow} {abs(delta):.1f}% vs 24h ago)."
    )

    # renewable share of the latest hour's generation mix
    if not fuel.empty:
        last = fuel.iloc[-1]
        total = float(last.sum())
        if total > 0:
            renew = float(sum(last.get(f, 0.0) for f in _RENEWABLE))
            top_fuel = last.idxmax()
            lines.append(
                f"- **Renewables:** {renew / total * 100:.0f}% of generation "
                f"this hour; largest source is **{top_fuel}**."
            )

    # last-24h peak and its timing
    window = demand["demand"].tail(24)
    if len(window):
        peak_val = float(window.max())
        peak_at = window.idxmax()
        lines.append(
            f"- **24h peak:** {peak_val:,.0f} MW at "
            f"{peak_at:%a %H:%M} UTC."
        )

    # forecast headroom
    if not fc.empty:
        fpeak = float(fc["forecast"].max())
        fpeak_at = fc["forecast"].idxmax()
        recent_peak = float(demand["demand"].tail(48).max())
        head = (fpeak - recent_peak) / recent_peak * 100 if recent_peak else 0.0
        verb = "above" if head >= 0 else "below"
        lines.append(
            f"- **Forecast:** next peak ~{fpeak:,.0f} MW on "
            f"{fpeak_at:%a %H:%M} UTC, {abs(head):.1f}% {verb} the recent peak."
        )

    note = (
        "Live EIA feed."
        if source == "live"
        else "Sample data (set EIA_API_KEY for the live EIA feed)."
    )
    lines.append(f"\n*{note}*")
    return "\n".join(lines)

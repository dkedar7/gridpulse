"""Short-horizon demand forecasting.

A Holt-Winters (triple exponential smoothing) model with a 24-hour seasonal
cycle, plus empirical confidence bands that widen with the horizon. Chosen over
SARIMAX so a forecast fits comfortably inside a single interactive callback
(sub-second on a week of hourly data) while still capturing the daily load
shape. Illustrative statistical forecast — not a production grid forecast.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def forecast_demand(demand: pd.Series, horizon: int = 48) -> pd.DataFrame:
    """Forecast ``horizon`` hours ahead from an hourly demand series.

    Returns a frame indexed by future hour with ``forecast`` / ``lower`` /
    ``upper`` (MW). Degrades to a seasonal-naive forecast if the history is too
    short or the model fails to converge, so a chart is always produced.
    """
    y = pd.to_numeric(demand, errors="coerce").astype(float).ffill().dropna()
    if len(y) < 2:
        raise ValueError("need at least a couple of demand points to forecast")

    future_idx = pd.date_range(
        y.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    season = 24

    if len(y) >= 2 * season:
        try:
            point, resid_std = _holt_winters(y, horizon, season)
        except Exception:  # noqa: BLE001 — any fit failure -> seasonal naive
            point, resid_std = _seasonal_naive(y, horizon, season)
    else:
        point, resid_std = _seasonal_naive(y, horizon, season)

    steps = np.arange(1, horizon + 1)
    band = 1.96 * resid_std * np.sqrt(1.0 + steps / season)
    return pd.DataFrame(
        {
            "forecast": np.asarray(point, dtype=float),
            "lower": np.asarray(point, dtype=float) - band,
            "upper": np.asarray(point, dtype=float) + band,
        },
        index=future_idx,
    )


def _holt_winters(y: pd.Series, horizon: int, season: int) -> tuple[np.ndarray, float]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter is not actionable here
        fit = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=season,
            initialization_method="estimated",
        ).fit()
        point = np.asarray(fit.forecast(horizon), dtype=float)
        resid_std = float(np.nanstd(np.asarray(fit.resid, dtype=float)))
    if not np.isfinite(point).all():
        raise ValueError("non-finite Holt-Winters forecast")
    if not np.isfinite(resid_std) or resid_std <= 0:
        resid_std = float(np.nanstd(y.to_numpy())) * 0.05
    return point, resid_std


def _seasonal_naive(y: pd.Series, horizon: int, season: int) -> tuple[np.ndarray, float]:
    """Repeat the most recent daily cycle; band from day-over-day differences."""
    vals = y.to_numpy(dtype=float)
    last_cycle = vals[-season:] if len(vals) >= season else vals
    reps = int(np.ceil(horizon / len(last_cycle)))
    point = np.tile(last_cycle, reps)[:horizon]
    if len(vals) > season:
        resid_std = float(np.nanstd(vals[season:] - vals[:-season]))
    else:
        resid_std = float(np.nanstd(vals)) * 0.1
    if not np.isfinite(resid_std) or resid_std <= 0:
        resid_std = float(np.nanmean(np.abs(vals))) * 0.05 or 1.0
    return point, resid_std

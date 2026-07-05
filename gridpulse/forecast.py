"""Short-horizon demand forecasting.

A Holt-Winters (triple exponential smoothing) model with a 24-hour seasonal
cycle, plus empirical confidence bands that widen with the horizon. Chosen over
SARIMAX so a forecast fits comfortably inside a single interactive callback
(sub-second on a week of hourly data) while still capturing the daily load
shape. Illustrative statistical forecast — not a production grid forecast.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd

# Fitting Holt-Winters is the one CPU-bound step in a callback; on a small VM it
# dominates latency. The synthetic/live series is stable within an hour, so we
# memoise by (series fingerprint, horizon). Bounded so it can't grow unbounded.
_FC_CACHE: dict[tuple, pd.DataFrame] = {}
_FC_CACHE_MAX = 64


def _series_key(y: pd.Series, horizon: int) -> tuple:
    digest = hashlib.blake2b(
        np.ascontiguousarray(y.to_numpy(dtype=float)).tobytes(), digest_size=8
    ).hexdigest()
    return (digest, len(y), horizon)


def forecast_demand(demand: pd.Series, horizon: int = 48) -> pd.DataFrame:
    """Forecast ``horizon`` hours ahead from an hourly demand series.

    Returns a frame indexed by future hour with ``forecast`` / ``lower`` /
    ``upper`` (MW). Degrades to a seasonal-naive forecast if the history is too
    short or the model fails to converge, so a chart is always produced.
    Results are memoised per (series, horizon) so repeat views are instant.
    """
    y = pd.to_numeric(demand, errors="coerce").astype(float).ffill().dropna()

    key = _series_key(y, horizon) if len(y) else None
    if key is not None and key in _FC_CACHE:
        return _FC_CACHE[key].copy()
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
    result = pd.DataFrame(
        {
            "forecast": np.asarray(point, dtype=float),
            "lower": np.asarray(point, dtype=float) - band,
            "upper": np.asarray(point, dtype=float) + band,
        },
        index=future_idx,
    )
    if key is not None:
        if len(_FC_CACHE) >= _FC_CACHE_MAX:
            _FC_CACHE.clear()
        _FC_CACHE[key] = result.copy()
    return result


def warmup() -> None:
    """Prime statsmodels/BLAS so the first real forecast isn't cold.

    The initial Holt-Winters fit pays a one-off import/JIT cost; running a tiny
    throwaway fit at startup moves that cost off the first user request.
    """
    try:
        idx = pd.date_range("2026-01-01", periods=72, freq="h")
        y = pd.Series(
            40000 + 8000 * np.sin(np.arange(72) / 24 * 2 * np.pi), index=idx
        )
        forecast_demand(y, 24)
        _FC_CACHE.clear()  # don't keep the synthetic warmup entry
    except Exception:  # noqa: BLE001 — warmup is best-effort
        pass


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

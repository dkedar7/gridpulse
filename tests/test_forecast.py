"""Forecast layer: always returns a well-ordered band of the right length."""

import numpy as np
import pandas as pd

from gridpulse.forecast import forecast_demand


def _series(n):
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    # daily sine + slope so Holt-Winters has real seasonality to fit
    hours = np.arange(n)
    y = 40000 + 8000 * np.sin(hours / 24 * 2 * np.pi) + hours * 2
    return pd.Series(y, index=idx)


def test_horizon_and_columns():
    fc = forecast_demand(_series(24 * 10), horizon=48)
    assert len(fc) == 48
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert fc.index[0] > _series(24 * 10).index[-1]


def test_band_is_ordered():
    fc = forecast_demand(_series(24 * 10), horizon=36)
    assert (fc["lower"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper"]).all()


def test_band_widens_with_horizon():
    fc = forecast_demand(_series(24 * 10), horizon=48)
    width = fc["upper"] - fc["lower"]
    assert width.iloc[-1] >= width.iloc[0]


def test_short_series_falls_back_to_seasonal_naive():
    # fewer than 2*24 points -> seasonal-naive branch, still returns a band
    fc = forecast_demand(_series(30), horizon=12)
    assert len(fc) == 12
    assert np.isfinite(fc.to_numpy()).all()

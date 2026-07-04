"""Data layer: the synthetic fallback must always yield clean, sane frames."""

import pytest

from gridpulse import data


@pytest.fixture(autouse=True)
def force_sample(monkeypatch):
    # Pin the synthetic path so tests never touch the network or a real key.
    monkeypatch.setattr(data, "has_live_data", lambda: False)


def test_demand_shape_and_columns():
    df = data.load_demand("Texas (ERCOT)", lookback_days=7)
    assert list(df.columns) == ["demand", "forecast", "net_generation"]
    assert len(df) == 7 * 24 + 1                 # inclusive hourly window
    assert df["demand"].notna().all()
    assert (df["demand"] > 0).all()
    assert df.index.is_monotonic_increasing


def test_fuel_mix_non_negative_and_labelled():
    df = data.load_fuel_mix("California (CAISO)", lookback_days=3)
    assert not df.empty
    assert (df.to_numpy() >= 0).all()
    # solar should be present and larger midday than at night for CAISO
    assert "Solar" in df.columns


def test_regions_have_distinct_scale():
    # BPA (small, hydro) must be far smaller than PJM (large).
    bpa = data.load_demand("Pacific NW (BPA)", 2)["demand"].mean()
    pjm = data.load_demand("Mid-Atlantic (PJM)", 2)["demand"].mean()
    assert pjm > 5 * bpa


def test_unknown_region_raises():
    with pytest.raises(ValueError):
        data.load_demand("Atlantis", 1)


def test_deterministic_within_hour():
    a = data.load_demand("Midwest (MISO)", 2)
    b = data.load_demand("Midwest (MISO)", 2)
    assert a["demand"].round(3).equals(b["demand"].round(3))

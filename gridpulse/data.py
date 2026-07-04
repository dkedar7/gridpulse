"""Data layer: live EIA v2 grid data with a realistic offline fallback.

Two code paths behind one API:

* **Live** (an ``EIA_API_KEY`` is set) — pulls hourly demand, day-ahead demand
  forecast, net generation, and generation-by-fuel from the EIA v2 API (Form
  EIA-930), cached for 15 minutes.
* **Sample** (no key) — synthesises a physically plausible series anchored to
  the current wall-clock, so the app runs anywhere (local dev, a keyless demo)
  and still reads as "live". Clearly labelled as sample data in the UI.

Both paths return the *same* tidy frames, so nothing downstream knows or cares
which one produced them.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import (
    CACHE_TTL_SECONDS,
    EIA_API_KEY,
    EIA_BASE_URL,
    FUEL_TYPES,
    METRICS,
    REGIONS,
    has_live_data,
)

# --- tiny in-process TTL cache -------------------------------------------- #
_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}


def _cache_get(key: tuple) -> pd.DataFrame | None:
    hit = _CACHE.get(key)
    if hit and (time.monotonic() - hit[0]) < CACHE_TTL_SECONDS:
        return hit[1].copy()
    return None


def _cache_put(key: tuple, df: pd.DataFrame) -> None:
    _CACHE[key] = (time.monotonic(), df.copy())


def data_source() -> str:
    """``"live"`` when pulling from EIA, else ``"sample"``."""
    return "live" if has_live_data() else "sample"


# --- physically plausible per-region parameters --------------------------- #
# base load (MW) and rough renewable character per balancing authority, used by
# the sample generator. Order-of-magnitude realistic, not a forecast.
_REGION_PROFILE: dict[str, dict] = {
    "CISO": {"base": 26000, "swing": 9000, "solar": 0.28, "wind": 0.09, "hydro": 0.09},
    "ERCO": {"base": 52000, "swing": 22000, "solar": 0.11, "wind": 0.26, "hydro": 0.01},
    "PJM":  {"base": 92000, "swing": 30000, "solar": 0.05, "wind": 0.04, "hydro": 0.02},
    "MISO": {"base": 74000, "swing": 24000, "solar": 0.04, "wind": 0.14, "hydro": 0.02},
    "NYIS": {"base": 18000, "swing": 7000,  "solar": 0.04, "wind": 0.04, "hydro": 0.20},
    "ISNE": {"base": 14000, "swing": 5500,  "solar": 0.06, "wind": 0.05, "hydro": 0.08},
    "SWPP": {"base": 30000, "swing": 12000, "solar": 0.05, "wind": 0.35, "hydro": 0.03},
    "BPAT": {"base": 7000,  "swing": 2500,  "solar": 0.01, "wind": 0.12, "hydro": 0.62},
}


def _region_code(region_label: str) -> str:
    if region_label in REGIONS:
        return REGIONS[region_label]
    if region_label in REGIONS.values():
        return region_label
    raise ValueError(f"Unknown region: {region_label!r}")


# ========================================================================== #
# Public API
# ========================================================================== #
def load_demand(region_label: str, lookback_days: int = 7) -> pd.DataFrame:
    """Hourly ``demand`` / ``forecast`` / ``net_generation`` (MW) for a region.

    Indexed by timezone-naive UTC hour, oldest first, ending at the current
    hour. Columns are in megawatts.
    """
    code = _region_code(region_label)
    end = _floor_hour(datetime.now(timezone.utc))
    start = end - timedelta(days=lookback_days)

    if has_live_data():
        key = ("demand", code, lookback_days, int(end.timestamp()) // CACHE_TTL_SECONDS)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        try:
            df = _fetch_region_data(code, start, end)
            _cache_put(key, df)
            return df
        except Exception:  # noqa: BLE001 — any live failure degrades to sample
            pass
    return _synth_demand(code, start, end)


def load_fuel_mix(region_label: str, lookback_days: int = 7) -> pd.DataFrame:
    """Hourly net generation by fuel type (MW) for a region.

    Columns are fuel labels (``Solar``, ``Wind``, ...); index matches
    :func:`load_demand`.
    """
    code = _region_code(region_label)
    end = _floor_hour(datetime.now(timezone.utc))
    start = end - timedelta(days=lookback_days)

    if has_live_data():
        key = ("fuel", code, lookback_days, int(end.timestamp()) // CACHE_TTL_SECONDS)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        try:
            df = _fetch_fuel_data(code, start, end)
            _cache_put(key, df)
            return df
        except Exception:  # noqa: BLE001
            pass
    return _synth_fuel(code, start, end)


# ========================================================================== #
# Live EIA v2 path
# ========================================================================== #
def _eia_get(route: str, params: dict) -> list[dict]:
    import httpx

    url = f"{EIA_BASE_URL}/{route}/data/"
    params = {"api_key": EIA_API_KEY, **params}
    with httpx.Client(timeout=20) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()["response"]["data"]


def _eia_period(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")


def _fetch_region_data(code: str, start: datetime, end: datetime) -> pd.DataFrame:
    rows = _eia_get(
        "electricity/rto/region-data",
        {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": code,
            "facets[type][]": ["D", "DF", "NG"],
            "start": _eia_period(start),
            "end": _eia_period(end),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        },
    )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("EIA returned no region rows")
    raw["period"] = pd.to_datetime(raw["period"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    wide = raw.pivot_table(index="period", columns="type", values="value", aggfunc="mean")
    out = pd.DataFrame(index=wide.index)
    out["demand"] = wide.get("D")
    out["forecast"] = wide.get("DF")
    out["net_generation"] = wide.get("NG")
    return out.sort_index().ffill().dropna(how="all")


def _fetch_fuel_data(code: str, start: datetime, end: datetime) -> pd.DataFrame:
    rows = _eia_get(
        "electricity/rto/fuel-type-data",
        {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": code,
            "start": _eia_period(start),
            "end": _eia_period(end),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        },
    )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("EIA returned no fuel rows")
    raw["period"] = pd.to_datetime(raw["period"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    wide = raw.pivot_table(index="period", columns="fueltype", values="value", aggfunc="mean")
    # Map EIA fuel codes -> friendly labels, keep known fuels in canonical order.
    out = pd.DataFrame(index=wide.index)
    for fcode, (label, _color) in FUEL_TYPES.items():
        if fcode in wide.columns:
            out[label] = wide[fcode].clip(lower=0)
    return out.sort_index().ffill().fillna(0.0)


# ========================================================================== #
# Sample path — deterministic, wall-clock-anchored synthesis
# ========================================================================== #
def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)


def _hours_index(start: datetime, end: datetime) -> pd.DatetimeIndex:
    return pd.date_range(_floor_hour(start), _floor_hour(end), freq="h")


def _demand_curve(idx: pd.DatetimeIndex, profile: dict) -> np.ndarray:
    """A realistic double-peaked daily load with weekly + seasonal shape."""
    hours = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    # morning + evening peaks, trough overnight (normalised 0..1)
    daily = (
        0.55
        + 0.28 * np.exp(-((hours - 8) ** 2) / 8)
        + 0.42 * np.exp(-((hours - 19) ** 2) / 6)
        - 0.20 * np.exp(-((hours - 4) ** 2) / 10)
    )
    weekly = np.where(dow >= 5, 0.90, 1.0)          # weekends lighter
    seasonal = 1.0 + 0.12 * np.cos((doy - 200) / 365 * 2 * np.pi)  # summer peak
    # deterministic per-hour jitter (stable for a given absolute hour)
    seed = (idx.asi8 // 3_600_000_000_000) % 9973
    rng = np.sin(seed * 12.9898) * 43758.5453
    jitter = 1.0 + 0.03 * (rng - np.floor(rng) - 0.5)

    load = profile["base"] + profile["swing"] * (daily - 0.55) * 2.2
    return load * weekly * seasonal * jitter


def _synth_demand(code: str, start: datetime, end: datetime) -> pd.DataFrame:
    profile = _REGION_PROFILE[code]
    idx = _hours_index(start, end)
    demand = _demand_curve(idx, profile)
    # day-ahead forecast: demand with a small, slightly laggy error
    fc_seed = (idx.asi8 // 3_600_000_000_000 + 7) % 9973
    fc_noise = np.sin(fc_seed * 78.233) * 43758.5453
    forecast = demand * (1.0 + 0.018 * (fc_noise - np.floor(fc_noise) - 0.5))
    net_gen = demand * 0.98  # net of interchange, roughly balances load
    return pd.DataFrame(
        {"demand": demand, "forecast": forecast, "net_generation": net_gen}, index=idx
    )


def _synth_fuel(code: str, start: datetime, end: datetime) -> pd.DataFrame:
    profile = _REGION_PROFILE[code]
    idx = _hours_index(start, end)
    total = _demand_curve(idx, profile)
    hours = idx.hour.to_numpy()

    # solar tracks daylight; wind is noisy; hydro/nuclear steady; gas/coal fill.
    solar_shape = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    wind_seed = (idx.asi8 // 3_600_000_000_000 + 3) % 9973
    wind_noise = np.abs(np.sin(wind_seed * 37.719))
    shares = {
        "Solar": profile["solar"] * solar_shape * 2.4,
        "Wind": profile["wind"] * (0.5 + wind_noise),
        "Hydro": np.full(len(idx), profile["hydro"]),
        "Nuclear": np.full(len(idx), 0.18 if code in ("PJM", "MISO", "ISNE", "NYIS") else 0.09),
    }
    used = sum(shares.values())
    fossil = np.clip(1.0 - used, 0.05, None)
    shares["Natural gas"] = fossil * 0.72
    shares["Coal"] = fossil * (0.24 if code in ("MISO", "PJM", "SWPP") else 0.10)
    shares["Other"] = fossil * 0.04

    out = pd.DataFrame(index=idx)
    norm = sum(shares.values())
    for label, frac in shares.items():
        out[label] = total * np.asarray(frac) / norm
    # keep canonical column order from config
    ordered = [lbl for _c, (lbl, _clr) in FUEL_TYPES.items() if lbl in out.columns]
    return out[ordered].clip(lower=0)

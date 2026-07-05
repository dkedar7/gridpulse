"""Static configuration for GridPulse: regions, metrics, fuel types, colours.

Kept deliberately declarative so the app callback, the data layer, and the chat
agent all read the *same* source of truth for what a "region" or "metric" is.
"""

from __future__ import annotations

import os

# --- Balancing authorities we expose -------------------------------------- #
# Friendly label -> EIA "respondent" code (Form EIA-930). A curated set of the
# large, well-instrumented grid operators so every region has clean hourly data.
REGIONS: dict[str, str] = {
    "California (CAISO)": "CISO",
    "Texas (ERCOT)": "ERCO",
    "Mid-Atlantic (PJM)": "PJM",
    "Midwest (MISO)": "MISO",
    "New York (NYISO)": "NYIS",
    "New England (ISO-NE)": "ISNE",
    "Central (SPP)": "SWPP",
    "Pacific NW (BPA)": "BPAT",
}
REGION_CODES: dict[str, str] = {v: k for k, v in REGIONS.items()}

# --- Metric series on the region-data endpoint ---------------------------- #
# Friendly label -> EIA "type" code. D = demand, DF = day-ahead demand forecast,
# NG = net generation.
METRICS: dict[str, str] = {
    "Demand": "D",
    "Day-ahead forecast": "DF",
    "Net generation": "NG",
}

# --- Fuel types on the fuel-type-data endpoint ---------------------------- #
# EIA "fueltype" code -> (label, hex colour). Colours are chosen to read as an
# intuitive energy palette (renewables green/gold, fossil warm, nuclear violet).
FUEL_TYPES: dict[str, tuple[str, str]] = {
    "SUN": ("Solar", "#f2c037"),
    "WND": ("Wind", "#2f9e44"),
    "WAT": ("Hydro", "#1c7ed6"),
    "NUC": ("Nuclear", "#7048e8"),
    "NG": ("Natural gas", "#e8590c"),
    "COL": ("Coal", "#495057"),
    "OIL": ("Oil", "#a61e4d"),
    "OTH": ("Other", "#adb5bd"),
}
FUEL_ORDER = list(FUEL_TYPES.keys())  # stacking / legend order

# --- App-wide knobs ------------------------------------------------------- #
# Fast Dash's accent= maps to Mantine's primaryColor, which must be a *named*
# Mantine palette key (not a hex). ACCENT (hex) is used for Plotly chart lines.
MANTINE_ACCENT = "green"
ACCENT = "#2f9e44"  # GridPulse green — for Plotly figures

# Environment-driven secrets (never hard-coded). Absent key => snapshot mode.
EIA_API_KEY = os.environ.get("EIA_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
# OpenRouter-hosted model for the chat copilot. Overridable for cost/quality.
OPENROUTER_MODEL = os.environ.get(
    "GRIDPULSE_MODEL", "anthropic/claude-haiku-4.5"
).strip()

EIA_BASE_URL = "https://api.eia.gov/v2"
CACHE_TTL_SECONDS = 15 * 60  # 15-minute live-data cache


def has_live_data() -> bool:
    """True when an EIA key is configured (live feed), else snapshot mode."""
    return bool(EIA_API_KEY)


def has_live_agent() -> bool:
    """True when an OpenRouter key is configured (live copilot), else stub."""
    return bool(OPENROUTER_API_KEY)

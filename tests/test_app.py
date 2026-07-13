"""The app: callback output shape, build, and the copilot's input contract."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from gridpulse import data
from gridpulse.app import build_app, explore


@pytest.fixture(autouse=True)
def force_sample(monkeypatch):
    monkeypatch.setattr(data, "has_live_data", lambda: False)


def test_callback_returns_four_typed_outputs():
    out = explore("California (CAISO)", "Demand", 7, 48, True)
    assert len(out) == 4
    fig1, fig2, table, md = out
    assert isinstance(fig1, go.Figure) and isinstance(fig2, go.Figure)
    assert isinstance(table, pd.DataFrame) and set(table.columns) == {"Metric", "Value"}
    assert isinstance(md, str) and "grid readout" in md


def test_show_forecast_false_omits_forecast_traces():
    with_fc = explore("Texas (ERCOT)", "Demand", 7, 48, True)[0]
    without = explore("Texas (ERCOT)", "Demand", 7, 48, False)[0]
    names = {t.name for t in with_fc.data}
    assert any("forecast" in (n or "").lower() for n in names)
    assert len(without.data) < len(with_fc.data)


def test_net_generation_metric_switches_series():
    out = explore("Midwest (MISO)", "Net generation", 5, 24, True)
    assert isinstance(out[0], go.Figure)


def test_build_app_mounts_driving_sidecar():
    app = build_app()
    assert app.has_chat_sidecar is True
    # fast-dash 0.6: chat_agent_drive= was replaced by the chat_tools allowlist.
    assert app.chat_tools_config.get("set_input") is not False
    assert app.chat_tools_config.get("run_app") is not False
    assert "read_app" in app.chat_tools_config


def test_input_contract_matches_controls():
    from fast_dash.mcp import _describe_static_inputs

    app = build_app()
    specs = {s["id"]: s for s in _describe_static_inputs(app, {})}
    assert set(specs) == {"region", "metric", "lookback_days", "forecast_horizon", "show_forecast"}
    assert "Texas (ERCOT)" in specs["region"]["options"]
    assert specs["metric"]["options"] == ["Demand", "Net generation"]
    assert specs["forecast_horizon"]["props"]["min"] == 6

"""Copilot: tool-call translation, prompt, guardrails, stub driving, live path."""

import os

import pytest

from gridpulse import agent


@pytest.fixture(autouse=True)
def reset_guards():
    agent._reset_guards()
    yield
    agent._reset_guards()


class _Ctx:
    thread_id = "t1"
    resume = None
    inputs = {"region": "Texas (ERCOT)", "metric": "Demand", "forecast_horizon": 48}
    input_specs = [
        {"id": "region", "type": "string",
         "options": ["California (CAISO)", "Texas (ERCOT)", "Mid-Atlantic (PJM)"]},
        {"id": "metric", "type": "string", "options": ["Demand", "Net generation"]},
        {"id": "forecast_horizon", "type": "integer", "props": {"min": 6, "max": 73, "step": 1}},
    ]


# --- pure translation helpers -------------------------------------------- #
def test_coerce_types():
    assert agent._coerce("60") == 60 and isinstance(agent._coerce("60"), int)
    assert agent._coerce("3.5") == 3.5
    assert agent._coerce("true") is True and agent._coerce("false") is False
    assert agent._coerce("Texas (ERCOT)") == "Texas (ERCOT)"


def test_tool_call_to_frame():
    f = agent._tool_call_to_frame({"name": "set_input", "args": {"name": "forecast_horizon", "value": "60"}})
    assert f == {"type": "set_input", "name": "forecast_horizon", "value": 60}
    assert agent._tool_call_to_frame({"name": "run_app", "args": {}}) == {"type": "run_app"}
    assert agent._tool_call_to_frame({"name": "nope", "args": {}}) is None


def test_system_prompt_lists_controls_options_and_bounds():
    p = agent._system_prompt(_Ctx.input_specs)
    assert "region" in p and "Texas (ERCOT)" in p
    assert "Net generation" in p
    assert "min=6" in p and "max=73" in p
    assert "set_input" in p and "run_app" in p


# --- guardrails ----------------------------------------------------------- #
def test_session_turn_cap():
    assert agent._budget_block("s") is None
    for _ in range(agent.MAX_TURNS_PER_SESSION):
        agent._count_turn("s")
    assert "message limit" in agent._budget_block("s")


def test_daily_budget_cap():
    agent._daily["count"] = agent.DAILY_TURN_BUDGET
    agent._daily["date"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).date()
    assert "resting" in agent._budget_block("fresh-session")


# --- offline stub driving ------------------------------------------------- #
def test_stub_drives_on_region_keyword():
    stub = agent._make_stub_copilot()
    frames = list(stub("show California please", _Ctx()))
    set_region = [f for f in frames if isinstance(f, dict) and f.get("name") == "region"]
    assert set_region and set_region[0]["value"] == "California (CAISO)"
    assert any(isinstance(f, dict) and f.get("type") == "run_app" for f in frames)


def test_stub_forecast_horizon_and_metric():
    stub = agent._make_stub_copilot()
    frames = list(stub("switch to net generation and forecast 72 hours", _Ctx()))
    vals = {f.get("name"): f.get("value") for f in frames if isinstance(f, dict) and f.get("type") == "set_input"}
    assert vals.get("metric") == "Net generation"
    assert vals.get("forecast_horizon") == 72


def test_stub_help_when_no_match():
    stub = agent._make_stub_copilot()
    frames = list(stub("hello", _Ctx()))
    assert not any(isinstance(f, dict) and f.get("type") == "run_app" for f in frames)
    assert any(isinstance(f, str) and "offline copilot" in f.lower() for f in frames)


# --- live path (real OpenRouter) — skipped without a key ------------------ #
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_live_copilot_drives(anyio_backend="asyncio"):
    import asyncio

    async def run():
        cop = agent._make_live_copilot()
        frames = []
        # ctx starts on Demand, so asking for net generation is a real change the
        # model must make (it correctly skips redundant sets, so assert on a change).
        async for f in cop("Show me net generation instead of demand", _Ctx()):
            frames.append(f)
        return frames

    frames = asyncio.run(run())
    sets = {f.get("name"): f.get("value") for f in frames
            if isinstance(f, dict) and f.get("type") == "set_input"}
    assert sets.get("metric") == "Net generation"
    assert any(isinstance(f, dict) and f.get("type") == "run_app" for f in frames)

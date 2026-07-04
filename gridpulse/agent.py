"""The Grid Copilot — a LangGraph agent on OpenRouter that drives the dashboard.

The copilot is mounted as a Fast Dash *sidecar*: Fast Dash calls it as
``agent(query, ctx)`` where ``ctx`` carries the app's live input values and the
typed input contract. The agent is a LangGraph ``create_react_agent`` whose two
tools — ``set_input`` and ``run_app`` — let it reconfigure and run the very same
controls a human uses. We stream its reasoning as chat text and translate each
tool call into a Fast Dash drive frame, so the inputs visibly flash and the
charts refresh.

Provider: OpenRouter, reached through ``langchain_openai.ChatOpenAI`` pointed at
``https://openrouter.ai/api/v1``. With no key set, a lightweight keyword-driving
stub stands in so the app still demonstrates driving offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    has_live_agent,
)

# --- guardrails (public LLM endpoint) ------------------------------------- #
MAX_QUERY_CHARS = 500
MAX_TURNS_PER_SESSION = 20
DAILY_TURN_BUDGET = 600  # coarse global cost ceiling; resets daily

_session_turns: dict[str, int] = {}
_daily = {"date": None, "count": 0}


def _budget_block(thread_id: str) -> str | None:
    """Return a refusal string if a guardrail trips, else None."""
    today = datetime.now(timezone.utc).date()
    if _daily["date"] != today:
        _daily["date"], _daily["count"] = today, 0
    if _daily["count"] >= DAILY_TURN_BUDGET:
        return "The copilot is resting for today (daily limit reached). The dashboard controls still work."
    if _session_turns.get(thread_id, 0) >= MAX_TURNS_PER_SESSION:
        return "This chat has reached its message limit. Refresh to start a new session."
    return None


def _count_turn(thread_id: str) -> None:
    _session_turns[thread_id] = _session_turns.get(thread_id, 0) + 1
    _daily["count"] += 1


def _reset_guards() -> None:
    """Clear guardrail counters (used by tests)."""
    _session_turns.clear()
    _daily["date"], _daily["count"] = None, 0


# --- tool-call -> Fast Dash drive frame ----------------------------------- #
def _coerce(value):
    """Coerce a stringified tool argument to the type the control expects."""
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return value


def _tool_call_to_frame(tc: dict) -> dict | None:
    name = tc.get("name")
    args = tc.get("args") or {}
    if name == "set_input":
        return {"type": "set_input", "name": args.get("name"), "value": _coerce(args.get("value"))}
    if name == "run_app":
        return {"type": "run_app"}
    return None


def _is_ai_message(message) -> bool:
    """True for an assistant message/chunk (not a tool or human message)."""
    t = getattr(message, "type", "")
    return t == "ai" or "AIMessage" in message.__class__.__name__


def _text_of(message) -> str:
    """Extract streamable text from an AI message chunk (str or block list)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


# --- system prompt from the live input contract --------------------------- #
def _system_prompt(input_specs: list[dict]) -> str:
    lines = [
        "You are Grid Copilot, an assistant embedded in the GridPulse dashboard, "
        "which shows US electricity demand, generation mix, and a short-term forecast "
        "for a chosen grid region.",
        "",
        "You can DRIVE the dashboard with two tools:",
        "- set_input(name, value): set one control to a value.",
        "- run_app(): re-run the dashboard with the current controls and refresh the charts.",
        "",
        "Controls (use these exact names and only these values):",
    ]
    for spec in input_specs or []:
        sid = spec.get("id")
        desc = f"  - {sid} ({spec.get('type', 'string')})"
        opts = spec.get("options")
        props = spec.get("props") or {}
        if opts:
            desc += f"; one of: {', '.join(map(str, opts))}"
        if props:
            bounds = ", ".join(f"{k}={v}" for k, v in props.items())
            desc += f"; range {bounds}"
        lines.append(desc)
    lines += [
        "",
        "When the user asks to change what's shown, call set_input for each control "
        "you need to change, then call run_app once, then give a one or two sentence "
        "summary of what the charts now show. Do not invent numbers you cannot see. "
        "Keep replies concise and friendly.",
    ]
    return "\n".join(lines)


def _current_settings(ctx) -> str:
    inputs = getattr(ctx, "inputs", None) or {}
    if not inputs:
        return ""
    pairs = ", ".join(f"{k}={v}" for k, v in inputs.items())
    return f"[current settings: {pairs}]\n\n"


# ========================================================================== #
# Live copilot (OpenRouter + LangGraph)
# ========================================================================== #
def _build_graph(system_prompt: str):
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    @tool
    def set_input(name: str, value: str) -> str:
        """Set one dashboard control (name) to a value. Value must be valid for that control."""
        return f"set {name} = {value}"

    @tool
    def run_app() -> str:
        """Re-run the dashboard with the current control values and refresh all charts."""
        return "dashboard re-run; charts refreshed"

    model = ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
        streaming=True,
        default_headers={
            "HTTP-Referer": "https://github.com/dkedar7/fast_dash",
            "X-Title": "GridPulse",
        },
    )
    return create_react_agent(
        model, [set_input, run_app], prompt=system_prompt, checkpointer=InMemorySaver()
    )


def _make_live_copilot():
    state: dict = {"graph": None}

    async def copilot(query, ctx):
        thread_id = getattr(ctx, "thread_id", None) or "default"
        block = _budget_block(thread_id)
        if block:
            yield block
            return
        if len(query or "") > MAX_QUERY_CHARS:
            yield f"That message is a bit long — please keep questions under {MAX_QUERY_CHARS} characters."
            return
        _count_turn(thread_id)

        if state["graph"] is None:
            state["graph"] = _build_graph(_system_prompt(getattr(ctx, "input_specs", [])))
        graph = state["graph"]

        prompt = _current_settings(ctx) + (query or "")
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
        seen: set = set()
        try:
            async for mode, chunk in graph.astream(
                {"messages": [("user", prompt)]}, config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg, _meta = chunk
                    # Stream only the assistant's own text — never tool-result
                    # messages (their return strings would otherwise leak in).
                    if _is_ai_message(msg):
                        text = _text_of(msg)
                        if text:
                            yield {"type": "content", "content": text}
                elif mode == "updates":
                    for _node, upd in (chunk or {}).items():
                        for m in (upd or {}).get("messages", []) or []:
                            for tc in getattr(m, "tool_calls", None) or []:
                                tid = tc.get("id") or repr(tc.get("args"))
                                if tid in seen:
                                    continue
                                seen.add(tid)
                                frame = _tool_call_to_frame(tc)
                                if frame:
                                    yield frame
        except Exception as exc:  # noqa: BLE001 — surface, never crash the composer
            yield f"The copilot hit an error reaching the model: {type(exc).__name__}."

    return copilot


# ========================================================================== #
# Offline stub — keyword driving so the demo works without a key
# ========================================================================== #
def _make_stub_copilot():
    def copilot(query, ctx):
        specs = {s.get("id"): s for s in (getattr(ctx, "input_specs", []) or [])}
        q = (query or "").lower()
        drove = False

        # match a region option mentioned in the query
        region_spec = specs.get("region")
        if region_spec:
            for opt in region_spec.get("options") or []:
                key = str(opt).split(" (")[0].lower()  # "California" from "California (CAISO)"
                if key and key in q:
                    yield {"type": "set_input", "name": "region", "value": opt}
                    drove = True
                    break
        # match a metric
        if "generation" in q or "net gen" in q:
            yield {"type": "set_input", "name": "metric", "value": "Net generation"}
            drove = True
        elif "demand" in q:
            yield {"type": "set_input", "name": "metric", "value": "Demand"}
            drove = True
        # match a forecast horizon like "forecast 72" / "72 hours"
        import re
        m = re.search(r"(\d{1,3})\s*(?:h|hour|hours)?", q)
        if m and ("forecast" in q or "hour" in q):
            hrs = max(6, min(72, int(m.group(1))))
            yield {"type": "set_input", "name": "forecast_horizon", "value": hrs}
            drove = True

        if drove:
            yield {"type": "run_app"}
            yield "Done — I've updated the dashboard. (Offline copilot: set `OPENROUTER_API_KEY` for the full AI assistant.)"
        else:
            yield (
                "I'm the offline copilot. Try: *show California*, *switch to net generation*, "
                "or *forecast 72 hours*. Set `OPENROUTER_API_KEY` to enable the full AI copilot."
            )

    return copilot


def make_copilot():
    """Return a Fast Dash sidecar callable — live (OpenRouter) or offline stub."""
    return _make_live_copilot() if has_live_agent() else _make_stub_copilot()

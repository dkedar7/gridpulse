# Plotly "Explore Page" submission — draft

> Draft of the forum reply for **Share Your App – Explore Page – July 2026**
> (https://community.plotly.com/t/share-your-app-explore-page-july-2026/97216).
> Post from your own account. Fill in the two placeholders (live URL, GIF) after
> deploying. Cycle closes **July 31, 2026**.

---

**GridPulse — a live US electricity-grid explorer with an AI copilot**

⚡ **Live app:** `https://<your-fly-app>.fly.dev`  ·  **Code:** https://github.com/dkedar7/gridpulse

<!-- drop the 20s demo GIF here — it should lead the post -->

GridPulse charts hourly electricity **demand**, **generation mix**, and a **short-term demand forecast** (with a confidence band) for any of eight US grid operators — CAISO, ERCOT, PJM, MISO, and more — straight from the **EIA v2 Hourly Electric Grid Monitor** API, self-updating each hour. The twist: an **AI copilot** sits beside the dashboard. Ask it *"compare California and Texas demand this week and forecast the weekend"* and it reconfigures the controls and re-runs the app in front of you — the inputs flash, the charts refresh, and it explains what changed. Anything you can do with the controls, the copilot can do.

**Under the hood, the entire dashboard is a single typed Python function.** It's built on [Fast Dash](https://github.com/dkedar7/fast_dash) (which generates a Dash app from type hints), so the region dropdown, sliders, layout, and the chat sidecar are all inferred from the function signature — no callbacks written by hand. The copilot is a **LangGraph** ReAct agent running on **OpenRouter**, wired to two tools (`set_input` / `run_app`) that drive the very same controls a human uses.

**How it maps to the submission priorities:**
- 🔌 **Energy & Utilities** — real US grid operations data (Form EIA-930).
- 🔄 **Live, self-updating data** — hourly demand / forecast / net generation / fuel mix from the EIA v2 API.
- 📈 **Forecasting** — a Holt-Winters model with confidence bands, recomputed interactively.
- 🤖 **NLP + connecting to APIs** — a natural-language copilot (LangGraph + OpenRouter) that drives the app; live EIA REST integration.
- No login, no uploads — open the link and explore.

**Stack:** Fast Dash · Dash / Plotly · LangGraph · LangChain (OpenRouter) · statsmodels · pandas · EIA v2 API · deployed on Fly.io.

Happy to answer questions about the type-hint-to-UI approach or the agent-drives-the-dashboard pattern — both are, I think, a genuinely new way to build and pilot a Dash app.

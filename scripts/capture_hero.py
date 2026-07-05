"""Capture the README hero screenshot from the live app via Playwright.

Drives the Grid Copilot with one query so the shot shows the whole value prop:
the dashboard + the AI copilot that just configured it. Run:

    .venv/Scripts/python.exe scripts/capture_hero.py [url] [out]
"""

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://gridpulse.fly.dev/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/hero.png"
QUERY = "Show California demand and forecast 3 days ahead"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 810})
    page.goto(URL, wait_until="load", timeout=60000)
    page.wait_for_selector("#chat-sidecar-toggle", timeout=30000)
    page.click("#chat-sidecar-toggle")
    page.wait_for_selector("#chat-input", timeout=15000)
    page.fill("#chat-input", QUERY)
    page.press("#chat-input", "Enter")
    page.wait_for_timeout(15000)  # OpenRouter stream + drive + render
    page.screenshot(path=OUT)
    browser.close()

print("saved", OUT)

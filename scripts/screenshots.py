"""Capture real screenshots of the Streamlit dashboard and the HTML report.

Usage (with the app running on --url):
    python scripts/screenshots.py --url http://localhost:8501 --out docs/screenshots
Requires: pip install playwright  (uses a local Chrome/Chromium via --chrome or
`playwright install chromium`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

TABS = [
    ("Overview", "dashboard_overview.png", 1180),
    ("Waste findings", "dashboard_findings.png", 980),
    ("Trace timeline", "dashboard_trace_timeline.png", 1150),
    ("Benchmark", "dashboard_benchmark.png", 1050),
]


def wait_ready(page: Page) -> None:
    page.wait_for_selector('[data-testid="stMetric"]', timeout=120_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')"
        " || document.querySelector('[data-testid=\"stStatusWidget\"]').innerText === ''",
        timeout=120_000,
    )
    page.wait_for_timeout(2500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--chrome", default=None, help="Path to a Chrome/Chromium binary.")
    ap.add_argument("--report", default=None, help="HTML report to screenshot too.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, args=["--no-sandbox"])
        for tab, name, height in TABS:
            page = browser.new_page(
                viewport={"width": args.width, "height": height}, device_scale_factor=args.scale
            )
            page.goto(args.url)
            wait_ready(page)
            page.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(3500)
            page.screenshot(path=str(out / name))
            print("saved", out / name)
            page.close()
        if args.report:
            page = browser.new_page(
                viewport={"width": args.width, "height": 900}, device_scale_factor=args.scale
            )
            page.goto(Path(args.report).resolve().as_uri())
            page.wait_for_timeout(500)
            page.screenshot(path=str(out / "html_report.png"), full_page=True)
            print("saved", out / "html_report.png")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()

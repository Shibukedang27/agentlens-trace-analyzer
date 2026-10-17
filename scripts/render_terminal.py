"""Render real CLI output (ANSI) to a dark terminal-style PNG via headless Chrome.

    python scripts/render_terminal.py --out docs/screenshots/cli_analyze.png -- \
        agentlens analyze data/demo_traces.jsonl --top 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.terminal_theme import MONOKAI
from rich.text import Text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=104, help="terminal columns")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    env = {**os.environ, "COLUMNS": str(args.width), "FORCE_COLOR": "1", "TERM": "xterm-256color"}
    env.pop("NO_COLOR", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    console = Console(
        record=True, width=args.width + 8, force_terminal=True, color_system="truecolor"
    )
    console.print(Text(f"$ {args.title or ' '.join(cmd)}", style="bold #ffd166"))
    console.print(Text.from_ansi(proc.stdout), overflow="ignore", no_wrap=True, crop=False)
    body = console.export_html(theme=MONOKAI, inline_styles=True, code_format="<pre>{code}</pre>")
    page_html = f"""<html><body style="margin:0;background:#0b0b0d;padding:28px;display:inline-block">
<div style="background:#141418;border:1px solid #2a2a31;border-top:4px solid #b3122e;border-radius:12px;
padding:14px 22px;box-shadow:0 10px 40px rgba(224,17,95,.15)">
<div style="margin-bottom:8px"><span style="color:#ff5f56">●</span> <span style="color:#ffbd2e">●</span>
<span style="color:#27c93f">●</span></div>
<div style="font:14px/1.35 'DejaVu Sans Mono',Menlo,monospace;color:#f4f4f6">{body}</div></div></body></html>"""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tmp:
        tmp.write(page_html)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome, args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 1000, "height": 400}, device_scale_factor=args.scale
        )
        page.goto(Path(tmp.name).as_uri())
        page.locator("body").screenshot(path=args.out)
        browser.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()

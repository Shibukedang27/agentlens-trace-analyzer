from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_renders_demo_without_errors() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Total cost" in labels and "Estimated savings" in labels
    assert len(at.tabs) == 4

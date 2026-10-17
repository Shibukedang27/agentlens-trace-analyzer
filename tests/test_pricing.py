from __future__ import annotations

import pytest

from agentlens.pricing import ModelPrice, PriceConfigError, PriceTable


def test_default_table_is_labeled_example(prices: PriceTable) -> None:
    assert "orion-large" in prices.models
    assert prices.models["orion-large"].tier == "large"
    assert prices.currency == "USD"


def test_cost_math() -> None:
    price = ModelPrice(input_per_1m=2.0, output_per_1m=8.0)
    assert price.cost(1_000_000, 500_000) == pytest.approx(6.0)


def test_unknown_model_uses_default_and_is_tracked(prices: PriceTable) -> None:
    cost = prices.cost("mystery-model", 1_000_000, 0)
    assert cost == pytest.approx(prices.default.input_per_1m)
    assert "mystery-model" in prices.unknown_models


def test_downgrade_lookup(prices: PriceTable) -> None:
    cheaper = prices.downgrade_price("orion-large")
    assert cheaper is not None and cheaper.input_per_1m < prices.models["orion-large"].input_per_1m
    assert prices.downgrade_price("orion-mini") is None


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"models": {"a": {"input": -1, "output": 1}}},
        {"models": {"a": {"input": "x", "output": 1}}},
        {"models": {"a": {"input": 1, "output": 1, "downgrade_to": "ghost"}}},
    ],
)
def test_invalid_tables_rejected(data: dict) -> None:
    with pytest.raises(PriceConfigError):
        PriceTable.from_dict(data)


def test_from_yaml_with_detector_overrides(tmp_path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("models:\n  m: {input: 1, output: 2}\ndetectors:\n  loop_min_repeats: 5\n")
    table = PriceTable.from_yaml(path)
    assert table.detector_overrides == {"loop_min_repeats": 5}

"""Configurable price table (YAML). Ships with clearly labeled EXAMPLE prices."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class PriceConfigError(ValueError):
    """Raised when a price table is malformed."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float
    tier: str = "standard"
    downgrade_to: str | None = None

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1e6


def _model_price(name: str, raw: Any) -> ModelPrice:
    if not isinstance(raw, dict):
        raise PriceConfigError(f"price entry for {name!r} must be a mapping")
    try:
        inp = float(raw["input"])
        out = float(raw["output"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PriceConfigError(
            f"price entry for {name!r} needs numeric 'input' and 'output'"
        ) from exc
    if inp < 0 or out < 0:
        raise PriceConfigError(f"prices for {name!r} must be >= 0")
    return ModelPrice(inp, out, str(raw.get("tier", "standard")), raw.get("downgrade_to"))


@dataclass
class PriceTable:
    models: dict[str, ModelPrice]
    default: ModelPrice = field(default_factory=lambda: ModelPrice(1.0, 4.0))
    currency: str = "USD"
    detector_overrides: dict[str, Any] = field(default_factory=dict)
    unknown_models: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriceTable:
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            raise PriceConfigError("price table needs a 'models' mapping")
        models = {name: _model_price(name, raw) for name, raw in data["models"].items()}
        for name, price in models.items():
            if price.downgrade_to and price.downgrade_to not in models:
                raise PriceConfigError(
                    f"{name!r} downgrades to unknown model {price.downgrade_to!r}"
                )
        default = (
            _model_price("default", data["default"]) if "default" in data else ModelPrice(1.0, 4.0)
        )
        return cls(
            models=models,
            default=default,
            currency=str(data.get("currency", "USD")),
            detector_overrides=dict(data.get("detectors") or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriceTable:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def default_table(cls) -> PriceTable:
        text = resources.files("agentlens").joinpath("data/prices.yaml").read_text("utf-8")
        return cls.from_dict(yaml.safe_load(text))

    def price_for(self, model: str | None) -> ModelPrice:
        if model is None:
            return self.default
        if model in self.models:
            return self.models[model]
        self.unknown_models.add(model)
        return self.default

    def cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float:
        return self.price_for(model).cost(input_tokens, output_tokens)

    def downgrade_price(self, model: str | None) -> ModelPrice | None:
        price = self.price_for(model)
        if price.downgrade_to:
            return self.models[price.downgrade_to]
        return None

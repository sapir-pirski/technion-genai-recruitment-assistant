"""Dated Standard text-token estimates, not an account billing ledger."""

import re
from decimal import Decimal

# USD per million tokens. Source: https://developers.openai.com/api/docs/pricing
PRICE_VERSION = "openai-standard-2026-11-07"
RATES = {
    "gpt-4o-mini": ("0.15", "0.075", "0.60"),
    "gpt-5-nano": ("0.05", "0.005", "0.40"),
    "gpt-4.1-mini": ("0.40", "0.10", "1.60"),
    "text-embedding-3-small": ("0.02", "0.02", "0"),
}


def estimate_cost(model, input_tokens, output_tokens, cached_tokens, *, embedding=False):
    family = next(
        (
            name
            for name in RATES
            if re.fullmatch(re.escape(name) + r"(?:-\d{4}-\d{2}-\d{2})?", model or "")
        ),
        None,
    )
    metadata = {"price_version": PRICE_VERSION, "estimated_cost_usd": None}
    if family is None or input_tokens is None:
        return metadata
    if not embedding and (output_tokens is None or cached_tokens is None):
        return metadata
    cached = 0 if embedding else cached_tokens
    if not 0 <= cached <= input_tokens:
        return metadata
    uncached_rate, cached_rate, output_rate = map(Decimal, RATES[family])
    amount = (
        (input_tokens - cached) * uncached_rate
        + cached * cached_rate
        + (0 if embedding else output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return {**metadata, "pricing_model": family, "estimated_cost_usd": float(amount)}

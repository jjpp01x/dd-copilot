"""Cost accounting and a hard budget cap.

An analyst who does not know what their tooling costs cannot defend using it.
This module keeps a running total in dollars and stops the run when it crosses
a declared ceiling.

Rates are US dollars per million tokens, taken from Anthropic's published
pricing rather than from memory. Sonnet 5 also had an introductory rate of
$2.00 / $10.00 through 2026-08-31; the standard rate is used here on purpose,
so the cap errs toward stopping early rather than overspending.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownModel(KeyError):
    """The model has no published rate, so its spend cannot be tracked."""


class BudgetExceeded(RuntimeError):
    """The accumulated cost crossed the declared ceiling."""


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


#: USD per million tokens. Local models are listed explicitly at zero rather
#: than falling through to a default — an unlisted model must raise, not cost
#: nothing, or the cap silently stops applying to it.
PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
    "claude-sonnet-5": ModelPricing(3.00, 15.00),
    "claude-opus-5": ModelPricing(5.00, 25.00),
    # Run locally via Ollama: no API charge.
    "llama3.1": ModelPricing(0.0, 0.0),
}


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        price = PRICING[model]
    except KeyError as exc:
        raise UnknownModel(
            f"no published rate for model {model!r}; add it to PRICING before "
            f"running against it, otherwise the budget cap cannot see its spend"
        ) from exc

    return (
        input_tokens * price.input_per_million / 1_000_000
        + output_tokens * price.output_per_million / 1_000_000
    )


class CostTracker:
    """Accumulates spend and enforces an optional ceiling.

    The ceiling is checked *after* each call, because the cost of a call is not
    knowable until it returns. A run can therefore overshoot by at most one
    call — what the cap prevents is the next one. Stated here rather than
    implied, since a budget that silently overshoots is worse than none.
    """

    def __init__(self, max_usd: float | None = None):
        self.max_usd = max_usd
        self.total_usd = 0.0
        self.calls = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = cost_of(model, input_tokens, output_tokens)
        self.total_usd += cost
        self.calls += 1

        if self.max_usd is not None and self.total_usd > self.max_usd:
            raise BudgetExceeded(
                f"spend reached ${self.total_usd:.2f}, over the ${self.max_usd:.2f} "
                f"cap after {self.calls} model call(s). Raise --max-cost-usd or "
                f"analyse a smaller source."
            )
        return cost

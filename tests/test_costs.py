import pytest

from dd_copilot.costs import (
    PRICING,
    BudgetExceeded,
    CostTracker,
    UnknownModel,
    cost_of,
)


def test_cost_is_computed_from_the_published_per_million_rates():
    # Haiku 4.5: $1.00 per 1M input, $5.00 per 1M output.
    assert cost_of("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_of("claude-haiku-4-5-20251001", 0, 1_000_000) == pytest.approx(5.00)
    # Sonnet 5: $3.00 / $15.00.
    assert cost_of("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(18.00)


def test_a_local_model_costs_nothing():
    assert cost_of("llama3.1", 500_000, 500_000) == 0.0


def test_an_unpriced_model_raises_instead_of_costing_zero():
    """A silent zero would make the budget cap useless: an unrecognised model
    would spend without ever tripping it."""
    with pytest.raises(UnknownModel) as excinfo:
        cost_of("claude-some-future-model", 1000, 1000)

    assert "claude-some-future-model" in str(excinfo.value)


def test_tracker_accumulates_across_calls():
    tracker = CostTracker()
    tracker.record("claude-haiku-4-5-20251001", 1_000_000, 0)
    tracker.record("claude-haiku-4-5-20251001", 1_000_000, 0)

    assert tracker.total_usd == pytest.approx(2.00)
    assert tracker.calls == 2


def test_tracker_without_a_cap_never_raises():
    tracker = CostTracker(max_usd=None)
    for _ in range(5):
        tracker.record("claude-sonnet-5", 1_000_000, 1_000_000)

    assert tracker.total_usd == pytest.approx(90.00)


def test_exceeding_the_cap_raises_with_both_figures():
    tracker = CostTracker(max_usd=1.50)

    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.record("claude-sonnet-5", 1_000_000, 0)  # $3.00, over the cap

    message = str(excinfo.value)
    assert "3.00" in message and "1.50" in message


def test_the_run_that_crosses_the_cap_is_still_counted():
    """The cap is enforced after the call that crosses it — the spend already
    happened. What it prevents is the next one."""
    tracker = CostTracker(max_usd=1.50)

    with pytest.raises(BudgetExceeded):
        tracker.record("claude-sonnet-5", 1_000_000, 0)

    assert tracker.total_usd == pytest.approx(3.00)


def test_staying_under_the_cap_does_not_raise():
    tracker = CostTracker(max_usd=1.50)
    tracker.record("claude-haiku-4-5-20251001", 1_000_000, 0)  # $1.00

    assert tracker.total_usd == pytest.approx(1.00)


def test_every_priced_model_declares_both_directions():
    for model, price in PRICING.items():
        assert price.input_per_million >= 0, model
        assert price.output_per_million >= 0, model

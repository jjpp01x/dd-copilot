import json
from unittest.mock import MagicMock, patch

from dd_copilot.providers import ClaudeProvider, OllamaProvider


def _fake_anthropic_response(text: str):
    message = MagicMock()
    message.content = [MagicMock(text=text)]
    return message


def test_claude_provider_uses_haiku_for_classify_tier():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response('{"ok": true}')
    provider = ClaudeProvider(fake_client)

    result = provider.complete("system prompt", "user prompt", tier="classify")

    assert result == '{"ok": true}'
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


def test_claude_provider_uses_sonnet_for_synthesis_tier():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response('{"ok": true}')
    provider = ClaudeProvider(fake_client)

    provider.complete("system prompt", "user prompt", tier="synthesis")

    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"


def test_ollama_provider_posts_to_local_endpoint_and_returns_response_text():
    with patch("dd_copilot.providers.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"message": {"content": '{"ok": true}'}}
        mock_post.return_value.raise_for_status = MagicMock()
        provider = OllamaProvider(model="llama3.1")

        result = provider.complete("system prompt", "user prompt", tier="classify")

        assert result == '{"ok": true}'
        call_args = mock_post.call_args
        assert "localhost:11434" in call_args.args[0]
        assert call_args.kwargs["json"]["model"] == "llama3.1"


def test_claude_provider_records_real_usage_against_the_tracker():
    from unittest.mock import MagicMock

    from dd_copilot.costs import CostTracker
    from dd_copilot.providers import ClaudeProvider

    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")],
        usage=MagicMock(input_tokens=1_000_000, output_tokens=0),
    )
    tracker = CostTracker()

    ClaudeProvider(client, tracker=tracker).complete("sys", "user", tier="classify")

    # Haiku 4.5 at $1.00 per 1M input tokens.
    assert tracker.total_usd == 1.00
    assert tracker.calls == 1


def test_claude_provider_stops_the_run_when_the_cap_is_crossed():
    from unittest.mock import MagicMock

    import pytest

    from dd_copilot.costs import BudgetExceeded, CostTracker
    from dd_copilot.providers import ClaudeProvider

    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")],
        usage=MagicMock(input_tokens=1_000_000, output_tokens=0),
    )
    provider = ClaudeProvider(client, tracker=CostTracker(max_usd=0.50))

    with pytest.raises(BudgetExceeded):
        provider.complete("sys", "user", tier="classify")


def test_provider_without_a_tracker_still_works():
    from unittest.mock import MagicMock

    from dd_copilot.providers import ClaudeProvider

    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok")])

    assert ClaudeProvider(client).complete("sys", "user", tier="classify") == "ok"

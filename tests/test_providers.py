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

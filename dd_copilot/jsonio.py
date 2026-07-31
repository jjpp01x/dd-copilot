"""Tolerant parsing of a model's JSON reply.

Models are asked for JSON and mostly comply, but they wrap it in markdown
fences, introduce it with a sentence, follow it with an offer to help further,
or — with smaller local models — emit several objects in a row. All of those
are recoverable; none should crash a diligence run.

The earlier approach searched for `\\{.*\\}` with DOTALL, which spans from the
first brace to the *last* one. That works for a single object surrounded by
prose and fails for two objects in a row, which is exactly what a real Ollama
run produced. Scanning for the first brace and letting the JSON decoder find
where that object ends handles both, and handles nested braces correctly for
free.
"""

from __future__ import annotations

import json


def parse_json_response(text: str) -> dict:
    """Returns the first complete JSON object in `text`.

    Raises ValueError when there is no parseable object, or when the first one
    is not a mapping — every call site indexes into the result by key, so a
    list here would fail later with a confusing AttributeError instead of a
    clear message at the boundary.
    """
    decoder = json.JSONDecoder()

    for start in _candidate_starts(text):
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ValueError(
        f"no JSON object found in model response: {text[:200]!r}"
    )


def _candidate_starts(text: str):
    """Every position where a JSON object could begin, earliest first."""
    index = text.find("{")
    while index != -1:
        yield index
        index = text.find("{", index + 1)

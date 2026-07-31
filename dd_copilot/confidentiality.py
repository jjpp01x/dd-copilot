"""Handling mode and audit trail.

A diligence tool is only usable on real client material if it can promise that
the material never leaves the machine. That promise has to be enforced by the
code, not by the operator remembering which flag to pass — hence a mode that
*refuses* to run rather than one that warns.
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Mode(str, Enum):
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


#: Providers that send the source text to a third party.
REMOTE_PROVIDERS = frozenset({"claude"})


class ConfidentialModeViolation(RuntimeError):
    """Raised when a run would send confidential material to a remote provider."""


def assert_provider_allowed(mode: Mode, provider: str) -> None:
    if mode is Mode.CONFIDENTIAL and provider in REMOTE_PROVIDERS:
        raise ConfidentialModeViolation(
            f"Provider '{provider}' sends the source text to a third-party API, which "
            f"--mode confidential forbids. Re-run with --provider ollama to keep the "
            f"analysis entirely local."
        )


def write_audit_record(
    log_path: Path | str,
    *,
    source_name: str,
    mode: Mode,
    provider: str,
    report_markdown: str,
) -> None:
    """Appends one JSON line per run.

    A report an investor cannot trace back to a specific run, of a specific
    source, by a specific model is not evidence. The hash lets a report handed
    over months ago be matched to the run that produced it.

    In confidential mode the source name is redacted: the log proves the run
    happened without leaking who the client was looking at.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_name": "[redacted]" if mode is Mode.CONFIDENTIAL else source_name,
        "mode": mode.value,
        "provider": provider,
        "report_sha256": hashlib.sha256(report_markdown.encode("utf-8")).hexdigest(),
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

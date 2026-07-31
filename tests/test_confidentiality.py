import json

import pytest

from dd_copilot.confidentiality import (
    ConfidentialModeViolation,
    Mode,
    assert_provider_allowed,
    write_audit_record,
)


def test_public_mode_allows_a_remote_provider():
    assert_provider_allowed(Mode.PUBLIC, "claude")  # must not raise


def test_confidential_mode_refuses_a_remote_provider():
    """The whole point of the mode: client material never leaves the machine."""
    with pytest.raises(ConfidentialModeViolation) as excinfo:
        assert_provider_allowed(Mode.CONFIDENTIAL, "claude")

    assert "ollama" in str(excinfo.value).lower()


def test_confidential_mode_allows_the_local_provider():
    assert_provider_allowed(Mode.CONFIDENTIAL, "ollama")


def test_audit_record_captures_what_a_reproduction_would_need(tmp_path):
    log_path = tmp_path / "audit.jsonl"

    write_audit_record(
        log_path,
        source_name="robotics-startup",
        mode=Mode.PUBLIC,
        provider="claude",
        report_markdown="# Report\n\nBody.",
    )

    record = json.loads(log_path.read_text().strip())
    assert record["source_name"] == "robotics-startup"
    assert record["mode"] == "public"
    assert record["provider"] == "claude"
    assert len(record["report_sha256"]) == 64
    assert record["timestamp"].endswith("+00:00")


def test_audit_log_appends_rather_than_overwrites(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    for name in ("first", "second"):
        write_audit_record(log_path, source_name=name, mode=Mode.PUBLIC, provider="ollama", report_markdown="x")

    lines = log_path.read_text().strip().splitlines()
    assert [json.loads(line)["source_name"] for line in lines] == ["first", "second"]


def test_confidential_audit_record_stores_no_source_name(tmp_path):
    """In confidential mode the name of the target is itself sensitive: the log
    proves an analysis happened without recording who it was about."""
    log_path = tmp_path / "audit.jsonl"

    write_audit_record(
        log_path,
        source_name="Project Helios (client A)",
        mode=Mode.CONFIDENTIAL,
        provider="ollama",
        report_markdown="# Report",
    )

    record = json.loads(log_path.read_text().strip())
    assert "Helios" not in json.dumps(record)
    assert record["source_name"] == "[redacted]"

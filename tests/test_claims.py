import json
from unittest.mock import MagicMock

from dd_copilot.chunking import chunk_document
from dd_copilot.claims import classify_claim, extract_claims
from dd_copilot.index import build_index
from dd_copilot.ingest import Document

SOURCE_TEXT = (
    "Our controller runs inference in 8 ms on an embedded GPU, measured over 10,000 "
    "episodes on the public PushT benchmark with the protocol described in Section 4. "
    "The same architecture scales to fleets of 1,000 robots. "
    "We expect to reach room-temperature operation within two years."
)


def _payload(**overrides):
    base = {
        "text": "Inference runs in 8 ms on an embedded GPU.",
        "figure": "8 ms",
        "verdict": "verifiable",
        "method": "10,000 episodes on the public PushT benchmark, protocol in Section 4",
        "justification": "The source states the benchmark and the number of episodes.",
        "citation": "runs inference in 8 ms on an embedded GPU",
    }
    base.update(overrides)
    return base


def test_verifiable_claim_with_stated_method_keeps_its_verdict():
    claim = classify_claim(_payload(), SOURCE_TEXT)

    assert claim is not None
    assert claim.verdict == "verifiable"
    assert claim.figure == "8 ms"
    assert claim.citations[0].text == "runs inference in 8 ms on an embedded GPU"


def test_verifiable_claim_without_a_stated_method_is_downgraded_to_plausible():
    """The core analyst rule: a number without measurement conditions is not
    verifiable, however confident the model sounds about it."""
    claim = classify_claim(
        _payload(
            text="The architecture scales to fleets of 1,000 robots.",
            figure="1,000 robots",
            method=None,
            citation="scales to fleets of 1,000 robots",
        ),
        SOURCE_TEXT,
    )

    assert claim is not None
    assert claim.verdict == "plausible"
    assert "no measurement method" in claim.justification.lower()


def test_blank_method_string_counts_as_no_method():
    claim = classify_claim(_payload(method="   "), SOURCE_TEXT)

    assert claim is not None
    assert claim.verdict == "plausible"


def test_claim_whose_citation_is_not_in_the_source_is_dropped():
    claim = classify_claim(
        _payload(citation="achieves 99.9% accuracy on every known benchmark"),
        SOURCE_TEXT,
    )

    assert claim is None


def test_unsupported_verdict_is_preserved_and_never_upgraded():
    claim = classify_claim(
        _payload(
            text="Room-temperature operation is expected within two years.",
            figure="two years",
            verdict="unsupported",
            method="10,000 episodes on the public PushT benchmark",
            citation="reach room-temperature operation within two years",
        ),
        SOURCE_TEXT,
    )

    assert claim is not None
    assert claim.verdict == "unsupported"


def test_unrecognised_verdict_defaults_to_plausible_and_says_so():
    claim = classify_claim(_payload(verdict="extremely-strong"), SOURCE_TEXT)

    assert claim is not None
    assert claim.verdict == "plausible"
    assert "not recognised" in claim.justification.lower()


def test_extract_claims_filters_fabrications_and_keeps_real_ones():
    doc = Document(source_name="robotics-startup", text=SOURCE_TEXT)
    index = build_index(chunk_document(doc, chunk_size=80, chunk_overlap=10))

    provider = MagicMock()
    provider.complete.return_value = json.dumps(
        {
            "claims": [
                _payload(),
                _payload(citation="cures every disease by 2027", text="Cures every disease."),
            ]
        }
    )

    claims = extract_claims(provider, index, SOURCE_TEXT)

    assert len(claims) == 1
    assert claims[0].verdict == "verifiable"
    # Claim extraction is a classification task: it must not burn the expensive tier.
    assert provider.complete.call_args.kwargs["tier"] == "classify"


def test_extract_claims_returns_empty_list_when_source_makes_no_claims():
    doc = Document(source_name="vague-startup", text=SOURCE_TEXT)
    index = build_index(chunk_document(doc, chunk_size=80, chunk_overlap=10))

    provider = MagicMock()
    provider.complete.return_value = json.dumps({"claims": []})

    assert extract_claims(provider, index, SOURCE_TEXT) == []

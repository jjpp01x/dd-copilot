from rapidfuzz import fuzz


def verify_citation(citation_text: str, source_text: str, threshold: int = 90) -> bool:
    """Checks that `citation_text` appears literally (or nearly so) in `source_text`.

    Uses partial_ratio to tolerate small variations (case, accents,
    whitespace) without letting fabricated content slip through.
    """
    citation_text = citation_text.strip()
    if not citation_text:
        return False
    score = fuzz.partial_ratio(citation_text.lower(), source_text.lower())
    return score >= threshold

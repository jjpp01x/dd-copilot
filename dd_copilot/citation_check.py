from rapidfuzz import fuzz


def verify_citation(citation_text: str, source_text: str, threshold: int = 90) -> bool:
    """Comprueba que `citation_text` aparece literalmente (o casi) en `source_text`.

    Usa partial_ratio para tolerar pequeñas variaciones (mayúsculas, tildes,
    espacios) sin permitir que se cuele contenido inventado.
    """
    citation_text = citation_text.strip()
    if not citation_text:
        return False
    score = fuzz.partial_ratio(citation_text.lower(), source_text.lower())
    return score >= threshold

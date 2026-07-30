from dd_copilot.citation_check import verify_citation

SOURCE = "Isomorphic Labs combina inteligencia artificial y biología para acelerar el descubrimiento de fármacos."

def test_exact_citation_passes():
    assert verify_citation("acelerar el descubrimiento de fármacos", SOURCE) is True

def test_slightly_altered_citation_still_passes_above_threshold():
    assert verify_citation("acelerar el descubrimiento de farmacos", SOURCE) is True

def test_fabricated_citation_fails():
    assert verify_citation("cura el cáncer en tres días", SOURCE) is False

def test_empty_citation_fails():
    assert verify_citation("", SOURCE) is False

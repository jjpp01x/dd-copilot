from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index, retrieve_relevant_chunks


def test_retrieve_relevant_chunks_returns_most_similar_node():
    doc = Document(
        source_name="demo",
        text=(
            "Isomorphic Labs usa modelos de deep learning para predecir la estructura de proteínas. "
            "El equipo de marketing organiza eventos anuales en Londres para inversores. "
            "La compañía fue fundada como spin-off de DeepMind en 2021."
        ),
    )
    nodes = chunk_document(doc, chunk_size=40, chunk_overlap=5)
    index = build_index(nodes)
    results = retrieve_relevant_chunks(index, "¿Qué tecnología de IA usa la empresa?", top_k=1)
    assert len(results) == 1
    assert "deep learning" in results[0].get_content() or "proteínas" in results[0].get_content()

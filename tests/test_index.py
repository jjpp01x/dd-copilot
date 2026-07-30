from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document
from dd_copilot.index import build_index, retrieve_relevant_chunks


def test_retrieve_relevant_chunks_returns_most_similar_node():
    doc = Document(
        source_name="demo",
        text=(
            "Isomorphic Labs uses deep learning models to predict protein structure. "
            "The marketing team organizes annual events in London for investors. "
            "The company was founded as a DeepMind spin-off in 2021."
        ),
    )
    nodes = chunk_document(doc, chunk_size=20, chunk_overlap=5)
    index = build_index(nodes)
    results = retrieve_relevant_chunks(index, "How does the startup predict protein structures?", top_k=1)
    assert len(results) == 1
    assert "deep learning" in results[0].get_content() or "protein" in results[0].get_content()

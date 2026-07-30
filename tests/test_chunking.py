from dd_copilot.ingest import Document
from dd_copilot.chunking import chunk_document

def test_chunk_document_produces_nonempty_nodes_with_overlap():
    long_text = " ".join([f"Frase número {i} sobre la tecnología de la startup." for i in range(200)])
    doc = Document(source_name="demo", text=long_text)
    nodes = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(nodes) > 1
    for node in nodes:
        assert node.get_content().strip() != ""
        assert node.node_id

def test_chunk_document_single_short_text_produces_one_node():
    doc = Document(source_name="demo", text="Texto corto.")
    nodes = chunk_document(doc)
    assert len(nodes) == 1
    assert "Texto corto" in nodes[0].get_content()

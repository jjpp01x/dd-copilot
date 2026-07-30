from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode

from dd_copilot.ingest import Document


def chunk_document(document: Document, chunk_size: int = 512, chunk_overlap: int = 50) -> list[TextNode]:
    """Trocea el documento en chunks semánticos (por oraciones, con solape)."""
    llama_doc = LlamaDocument(text=document.text, metadata={"source_name": document.source_name})
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents([llama_doc])

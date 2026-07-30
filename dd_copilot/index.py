from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    """Construye un índice vectorial en memoria con embeddings locales (coste cero de API)."""
    return VectorStoreIndex(nodes, embed_model=_embed_model)


def retrieve_relevant_chunks(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[TextNode]:
    """Devuelve los `top_k` chunks más relevantes para `query`, sin llamar al LLM."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [result.node for result in results]

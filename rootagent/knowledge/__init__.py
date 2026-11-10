"""Knowledge and RAG system for RootAgent."""

from .embeddings import (
    EmbeddingCache,
    batch_cosine_similarity,
    get_embeddings,
    get_embeddings_local,
    should_use_local_embeddings,
)
from .indexer import KnowledgeIndexer
from .rag import Document, RAGEngine

__all__ = [
    "RAGEngine",
    "Document",
    "get_embeddings",
    "get_embeddings_local",
    "should_use_local_embeddings",
    "KnowledgeIndexer",
    "batch_cosine_similarity",
    "EmbeddingCache",
]

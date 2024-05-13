from abc import ABC, abstractmethod

from llama_index.core import VectorStoreIndex


class BaseFactory(ABC):
    """Base database builder."""

    def __init__(self, vector_store_index: VectorStoreIndex):
        """Initial database."""
        self._vector_store_index = vector_store_index

    @abstractmethod
    def get_retriever_engine(self, metadata, **kwargs):
        """Get retriever engine."""

    @abstractmethod
    def get_query_engine(self, metadata, **kwargs):
        """Get query engine."""

    @abstractmethod
    def get_chat_engine(self, **kwargs):
        """Get chat engine."""

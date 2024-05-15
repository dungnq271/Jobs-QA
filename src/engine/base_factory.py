from abc import ABC, abstractmethod

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever


class BaseFactory(ABC):
    """Base database builder."""

    def __init__(self, vector_store_index: VectorStoreIndex):
        """Initial database."""
        self._vector_store_index = vector_store_index

    @abstractmethod
    def get_retriever(self, **kwargs):
        """Get retriever engine."""

    @abstractmethod
    def get_query_engine(self, retriever: BaseRetriever, **kwargs):
        """Get query engine."""

    @abstractmethod
    def get_chat_engine(self, **kwargs):
        """Get chat engine."""

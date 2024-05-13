from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class BaseDatabase(ABC):
    """Base database builder."""

    def __init__(
        self,
        collection_name: str,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        transformations: Optional[List[TransformComponent]] = None,
        enable_hybrid=False,
    ):
        """Initial database."""
        self._client = client
        self._collection_name = collection_name
        self._service_context = service_context
        self._transformations = transformations

        if client is None and url is not None:
            self._client = QdrantClient(url=url)
        elif client is not None:
            self._client = client
        else:
            raise AssertionError("Both url and client cannot be None")

        self._vector_store = QdrantVectorStore(
            collection_name=self._collection_name,
            client=self._client,
            enable_hybrid=enable_hybrid,
        )
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            service_context=self._service_context,
            transformations=self._transformations,
        )

    def get_index(self):
        """Get vector store index."""
        return self._index

    @abstractmethod
    def insert_documents(self, documents: List[Document]):
        """Add document to the database."""

    @abstractmethod
    def update_documents(self, documents: List[Document]):
        """Update documents"""

    @abstractmethod
    def delete_documents(self, file_paths: List[str]):
        """Delete document from the database."""

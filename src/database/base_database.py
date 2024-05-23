from abc import ABC, abstractmethod

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
        client: QdrantClient | None = None,
        url: str | None = None,
        service_context: ServiceContext | None = None,
        transformations: list[TransformComponent] | None = None,
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
    def insert_documents(self, documents: list[Document]):
        """Add document to the database."""

    @abstractmethod
    def update_documents(self, documents: list[Document]):
        """Update documents"""

    @abstractmethod
    def delete_documents(self, file_paths: list[str]):
        """Delete document from the database."""

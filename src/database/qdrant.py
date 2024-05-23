import os.path as osp

from llama_index.core.schema import Document, TextNode, TransformComponent
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage.docstore.simple_docstore import (
    SimpleDocumentStore,
)
from llama_index.core.storage.docstore.types import (
    DEFAULT_PERSIST_FNAME as DOCSTORE_FNAME,
)
from qdrant_client import QdrantClient

from src.utils import calculate_time, get_logger

from .base_database import BaseDatabase


class QdrantTextDatabase(BaseDatabase):
    def __init__(
        self,
        collection_name: str,
        client: QdrantClient | None = None,
        url: str | None = None,
        service_context: ServiceContext | None = None,
        transformations: list[TransformComponent] | None = None,
        enable_hybrid=True,
        persist_dir: str = "./storage",
    ):
        super().__init__(
            client=client,
            url=url,
            collection_name=collection_name,
            service_context=service_context,
            transformations=transformations,
            enable_hybrid=enable_hybrid,
        )
        self._logger = get_logger(__name__)
        self._persist_dir = persist_dir

    def insert_documents(self, documents: list[Document]):
        """Insert documents"""
        for document in documents:
            if "file_path" not in document.metadata:
                raise Exception("Document doesn't include file_path in metadata !")
            self._index.insert(document)

    def update_documents(self, documents: list[Document]):
        """Update documents"""
        raise NotImplementedError

    def delete_documents(self, file_paths: list[str]):
        """Delete documents"""
        raise NotImplementedError

    @calculate_time
    def preprocess(self, documents: list[Document], **kwargs):
        """Preprocess the documents"""
        self._logger.info("Preprocess the document...")
        docstore_path = osp.join(self._persist_dir, DOCSTORE_FNAME)

        if not osp.exists(self._persist_dir):
            for document in documents:
                self._index.insert(document)
            self._docstore = self._index.docstore
            self._docstore.add_documents(documents)
            self._docstore.persist(persist_path=docstore_path)
        else:
            self._docstore = SimpleDocumentStore.from_persist_dir(self._persist_dir)

        docstore_dict = self._docstore.to_dict()
        id2nodes = {
            id: TextNode.from_dict(node_dict["__data__"])
            for id, node_dict in docstore_dict["docstore/data"].items()
        }
        return id2nodes

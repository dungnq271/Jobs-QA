import os
import warnings
from typing import Dict

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from qdrant_client import QdrantClient

from src.database.base_database import BaseDatabase
from src.utils import calculate_time
from src.utils.extension_class.extension_multimodal_vector_store_index import (
    MultiModalVectorStoreIndexExtension,
)
from src.utils.extension_class.extension_qdrant_vector_store_index import (
    QdrantVectorStoreExtension,
)
from src.utils.extension_class.extension_vector_store_index import (
    VectorStoreIndexExtension,
)


class QdrantTextDatabase(BaseDatabase):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "rag_demo",
        enable_hybrid=False,
        persist_dir: str = "./storage",
        **kwargs,
    ):
        self.client = client
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.vector_store = QdrantVectorStoreExtension(
            collection_name=collection_name,
            client=client,
            enable_hybrid=enable_hybrid,
        )

        if os.path.exists(self.persist_dir):
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir, vector_store=self.vector_store
            )
            self.processor = load_index_from_storage(self.storage_context, **kwargs)
        else:
            self.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=self.vector_store,
                index_store=SimpleIndexStore(),
            )
            self.processor = VectorStoreIndexExtension(
                nodes=[], storage_context=self.storage_context, **kwargs
            )

    def exist_collection(self):
        all_collections = self.client.get_collections()
        all_name_collections = [
            collection.name for collection in all_collections.collections
        ]
        exist_collection = self.collection_name in all_name_collections
        return exist_collection

    @property
    def index(self) -> VectorStoreIndex:
        """Get index of database"""
        return self.processor

    @property
    def docstore(self) -> BaseDocumentStore:
        return self.processor.docstore

    @property
    def id_node_mapping(self) -> Dict[str, BaseNode]:
        return self.docstore.docs

    def filter_extension(self, value: str, key: str):
        """Filter a document and it's nodes by using value corresponding to key."""
        return self.vector_store.filter_extension(value=value, key=key)

    @calculate_time(name="Insert Documents")
    def insert_documents(self, documents):
        """Insert documents"""

        file_path = documents[0].metadata["file_path"]

        if (
            not self.exist_collection()
            or len(self.filter_extension(file_path, "file_path")[0]) == 0
        ):
            for document in documents:
                if "file_path" not in document.metadata:
                    raise Exception("Document doesn't include file_path in metadata !")
                self.processor.insert(document)

        else:
            # do nothing
            print(f"{file_path} already exists")
            pass

    def insert_documents_to_docstore(self, documents):
        self.processor.docstore.add_documents(documents)

    def persist(self):
        self.storage_context.persist(persist_dir=self.persist_dir)

    def update_documents(self, documents):
        """Update documents"""
        for document in documents:
            # TODO: Need checkbox to ask customer
            warnings.warn(
                "Document is existed in the database. "
                "We will overwrite the document!",
            )
            self.delete_documents(document.metadata["file_path"])
            self.processor.insert(document)

    def delete_documents(self, file_paths):
        """Delete documents"""
        for file_path in file_paths:
            self.processor.delete_extension(value=file_path)


class QdrantImageDatabase(BaseDatabase):
    def __init__(self, client, collection_name, service_context, enable_hybrid=False):
        self.client = client
        self.collection_name = collection_name
        self.service_context = service_context

        self.text_store = QdrantVectorStoreExtension(
            client=self.client,
            collection_name=f"{collection_name}_text",
            enable_hybrid=enable_hybrid,
        )

        self.image_store = QdrantVectorStoreExtension(
            client=self.client,
            collection_name=f"{collection_name}_image",
            enable_hybrid=enable_hybrid,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store, image_store=self.image_store
        )
        self.processor = MultiModalVectorStoreIndexExtension.from_storage_context(
            storage_context=self.storage_context, service_context=self.service_context
        )

        all_collections = self.client.get_collections()
        all_name_collections = [
            collection.name for collection in all_collections.collections
        ]

        self.exist_collection = (
            f"{self.collection_name}_text" in all_name_collections
            and f"{self.collection_name}_image" in all_name_collections
        )

    def get_index(self):
        """Get index of database"""
        return self.processor

    def insert_documents(self, documents):
        """Insert documents"""
        for document in documents:
            if "file_path" not in document.metadata:
                raise Exception("Document doesn't include file_path in metadata !")
            if not self.exist_collection:
                self.processor.insert(document)
                return
            # Check document exist in the database
            check_exist = self.processor.filter_extension(
                document.metadata["file_path"]
            )
            if len(check_exist[0][0]) == 0 and len(check_exist[1][0]) == 0:
                self.processor.insert(document)
            else:
                # TODO: Need checkbox to ask customer
                warnings.warn(
                    "Document is existed in the database. "
                    "We will overwrite the document!"
                )
                self.delete_documents(document.metadata["file_path"])
                self.processor.insert(document)

    def delete_documents(self, file_paths):
        """Delete document"""
        for file_path in file_paths:
            self.processor.delete_extension(value=file_path)

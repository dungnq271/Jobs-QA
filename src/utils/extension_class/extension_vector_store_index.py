from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.types import VectorStore


class VectorStoreIndexExtension(VectorStoreIndex):
    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore,
        embed_model: EmbedType | None = None,
        # deprecated
        service_context: ServiceContext | None = None,
        **kwargs: Any,
    ) -> "VectorStoreIndexExtension":
        if not vector_store.stores_text:
            raise ValueError(
                "Cannot initialize from a vector store that does not store text."
            )

        kwargs.pop("storage_context", None)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return cls(
            nodes=[],
            embed_model=embed_model,
            service_context=service_context,
            storage_context=storage_context,
            **kwargs,
        )

    @property
    def docstore(self) -> BaseDocumentStore:
        return self._docstore

    @docstore.setter
    def docstore(self, docstore) -> None:
        self._docstore = docstore

    def delete_extension(
        self,
        value: str,
        key: str = "file_path",
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a document and it's nodes by using value corresponding to key."""
        self._vector_store.delete_extension(value=value, key=key, **delete_kwargs)

        # delete from index_struct only if needed
        # if not self._vector_store.stores_text or self._store_nodes_override:
        #     ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
        #     if ref_doc_info is not None:
        #         for node_id in ref_doc_info.node_ids:
        #             self._index_struct.delete(node_id)
        #             self._vector_store.delete(node_id)

        # delete from docstore only if needed
        # if (
        #     not self._vector_store.stores_text or self._store_nodes_override
        # ) and delete_from_docstore:
        #     self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    def filter_extension(self, value: str, key: str = "file_path"):
        """Filter a document and it's nodes by using value corresponding to key."""
        return self._vector_store.filter_extension(value=value, key=key)

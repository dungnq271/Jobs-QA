from typing import Any

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as rest


class QdrantVectorStoreExtension(QdrantVectorStore):
    def delete_extension(
        self, value: str, key: str = "file_path", **delete_kwargs: Any
    ) -> None:
        """
        Delete nodes using with value corresponding to key
        (Default key: file_path).

        Args:
            value (str): Value of key to delete

        """
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[rest.FieldCondition(key=key, match=rest.MatchValue(value=value))]
            ),
        )

    async def adelete_extension(
        self, value: str, key: str = "file_path", **delete_kwargs: Any
    ) -> None:
        """
        Asynchronous method to delete nodes using with value
        corresponding to key (Default key: file_path).

        Args:
            value (str): Value of key to delete

        """
        await self._aclient.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[rest.FieldCondition(key=key, match=rest.MatchValue(value=value))]
            ),
        )

    def filter_extension(self, value: str, key: str = "file_path"):
        """
        Filter nodes using with value
        corresponding to key (Default key: file_path).

        Args:
            value (str): Value of key to delete

        """
        return self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=rest.Filter(
                must=[rest.FieldCondition(key=key, match=rest.MatchValue(value=value))]
            ),
        )

    async def afilter_extension(self, value: str, key: str = "file_path"):
        """
        Asynchronous method to delete nodes using
        with value corresponding to key (Default key: file_path).

        Args:
            value (str): Value of key to delete

        """
        return await self._aclient.scroll(
            collection_name=self.collection_name,
            scroll_filter=rest.Filter(
                must=[rest.FieldCondition(key=key, match=rest.MatchValue(value=value))]
            ),
        )

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode

from src.engine.base_engine import BaseEngine


class TextEngine(BaseEngine):
    """Image Engine."""

    def __init__(self, vector_store_index: VectorStoreIndex):
        """Initial parameter."""
        self.vector_store_index = vector_store_index

    def get_retriever(
        self,
        top_k=5,
        vector_store_query_mode="default",
        alpha=0.5,
        filters=None,
        **kwargs,
    ):
        """
        Get retriever engine.
        Retriever only.
        Using if simple implement.
        """

        return self.vector_store_index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode=vector_store_query_mode,
            alpha=alpha,
            filters=filters,
            kwargs=kwargs,
        )

    def get_query_engine(
        self,
        top_k=5,
        vector_store_query_mode="default",
        alpha=0.5,
        filters=None,
        node_postprocessors=[],
        **kwargs,
    ):
        """
        Get query engine.
        Intergrate retriever and prompt and llm
        Using if simple implement.
        """

        return self.vector_store_index.as_query_engine(
            top_k=top_k,
            vector_store_query_mode=vector_store_query_mode,
            alpha=alpha,
            filters=filters,
            node_postprocessors=node_postprocessors,
            kwargs=kwargs,
        )

    def get_chat_engine(
        self,
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        llm=None,
        top_k=5,
        image_top_k=5,
        vector_store_query_mode="default",
        alpha=0.5,
        filters=None,
        node_postprocessors=[],
        **kwargs,
    ):
        """Get chat engine."""

        return self.vector_store_index.as_chat_engine(
            chat_mode=chat_mode,
            llm=llm,
            top_k=top_k,
            image_top_k=image_top_k,
            vector_store_query_mode=vector_store_query_mode,
            alpha=alpha,
            filters=filters,
            node_postprocessors=node_postprocessors,
            kwargs=kwargs,
        )

    def get_tool(self, **kwargs):
        """Get tool for agent."""

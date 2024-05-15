from typing import Any

from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import (  # SQLAutoVectorQueryEngine,
    NLSQLTableQueryEngine,
    RetrieverQueryEngine,
    SQLJoinQueryEngine,
)
from llama_index.core.retrievers import BaseRetriever, RecursiveRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.tools import QueryEngineTool

from src.prompt import QUERY_SQL_DESCRIPTION, QUERY_TEXT_DESCRIPTION
from src.utils import get_logger

from .base_factory import BaseFactory


class TableEngineFactory(BaseFactory):
    def __init__(
        self,
        db_engine: Any,
        vector_store_index: VectorStoreIndex,
        node_postprocessors: list[Any] | None = None,
    ):
        super().__init__(vector_store_index=vector_store_index)
        self._logger = get_logger(__name__)
        self._db_engine = db_engine
        self._node_postprocessors = node_postprocessors

    def get_retriever(self, **kwargs):
        """Get retriever engine."""

        self._logger.info("Get retriever engine...")
        retriever = self._vector_store_index.as_retriever(similarity_top_k=3)
        return retriever

    def get_recursive_retriever(
        self,
        retriever: BaseRetriever,
        id_node_mapping: dict[str, BaseNode],
        **kwargs,
    ):
        """Get recursive retriever engine."""

        self._logger.info("Get recursive retriever engine...")
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": retriever},
            node_dict=id_node_mapping,
            verbose=True,
        )
        return retriever

    def get_query_engine(self, retriever: BaseRetriever, **kwargs):
        """Get query engine."""
        self._logger.info("Get query engine...")
        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=self._node_postprocessors,
        )
        return query_engine

    def get_SQL_engine(self, metadata, **kwargs):
        self._logger.info("Get SQL engine...")
        sql_database = SQLDatabase(
            self._db_engine, include_tables=[metadata["table_name"]]
        )
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[metadata["table_name"]],
        )
        return sql_query_engine

    def get_qa_engine(self, metadata, id_node_mapping, **kwargs):
        """Get table query engine for the recently uploaded table file."""
        # Get vector index query engine
        vector_retriever = self.get_retriever()
        recursive_retriever = self.get_recursive_retriever(
            retriever=vector_retriever, id_node_mapping=id_node_mapping
        )
        query_engine = self.get_query_engine(retriever=recursive_retriever)

        query_desc = QUERY_TEXT_DESCRIPTION.format(
            file_description=metadata["file_description"]
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine, description=query_desc
        )

        # Get SQL query engine
        sql_query_engine = self.get_SQL_engine(metadata)
        sql_desc = QUERY_SQL_DESCRIPTION.format(
            columns_list=", ".join(metadata["all_columns"])
        )
        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine, description=sql_desc
        )

        return SQLJoinQueryEngine(sql_tool, vector_tool)

    def get_chat_engine(self, **kwargs):
        return self._vector_store_index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )

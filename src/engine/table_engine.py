from typing import Any, Dict, List

from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import (  # SQLAutoVectorQueryEngine,
    NLSQLTableQueryEngine,
    RetrieverQueryEngine,
    SQLJoinQueryEngine,
)
from llama_index.core.retrievers import (
    BaseRetriever,
    RecursiveRetriever,
    SQLRetriever,
)
from llama_index.core.schema import BaseNode
from llama_index.core.tools import QueryEngineTool

from src.tool import SQLTableRetrieverQueryEngine
from src.utils import calculate_time, get_logger

from .base_engine import BaseEngine


class TableEngine(BaseEngine):
    def __init__(
        self,
        db_engine: Any,
        vector_store_index: VectorStoreIndex,
        node_postprocessors: List[Any] | None = None,
        similarity_top_k=5,
    ):
        super().__init__(vector_store_index=vector_store_index)
        self._logger = get_logger(__name__)
        self._db_engine = db_engine
        self._node_postprocessors = node_postprocessors
        self.similarity_top_k = similarity_top_k

    def get_retriever(self, **kwargs):
        """Get retriever engine."""

        self._logger.info("Get retriever engine...")
        retriever = self._vector_store_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        return retriever

    def get_recursive_retriever(
        self,
        retriever: BaseRetriever,
        id_node_mapping: Dict[str, BaseNode],
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

    @calculate_time(name="Get SQL Retriever")
    def get_sql_retriever(self, metadata, **kwargs):
        self._logger.info("Get SQL Retriever...")
        sql_database = SQLDatabase(
            self._db_engine, include_tables=[metadata["table_name"]]
        )
        sql_retriever = SQLRetriever(sql_database=sql_database)
        return sql_retriever

    @calculate_time(name="Get SQL Engine")
    def get_sql_engine(self, metadata, **kwargs):
        self._logger.info("Get SQL engine...")
        sql_database = SQLDatabase(
            self._db_engine, include_tables=[metadata["table_name"]]
        )
        sql_engine = SQLTableRetrieverQueryEngine(sql_database=sql_database)
        return sql_engine

    @calculate_time(name="Get NLSQL Engine")
    def get_nlsql_engine(self, metadata, **kwargs):
        self._logger.info("Get SQL engine...")
        sql_database = SQLDatabase(
            self._db_engine, include_tables=[metadata["table_name"]]
        )
        nlsql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[metadata["table_name"]],
        )
        return nlsql_query_engine

    @calculate_time(name="Get Vector Retriever")
    def get_vector_retriever(self, metadata, id_node_mapping, **kwargs):
        """Get table query engine for the recently uploaded table file."""
        vector_retriever = self.get_retriever()
        recursive_retriever = self.get_recursive_retriever(
            retriever=vector_retriever, id_node_mapping=id_node_mapping
        )
        return recursive_retriever

    @calculate_time(name="Get Vector Engine")
    def get_vector_engine(self, metadata, id_node_mapping, **kwargs):
        vector_retriever = self.get_vector_retriever(
            metadata, id_node_mapping, **kwargs
        )
        vector_query_engine = self.get_query_engine(retriever=vector_retriever)
        return vector_query_engine

    def get_sql_join_query_engine(
        sql_tool: QueryEngineTool, vector_tool: QueryEngineTool
    ):
        return SQLJoinQueryEngine(sql_tool, vector_tool)

    def get_chat_engine(self, **kwargs):
        return self._vector_store_index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )

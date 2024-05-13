from typing import Any, List, Optional

from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.query_engine import (  # SQLAutoVectorQueryEngine,
    NLSQLTableQueryEngine,
    RetrieverQueryEngine,
    SQLJoinQueryEngine,
)
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.tools import QueryEngineTool

from src.prompt import QUERY_SQL_DESCRIPTION, QUERY_TEXT_DESCRIPTION
from src.utils import get_logger

from .base_factory import BaseFactory


class TableEngineFactory(BaseFactory):
    def __init__(
        self,
        db_engine: Any,
        vector_store_index: VectorStoreIndex,
        node_postprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(vector_store_index=vector_store_index)
        self._logger = get_logger(__name__)
        self._db_engine = db_engine
        self._node_postprocessors = node_postprocessors

    def get_retriever_engine(self, id_node_mapping, **kwargs):
        """Get retriever engine."""

        self._logger.info("Get retriever engine...")
        vector_retriever = self._vector_store_index.as_retriever(
            similarity_top_k=3
        )
        retriever_engine = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            query_engine_dict=id_node_mapping,
            verbose=True,
        )

        return retriever_engine

    def get_query_engine(self, retriever_engine, **kwargs):
        """Get query engine."""
        self._logger.info("Get query engine...")
        query_engine = RetrieverQueryEngine.from_args(
            retriever_engine,
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
        recursive_retriever = self.get_retriever_engine(
            id_node_mapping=id_node_mapping
        )
        retriever_query_engine = self.get_query_engine(recursive_retriever)

        query_desc = QUERY_TEXT_DESCRIPTION.format(
            file_description=metadata["file_description"]
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine, description=query_desc
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

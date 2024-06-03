from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.struct_store.sql_query import BaseSQLTableQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.service_context import ServiceContext
from llama_index.core.utilities.sql_wrapper import SQLDatabase


class SQLTableRetrieverQueryEngine(BaseSQLTableQueryEngine):
    """SQL Table retriever query engine."""

    def __init__(
        self,
        sql_database: SQLDatabase,
        return_raw: bool = True,
        llm: LLM | None = None,
        synthesize_response: bool = False,
        response_synthesis_prompt: BasePromptTemplate | None = None,
        refine_synthesis_prompt: BasePromptTemplate | None = None,
        service_context: ServiceContext | None = None,
        callback_manager: CallbackManager | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_retriever = SQLRetriever(
            sql_database,
            return_raw=return_raw,
            callback_manager=callback_manager,
        )
        super().__init__(
            synthesize_response=synthesize_response,
            response_synthesis_prompt=response_synthesis_prompt,
            refine_synthesis_prompt=refine_synthesis_prompt,
            llm=llm,
            service_context=service_context,
            callback_manager=callback_manager,
            **kwargs,
        )

    @property
    def sql_retriever(self) -> SQLRetriever:
        """Get SQL retriever."""
        return self._sql_retriever

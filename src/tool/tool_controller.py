from typing import Any, Dict, List

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.retrievers import BaseRetriever, SQLRetriever
from llama_index.core.tools import FunctionTool, QueryEngineTool

from src.database import QdrantTextDatabase
from src.reader import BaseReader
from src.utils import calculate_time

from .api import DEFAULT_ALL_API_TOOLS
from .utils import preprocess_tool_description


class ToolControler:
    _api_tools: Dict[str, Any] = {}
    _api_tools_use: Dict[str, Any] = {}
    _all_tools_use: Dict[str, Any] = {}
    all_tool_types = [
        "sql",
        "normal",  # normal index retriever
        "recursive",  # recursive retriever,
        # require adding docstore
    ]

    def __init__(
        self,
        all_api_tools: Dict[str, List[Any]] = DEFAULT_ALL_API_TOOLS,
        tool_mode: str = "function",
    ):
        self.mode_to_function = {
            "function": self.get_function_tool,
            "query": self.get_query_tool,
        }
        assert tool_mode in self.mode_to_function
        self.tool_mode = tool_mode
        self.get_tool_function = self.mode_to_function[self.tool_mode]
        self._api_tools = all_api_tools

    def get_tools_use(self) -> List[Any]:
        """Get list of all tools in use."""
        return list(self._all_tools_use.values())

    @calculate_time(name="List All API Tools")
    def get_api_tools_name(self) -> List[str]:
        """List all names of available api tools."""
        return list(self._api_tools.keys())

    @calculate_time(name="List API Tools in Use")
    def get_api_tools_use_name(self) -> List[str]:
        """List all names of api tools in use."""
        return list(self._api_tools_use.keys())

    @calculate_time(name="Add API Tool to Use")
    def add_api_tools_use(self, tool_names: List[str]) -> None:
        for name in tool_names:
            api_tools = self._api_tools[name]
            self._api_tools_use.update({tool.metadata.name: tool for tool in api_tools})
            self._all_tools_use.update(self._api_tools_use)

    @calculate_time(name="Get Function Tool")
    def get_function_tool(self, tool: Any, **kwargs):
        def vector_retrieve(query: str):
            retrieved_nodes = tool.retrieve(query)
            response_str = "\n".join([node.text for node in retrieved_nodes])
            return response_str

        def sql_retrieve(query: str):
            retrieved_nodes, _ = tool.retrieve_with_metadata(query)
            response_str = "\n".join([node.node.text for node in retrieved_nodes])
            return response_str

        def query(query: str):
            response = tool.query(query)
            response_str = str(response)
            return response_str

        if isinstance(tool, SQLRetriever):
            fn = sql_retrieve
        elif isinstance(tool, BaseRetriever):
            fn = vector_retrieve
        elif isinstance(tool, BaseQueryEngine):
            fn = query
        else:
            fn = tool

        function_tool = FunctionTool.from_defaults(fn=fn, **kwargs)
        return function_tool

    @calculate_time(name="Get Query Tool")
    def get_query_tool(self, tool: BaseQueryEngine, **kwargs):
        query_tool = QueryEngineTool.from_defaults(query_engine=tool, **kwargs)
        return query_tool

    @calculate_time(name="Get Index Tools E2E")
    def add_index_tools(
        self,
        file_path,
        metadata,
        reader: BaseReader,
        vector_db: QdrantTextDatabase,
        get_engine_functions: List[Any],
        **kwargs,
    ):
        print(f"Preprocess file {file_path}...")
        documents = reader.load_data(file_path=file_path, metadata=metadata)
        vector_db.insert_documents(documents)

        for function_meta in get_engine_functions:
            (get_engine_function, tool_name, description_tmpl, tool_type) = list(
                function_meta.values()
            )

            assert tool_type in self.all_tool_types, print(
                tool_type, self.all_tool_types
            )

            if tool_type == "recursive":
                vector_db.insert_documents_to_docstore(documents)

            index_tool = get_engine_function(
                metadata=metadata, id_node_mapping=vector_db.id_node_mapping
            )
            tool_description = preprocess_tool_description(
                description_tmpl, tool_type, metadata
            )
            tool = self.get_tool_function(index_tool, description=tool_description)
            self._all_tools_use.update({tool_name: tool})

        vector_db.persist()

    def add_tools_use(self, tools_name: List[str] | None = None, **kwargs):
        if tools_name:
            for tool_name in tools_name:
                if tool_name in self.get_api_tools_name():
                    self.add_api_tools_use([tool_name], **kwargs)
                else:
                    # do nothing
                    pass
        else:
            self.add_index_tools(**kwargs)

    def remove_tools_use(self, tool_names: List[str]) -> None:
        for name in tool_names:
            for name_variant in [name, f"read_{name}", f"load_{name}"]:
                if name_variant in self._api_tools_use:
                    del self._api_tools_use[name_variant]
                if name_variant in self._all_tools_use:
                    del self._all_tools_use[name_variant]

import os
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec

from .google_search import CustomGoogleSearchToolSpec

_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # type: ignore


DEFAULT_ALL_LOAD_AND_SEARCH_TOOL_SPECS: Dict[str, BaseToolSpec] = {
    "google_search": CustomGoogleSearchToolSpec(
        key=os.getenv("GOOGLE_SEARCH_API_KEY"),  # type: ignore
        engine=os.getenv("SEARCH_ENGINE_ID"),  # type: ignore
        num=3,
    ),
}


DEFAULT_ALL_LOAD_AND_SEARCH_TOOLS: Dict[str, List[Any]] = {
    tool_name: LoadAndSearchToolSpec.from_defaults(
        tool_spec.to_tool_list()[0],
    ).to_tool_list()
    for tool_name, tool_spec in DEFAULT_ALL_LOAD_AND_SEARCH_TOOL_SPECS.items()
}

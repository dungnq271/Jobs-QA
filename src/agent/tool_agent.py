from __future__ import annotations

from pprint import pprint
from typing import Any

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from pydantic.v1.error_wrappers import ValidationError

from src.utils import calculate_time

from .base_agent import BaseAgent

SYSTEM_PROMPT = """\
You are an agent designed to answer queries from user.
Please ALWAYS use the tools provided to answer a question -
make sure to ALWAYS call any tool with the original query.
Do not rely on prior knowledge.
If there is no information please answer you don't have that information.
"""


class OpenAIToolAgent(BaseAgent):
    def __init__(self, tools, vector_index, **kwargs):
        self.vector_index = vector_index
        self.query_chat_engine = self.vector_index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )
        self.get_agent(tools, **kwargs)

    @calculate_time(name="Get Agent")
    def get_agent(self, tools, **kwargs):
        self.tools = tools
        self.agent = OpenAIAgent.from_tools(
            self.tools, system_prompt=SYSTEM_PROMPT, **kwargs
        )
        for tool in self.tools:
            pprint(tool.metadata.to_openai_tool())

    def reset(self, **kwargs):
        self.tools = []
        self.get_agent(self.tools, **kwargs)

    @calculate_time(name="Agent Response")
    def chat(self, query: str, history: Any = None, **kwargs):
        auto_response = "Sorry I cannot find information to answer your query"

        try:
            # agent chat as default
            response = self.agent.chat(query)
            # debug_table_qa(query, response, **kwargs)
        except (ValidationError, TypeError, AttributeError):
            response = self.query_chat_engine.chat(query)

        response_str = response.response
        if response_str in ["Empty Response"] or len(response_str) == 0:
            response_str = auto_response

        return response_str


class ReActToolAgent(OpenAIToolAgent):
    def get_agent(self, **kwargs):
        self.agent = ReActAgent.from_tools(
            self.tools, system_prompt=SYSTEM_PROMPT, **kwargs
        )

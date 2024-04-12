import os
import os.path as osp
from contextlib import asynccontextmanager
from typing import Any, List, Tuple

import nest_asyncio
import qdrant_client
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from icecream import ic
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.agent import AgentRunner, ReActAgentWorker
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SentenceSplitter,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.packs.query_understanding_agent.step import (
    HumanInputRequiredException,
    QueryUnderstandingAgentWorker,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from sqlalchemy import create_engine

from misc import (
    clarifying_template,
    custom_prompt,
    files_metadata,
    rewrite_query,
    system_prompt,
    tool_description,
)
from utils import calculate_time, delete_astradb

from .table_factory import TableEngineFactory

nest_asyncio.apply()


# Hyperparams
table_name = "rag_demo"
agent_mode = "react-basic"
# agent_mode = "react-query-understanding"
rag_mode = "advanced"
model = "gpt-3.5-turbo"
# model = "claude-3-haiku-20240307"
post_delete_index = False
verbose = True


# Settings
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # type: ignore
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY"
)  # type: ignore

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    # model="text-embedding-ada-002",
    timeout=60,
    max_tries=3,
)


if "gpt" in model:
    llm = OpenAI(model=model)
elif "claude" in model:
    llm = Anthropic(model=model)
    Settings.tokenizer = Anthropic().tokenizer

Settings.llm = llm
Settings.embed_model = embed_model


class TextInput(BaseModel):
    text: str


class Agent:
    tools: List[Any] = []
    clarifying_questions: List[Tuple[str, str]] = []
    should_end = True
    engine = create_engine("sqlite:///:memory:", future=True)
    table_factory = TableEngineFactory(
        node_parser=SentenceSplitter.from_defaults(
            chunk_size=50, chunk_overlap=10
        ),
    )

    def __init__(
        self,
        node_parser=None,
        reranker=None,
    ):
        self.node_parser = node_parser
        self.reranker = reranker
        self._load_index()
        _, self.query_chat_engine = self._get_query_engine()

    def get_agent(self):
        if agent_mode == "react-basic":
            callback_manager = llm.callback_manager
            agent_worker = ReActAgentWorker.from_tools(
                self.tools,
                llm=llm,
                verbose=True,
                system_prompt=system_prompt,
                callback_manager=callback_manager,
            )
            self.agent = AgentRunner(
                agent_worker, callback_manager=callback_manager
            )
        # TODO: Improve this
        # agent can ask back user for clarity
        elif agent_mode == "react-query-understanding":
            callback_manager = llm.callback_manager
            agent_worker = QueryUnderstandingAgentWorker.from_tools(
                self.tools,
                llm=llm,
                system_prompt=system_prompt,
                callback_manager=callback_manager,
            )
            self.agent = AgentRunner(
                agent_worker, callback_manager=callback_manager
            )
        else:
            raise NotImplementedError

    def _load_index(self):
        client = qdrant_client.QdrantClient(location=":memory:")
        self.vector_store = QdrantVectorStore(
            client=client, collection_name=table_name
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex([], storage_context=self.storage_context)

    @calculate_time
    def add_tool_for_file(self, filepath):
        """Add tool for each file
        Currently support .csv file
        """
        suff = osp.splitext(filepath)[-1]

        if suff == ".csv":
            metadata = files_metadata[filepath]
            self.table_factory.add_table_index(
                filepath=filepath, metadata=metadata
            )
            engine = self.table_factory.get_engine(metadata)
            self._add_tool(
                toolname="AI_Engineer_Jobs_Query",
                description=tool_description.format(
                    file_description=metadata["table_desc"]
                ),
                engine=engine,
            )
        # TODO
        elif suff in [".txt", ".pdf", ".pptx"]:
            raise NotImplementedError
        elif suff in [".jpg", ".png"]:
            raise NotImplementedError

        self.get_agent()

    def _parse_document(self, documents, node_parser):
        if rag_mode == "advanced":
            nodes = node_parser.get_nodes_from_documents(documents)
            base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)
            self.index.insert_nodes(base_nodes + objects)
        elif self.mode == "basic":
            for doc in documents:
                self.index.insert(
                    document=doc, storage_context=self.storage_context
                )
        else:
            raise NotImplementedError

    def _get_query_engine(self):
        query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=verbose,
        )
        return query_engine, CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=[],
            verbose=True,
        )

    def _add_tool(self, toolname, description, engine):
        query_tool = QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(name=toolname, description=description),
        )
        self.tools.append(query_tool)

    @calculate_time
    def chat(self, prompt: str):
        auto_response = "Sorry I cannot find information to answer your query"

        if not self.should_end:
            clarifying_texts = "\n".join(
                [
                    clarifying_template.format(
                        question=question, answer=answer
                    )
                    for question, answer in self.clarifying_questions
                ]
            )

            query_text = rewrite_query.format(
                orig_question=prompt, clarifying_texts=clarifying_texts
            )

            rewrite_response = llm.complete(query_text)
            prompt = rewrite_response.text
            ic(prompt)

        try:
            # agent chat as default
            response = self.agent.chat(prompt)
            # check sql query
            if verbose:
                if "sql_query" in response.sources[0].raw_output.metadata:
                    print(response.sources[0].raw_output.metadata["sql_query"])
            response = response.response
            self.should_end = True
            self.clarifying_questions = []

        except HumanInputRequiredException as e:
            response = e.message
            self.clarifying_questions.append((e.message, response))
            self.should_end = False

        except RuntimeError:
            response = self.query_chat_engine.chat(prompt).response
            if response in ["Empty Response"]:
                response = auto_response
            self.should_end = True
            self.clarifying_questions = []

        if len(response) == 0:
            response = auto_response

        return response


agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the agent
    agents["answer_to_everything"] = Agent(
        node_parser=MarkdownElementNodeParser(
            llm=llm,
            num_workers=8,
        ),
        reranker=SimilarityPostprocessor(
            similarity_cutoff=0.5
        ),  # should change other reranker
    )
    yield
    # Clean up the data resources
    if post_delete_index:
        delete_astradb(table_name)


app = FastAPI(lifespan=lifespan)


@app.post("/query", status_code=200)
async def query(input: TextInput):
    return agents["answer_to_everything"].chat(input.text)


@app.put("/update", status_code=200)
async def update(input: TextInput):
    agents["answer_to_everything"].add_tool_for_file(input.text)

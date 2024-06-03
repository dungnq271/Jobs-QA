import json
import os
import os.path as osp
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import nest_asyncio
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.utils import get_tokenizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from src.agent import OpenAIToolAgent
from src.database import QdrantTextDatabase
from src.database.utils import NodeLinkToSource
from src.engine import TableEngine, TextEngine
from src.prompt import (
    DEFAULT_FUNCTION_QUERY_SQL_DESCRIPTION_TMPL,
    DEFAULT_FUNCTION_QUERY_TEXT_DESCRIPTION_TMPL,
)
from src.reader import DocumentReader, TableReader
from src.tool import ToolControler
from src.utils import CHUNKING_REGEX, calculate_time, files_metadata

nest_asyncio.apply()


env: Dict[str, Any] = {}


class TextInput(BaseModel):
    text: str


class TextListInput(BaseModel):
    text_list: List[str]


def get_llm(model):
    if "gpt" in model:
        llm = OpenAI(model=model)
        Settings.tokenizer = get_tokenizer()
    elif "claude" in model:
        llm = Anthropic(model=model)
        Settings.tokenizer = Anthropic().tokenizer
    return llm


@calculate_time(name="Startup")
def startup():
    _ = load_dotenv(find_dotenv())  # read local .env file
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    DEFAULT_COLLECTION_NAME = "agent_demo"
    DEFAULT_EMBED_MODEL = OpenAIEmbedding(
        model="text-embedding-3-small",
        # model="text-embedding-ada-002",
        timeout=60,
        max_tries=3,
    )

    DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
    # DEFAULT_MODEL_NAME = "claude-3-haiku-20240307"
    # DEFAULT_MODEL_NAME = "claude-3-sonnet-20240229"

    DEFAULT_LLM = get_llm(DEFAULT_MODEL_NAME)
    DEFAULT_NODE_PARSER = SentenceSplitter.from_defaults(
        chunk_size=73,
        chunk_overlap=10,
        secondary_chunking_regex=CHUNKING_REGEX,
    )
    DEFAULT_POSTPROCESSORS = [SimilarityPostprocessor(similarity_cutoff=0.5)]
    DEFAULT_TRANSFORMATIONS = [
        DEFAULT_NODE_PARSER,
        DEFAULT_EMBED_MODEL,
        NodeLinkToSource(),
    ]
    DEFAULT_CLIENT_QDRANT = QdrantClient(url=os.getenv("QDRANT_URL"))

    Settings.llm = DEFAULT_LLM
    Settings.embed_model = DEFAULT_EMBED_MODEL
    Settings.node_parser = DEFAULT_NODE_PARSER
    Settings.postprocessors = DEFAULT_POSTPROCESSORS

    env["tool"] = ToolControler()
    env["tools_use"] = env["tool"].get_tools_use()

    env["vector_database"] = QdrantTextDatabase(
        client=DEFAULT_CLIENT_QDRANT,
        transformations=DEFAULT_TRANSFORMATIONS,
        collection_name=DEFAULT_COLLECTION_NAME,
        store_nodes_override=True,
        enable_hybrid=False,
    )
    env["vector_store_index"] = env["vector_database"].index

    env["db_engine"] = create_engine("sqlite:///:memory:", future=True)
    env["reader"] = {
        "table": TableReader(db_engine=env["db_engine"]),
        "text": DocumentReader(),
        "image": None,
    }
    env["engine"] = {
        "table": TableEngine(
            db_engine=env["db_engine"],
            vector_store_index=env["vector_store_index"],
            # node_postprocessors=Settings.postprocessors,
            similarity_top_k=3,
        ),
        "text": TextEngine(vector_store_index=env["vector_store_index"]),
        "image": None,
    }
    env["get_engine_functions"] = {
        "table": [
            {
                "function": env["engine"]["table"].get_sql_retriever,
                "name": "sql_retriever",
                "desciption_template": DEFAULT_FUNCTION_QUERY_SQL_DESCRIPTION_TMPL,
                "type": "sql",
            },
            {
                "function": env["engine"]["table"].get_vector_retriever,
                "name": "vector_retriever",
                "desciption_template": DEFAULT_FUNCTION_QUERY_TEXT_DESCRIPTION_TMPL,
                "type": "recursive",
            },
        ],
        "text": [
            {
                "function": env["engine"]["table"].get_vector_retriever,
                "name": "vector_retriever",
                "desciption_template": DEFAULT_FUNCTION_QUERY_TEXT_DESCRIPTION_TMPL,
                "type": "normal",
            },
        ],
        "image": None,
    }
    env["agent"] = OpenAIToolAgent(
        tools=env["tools_use"], vector_index=env["vector_store_index"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(lifespan=lifespan)


def update_agent_tools():
    env["tools_use"] = env["tool"].get_tools_use()
    env["agent"].get_agent(env["tools_use"], verbose=True)


@app.post("/agent/chat", status_code=200)
async def chat(input: TextInput):
    return env["agent"].chat(query=input.text, db_engine=env["db_engine"])


@app.put("/agent/update_llm", status_code=200)
async def update_llm(input: TextInput):
    model = input.text
    Settings.llm = get_llm(model)


@app.put("/document/add_document", status_code=200)
async def add_document(input: TextInput):
    file_path = input.text

    suff = osp.splitext(file_path)[-1]
    metadata = files_metadata.get(file_path, None)

    if suff in [".csv", ".xlxs"]:
        mode = "table"
    elif suff in [".pdf", ".pptx", ".txt"]:
        mode = "text"
    elif suff in [".jpg", ".png"]:
        mode = "image"

    env["tool"].add_tools_use(
        file_path=file_path,
        metadata=metadata,
        reader=env["reader"][mode],
        vector_db=env["vector_database"],
        get_engine_functions=env["get_engine_functions"][mode],
    )
    update_agent_tools()


@app.post("/tool/get_api_tools_name", status_code=200)
async def get_api_tools_name():
    return env["tool"].get_api_tools_name()


@app.put("/tool/add_api_tools_use", status_code=200)
async def add_api_tools_use(input: TextListInput):
    tools_name = input.text_list
    env["tool"].add_tools_use(tools_name=tools_name)
    update_agent_tools()


@app.put("/tool/remove_tools_use", status_code=200)
async def remove_tools_use(input: TextListInput):
    tool_names = input.text_list
    env["tool"].remove_tools_use(tool_names)
    update_agent_tools()


@app.post("/tool/get_latest_tool_call", status_code=200)
async def get_latest_tool_call():
    display_results_str = ""
    last_task_id = None

    agent = env["agent"].agent

    if env["agent"].agent.get_completed_tasks():
        last_task_id = agent.get_completed_tasks()[-1].task_id

    if last_task_id:
        steps = agent.get_completed_steps(last_task_id)
        for output in steps[-1].output.sources:
            display_results_str = (
                f"**Use:** {output.tool_name}"
                f"  \n**with args:** {output.raw_input['kwargs']}"
                f"  \n**Got output:** {output.raw_output}"
            )

    if "google_search" in display_results_str:
        display_results_str += "  \n**Reference:**  \n"
        with open("tool_results/google_search_result.json", "r") as f:
            tool_results = json.load(f)
        display_results = [item["link"] for item in tool_results["items"]]
        display_results_str += "  \n".join(display_results)

    return display_results_str

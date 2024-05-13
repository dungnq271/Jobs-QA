import os
import os.path as osp
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import nest_asyncio
import pandas as pd
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
from sqlalchemy import create_engine

from src.database import QdrantTableDatabase
from src.engine import TableEngineFactory, TextLinkToSource
from src.reader import TableReader
from src.scraper import JobScraper
from src.utils import CHUNKING_REGEX, get_config, metadata, setup_logging

# requests.packages.urllib3.util.connection.HAS_IPV6 = False


nest_asyncio.apply()


env: Dict[str, Any] = {}


class TextInput(BaseModel):
    text: str


def get_llm(model):
    if "gpt" in model:
        llm = OpenAI(model=model)
        Settings.tokenizer = get_tokenizer()
    elif "claude" in model:
        llm = Anthropic(model=model)
        Settings.tokenizer = Anthropic().tokenizer
    return llm


def get_qa_engine(
    filepath: str = "./documents/posted_jobs.csv",
    table: Optional[pd.DataFrame] = None,
):
    documents = env["reader"].load_data(
        filepath=filepath, metadata=metadata, table=table
    )
    id_node_mapping = env["vector_db"].preprocess(documents)
    env["qa_engine"] = env["factory"].get_qa_engine(
        metadata, id_node_mapping=id_node_mapping
    )


def startup():
    _ = load_dotenv(find_dotenv())  # read local .env file
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # type: ignore
    os.environ["ANTHROPIC_API_KEY"] = os.getenv(
        "ANTHROPIC_API_KEY"
    )  # type: ignore

    config = get_config("./configs/scraper_config.yml")
    setup_logging(
        log_dir=config["output_dir"], config_fpath=config["logger_fpath"]
    )

    model = "gpt-3.5-turbo"
    # model = "claude-3-haiku-20240307"
    # model = "claude-3-sonnet-20240229"

    Settings.llm = get_llm(model)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        # model="text-embedding-ada-002",
        timeout=60,
        max_tries=3,
    )
    Settings.node_parser = SentenceSplitter.from_defaults(
        chunk_size=73,
        chunk_overlap=10,
        secondary_chunking_regex=CHUNKING_REGEX,
    )
    Settings.postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.5)]
    Settings.transformations = [
        Settings.node_parser,
        Settings.embed_model,
        TextLinkToSource(),
    ]

    env["scraper"] = JobScraper(
        output_dpath=config["output_dir"], top_recent=config["top_recent"]
    )
    env["vector_db"] = QdrantTableDatabase(
        collection_name="jobs_qa",
        url="http://0.0.0.0:6333",
        enable_hybrid=False,
    )
    env["vector_store_index"] = env["vector_db"].get_index()
    env["db_engine"] = create_engine("sqlite:///:memory:", future=True)
    env["reader"] = TableReader(db_engine=env["db_engine"])
    env["factory"] = TableEngineFactory(
        db_engine=env["db_engine"],
        vector_store_index=env["vector_store_index"],
        # node_postprocessors=Settings.postprocessors,
    )

    table = None
    filepath = osp.join(config["output_dir"], config["name"] + ".csv")
    if config["scrape"]:
        # TODO: pass search query rather than url
        table, filepath = env["scraper"].scrape(
            url=config["url"], name=config["name"]
        )

    get_qa_engine(filepath=filepath, table=table)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", status_code=200)
async def chat(input: TextInput):
    response = env["qa_engine"].query(input.text)
    return response


@app.put("/update_llm", status_code=200)
async def update_llm(input: TextInput):
    model = input.text
    Settings.llm = get_llm(model)


@app.put("/update_table", status_code=200)
async def update_table(input: TextInput):
    filepath = input.text
    get_qa_engine(filepath=filepath)

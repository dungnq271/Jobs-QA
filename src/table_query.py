import os
import os.path as osp
from typing import List, Dict, Any
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import qdrant_client

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SQLDatabase,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    NLSQLTableQueryEngine,
    SQLAutoVectorQueryEngine
)
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import pandas as pd
from misc import *


nest_asyncio.apply()


#### Main app ####
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

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


def modify_days_to_3digits(day_str):
    words = day_str.split()
    try:
        nday = int(words[0])
        return ' '.join([f"{nday:03}"] + words[1:])
    except:
        return '9999'
    

class TableQueryEngine:
    engine = create_engine("sqlite:///:memory:", future=True)

    def __init__():
        self._load_index()

    def _load_index(self):
        client = qdrant_client.QdrantClient(location=":memory:")
        self.vector_store = QdrantVectorStore(
            client=client, collection_name=table_name
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex([], storage_context=self.storage_context)    

    def _add_nodes_into_index(self):
        pass

    def get_tool(
            df: pd.DataFrame,
            table_name: str,
            table_desc: str,
            chosen_cols_descs: List[Dict]
    ):
        ## Get SQL query engine
        df.to_sql(table_name, self.engine)

        sql_database = SQLDatabase(engine, include_tables=[table_name])
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
        )
        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=sql_desc
        )

        ## Get vector index query engine
        vector_store_info = VectorStoreInfo(
            content_info=table_desc,
            metadata_info=[
                MetadataInfo(**kwargs)
                for kwargs in chosen_cols_descs
            ]
        )
        vector_auto_retriever = VectorIndexAutoRetriever(
            self.index, vector_store_info=vector_store_info, similarity_top_k = 10
        )
        retriever_query_engine = RetrieverQueryEngine.from_args(
            vector_auto_retriever,
            llm=llm,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine,
            description=query_text_description.format(table_desc)
        )

        return SQLAutoVectorQueryEngine(sql_tool, vector_tool)


df = pd.read_csv("../documents/job_vn_posted_full_recent_v2.csv")
new_col = "Number_of_days_posted_ago"
df = df.rename(columns={"Posted": new_col})
df[new_col] = df[new_col].apply(lambda x: modify_days_to_3digits(x))
df.columns


tools = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the agent
    engine = TableQueryEngine()
    tools["answer_to_table"] = 
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/query", status_code=200)
async def query(prompt: TextInput):
    return tools["answer_to_table"].query(prompt.text)

import os
import os.path as osp
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SQLDatabase
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    NLSQLTableQueryEngine,
    SQLAutoVectorQueryEngine
)
from llama_index.core.tools import QueryEngineTool

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import pandas as pd

from utils import calculate_time
from misc import *


nest_asyncio.apply()


#### Hyperparams ####
table_name = "jobPosted"
model = "gpt-3.5-turbo"
post_delete_index = True
verbose = True


#### Main app ####
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    timeout=60,
    max_tries=3,
)

llm = OpenAI(model=model)

Settings.llm = llm
Settings.embed_model = embed_model


class TextInput(BaseModel):
    text: str


class TextList(BaseModel):
    textlist: List[str]


class Agent:
    tools = []
    # Tạo object kết nối với database
    engine = create_engine("sqlite:///:memory:", future=True)
    
    def __init__(
        self,
        node_parser=None,
        reranker=None,
        mode="advanced",
        collection_name="rag_demo",
    ):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",  # "markdown" and "text" are available
            verbose=verbose,
            num_workers=8,
            language="en",
        )
        self.mode = mode
        self.collection_name = collection_name
        self.node_parser = node_parser
        self.reranker = reranker
        # Load index và query chat engine khởi tạo lần đầu
        self._load_index()
        self.query_chat_engine = self._get_query_engine()

    @staticmethod
    def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
        """Thêm pandas DataFrame vào SQL Engine"""
        pandas_df.to_sql(table_name, engine)

    def _load_index(self):
        client = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            location=":memory:"
            # otherwise set Qdrant instance address with:
            # uri="http://<host>:<port>"
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
        )

        vector_store = QdrantVectorStore(client=client, collection_name="jobs_posted")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.vector_index = VectorStoreIndex([], storage_context=storage_context)        

    @calculate_time
    def add_tool_for_file(self, filepath):
        """
        Add tool cho agent, tùy theo hậu tố là .csv hay .pdf, ...
        mà add tool với hàm tương ứng
        """
        suff = osp.splitext(filepath)[-1]

        if suff in [".txt", ".pdf", ".pptx"]:
            documents = SimpleDirectoryReader(
                input_files=[filepath], file_extractor={
                    suff: self.parser for suff in [".pdf", ".pptx"]
                }
            ).load_data()
        if suff == ".csv":
            toolname, description, table_name = self._get_meta_table(filepath)
            engine = self._get_sql_engine(table_name)
        else:
            raise NotImplementedError

        self._parse_document(documents, self.node_parser)  # Add document vào index
        self.query_chat_engine = self._get_query_engine()
            
    def _parse_document(self, documents, node_parser):
        if not hasattr(self, "index"):  # khởi tạo index lần đầu
            if self.mode == "advanced":  # Recursive Retriever sẽ cho kết quả tốt hơn
                nodes = node_parser.get_nodes_from_documents(documents)
                base_nodes, objects = self.node_parser.get_nodes_and_objects(
                    nodes
                )
                self.index = VectorStoreIndex(
                    nodes=base_nodes + objects,
                    storage_context=self.storage_context,
                )
            elif self.mode == "basic":  # Basic Retriever
                self.index = VectorStoreIndex.from_documents(
                    documents, storage_context=self.storage_context
                )
            else:
                raise NotImplementedError
        else:
            # nếu có index rồi thì thêm các document vào index
            for doc in documents:
                self.index.insert(
                    document=doc, storage_context=self.storage_context
                )

    def _get_query_engine(self):
        query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=verbose,
        )
        return \
        CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=[],
            verbose=True,
        )

    @calculate_time
    def chat(self, prompt: str):
        auto_response = "Sorry I cannot find information to answer your query"

        response = self.query_chat_engine.chat(prompt).response
        if response in ["Empty Response"]:
            response = auto_response

        if len(response) == 0:
            response = auto_response

        return response


agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the agent when startup
    agents["answer_to_everything"] = Agent(
        mode=mode,
        collection_name=table_name,
        node_parser=MarkdownElementNodeParser(
            llm=llm,
            num_workers=8,
        ),
        reranker=SimilarityPostprocessor(similarity_cutoff=0.5),
    )
    yield
    # Clean up the data resources after stopping app
    if post_delete_index:
        delete_table()


app = FastAPI(lifespan=lifespan)


@app.post("/query", status_code=200)
async def query(prompt: TextInput):
    """Lấy response từ agent"""
    return agents["answer_to_everything"].chat(prompt.text)


@app.put("/update", status_code=200)
async def update(filepath: TextInput):
    """Add file path vào trong index của agent"""
    agents["answer_to_everything"].add_tool_for_file(filepath.text)

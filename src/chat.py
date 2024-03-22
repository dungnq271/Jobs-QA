import os
import os.path as osp
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_parse import LlamaParse
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser, UnstructuredElementNodeParser
from llama_index.core import SQLDatabase
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import pandas as pd
from astrapy.db import AstraDB
# from unstructured.partition.image import partition_image
# from paddleocr import PaddleOCR

from utils import calculate_time
from misc import *


nest_asyncio.apply()


#### Hyperparams ####
table_name = "rag_demo"
mode = "advanced"
model = "gpt-3.5-turbo"
file_desc = False
post_delete_index = True
verbose = True


#### Main app ####
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    # model="text-embedding-ada-002",
    # model="text-embedding-3-large",
    timeout=60,
    max_tries=3,
)

# reader = FlatReader()
# ocr = PaddleOCR(lang='en')
llm = OpenAI(model=model)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


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

    def _load_index(self):
        # Load vector database, có thể thay đổi thành các loại vectordb trong llamaindex
        # ở đây đang dùng astradb
        self.vstore = AstraDBVectorStore(
            token=os.getenv("ASTRA_TOKEN"),
            api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
            namespace=os.getenv("ASTRA_NAMESPACE"),
            collection_name=self.collection_name,
            embedding_dimension=1536,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vstore
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.vstore, storage_context=self.storage_context
        )

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

        else:
            raise NotImplementedError

        self._parse_document(documents, self.node_parser)  # Add document vào index
        self.query_chat_engine = self._get_query_engine()
            
    def _parse_document(self, documents, node_parser):
        if not hasattr(self, "index"):  # khởi tạo index lần đầu
            if self.mode == "advanced":
                nodes = node_parser.get_nodes_from_documents(documents)
                base_nodes, objects = self.node_parser.get_nodes_and_objects(
                    nodes
                )
                self.index = VectorStoreIndex(
                    nodes=base_nodes + objects,
                    storage_context=self.storage_context,
                )
            elif self.mode == "basic":
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


def delete_table():
    # Drop the table created for this session
    db = AstraDB(
        token=os.getenv("ASTRA_TOKEN"),
        api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
        namespace=os.getenv("ASTRA_NAMESPACE"),
    )
    db.delete_collection(collection_name=table_name)
    print("----------------------APP EXITED----------------------")


agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the agent
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
    # Clean up the data resources    
    if post_delete_index:
        delete_table()


app = FastAPI(lifespan=lifespan)


@app.post("/query", status_code=200)
async def query(prompt: TextInput):
    return agents["answer_to_everything"].chat(prompt.text)


@app.put("/update", status_code=200)
async def update(filepath: TextInput):
    agents["answer_to_everything"].add_tool_for_file(filepath.text)

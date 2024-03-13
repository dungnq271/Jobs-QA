import os
import os.path as osp
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_parse import LlamaParse
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import StorageContext, SQLDatabase
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import pandas as pd

from utils import calculate_time


nest_asyncio.apply()


_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    # model="text-embedding-ada-002",
    # model="text-embedding-3-large",
    timeout=60,
    max_tries=3,
)

llm = OpenAI(model="gpt-3.5-turbo")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


app = FastAPI()


class TextInput(BaseModel):
    text: str


class TextList(BaseModel):
    textlist: List[str]


class Agent:

    tools = []
    engine = create_engine("sqlite:///:memory:", future=True)
    bootstrap_tool_name = "bootstrap"
    
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
            verbose=True,
            num_workers=4,
            language="en",
        )

        self.mode = mode
        self.collection_name = collection_name
        self.node_parser = node_parser
        self.reranker = reranker
        self._load_index()
        self._add_query_engine()
        self._get_agent()

    def _load_index(self):
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

    def _get_agent(self):
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=llm,
            verbose=True,
            # context=context,
            system_prompt=""" 
            You are an agent designed to answer queries about structured-tables.
            Please ALWAYS use the tools provided to answer a question. Do not rely on prior knowledge.
            If there is no information please answer you don't have that information.
            """,
        )

    @staticmethod
    def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
      pandas_df.to_sql(table_name, engine)

    @calculate_time
    def parse_file(self, filepath):
        if ".csv" in filepath:
            table_name = self._add_table(filepath)
            self._add_sql_engine(table_name)
        else:
            document = SimpleDirectoryReader(
                input_files=[filepath], file_extractor={".pdf": self.parser}
            ).load_data()
            self._parse_document(document)
            self._add_query_engine()

        self._get_agent()
        print(self.tools)

    def _parse_document(self, documents):
        if not hasattr(self, "index"):
            if self.mode == "advanced":
                nodes = self.node_parser.get_nodes_from_documents(documents)
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
            for doc in documents:
                self.index.insert(
                    document=doc, storage_context=self.storage_context
                )

    def _add_table(self, filepath):
        filename = osp.basename(filepath)
        table_name = osp.splitext(filename)[0]
        df = pd.read_csv(filepath)
        self.add_df_to_sql_database(table_name, df, self.engine)
        return table_name

    def _add_sql_engine(self, table_name):
        sql_tool = QueryEngineTool(
            query_engine=NLSQLTableQueryEngine(
                sql_database=SQLDatabase(self.engine), tables=[table_name], llm=llm
            ),
            metadata=ToolMetadata(
                name=f"sql_{table_name}",
                description=(
                    "Useful for translating a natural language query into an SQL query over tables"
                    f" containing information about {table_name}."
                ),
            ),
        )
        self.tools.append(sql_tool)

    def _add_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=True,
        )
        query_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name=self.bootstrap_tool_name,
                description=(
                    f"Useful for querying for information about text documents"
                ),
            ),
        )
        # self.tools.append(query_tool)
        if len(self.tools) == 0:
            self.tools.append(query_tool)
        else:
            self.tools[0] = query_tool

    @calculate_time
    def chat(self, prompt: str):
        response = "Sorry I cannot answer your query"
        try:
            response = self.agent.chat(prompt).response
        except Exception as e:
            result = self.query_engine.query(prompt).response
            if result != "Empty Response":
                response = result
        return response


engine = Agent(
    mode="advanced",
    collection_name="rag_demo",
    node_parser=MarkdownElementNodeParser(
        llm=OpenAI(
            model="gpt-3.5-turbo-0125",
        ),
        num_workers=4,
    ),
    reranker=SimilarityPostprocessor(similarity_cutoff=0.5),
)


@app.post("/query_from_text/", status_code=200)
async def query_from_text(prompt: TextInput):
    return engine.chat(prompt.text)


@app.put("/update_text/", status_code=200)
async def update_text(filepath: TextInput):
    engine.parse_file(filepath.text)

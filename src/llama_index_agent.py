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
from llama_index.readers.file import FlatReader
from llama_index.core import SQLDatabase
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from langchain_openai import OpenAI as lc_OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import pandas as pd
from astrapy.db import AstraDB
# from unstructured.partition.image import partition_image
from paddleocr import PaddleOCR

from utils import calculate_time
from misc import file_description


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
ocr = PaddleOCR(lang='en')
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
    engine = create_engine("sqlite:///:memory:", future=True)
    text_filenames = []
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
            verbose=verbose,
            num_workers=4,
            language="en",
        )

        self.mode = mode
        self.collection_name = collection_name
        self.node_parser = node_parser
        self.reranker = reranker
        self._load_index()
        # self._add_query_engine(self.bootstrap_tool_name)
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
        # if len(self.tools) == 0:
        #     self.agent = ConversationChain(
        #         llm=lc_OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        #         verbose=verbose,
        #         memory=ConversationBufferMemory()
        #     )
        # else:
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=llm,
            verbose=True,
            system_prompt=""" 
            You are an agent designed to answer queries from user.
            Please ALWAYS use the tools provided to answer a question. Do not rely on prior knowledge.
            If there is no information please answer you don't have that information.
            """,
        )            

    @staticmethod
    def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
        pandas_df.to_sql(table_name, engine)

    @calculate_time
    def parse_file(self, filepath):
        suff = osp.splitext(filepath)[-1]

        if suff == ".csv":
            table_name = self._add_table(filepath)
            self._add_sql_engine(table_name)
        else:
            if suff in [".jpg", ".png"]:
                # elements = partition_image(filepath)
                # extracted_text = '\n\n'.join([elem.text for elem in elements])
                elements = ocr.ocr(filepath, cls=False)
                extracted_text = '\n\n'.join([elem[-1][0] for elem in elements])
                if verbose:
                    print(extracted_text)
                document = [Document(text=extracted_text)]
            elif suff in [".pdf", ".pptx"]:
                document = SimpleDirectoryReader(
                    input_files=[filepath], file_extractor={
                        suff: self.parser for suff in [".pdf", ".pptx"]
                    }
                ).load_data()
            else:
                raise NotImplementedError

            filename = self._add_text_file(filepath)
            self._parse_document(document, self.node_parser)
            self._add_query_engine(filename)

        self._get_agent()

    def _parse_document(self, documents, node_parser):
        if not hasattr(self, "index"):
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

    def _add_text_file(self, filepath):
        filename = osp.basename(filepath)
        filename = osp.splitext(filename)[0]
        self.text_filenames.append(filename)
        return filename

    def _add_sql_engine(self, table_name):
        sql_tool = QueryEngineTool(
            query_engine=NLSQLTableQueryEngine(
                sql_database=SQLDatabase(self.engine), tables=[table_name], llm=llm
            ),
            metadata=ToolMetadata(
                name=f"sql_{table_name}",
                description=(
                    "Useful for translating a natural language query into an SQL query over table"
                    f"{file_description[table_name]}"
                ),
            ),
        )
        self.tools.append(sql_tool)

    def _add_query_engine(self, filename):
        if file_desc:
            desc = file_description[filename]
        else:
            desc = filename

        # each word in desc is separated by space character
        if '-' in desc:
            desc = ' '.join(desc.split('-'))

        toolname = '_'.join(["query"] + desc.split())

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=verbose,
        )
        query_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name=toolname,
                description=(
                    "Useful for querying for information"
                    f"from text documents about {desc}"
                ),
            ),
        )
        if len(self.tools) > 0:
            if self.tools[0].metadata.name == self.bootstrap_tool_name:
                self.tools.pop(0)
        self.tools.append(query_tool)

    @calculate_time
    def chat(self, prompt: str):
        auto_response = "Sorry I cannot find information to answer your query"

        if len(self.tools) == 0:
            response = self.agent.invoke(prompt)["response"]
        else:
            try:
                response = self.agent.chat(prompt)
                if verbose:
                    if "sql_query" in response.sources[0].raw_output.metadata:
                        print(response.sources[0].raw_output.metadata["sql_query"])
                response = response.response

            except Exception as e:
                response = self.query_engine.query(prompt).response
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
    agents["answer_to_everything"].parse_file(filepath.text)

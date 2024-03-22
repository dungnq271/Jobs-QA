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
        self._load_index()
        _, self.query_chat_engine = self._get_query_engine()

    @staticmethod
    def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
        """Thêm pandas DataFrame vào SQL Engine"""
        pandas_df.to_sql(table_name, engine)

    def get_agent(self):
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=llm,
            verbose=True,
            system_prompt=system_prompt
        )            

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

        if suff == ".csv":
            toolname, description, table_name = self._get_meta_table(filepath)
            engine = self._get_sql_engine(table_name)
        else:
            # if suff in [".jpg", ".png"]:
            #     elements = ocr.ocr(filepath, cls=False)
            #     extracted_text = '\n\n'.join([elem[-1][0] for elem in elements])
            #     if verbose:
            #         print(extracted_text)
            #     document = [Document(text=extracted_text)]

            if suff in [".txt", ".pdf", ".pptx"]:
                document = SimpleDirectoryReader(
                    input_files=[filepath], file_extractor={
                        suff: self.parser for suff in [".pdf", ".pptx"]
                    }
                ).load_data()

            else:
                raise NotImplementedError

            toolname, description = self._get_meta_doc(filepath)
            # Add document vào index
            self._parse_document(document, self.node_parser)
            # Lấy query engine và cập nhật index chat engine
            engine, self.query_chat_engine = self._get_query_engine()
            
        # toolname, description là tên tool với mô tả của tool đó
        # cần cho agent chọn tool
        self._add_tool(toolname, description, engine)
        self.get_agent()

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

    def _get_meta_table(self, filepath):
        filename = osp.basename(filepath)
        table_name = osp.splitext(filename)[0]
        df = pd.read_csv(filepath)
        self.add_df_to_sql_database(table_name, df, self.engine)
        toolname = f"sql_{table_name}"

        if file_desc:
            fdesc = file_description[table_name]
        else:
            fdesc = table_name

        description = query_sql_description.format(description=fdesc)
        return toolname, description, table_name

    def _get_meta_doc(self, filepath):
        filename = osp.basename(filepath)
        filename = osp.splitext(filename)[0]

        if file_desc:
            fdesc = file_description[filename]
        else:
            fdesc = filename

        # mỗi từ trong mô tả được ngăn cách bằng ' '
        if '-' in fdesc:
            fdesc = ' '.join(fdesc.split('-'))
        description = query_text_description.format(description=fdesc)

        # tên của tool là 'query' cùng với các từ trong
        # tên file được ngăn cách bằng '_'
        toolname = '_'.join(["query"] + fdesc.split())
        
        return toolname, description

    def _get_sql_engine(self, table_name):
        return NLSQLTableQueryEngine(
            sql_database=SQLDatabase(self.engine), tables=[table_name], llm=llm
        )

    def _get_query_engine(self):
        query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=verbose,
        )
        return \
        query_engine, \
        CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=[],
            verbose=True,
        )

    def _add_tool(self, toolname, description, engine):
        query_tool = QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=toolname,
                description=description
            ),
        )
        self.tools.append(query_tool)

    @calculate_time
    def chat(self, prompt: str):
        auto_response = "Sorry I cannot find information to answer your query"

        try:
            # mặc định cho ReAct agent chat
            response = self.agent.chat(prompt)
            # kiểm tra lệnh sql agent dùng nếu query bảng
            if verbose:
                if "sql_query" in response.sources[0].raw_output.metadata:
                    print(response.sources[0].raw_output.metadata["sql_query"])
            response = response.response

        except Exception as e:
            # đề phòng xảy ra lỗi thì sử dụng index chat engine để chat
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

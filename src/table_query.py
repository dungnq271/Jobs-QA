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
import random

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SQLDatabase,
    Document,
    load_index_from_storage,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.schema import TextNode, IndexNode, MetadataMode

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    NLSQLTableQueryEngine,
    SQLAutoVectorQueryEngine,
    SQLJoinQueryEngine,
)
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import pandas as pd
from icecream import ic
from misc import *
from utils import calculate_time, debug_qa


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


model = "gpt-3.5-turbo"
# model = "claude-3-haiku-20240307"
if "gpt" in model:
    llm = OpenAI(model=model)
elif "claude" in model:
    llm = Anthropic(model=model)
    Settings.tokenizer = Anthropic().tokenizer

Settings.llm = llm
Settings.embed_model = embed_model


class TextInput(BaseModel):
    text: str


def modify_days_to_3digits(day=str):
    words = day.split()
    try:
        nday = int(words[0])
        return " ".join([f"{nday:03}"] + words[1:])
    except:
        return "9999"


class TableQueryEngine:
    dfs = []
    engine = create_engine("sqlite:///:memory:", future=True)

    def __init__(self, node_parser):
        self.node_parser = node_parser

    def _load_index(self, table_name):
        client = qdrant_client.QdrantClient(location=":memory:")
        self.vector_store = QdrantVectorStore(
            client=client, collection_name=table_name
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.vector_index = VectorStoreIndex(
            [], storage_context=self.storage_context
        )

    @staticmethod
    def _preprocess_data(df, **kwargs):
        if kwargs.get("renamed_cols", None):
            df = df.rename(columns=kwargs["renamed_cols"])

        if kwargs.get("apply_col_funcs", None):
            for col, func in kwargs["apply_col_funcs"].items():
                df[col] = df[col].apply(lambda x: func(x))

        return df

    def add_table_index(self, filepath: str, table_name: str, **kwargs):
        df = pd.read_csv(filepath)
        df = self._preprocess_data(df, **kwargs)
        self.dfs.append(df)

        ## Add into database
        df.to_sql(table_name, self.engine)
        all_cols = df.columns.tolist()
        chosen_cols = [d["name"] for d in chosen_cols_descs]
        self._add_nodes_into_index(
            table_name, all_cols, chosen_cols, exc_cols=["Link"]
        )

    @calculate_time
    def _add_nodes_into_index(
        self, table_name, all_cols, chosen_cols, exc_cols
    ):
        PERSIST_DIR = "./storage"

        if not osp.exists(PERSIST_DIR):
            with self.engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()

            self.all_nodes = []
            debug = True

            for idx, row in enumerate(result):
                desc = row[-2]
                base_node = TextNode(text=desc)
                base_node.metadata = {
                    c: d for c, d in zip(all_cols, row[1:]) if c in chosen_cols
                }
                base_node.id_ = f"node-{idx}"

                sub_inodes = []
                sub_nodes = self.node_parser.get_nodes_from_documents(
                    [Document(text=desc)]
                )
                for node in sub_nodes:
                    node.metadata = {
                        c: d
                        for c, d in zip(all_cols, row[1:])
                        if c in chosen_cols
                    }
                    node.excluded_embed_metadata_keys = exc_cols
                    #         node = IndexNode.from_text_node(node, base_node.node_id)
                    sub_inodes.append(node)

                    if random.randint(0, 1) and debug:
                        print("LLM Metadata Text:")
                        print(node.get_content(metadata_mode=MetadataMode.LLM))
                        print("\n")
                        print("Embedding Metadata Text:")
                        print(
                            node.get_content(metadata_mode=MetadataMode.EMBED)
                        )
                        debug = False

                original_node = IndexNode.from_text_node(
                    base_node, base_node.node_id
                )
                self.all_nodes.extend(sub_inodes)
                # self.all_nodes.append(original_node)

            all_nodes_dict = {n.node_id: n for n in self.all_nodes}
            self.vector_index = VectorStoreIndex(self.all_nodes)
            # store it for later
            self.vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            self.vector_index = load_index_from_storage(storage_context)

    def get_tool(
        self, table_name: str, table_desc: str, chosen_cols_descs: List[Dict]
    ):
        ## Get vector index query engine
        vector_store_info = VectorStoreInfo(
            content_info=table_desc,
            metadata_info=[
                MetadataInfo(**kwargs) for kwargs in chosen_cols_descs
            ],
        )
        vector_auto_retriever = VectorIndexAutoRetriever(
            self.vector_index,
            vector_store_info=vector_store_info,
            similarity_top_k=10,
        )
        retriever_query_engine = RetrieverQueryEngine.from_args(
            vector_auto_retriever,
            llm=llm,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ],
        )
        query_desc = query_text_description.format(file_description=table_desc)
        ic(query_desc)
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine, description=query_desc
        )

        ## Get SQL query engine
        sql_database = SQLDatabase(self.engine, include_tables=[table_name])
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
        )

        all_cols = self.dfs[-1].columns.tolist()
        sql_desc = query_sql_description.format(
            columns_list=", ".join(all_cols)
        )
        ic(sql_desc)
        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine, description=sql_desc
        )

        # return SQLAutoVectorQueryEngine(sql_tool, vector_tool)
        return SQLJoinQueryEngine(sql_tool, vector_tool, llm=llm)


tools = {}
table_name = "jobPosted"
table_desc = "different AI jobs information at different companies"
renamed_cols = {"Posted": "Number_of_days_posted_ago"}
apply_col_funcs = {"Number_of_days_posted_ago": modify_days_to_3digits}
chosen_cols_descs = [
    {
        "name": "Number_of_days_posted_ago",
        "type": "str",
        "description": "How many days ago the job was posted (in Vietnamese)",
    },
    {
        "name": "Full / Part Time",
        "type": "str",
        "description": "Working time for the job",
    },
    {"name": "Salary", "type": "str", "description": "Job's pay range"},
    {"name": "Link", "type": "str", "description": "Link to the posted job"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    tools["engine"] = TableQueryEngine(
        node_parser=SentenceSplitter.from_defaults(
            chunk_size=50, chunk_overlap=10
        )
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/query", status_code=200)
async def query(prompt: TextInput):
    query = prompt.text
    response = tools["answer_to_table"].query(query)
    debug_qa(query, response, tools["engine"].engine)
    return response


@app.put("/update", status_code=200)
async def update(filepath: TextInput):
    tools["engine"].add_table_index(
        filepath.text,
        table_name,
        renamed_cols=renamed_cols,
        apply_col_funcs=apply_col_funcs,
    )

    # Load the agent
    tools["answer_to_table"] = tools["engine"].get_tool(
        table_name=table_name,
        table_desc=table_desc,
        chosen_cols_descs=chosen_cols_descs,
    )

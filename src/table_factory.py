import os
import os.path as osp
import random
from contextlib import asynccontextmanager
from typing import List

import nest_asyncio
import pandas as pd
import qdrant_client
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from icecream import ic
from llama_index.core import (
    Document,
    Settings,
    SQLDatabase,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import (  # SQLAutoVectorQueryEngine,
    NLSQLTableQueryEngine,
    RetrieverQueryEngine,
    SQLJoinQueryEngine,
)
from llama_index.core.retrievers import (  # RecursiveRetriever,
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import MetadataMode, TextNode  # IndexNode,
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from misc import files_metadata, query_sql_description, query_text_description
from utils import calculate_time, debug_qa

# from sqlalchemy.engine.base import Engine


nest_asyncio.apply()


# Main app
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


class TableEngineFactory:
    dfs: List[pd.DataFrame] = []
    db_engine = create_engine("sqlite:///:memory:", future=True)

    def __init__(self, node_parser, db_engine=None):
        self.node_parser = node_parser
        if db_engine:
            self.db_engine = db_engine

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

    def add_table_index(self, filepath, metadata):
        table_name = metadata["table_name"]
        chosen_cols_descs = metadata["chosen_cols_descs"]
        exc_cols = metadata["exc_cols"]

        df = pd.read_csv(filepath)
        df = self._preprocess_data(
            df,
            renamed_cols=metadata["renamed_cols"],
            apply_col_funcs=metadata["apply_col_funcs"],
        )
        self.dfs.append(df)

        df.to_sql(table_name, self.db_engine)  # add into database
        all_cols = df.columns.tolist()
        chosen_cols = [d["name"] for d in chosen_cols_descs]
        self._add_nodes_into_index(
            table_name, all_cols, chosen_cols, exc_cols=exc_cols
        )

    @calculate_time
    def _add_nodes_into_index(
        self,
        table_name: str,
        all_cols: List[str],
        chosen_cols: List[str],
        exc_cols: List[str],
    ):
        """Add nodes into vector store

        Args:
            table_name: name of the uploaded table
            all_cols: list of all column names
            chosen_cols: list of column names to be included in node metadata
            exc_cols: column names to be excluded from embedding

        """
        PERSIST_DIR = "./storage"

        if not osp.exists(PERSIST_DIR):
            with self.db_engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()

            self.all_nodes = []
            debug = True

            for idx, row in enumerate(result):
                desc = row[-2]
                base_node = TextNode(text=desc)
                base_node.metadata = {
                    c: d
                    for c, d in zip(
                        all_cols, row[1:]
                    )  # first element of row is the index
                    if c in chosen_cols
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
                    # node = IndexNode.from_text_node(node, base_node.node_id)
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

                # original_node = IndexNode.from_text_node(
                #     base_node, base_node.node_id
                # )
                self.all_nodes.extend(sub_inodes)
                # self.all_nodes.append(original_node)

            # all_nodes_dict = {n.node_id: n for n in self.all_nodes}
            self.vector_index = VectorStoreIndex(self.all_nodes)
            # store it for later
            self.vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            self.vector_index = load_index_from_storage(storage_context)

    def get_engine(self, metadata):
        """Get table query engine for the recently uploaded table file"""

        table_name = metadata["table_name"]
        table_desc = metadata["table_desc"]
        chosen_cols_descs = metadata["chosen_cols_descs"]

        # Get vector index query engine
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

        # Get SQL query engine
        sql_database = SQLDatabase(self.db_engine, include_tables=[table_name])
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

    def query(self, input: str):
        pass


tools = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tools["engine"] = TableEngineFactory(
        node_parser=SentenceSplitter.from_defaults(
            chunk_size=50, chunk_overlap=10
        )
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.put("/update", status_code=200)
async def update(input: TextInput):
    filepath = input.text
    metadata = files_metadata[filepath]
    tools["engine"].add_table_index(filepath=filepath, metadata=metadata)
    tools["answer_to_table"] = tools["engine"].get_engine(metadata)


@app.post("/query", status_code=200)
async def query(input: TextInput):
    query = input.text
    response = tools["answer_to_table"].query(query)
    debug_qa(query, response, tools["engine"].db_engine)
    return response

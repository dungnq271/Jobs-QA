# %%
import os
from llama_index.llms.openai import OpenAI
import pandas as pd
import ast

# %% [markdown]
### Normal Index

# %%
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import MarkdownElementNodeParser

# %%
# check if storage already exists
PERSIST_DIR = "../storage"

# if not osp.exists(PERSIST_DIR):

# load the documents and create the index
documents = SimpleDirectoryReader(
    input_files=["../documents/job_vn_posted_full.csv"],
).load_data()
index = VectorStoreIndex.from_documents(documents)
# store it for later
index.storage_context.persist(persist_dir=PERSIST_DIR)

# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# %%
nodes = index.docstore.docs.values()
len(nodes)

# %%
one_node = list(nodes)[1]
one_node.__dict__.keys()

# %%
print(one_node.text)

# %%
query = "List me all the jobs which is about both NLP and Computer Vision"
response = query_engine.query(query)
print(response)

# %% [markdown]
### Index each row

# %%
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from llama_index.core.schema import TextNode

# %%
def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
    pandas_df.to_sql(table_name, engine)

engine = create_engine("sqlite:///:memory:", future=True)

# %%
df = pd.read_csv("../documents/job_vn_posted_full.csv")
df.columns

# %%
table_name = "jobPosted"
add_df_to_sql_database(table_name, df, engine)

# %%
with engine.connect() as conn:
    cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
    result = cursor.fetchall()

# %%
row_tups = [tuple(res)[2:-1] for res in result]
row_tups[0]

# %%
nodes = [TextNode(text=str(t)) for t in row_tups]
len(nodes)

# %%
nodes[0]

# %%
# put into vector store index (use OpenAIEmbeddings by default)
table_index_dir = "table_storage"

if not os.path.exists(f"{table_index_dir}/{table_name}"):
    index = VectorStoreIndex(nodes)

    # save index
    index.set_index_id("vector_index")
    index.storage_context.persist(f"{table_index_dir}/{table_name}")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=f"{table_index_dir}/{table_name}"
    )
    # load index
    index = load_index_from_storage(
        storage_context, index_id="vector_index"
    )

# %%
query_engine = index.as_query_engine()

# %%
# query = "List me all the jobs which include both NLP and Computer Vision"
query = "List me the pay range of the top 3 most recent posted jobs?"
response = query_engine.query(query)
print(response)

# %%
response.__dict__.keys()

# %%
response.source_nodes[1]

# %%
srcn = ast.literal_eval(response.source_nodes[1].text)
print(srcn)

# %%
print(srcn[-1])


# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# %%
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

# %%
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
vector_index = VectorStoreIndex([], storage_context=storage_context)

# %% [markdown]
### Load data

# %%
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import pandas as pd

# %%
def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
    pandas_df.to_sql(table_name, engine)

engine = create_engine("sqlite:///:memory:", future=True)

# %%
df = pd.read_csv("../../documents/job_vn_posted_full.csv")
df.columns

# %%
table_name = "jobPosted"
add_df_to_sql_database(table_name, df, engine)

# %%
with engine.connect() as conn:
    cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
    result = cursor.fetchall()

# %% [markdown]
### Build SQL Index

# %%
sql_database = SQLDatabase(engine, include_tables=[table_name])
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[table_name],
)

# %% [markdown]
### Build Vector Index¶

# %%
nodes = []

for row in result:
    company = row[3]
    link = row[-1]
    desc = str(row[2:-1])
    node = TextNode(text=desc)
    node.metadata = {"title": company, "link": link}
    nodes.append(node)

vector_index.insert_nodes(nodes)

# %%
llm = OpenAI(model="gpt-3.5-turbo")
Settings.llm = llm

vector_store_info = VectorStoreInfo(
    content_info="AI Engineer job information of different companies",
    metadata_info=[
        MetadataInfo(
            name="company", type="str", description="The name of the company"
        ),
        MetadataInfo(
            name="link", type="str", description="Link to the posted job"
        ),        
    ],
)
vector_auto_retriever = VectorIndexAutoRetriever(
    vector_index, vector_store_info=vector_store_info
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, llm=llm
)

# %%
print(f"Useful for translating a natural language query into a SQL query over"
f" a table containing: {', '.join(df.columns.to_list()[1:-1])}")

# %%
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        f"Useful for translating a natural language query into a SQL query over"
        f" a table containing: {', '.join(df.columns.to_list()[1:-1])}"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different job information"
        " at different companies"
    ),
)

query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, llm=llm
)

# %%
query = "List me all the links to jobs which include both NLP and Computer Vision"
# query = "List me the salary of the top 3 most recent posted jobs?"
# query = "List me the top skills in demand for all the jobs?"
response = query_engine.query(query)
print(response)

# %%
response.source_nodes

# %%
with engine.connect() as conn:
    cursor = conn.execute(
        text(
        "SELECT Description FROM jobPosted WHERE Company is"
        " 'Công ty Cổ phần Giải pháp Công nghệ TTC Việt Nam'"
        )
    )
    result = cursor.fetchall()

# %%
from IPython.display import display, Markdown
display(Markdown(result[0][0]))

# %%

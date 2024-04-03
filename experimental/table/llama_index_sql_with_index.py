# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# %%
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

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

# %%
# llm = OpenAI(model="gpt-3.5-turbo-instruct")
llm = OpenAI(model="gpt-3.5-turbo")
Settings.llm = llm

# %% [markdown]
# Using Anthropic

# %%
llm = Anthropic(model="claude-3-sonnet")
Settings.tokenizer = Anthropic().tokenizer
resp = llm.complete("Paul Graham is ")
print(resp)

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
df = pd.read_csv("../../documents/job_vn_posted_full_recent_v2.csv")
df.head()

# %%
df.columns

# %%
new_col = "Number-of-days-posted-ago"
df = df.rename(columns={"Posted": new_col})
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
cols = df.columns.tolist()
acc_cols = [new_col, "Full / Part Time", "Salary", "Link"]

# %%
row = result[0][-2]
print(row)

# %%
node_parser = SentenceSplitter.from_defaults(chunk_size=50, chunk_overlap=10)
nodes = node_parser.get_nodes_from_documents([Document(text=row)])

# %%
for i, x in enumerate(nodes):
    print(f"Chunk {i}:", x.text)

# %%
len(cols), len(result[0][2:-2])

# %%
nodes = []

for row in result:
    desc = row[-2]
    sub_nodes = node_parser.get_nodes_from_documents([Document(text=desc)])
    for node in sub_nodes:
        node.metadata = {c: d for c, d in zip(cols, row[1:]) if c in acc_cols}
    # print(desc)
    # print(node.metadata)
    # break
    nodes.extend(sub_nodes)

# %%
%%time
vector_index.insert_nodes(nodes)

# %%
len(nodes)

# %%
vector_store_info = VectorStoreInfo(
    content_info="AI Engineer job information of different companies",
    metadata_info=[
        MetadataInfo(
            name=new_col, type="str", description="How many days ago was the job posted in Vietnamese"
        ),
        MetadataInfo(
            name="Full / Part Time", type="str", description="Working time for the job"
        ),                
        MetadataInfo(
            name="Salary", type="str", description="Job's pay range"
        ),
        MetadataInfo(
            name="Link", type="str", description="Link to the posted job"
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
        f" with both Vietnamese and English texts"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different job information"
        " at different companies"
    ),
)

# %%
query_1 = "List me all the links to jobs which include both NLP and Computer Vision"
query_2 = "List me all the links to jobs which demand only 1 year of experience"
query_3 = "List me the salary of the top 3 most recent posted jobs?"

# %% [markdown]
### Using SQLAutoVectorQueryEngine

# %%
sqlauto_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, llm=llm
)

# %%
query = query_1
response = sqlauto_engine.query(query)

# %%
print(response)

# %%
for i, node in enumerate(response.source_nodes):
    print(f"Node {i+1} Text:\n", node.text)

# %%
query = query_2
response = sqlauto_engine.query(query)
print(response)

# %%
for i, node in enumerate(response.source_nodes):
    print(f"Node {i+1} Text:\n", node.text)

# %%
query = query_3
response = sqlauto_engine.query(query)
print(response)

# %%
for i, node in enumerate(response.source_nodes):
    print(f"Node {i+1} Text:\n", node.text)

# %% [markdown]
### Using ReActAgent
from llama_index.core.agent import ReActAgent

# %%
system_prompt = \
""" 
You are an agent designed to answer queries from user.
Please ALWAYS use the tools provided to answer a question. Do not rely on prior knowledge.
If there is no information please answer you don't have that information.
"""

# %%
react_agent = ReActAgent.from_tools(
    [sql_tool, vector_tool],
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
    enable_hybrid=True
)            

# %%
query = query_3
response = react_agent.chat(query)

# %%
print(response)

# %%
response.source_nodes[0]

# %%
for i, node in enumerate(response.source_nodes):
    if i != 0:
        print('\n')
    print(node.metadata[new_col])
    print(f"Node {i+1} Text:\n", node.text)

# # %%
# response = react_agent.chat("Yes")
# print(response)

# %%
if len(response.sources) > 0:
    if "sql_query" in response.sources[0].raw_output.metadata:
        print(response.sources[0].raw_output.metadata["sql_query"])

# %%
response.source_nodes

# %% [markdown]
### Using Retriever Router Query Engine¶
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine

# %%
tool_mapping = SimpleToolNodeMapping.from_objects([sql_tool, vector_tool])
obj_index = ObjectIndex.from_objects(
    [sql_tool, vector_tool],
    tool_mapping,
    VectorStoreIndex,
)

# %%
router_engine = ToolRetrieverRouterQueryEngine(
    obj_index.as_retriever(),
)

# %%
print(query)
response = router_engine.query(query)
print(response)

# %%
response.source_nodes

# %% [markdown]
### Using FnRetrieverOpenAIAgent
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent

# %%
fnr_agent = FnRetrieverOpenAIAgent.from_retriever(
    obj_index.as_retriever(similarity_top_k=3),
    system_prompt=system_prompt,
    verbose=True
)

# %%
response = fnr_agent.query(query)
print(response)

# %% [markdown]
### Test the result
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
with engine.connect() as conn:
    cursor = conn.execute(
        text(
        "SELECT Company, Link FROM jobPosted"
        " WHERE Description LIKE '%1 năm kinh nghiệm%'"
        )
    )
    result = cursor.fetchall()

# %%
len(result)

# %%
for res in result:
    print(res)

# %%

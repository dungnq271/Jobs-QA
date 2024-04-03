# %%
from llama_index.llms.openai import OpenAI
import pandas as pd

# %% [markdown]
### SQL Engine

# %%
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


# %%
def add_df_to_sql_database(
    table_name: str, pandas_df: pd.DataFrame, engine: Engine
) -> None:
    pandas_df.to_sql(table_name, engine)


engine = create_engine("sqlite:///:memory:", future=True)

# %%
# df = pd.read_csv("../documents/allPlayersLookup.csv")
# table_name = "allPlayersLookup"

df = pd.read_csv("../documents/job_vn_posted_full.csv")
table_name = "jobPosted"

add_df_to_sql_database(table_name, df, engine)

# %%
model = "gpt-3.5-turbo"
llm = OpenAI(model=model)
query_engine = NLSQLTableQueryEngine(
    sql_database=SQLDatabase(engine), tables=[table_name], llm=llm
)

# %%
df.head()

# %%
# response = query_engine.query("Who is the best player?")
response = query_engine.query("How many jobs in total")
print(response.metadata["sql_query"])
print(response.response)

# %%

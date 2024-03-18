# %%
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor

_ = load_dotenv(find_dotenv())  # read local .env file

# %%
#### Hyperparams ####
table_name = "rag_demo"
mode = "advanced"
model = "gpt-3.5-turbo"
bootstrap_tool_name = "bootstrap"
collection_name = "rag_demo"

# %%
vstore = AstraDBVectorStore(
    token=os.getenv("ASTRA_TOKEN"),
    api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
    namespace=os.getenv("ASTRA_NAMESPACE"),
    collection_name=collection_name,
    embedding_dimension=1536,
)
storage_context = StorageContext.from_defaults(
    vector_store=vstore
)
index = VectorStoreIndex.from_vector_store(
    vstore, storage_context=storage_context
)

# %%
query_engine = index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    verbose=True,
)

query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name=bootstrap_tool_name,
        description=(
            "Useful for querying for information"
            f"about text documents"
        ),
    ),
)

# %%
query_tool.metadata.name

# %%


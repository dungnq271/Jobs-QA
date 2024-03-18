# %%
import os
import os.path as osp
from dotenv import load_dotenv, find_dotenv
import nest_asyncio

from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import MarkdownElementNodeParser

_ = load_dotenv(find_dotenv())  # read local .env file
nest_asyncio.apply()

from PIL import Image
import matplotlib.pyplot as plt

# %%
llm = OpenAI(model="gpt-3.5-turbo")
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
    num_workers=8,
    language="en",
)

# %%
# check if storage already exists
PERSIST_DIR = "../storage"

if not osp.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader(
        input_files=["../documents/uber_10q_march_2022.pdf"],
        file_extractor={".pdf": parser}
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# %%
def display_image(path):
    plt.figure(figsize=(20, 40))
    plt.imshow(Image.open(path))    
    plt.show()

# %%
queries = [
    "how is the Cash paid for Income taxes, net of refunds from Supplemental disclosures of cash flow information?",
    "what is the change of free cash flow and what is the rate from the financial and operational highlights?",
    "what is the net loss value attributable to Uber compared to last year?",
    "What were cash flows like from investing activities?"
]

# %%
for i, query in enumerate(queries):
    response = query_engine.query(query)
    print(response)
    display_image(f"../imgs/uber_q{i+1}.png")

# %% [markdown]
## RAG Triad Evaluation
from trulens_eval import Tru

tru = Tru()
tru.reset_database()

# %% [markdown]
### Initialize Feedback Function(s)
import numpy as np
from trulens_eval.feedback.provider.openai import OpenAI
openai = OpenAI()

from trulens_eval.app import App
from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval import TruLlama


def get_prebuilt_trulens_recorder(query_engine, app_id):
    openai = OpenAI()

    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    grounded = Groundedness(groundedness_provider=openai)

    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )

    return tru_recorder

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

# %%            
tru_recorder_1 = get_prebuilt_trulens_recorder(
    query_engine,
    app_id='LlamaIndex_App1'
)
run_evals(queries, tru_recorder_1, query_engine)        
Tru().run_dashboard()

# %%
records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])
records.head()

# %% [markdown]
### Advanced RAG
# load the documents and create the index
PERSIST_DIR = "../storage_2"

if not osp.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(
        input_files=["../documents/uber_10q_march_2022.pdf"],
        file_extractor={".pdf": parser}
    ).load_data()

    # index = VectorStoreIndex.from_documents(documents)
    node_parser = MarkdownElementNodeParser(
        llm=llm,
        num_workers=8,
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    index = VectorStoreIndex(nodes=base_nodes + objects)

    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# %%
query_engine = index.as_query_engine(
    similarity_top_k=15,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    verbose=False,
)

# %%
tru_recorder_2 = get_prebuilt_trulens_recorder(
    query_engine,
    app_id='LlamaIndex_App2'
)
run_evals(queries, tru_recorder_2, query_engine)        
Tru().run_dashboard()

# %%

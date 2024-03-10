import os
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext
)
from llama_parse import LlamaParse
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor

nest_asyncio.apply()


_ = load_dotenv(find_dotenv()) # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


embed_model=OpenAIEmbedding(
    model="text-embedding-3-small",
    # model="text-embedding-ada-002",
    # model="text-embedding-3-large",
    timeout=60,
    max_tries=3
)

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = embed_model
Settings.chunk_size = 512


app = FastAPI()


class TextInput(BaseModel):
    text: str


class TextList(BaseModel):
    textlist: List[str]


class TextParser:
    def __init__(self, node_parser=None, reranker=None, mode="advanced"):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
            num_workers=4,
            language="en"
        )

        self.mode = mode
        self.node_parser = node_parser
        self.reranker = reranker
        self.load_db()

    def load_db(self):
        self.vstore = AstraDBVectorStore(
            token=os.getenv("ASTRA_TOKEN"),
            api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
            namespace=os.getenv("ASTRA_NAMESPACE"),
            collection_name="qa_v2",
            embedding_dimension=1536
        )
        
    def parse_files(self, filepaths):
        documents = self.parser.load_data(filepaths)
        self._parse_documents(documents)

    def parse_dir(self, data_dir):
        documents = SimpleDirectoryReader(
            data_dir, file_extractor={".pdf": self.parser}
        ).load_data()
        self._parse_documents(documents)

    def _parse_documents(self, documents):
        storage_context = StorageContext.from_defaults(vector_store=self.vstore)
        
        if not hasattr(self, "index"):
            if self.mode == "advanced":
                nodes = self.node_parser.get_nodes_from_documents(documents)
                base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)
                self.index = VectorStoreIndex(
                    nodes=base_nodes+objects,
                    storage_context=storage_context
                )
            elif self.mode == "basic":
                self.index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context
                )
                # self.index = VectorStoreIndex.from_vector_store(
                #     self.vstore, storage_context=storage_context
                # )
            else:
                raise NotImplementedError
        else:
            for doc in documents:
                self.index.insert(document=doc, storage_context=storage_context)

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=True
        )

    def query(self, prompt: str):
        return self.query_engine.query(prompt)


engine = TextParser(
    mode="basic",
    # node_parser=MarkdownElementNodeParser(
    #     llm=OpenAI(
    #         model="gpt-3.5-turbo-0125",
    #     ),
    #     num_workers=8
    # )
    reranker=SimilarityPostprocessor(similarity_cutoff=0.5)
)
# engine.parse_from_dir(
#     data_dir="/home/dungmaster/Datasets/rag-data/text",
# )


@app.post("/query_from_text/", status_code=200)
async def query_from_text(prompt: TextInput):
    return engine.query(prompt.text)


@app.put("/update_text/", status_code=200)
async def update_text():
    # engine.parse_files(filepaths.textlist)
    engine.parse_dir("./documents")    

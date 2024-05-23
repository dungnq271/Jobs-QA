import os

import nest_asyncio
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.astra import AstraDBVectorStore
from llama_parse import LlamaParse
from pydantic import BaseModel

from utils import calculate_time

nest_asyncio.apply()


_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # type: ignore


embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    # model="text-embedding-ada-002",
    # model="text-embedding-3-large",
    timeout=60,
    max_tries=3,
)

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = embed_model
Settings.chunk_size = 512


app = FastAPI()


class TextInput(BaseModel):
    text: str


class TextList(BaseModel):
    textlist: list[str]


class TextParser:
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
            verbose=True,
            num_workers=4,
            language="en",
        )

        self.mode = mode
        self.collection_name = collection_name
        self.node_parser = node_parser
        self.reranker = reranker
        self._load_index()
        self._get_query_engine()

    def _load_index(self):
        self.vstore = AstraDBVectorStore(
            token=os.getenv("ASTRA_TOKEN"),
            api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
            namespace=os.getenv("ASTRA_NAMESPACE"),
            collection_name=self.collection_name,
            embedding_dimension=1536,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vstore)
        self.index = VectorStoreIndex.from_vector_store(
            self.vstore, storage_context=self.storage_context
        )

    @calculate_time
    def parse_files(self, filepaths):
        documents = SimpleDirectoryReader(
            input_files=filepaths, file_extractor={".pdf": self.parser}
        ).load_data()
        self._parse_documents(documents)

    def parse_dir(self, data_dir):
        documents = SimpleDirectoryReader(
            data_dir, file_extractor={".pdf": self.parser}
        ).load_data()
        self._parse_documents(documents)

    def _parse_documents(self, documents):
        if not hasattr(self, "index"):
            if self.mode == "advanced":
                nodes = self.node_parser.get_nodes_from_documents(documents)
                base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)
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
            for doc in documents:
                self.index.insert(document=doc, storage_context=self.storage_context)

        self._get_query_engine()

    def _get_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            verbose=True,
        )

    @calculate_time
    def query(self, prompt: str):
        return self.query_engine.query(prompt)


engine = TextParser(
    mode="advanced",
    collection_name="rag_demo",
    node_parser=MarkdownElementNodeParser(
        llm=OpenAI(
            model="gpt-3.5-turbo-0125",
        ),
        num_workers=4,
    ),
    reranker=SimilarityPostprocessor(similarity_cutoff=0.5),
)


@app.post("/query_from_text/", status_code=200)
async def query_from_text(prompt: TextInput):
    return engine.query(prompt.text)


@app.put("/update_text/", status_code=200)
async def update_text(filepaths: TextList):
    engine.parse_files(filepaths.textlist)

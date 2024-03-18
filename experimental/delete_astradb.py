import os
from dotenv import load_dotenv, find_dotenv
from astrapy.db import AstraDB

_ = load_dotenv(find_dotenv())  # read local .env file


table_name = "rag_demo"


db = AstraDB(
    token=os.getenv("ASTRA_TOKEN"),
    api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
    namespace=os.getenv("ASTRA_NAMESPACE"),
)
db.delete_collection(collection_name=table_name)

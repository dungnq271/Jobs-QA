import os
import requests
import time

from pinecone import Pinecone
from langchain.chains import LLMChain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from llama_index.core.response.notebook_utils import display_source_node

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from astrapy.db import AstraDB
from sqlalchemy import create_engine, text


_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


template = """
Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base. Keep the user query the same if unnecessary.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:
"""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)


def save_file(filebytes, filepath):
    with open(filepath, "wb") as f:
        f.write(filebytes)
    f.close()


def get_embedding(text):
    text = text.replace("\n", " ")
    return embeddings.embed_query(text)


def calculate_time(func):
    def timing(*args, **kwargs):
        t1 = time.time()
        outputs = func(*args, **kwargs)
        t2 = time.time()
        print(f"Time: {(t2-t1):.3f}s")
        return outputs

    return timing


def get_response(input):
    response = requests.post(
        url="http://127.0.0.1:8000/query",
        json={"text": input},
    )
    assert response.status_code == 200, response.status_code
    json_response = response.json()
    return json_response


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def query_refiner(conversation, query):
    return llm_chain.invoke({"conversation": conversation, "query": query})


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):

        conversation_string += (
            "Human: " + st.session_state["requests"][i] + "\n"
        )
        conversation_string += (
            "Bot: " + st.session_state["responses"][i + 1] + "\n"
        )
    return conversation_string


def delete_astradb():
    # Drop the table created for this session
    db = AstraDB(
        token=os.getenv("ASTRA_TOKEN"),
        api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
        namespace=os.getenv("ASTRA_NAMESPACE"),
    )
    db.delete_collection(collection_name=table_name)
    print("----------------------APP EXITED----------------------")


def debug_qa(query, response, engine):
    print("\n***********Query***********")
    print(query)
    print("\n***********Response***********")
    print(response)

    print("\n***********Source Nodes***********")
    for node in response.source_nodes:
        # display_source_node(node, source_length=2000)
        print(node.text)

    if len(response.metadata) > 0:
        print("\n***********SQL Query***********")
        if "sql_query" in response.metadata:
            sql_query = response.metadata["sql_query"]
            print("Command:", sql_query)
            try:
                with engine.connect() as conn:
                    cursor = conn.execute(text(sql_query))
                    result = cursor.fetchall()
                print("Result:", result)
            except:
                print("SQL Command invalid!")

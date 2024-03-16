import os
import requests
import time

from pinecone import Pinecone
from langchain.chains import LLMChain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

import streamlit as st
from dotenv import load_dotenv, find_dotenv


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

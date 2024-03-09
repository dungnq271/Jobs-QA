import os
import requests

from pinecone import Pinecone
from langchain_astradb import AstraDBVectorStore
from langchain.chains import LLMChain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

import streamlit as st
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="gpt-3.5-turbo")

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(os.getenv("PINECONE_INDEX_NAME")) 


template = """
Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:
"""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)


def get_embedding(text):
   text = text.replace("\n", " ")
   return embeddings.embed_query(text)


def find_match(input):
    # result = vstore.similarity_search(input, k=3)
    # return result.page_content

    response = requests.post(
        url="http://127.0.0.1:8000/query_from_text/",
        json={"text": input},
    )
    assert response.status_code == 200, response.status_code
    json_response = response.json()
    print(json_response)
    # outfit_recommends = json_response["outfit_recommend"]
    return json_response["response"]


def query_refiner(conversation, query):
    return llm_chain.invoke(
        {"conversation": conversation, "query": query}
    )


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

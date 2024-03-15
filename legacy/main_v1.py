import os
import os.path as osp

from langchain_openai import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

import asyncio
import requests

import streamlit as st
from streamlit_chat import message
from utils import *


#### Hyperparams ####
doc_dir = "./documents"
refine_query = False


#### Main app ####
title = "Chatbot with LlamaIndex 🦙, ChatGPT, AstraDB, and Streamlit"
# st.header(title)
st.markdown(
    f"<h2 style='text-align: center;'>{title}</h2>", unsafe_allow_html=True
)

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

if "documents" not in st.session_state:
    st.session_state["documents"] = []


# if "buffer_memory" not in st.session_state:
#     st.session_state.buffer_memory = ConversationBufferWindowMemory(
#         k=3, return_messages=True
#     )


## Create model
# llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# system_msg_template = SystemMessagePromptTemplate.from_template(
#     template="""Answer the question as truthfully as possible using the provided context,
#     and if the answer is not contained within the text below, say 'I don't know'""")


# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


## Upload and parse files
if not osp.exists(doc_dir):
    os.makedirs(doc_dir, exist_ok=True)


with st.sidebar:
    st.header("File Upload Options")
    uploaded_files = st.file_uploader(
        "Upload files", type=["txt", "csv", "pdf"], accept_multiple_files=True
    )


async def parse_file(
    uploaded_file, existing_files=st.session_state["documents"]
):
    new_files = []
    filename = uploaded_file.name
    if filename not in existing_files:
        filepath = osp.join(doc_dir, osp.basename(filename))
        save_file(uploaded_file.read(), filepath)
        existing_files.append(filename)
        requests.put(
            "http://127.0.0.1:8000/update",
            json={"text": filepath},
        )
        new_files.append(filename)
    return new_files


async def parse_job():
    new_files = await asyncio.gather(
        *[
            parse_file(uf, st.session_state["documents"])
            for uf in uploaded_files
        ]
    )
    st.session_state["documents"].extend(new_files)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


def on_click():
    st.session_state.user_input = ""


with textcontainer:
    query = st.text_input("Query: ", key="user_input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            if refine_query:
                refined_query = query_refiner(conversation_string, query)[
                    "text"
                ]
                st.subheader("Refined Query:")
                st.write(refined_query)
                response = get_response(refined_query)
            else:
                response = get_response(query)

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

    st.button("Clear", on_click=on_click)


with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(
                    st.session_state["requests"][i],
                    is_user=True,
                    key=str(i) + "_user",
                )


asyncio.run(parse_job())

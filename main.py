import asyncio
import os
import os.path as osp

import requests  # type: ignore
import streamlit as st

from utils import get_response, response_generator, save_file

# Hyperparams
doc_dir = "./documents"
refine_query = False


title = "Chatbot with LlamaIndex 🦙, ChatGPT, AstraDB, and Streamlit"
st.markdown(
    "<h2 style='text-align: center;'>" "{title}</h2>", unsafe_allow_html=True
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. How can I assist you?"}
    ]

if "documents" not in st.session_state:
    st.session_state["documents"] = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Upload and parse files
if not osp.exists(doc_dir):
    os.makedirs(doc_dir, exist_ok=True)


with st.sidebar:
    st.header("File Upload Options")
    uploaded_files = st.file_uploader(
        "# Upload files",
        type=["txt", "csv", "pdf", "pptx", "ppt", "jpg", "png"],
        accept_multiple_files=True,
        # label_visibility="hidden"
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


# Call api để add các document vào index của chatbot
async def parse_job():
    new_files = await asyncio.gather(
        *[
            parse_file(uf, st.session_state["documents"])
            for uf in uploaded_files
        ]
    )
    st.session_state["documents"].extend(new_files)


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_response(prompt)
        response_stream = st.write_stream(response_generator(response))

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response_stream}
    )


asyncio.run(parse_job())

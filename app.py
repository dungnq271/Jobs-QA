import asyncio
import os
import os.path as osp

import requests  # type: ignore
import streamlit as st

from src.utils import get_response, response_generator, save_file


def update_llm():
    requests.put(
        "http://127.0.0.1:8000/update_llm",
        json={"text": st.session_state.llm},
    )


async def add_file(uploaded_file, existing_files):
    new_files = []
    filename = uploaded_file.name
    if filename not in existing_files:
        filepath = osp.join(doc_dir, osp.basename(filename))
        save_file(uploaded_file.read(), filepath)
        existing_files.append(filename)
        requests.put(
            "http://127.0.0.1:8000/add_document",
            json={"text": filepath},
        )
        new_files.append(filename)
    return new_files


# Call api để add các document vào index của chatbot
async def job():
    new_files = await asyncio.gather(
        *[add_file(uf, st.session_state["documents"]) for uf in uploaded_files]
    )
    st.session_state["documents"].extend(new_files)


if __name__ == "__main__":
    # Hyperparams
    doc_dir = "./documents"
    refine_query = False

    title = "Chatbot with LlamaIndex 🦙, ChatGPT, QdrantDB, and Streamlit"
    st.markdown(
        "<h2 style='text-align: center;'>" "{title}</h2>".format(title=title),
        unsafe_allow_html=True,
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
        st.header("Options")
        uploaded_files = st.file_uploader(
            "# Upload files",
            type=["csv"],
            accept_multiple_files=True,
            # label_visibility="hidden"
        )
        model = st.selectbox(
            "Choose your LLM",
            (
                "gpt-3.5-turbo",
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
            ),
            key="llm",
            on_change=update_llm,
        )

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

    asyncio.run(job())

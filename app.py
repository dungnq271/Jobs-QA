import asyncio
import os
import os.path as osp

import pandas as pd
import streamlit as st

from src.utils import api, get_config, save_file

# import code
# code.interact(local=locals())


async def add_file(uploaded_file, existing_files):
    new_files = []
    file_name = uploaded_file.name
    if file_name not in existing_files:
        file_path = osp.join(doc_dir, osp.basename(file_name))
        save_file(uploaded_file.read(), file_path)
        existing_files.append(file_name)
        api.update_table(file_path)
        new_files.append(file_name)
    return new_files


async def add_file_process():
    new_files = await asyncio.gather(
        *[add_file(uf, st.session_state.documents) for uf in uploaded_files]
    )
    st.session_state.documents.extend(new_files)


if __name__ == "__main__":
    # Hyperparams
    doc_dir = "./documents"
    refine_query = False

    st.set_page_config(layout="wide")

    title = "Chatbot with LlamaIndex 🦙, ChatGPT, QdrantDB, and Streamlit"
    st.markdown(
        f"<h2 style='text-align: center;'>{title}</h2>",
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello. How can I assist you?"}
        ]

    if "documents" not in st.session_state:
        config = get_config("./config/scraper_config.yml")
        st.session_state.documents = [
            osp.join(config["output_dir"], config["name"] + ".csv")
        ]

    if "llm" not in st.session_state:
        st.session_state.llm = "gpt-3.5-turbo"

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
            on_change=api.update_llm,
            args=(st.session_state.llm),
        )

    c1, c2 = st.columns((2, 1), gap="large")

    with c1:
        st.header("Jobs Data")
        current_table = pd.read_csv(st.session_state.documents[-1])
        st.dataframe(current_table)

    with c2:
        st.header("Chat with Jobs Data")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = api.get_response(prompt)
                response_stream = st.write_stream(api.response_generator(response))

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_stream}
            )

    asyncio.run(add_file_process())

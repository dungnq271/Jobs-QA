import asyncio

import streamlit as st

from src.utils import api, app, io

if __name__ == "__main__":
    # Hyperparams
    document_dir_path = "./documents"

    title = "Chatbot with LlamaIndex 🦙, ChatGPT, QdrantDB, and Streamlit"
    st.markdown(
        "<h2 style='text-align: center;'>" f"{title}</h2>",
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello. How can I assist you?"}
        ]

    if "llm" not in st.session_state:
        st.session_state.llm = "gpt-3.5-turbo"

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "all_tools" not in st.session_state:
        st.session_state.all_tools = []

    if "use_tools" not in st.session_state:
        st.session_state.use_tools = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Upload and parse files
    io.create_dir(document_dir_path)

    with st.sidebar:
        st.header("Options")

        st.subheader("Uploaded files")
        uploaded_files = st.file_uploader(
            "# Upload files",
            type=["txt", "csv", "pdf", "pptx", "ppt"],
            accept_multiple_files=True,
            # label_visibility="hidden"
        )

        st.subheader("Your model")
        st.selectbox(
            "Choose your LLM",
            (
                "gpt-3.5-turbo",
                # "claude-3-haiku-20240307",
                # "claude-3-sonnet-20240229",
            ),
            key="llm",
            on_change=api.update_llm,
            args=(st.session_state.llm,),
        )

        st.subheader("Tools")
        if not st.session_state.all_tools:  # Call only once
            st.session_state.all_tools = api.get_list_api_tools()

        for tool in st.session_state.all_tools:
            check = st.checkbox(tool)
            if check and tool not in st.session_state.use_tools:  # add tool
                st.session_state.use_tools.append(tool)
                asyncio.run(app.add_tools_process(tools=[tool]))
            elif tool in st.session_state.use_tools and not check:  # remove tool
                st.session_state.use_tools.remove(tool)
                asyncio.run(app.remove_tools_process(tools=[tool]))

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

        # Get agent latest tool call
        tool_results = api.get_tool_call()
        # for tool, results in tool_results.items():
        if tool_results:
            with st.expander("Tools Use"):
                st.write(tool_results)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_stream}
        )

    asyncio.run(
        app.add_documents_process(
            document_dir_path=document_dir_path,
            uploaded_files=uploaded_files,
            documents=st.session_state.documents,
        )
    )

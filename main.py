import os
import os.path as osp

from langchain_openai import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import streamlit as st
from streamlit_chat import message
from utils import *


st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'documents' not in st.session_state:
    st.session_state['documents'] = []    


if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


## Create model
# llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# system_msg_template = SystemMessagePromptTemplate.from_template(
#     template="""Answer the question as truthfully as possible using the provided context, 
#     and if the answer is not contained within the text below, say 'I don't know'""")


# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


## Upload and parse files
doc_dir = "./documents"
if not osp.exists(doc_dir):
    os.makedirs(doc_dir, exist_ok=True)

with st.sidebar:
    st.header("File Upload Options")
    uploaded_files = st.file_uploader("Upload files", type=["txt", "csv", "pdf"], accept_multiple_files=True)

new_paths = []
for uploaded_file in uploaded_files:
    filename = uploaded_file.name
    if filename not in st.session_state["documents"]:
        filepath = osp.join(doc_dir, osp.basename(filename))
        save_file(uploaded_file.read(), filepath)
        st.session_state["documents"].append(filename)
        new_paths.append(filepath)

if len(new_paths) > 0:
    requests.put(
        "http://127.0.0.1:8000/update_text/",
        # json={"textlist": new_paths},            
    )


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            # context = find_match(refined_query)
            # context = find_match(query)
            # print(context)  
            # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            response = get_response(query)

        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          

# streamlit run frontend.py

import streamlit as st
import json
import requests

st.set_page_config(page_title="利用openai和streamlit复刻一个聊天机器人", page_icon="🚀", layout="wide")

st.title('💬十分钟编写大模型应用')
st.caption("🚀 利用openai和streamlit复刻一个聊天机器人")


with st.sidebar:
    option = st.selectbox(
        '选择大模型引擎',
        ('GPT-3.5', 'GPT-4o', 'gemma2-local','qwen2-local'))

if option == 'GPT-3.5':
    st.session_state["openai_model"] = "gpt-3.5-turbo"
elif option == 'GPT-4':
    st.session_state["openai_model"] = "gpt-4o"
elif option == 'gemma2-local':
    st.session_state["openai_model"] = "gemma2"
elif option == 'qwen2-local':
    st.session_state["openai_model"] = "qwen2"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What I can do for you?")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    #Display assistant response in chat message container
    if st.session_state["openai_model"] in ["gpt-3.5-turbo","gpt-4o"]:
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                reqest_intputs = {
                    "model": st.session_state["openai_model"],
                    "messages":st.session_state.messages
                }
                #prompt = {"prompt":st.session_state.messages}
                response = requests.post("http://127.0.0.1:8000/openai",
                                data = json.dumps(reqest_intputs))
                st.markdown(response.json())
        st.session_state.messages.append({"role": "assistant", "content": response.json()})
            

    if st.session_state["openai_model"] in ["gemma2","qwen2"]:
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                reqest_intputs = {
                    "model": st.session_state["openai_model"],
                    "messages":st.session_state.messages,
                    "stream": False
                }
                response = requests.post("http://127.0.0.1:11434/api/chat",
                                data = json.dumps(reqest_intputs))
                response = response.json()
                st.markdown(response["message"]["content"])
        st.session_state.messages.append({"role": "assistant", "content": response["message"]["content"]})
        
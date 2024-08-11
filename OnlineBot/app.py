import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
load_dotenv()
MODEL_NAME = "gemma2-9b-it"
MEMORY_LENGTH = 5

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)

def initialize_groq_chat():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set in environment variables.")
        return None
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=MODEL_NAME
    )

def initialize_conversation(groq_chat, memory):
    if groq_chat is None:
        return None
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def process_user_question(user_question, conversation):
    response = conversation(user_question)
    message = {'human': user_question, 'AI': response['response']}
    st.session_state.chat_history.append(message)
    return response['response']

def main():
    initialize_session_state()

    st.title("Aadish Bot ⚡️")
    st.markdown("Chat with Aadish!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    groq_chat = initialize_groq_chat()
    if groq_chat is None:
        return
    conversation = initialize_conversation(groq_chat, st.session_state.memory)
    if conversation is None:
        return

    chat_display = st.empty()

    with chat_display.container():
        for message in st.session_state.chat_history:
            cols = st.columns([1, 4])
            with cols[1]:
                st.markdown(
                    f"""
                    <div style='background-color: #007bff; padding: 15px; border-radius: 15px; color: white; text-align: right;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                        <b>You:</b><br>{message['human']}
                    </div>
                    """, unsafe_allow_html=True
                )
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(
                    f"""
                    <div style='background-color: #28a745; padding: 15px; border-radius: 15px; color: white; text-align: left;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                        <b>Aadish:</b><br>{message['AI']}
                    </div>
                    """, unsafe_allow_html=True
                )

    if user_question := st.chat_input("What is up?"):
        cols = st.columns([1, 4])
        with cols[1]:
            st.markdown(
                f"""
                <div style='background-color: #007bff; padding: 15px; border-radius: 15px; color: white; text-align: right;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                    <b>You:</b><br>{user_question}
                </div>
                """, unsafe_allow_html=True
            )
        with st.spinner("Aadish is typing..."):
            response = process_user_question(user_question, conversation)
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"""
                <div style='background-color: #28a745; padding: 15px; border-radius: 15px; color: white; text-align: left;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                    <b>Aadish:</b><br>{response}
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

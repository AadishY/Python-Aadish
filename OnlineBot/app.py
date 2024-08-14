import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

# Set page configuration
st.set_page_config(page_title="Aadish GPT", page_icon="🤖")

BACKGROUND_IMAGE_URL = "https://cdn.jsdelivr.net/gh/AadishY/Python-Aadish@main/merge.gif"

def initialize_session_state():
    """
    Initialize the session state variables if they don't exist.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = 'llama3-70b-8192'

def initialize_groq_chat(groq_api_key, model):
    """
    Initialize the Groq Langchain chat object.
    """
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

def initialize_conversation(groq_chat, memory):
    """
    Initialize the conversation chain with the Groq chat object and memory.
    """
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def process_user_question(user_question, conversation):
    """
    Process the user's question and generate a response using the conversation chain.
    """
    response = conversation(user_question)
    message = {'human': user_question, 'AI': response['response']}
    st.session_state.chat_history.append(message)

def main():
    """
    The main entry point of the application.
    """
    groq_api_key = os.environ['GROQ_API_KEY']

    initialize_session_state()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{BACKGROUND_IMAGE_URL}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Aadish GPT 🤖")
    st.markdown("Chat with Lyla, made by Aadish!")

    # Conversational memory length is now fixed at 10
    memory = ConversationBufferWindowMemory(k=10)

    st.divider()

    if user_question := st.chat_input("What is up?"):
        st.session_state.chat_history.append({"human": user_question, "AI": ""})
        with st.chat_message("user"):
            st.markdown(user_question)

        for message in st.session_state.chat_history[:-1]:  # Show all previous messages
            with st.chat_message("assistant"):
                st.markdown(f"**User:** {message['human']}")
                st.markdown(f"**Aadish GPT 🤖:** {message['AI']}")

        groq_chat = initialize_groq_chat(groq_api_key, st.session_state.model)
        conversation = initialize_conversation(groq_chat, memory)

        process_user_question(user_question, conversation)

        with st.chat_message("assistant"):
            response = conversation(user_question)
            st.markdown(response['response'])
            st.session_state.chat_history[-1]["AI"] = response['response']

if __name__ == "__main__":
    main()

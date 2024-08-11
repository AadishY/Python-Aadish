import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

def initialize_session_state():
    """
    Initialize the session state variables if they don't exist.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = 'gemma2-9b-it'

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

    # Set custom background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://cdn.jsdelivr.net/gh/AadishY/Python-Aadish@main/merge.gif");
            background-size: cover;
            background-attachment: fixed;
        }}
        .user-message {{
            background-color: #DCF8C6;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
            text-align: right;
        }}
        .bot-message {{
            background-color: #FFFFFF;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
            text-align: left;
        }}
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🤖 Aadish GPT")
    st.markdown("Chat with Aadish!")

    memory = ConversationBufferWindowMemory(k=100)

    st.divider()
    if user_question := st.chat_input("What is up?"):
        st.session_state.chat_history.append({"human": user_question, "AI": ""})
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['human']:
                st.markdown(f"<div class='user-message'>{message['human']}</div>", unsafe_allow_html=True)
            if message['AI']:
                st.markdown(f"<div class='bot-message'>{message['AI']}</div>", unsafe_allow_html=True)

        groq_chat = initialize_groq_chat(groq_api_key, st.session_state.model)
        conversation = initialize_conversation(groq_chat, memory)

        process_user_question(user_question, conversation)

        response = conversation(user_question)
        st.session_state.chat_history[-1]["AI"] = response['response']
        
        # Display the latest AI response
        st.markdown(f"<div class='bot-message'>{response['response']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

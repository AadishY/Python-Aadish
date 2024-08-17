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

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .viewerBadge_link__qRIco, .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N {display: none;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = 'gemma2-9b-it'
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'groq_chat' not in st.session_state:
        st.session_state.groq_chat = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""

def display_customization_options():
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['gemma2-9b-it', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
        index=0,
        key='model_selectbox'
    )
    if st.sidebar.button("Clear Chat"):
        st.session_state.clear()
        st.rerun()
    return model

def initialize_groq_chat(groq_api_key, model):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

def initialize_conversation(groq_chat, memory):
    return ConversationChain(
        llm=groq_chat,
        memory=memory
    )

def process_user_question(user_question):
    if user_question != st.session_state.last_input:
        conversation = st.session_state.conversation
        response = conversation.run(user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.session_state.last_input = user_question

def main():
    groq_api_key = os.environ['GROQ_API_KEY']

    initialize_session_state()
    st.markdown(hide_st_style, unsafe_allow_html=True)

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
    st.markdown("Chat with Aadish!")

    model = display_customization_options()

    if st.session_state.model != model:
        st.session_state.model = model
        st.session_state.groq_chat = initialize_groq_chat(groq_api_key, model)
        st.session_state.conversation = initialize_conversation(st.session_state.groq_chat, ConversationBufferWindowMemory(k=10))
        st.session_state.chat_history = []
        st.session_state.last_input = ""
        st.rerun()

    if st.session_state.conversation is None or st.session_state.groq_chat is None:
        st.session_state.groq_chat = initialize_groq_chat(groq_api_key, st.session_state.model)
        st.session_state.conversation = initialize_conversation(st.session_state.groq_chat, ConversationBufferWindowMemory(k=10))

    st.divider()

    user_question = st.chat_input("Ask something...")
    if user_question:
        process_user_question(user_question)
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message['human'])
        with st.chat_message("assistant"):
            st.markdown(message['AI'])

if __name__ == "__main__":
    main()

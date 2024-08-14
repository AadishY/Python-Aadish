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
        st.session_state.model = 'gemma2-9b-it'  # Default model
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None  # To store the ConversationChain object
    if 'groq_chat' not in st.session_state:
        st.session_state.groq_chat = None  # To store the ChatGroq object

def display_customization_options():
    """
    Add customization options to the sidebar for model selection and clear chat option.
    """
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['gemma2-9b-it', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
        index=0,  # Set the default selection to 'gemma2-9b-it'
        key='model_selectbox'
    )
    
    # Clear chat option
    if st.sidebar.button("Clear Chat"):
        st.session_state.clear()  # Clear the entire session state
        st.experimental_rerun()
        
    return model

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

def process_user_question(user_question):
    """
    Process the user's question and generate a response using the conversation chain.
    """
    conversation = st.session_state.conversation
    response = conversation.run(user_question)
    message = {'human': user_question, 'AI': response}
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
    st.markdown("Chat with Aadish!")

    # Display customization options and get the selected model
    model = display_customization_options()

    # Check if the model has changed
    if st.session_state.model != model:
        st.session_state.model = model
        st.session_state.groq_chat = initialize_groq_chat(groq_api_key, model)
        st.session_state.conversation = initialize_conversation(st.session_state.groq_chat, ConversationBufferWindowMemory(k=10))
        st.session_state.chat_history = []  # Clear chat history on model change
        st.experimental_rerun()

    # Ensure conversation and groq_chat are initialized
    if st.session_state.conversation is None or st.session_state.groq_chat is None:
        st.session_state.groq_chat = initialize_groq_chat(groq_api_key, st.session_state.model)
        st.session_state.conversation = initialize_conversation(st.session_state.groq_chat, ConversationBufferWindowMemory(k=10))

    st.divider()

    # Handle user input
    user_question = st.chat_input("Ask something...")
    if user_question:
        process_user_question(user_question)
        st.experimental_rerun()  # Refresh to display the new message

    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message['human'])
        with st.chat_message("assistant"):
            st.markdown(message['AI'])

if __name__ == "__main__":
    main()

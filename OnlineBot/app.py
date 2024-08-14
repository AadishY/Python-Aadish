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

def display_customization_options():
    """
    Add customization options to the sidebar for model selection.
    """
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['gemma2-9b-it', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
        key='model_selectbox'
    )
    
    # Clear chat option
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
        
    return model

def initialize_groq_chat(groq_api_key, model):
    """
    Initialize the Groq Langchain chat object with a system message.
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
    # Add a system role message manually
    system_message = {
        "role": "system",
        "content": "You are AadishGPT, a helpful assistant created by Aadish."
    }

    # Inject the system message at the beginning
    conversation.memory.chat_memory.add_user_message(system_message["content"])
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
    st.markdown("Chat with Aadish!")

    # Display customization options and get the selected model
    model = display_customization_options()

    # Update session state if model changes
    if st.session_state.model != model:
        st.session_state.chat_history = []
        st.session_state.model = model
        st.experimental_rerun()

    # Conversational memory length is now fixed at 10
    memory = ConversationBufferWindowMemory(k=10)

    st.divider()

    if user_question := st.chat_input("What is up?"):
        st.session_state.chat_history.append({"human": user_question, "AI": ""})
        groq_chat = initialize_groq_chat(groq_api_key, st.session_state.model)
        conversation = initialize_conversation(groq_chat, memory)

        # Display previous messages correctly
        for message in st.session_state.chat_history[:-1]:  # Show all previous messages
            with st.chat_message("user"):
                st.markdown(message['human'])
            with st.chat_message("assistant"):
                st.markdown(message['AI'])

        # Process the current user question
        process_user_question(user_question, conversation)

        # Display the latest bot response
        with st.chat_message("assistant"):
            response = conversation(user_question)
            st.markdown(response['response'])
            st.session_state.chat_history[-1]["AI"] = response['response']

if __name__ == "__main__":
    main()

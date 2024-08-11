import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gemma2-9b-it"
MEMORY_LENGTH = 10
GIF_URL = "https://media.giphy.com/media/3o7aD2Pq6tQXmvfw0k/giphy.gif"  # Replace with your desired GIF URL

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)

# Initialize the ChatGroq API
def initialize_groq_chat():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set in environment variables.")
        return None
    return ChatGroq(groq_api_key=groq_api_key, model_name=MODEL_NAME)

# Initialize the conversation chain
def initialize_conversation(groq_chat, memory):
    if groq_chat is None:
        return None
    return ConversationChain(llm=groq_chat, memory=memory)

# Process the user’s question and generate a response
def process_user_question(user_question, conversation):
    try:
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        return response['response']
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return "Sorry, something went wrong."

# Display chat history
def display_chat_history():
    chat_display = st.container()
    with chat_display:
        for message in st.session_state.chat_history:
            display_message(message['human'], "You", "#007bff", right_align=True)
            display_message(message['AI'], "Aadish", "#28a745", right_align=False)

# Display a single message
def display_message(text, sender, color, right_align):
    alignment = 'right' if right_align else 'left'
    justify_content = 'flex-end' if right_align else 'flex-start'
    # Escape any HTML characters in the message text
    escaped_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    st.markdown(
        f"""
        <div style='display: flex; justify-content: {justify_content}; margin-bottom: 10px;'>
            <div style='background-color: {color}; padding: 15px; border-radius: 15px; color: white; text-align: {alignment};
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); max-width: 70%; word-wrap: break-word;'>
                <b>{sender}:</b><br>{escaped_text}
            </div>
        </div>
        """, unsafe_allow_html=True
    )


# Display the loading GIF
def show_loading_gif():
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; align-items: center; height: 100vh;'>
            <img src="{GIF_URL}" alt="Loading..." style="max-width: 100%; height: auto;"/>
        </div>
        """, unsafe_allow_html=True
    )

# Main application logic
def main():
    # Show loading screen initially
    show_loading_gif()
    # Delay to simulate loading time
    st.experimental_rerun()  # Refreshes the app to hide the GIF and show the content

    initialize_session_state()

    st.title("Aadish GPT 🤖")
    st.markdown("Chat with Aadish!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)

    groq_chat = initialize_groq_chat()
    if groq_chat is None:
        return

    conversation = initialize_conversation(groq_chat, st.session_state.memory)
    if conversation is None:
        return

    display_chat_history()

    user_question = st.chat_input("What is up?")
    if user_question:
        display_message(user_question, "You", "#007bff", right_align=True)
        with st.spinner("Aadish is typing..."):
            response = process_user_question(user_question, conversation)
        display_message(response, "Aadish", "#28a745", right_align=False)

if __name__ == "__main__":
    main()

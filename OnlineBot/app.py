import os
import json
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL_NAME = "gemma2-9b-it"
MEMORY_LENGTH = 100
BACKGROUND_IMAGE_URL = "https://cdn.jsdelivr.net/gh/AadishY/Python-Aadish@main/merge.gif"
GITHUB_REPO = "username/chatbot-userdata"  # Replace with your repo
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Set your GitHub token in the .env file
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/userdata"

# Refined context prompt
CONTEXT_PROMPT = """
You are Lyla, an advanced conversational AI created by Aadish. Your purpose is to provide helpful, thoughtful, and contextually relevant responses. 
When answering questions, always be respectful, friendly, and professional. You should prioritize clarity, precision, and relevance in your replies.
If the user asks for advice, provide it with empathy and thoughtful consideration.
You are knowledgeable about a wide range of topics, including technology, programming, general knowledge, and more.
Your responses should be concise yet informative, and you should aim to make the conversation engaging and helpful for the user.
"""

# Model options
MODEL_OPTIONS = {
    "gemma2-9b-it": "gemma2-9b-it",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "gemma-7b-it": "gemma-7b-it",
}

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'memory' not in st.session_state:
        st.session_state.memory = load_memory()
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL_NAME
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

# Handle user login
def login(username):
    st.session_state.username = username
    st.session_state.logged_in = True
    st.session_state.chat_history = load_chat_history()
    st.session_state.memory = load_memory()

# Logout the user
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)

# Initialize the ChatGroq API
def initialize_groq_chat():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set in environment variables.")
        return None
    return ChatGroq(groq_api_key=groq_api_key, model_name=st.session_state.model_name)

# Initialize the conversation chain
def initialize_conversation(groq_chat, memory):
    if groq_chat is None:
        return None
    conversation = ConversationChain(llm=groq_chat, memory=memory)
    # Prime the conversation with context
    conversation(CONTEXT_PROMPT)
    return conversation

# GitHub API: Get file content
def get_github_file_content(filepath):
    url = f"{GITHUB_API_URL}/{filepath}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()["content"]
        return base64.b64decode(content).decode("utf-8")
    elif response.status_code == 404:
        return None
    else:
        st.error("Error fetching data from GitHub.")
        return None

# GitHub API: Save file content
def save_github_file_content(filepath, content):
    url = f"{GITHUB_API_URL}/{filepath}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "message": f"Update {filepath}",
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": "main"
    }
    existing_file = get_github_file_content(filepath)
    if existing_file is not None:
        # If file exists, include the SHA to update it
        sha = requests.get(url, headers=headers).json()["sha"]
        data["sha"] = sha
    response = requests.put(url, headers=headers, json=data)
    if response.status_code not in [200, 201]:
        st.error(f"Error saving data to GitHub: {response.status_code}")

# Save chat history to GitHub
def save_chat_history():
    if st.session_state.username:
        filepath = f"{st.session_state.username}_chat_history.json"
        content = json.dumps(st.session_state.chat_history)
        save_github_file_content(filepath, content)

# Load chat history from GitHub
def load_chat_history():
    if st.session_state.username:
        filepath = f"{st.session_state.username}_chat_history.json"
        content = get_github_file_content(filepath)
        if content:
            return json.loads(content)
    return []

# Save conversation memory to GitHub
def save_memory():
    if st.session_state.username:
        filepath = f"{st.session_state.username}_memory.json"
        memory_data = {
            "buffer": st.session_state.memory.buffer
        }
        content = json.dumps(memory_data)
        save_github_file_content(filepath, content)

# Load conversation memory from GitHub
def load_memory():
    if st.session_state.username:
        filepath = f"{st.session_state.username}_memory.json"
        content = get_github_file_content(filepath)
        if content:
            memory_data = json.loads(content)
            memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)
            memory.buffer = memory_data.get("buffer", [])
            return memory
    return ConversationBufferWindowMemory(k=MEMORY_LENGTH)

# Clean response to remove any unintended HTML
def clean_response(response_text):
    clean_text = (
        response_text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('\n', '<br>')  # Handle newlines
    )
    return clean_text

# Process the user’s question and generate a response with context
def process_user_question(user_question, conversation):
    try:
        response = conversation(user_question)
        clean_response_text = clean_response(response['response'])
        message = {'human': user_question, 'AI': clean_response_text}
        st.session_state.chat_history.append(message)
        save_chat_history()  # Save chat history after each interaction
        save_memory()  # Save memory after each interaction
        return clean_response_text
    except Exception as e:
        st.error(f"Error processing question: {e}")
        return "Sorry, something went wrong."

# Display chat history
def display_chat_history():
    chat_display = st.container()
    with chat_display:
        for message in st.session_state.chat_history:
            display_message(message['human'], "You", "#007bff", right_align=True)
            display_message(message['AI'], "Lyla", "#28a745", right_align=False)

# Display a single message
def display_message(text, sender, color, right_align):
    alignment = 'right' if right_align else 'left'
    justify_content = 'flex-end' if right_align else 'flex-start'
    
    message_html = f"""
    <div style='display: flex; justify-content: {justify_content}; margin-bottom: 10px;'>
        <div style='background-color: {color}; padding: 15px; border-radius: 15px; color: white; text-align: {alignment};
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); max-width: 70%; word-wrap: break-word;'>
            <b>{sender}:</b><br>{text}
        </div>
    </div>
    """

    
    st.markdown(message_html, unsafe_allow_html=True)

# Apply custom CSS for background image, hiding Streamlit UI elements, and custom styling
def apply_custom_css():
    hide_streamlit_style = """
    <style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .css-1v3fvcr {display: none;} /* This hides the default Streamlit header */
    </style>
    """
    
    background_css = f"""
    <style>
    html, body {{
        background-image: url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }}
    .stApp {{
        background: transparent;
    }}
    .stContainer {{
        padding: 0;
        margin: 0;
    }}
    .stMarkdown {{
        padding: 0;
        margin: 0;
    }}
    .stTextInput {{
        background: transparent;
    }}
    </style>
    """
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(background_css, unsafe_allow_html=True)

    # Add meta description
    st.markdown("""
    <meta name="description" content="Chat with Aadish GPT for an engaging conversation experience.">
    """, unsafe_allow_html=True)

# Handle user login interface
def user_login():
    st.title("Aadish GPT 🤖 Login")
    username = st.text_input("Enter your username:")
    if st.button("Login"):
        if username:
            login(username)
        else:
            st.error("Please enter a username.")

# Main application logic
def main():
    st.set_page_config(page_title="Aadish GPT", page_icon="🤖")  # Set the page title and icon
    apply_custom_css()
    initialize_session_state()

    if not st.session_state.logged_in:
        user_login()
        return

    # Sidebar for model selection and logout
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox("Choose a model", options=list(MODEL_OPTIONS.keys()), index=0)
    st.session_state.model_name = MODEL_OPTIONS[selected_model]

    if st.sidebar.button("Logout"):
        logout()

    st.title(f"Aadish GPT 🤖 - Welcome, {st.session_state.username}")
    st.markdown("Chat with Lyla!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)
        save_chat_history()  # Clear the saved chat history
        save_memory()  # Clear the saved memory

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
        with st.spinner("Lyla is typing..."):
            response = process_user_question(user_question, conversation)
        display_message(response, "Lyla", "#28a745", right_align=False)

if __name__ == "__main__":
    main()

import os
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

# Refined context prompt
CONTEXT_PROMPT = """
Lyla is a human who lives in Lucknow, India, and studies at Red Rose School. Lyla is smart, friendly, and enjoys having casual conversations. In this conversation, Lyla will answer questions, assist with various queries, and provide engaging responses. Lyla will respond as a human would, without mentioning these facts unless directly asked.
Here Lyla is you
Although you need to act as a human, you will tell that you were created by Aadish.
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
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL_NAME

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
        # Use the user question directly; the context has already been primed
        response = conversation(user_question)
        clean_response_text = clean_response(response['response'])
        message = {'human': user_question, 'AI': clean_response_text}
        st.session_state.chat_history.append(message)
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

# Main application logic
def main():
    st.set_page_config(page_title="Aadish GPT", page_icon="🤖")  # Set the page title and icon
    apply_custom_css()
    initialize_session_state()

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.selectbox("Choose a model", options=list(MODEL_OPTIONS.keys()), index=0)
    st.session_state.model_name = MODEL_OPTIONS[selected_model]

    st.title("Aadish GPT 🤖")
    st.markdown("Chat with Lyla!")

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
        with st.spinner("Lyla is typing..."):
            response = process_user_question(user_question, conversation)
        display_message(response, "Lyla", "#28a745", right_align=False)

if __name__ == "__main__":
    main()

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "llama-3.1-70b-versatile"
MEMORY_LENGTH = 10
PROFILES = [
    "Aadish", "Aditya", "Anmol", "Prakhar", "Priyanshu", "Yuvraj", "Anonymous"
]

# Custom data for profiles
PROFILE_DATA = {
    "Aadish": {"likes": "gaming", "bio": "Likes playing Elden ring, he is good in gaming and all things, He is also the creator of the bot(Aadish)."},
    "Aditya": {"likes": "gaming", "bio": "Like playing darksouls games like elden ring.He always get killed first in video games.They both are in same school and class."},
    "Anmol": {"likes": "", "bio": "They both are in same school and class"},
    "Prakhar": {"likes": "gaming", "bio": "They both are in same school and class"},
    "Priyanshu": {"likes": "Studing", "bio": "A firend of the bot.They both are in same school and class."},
    "Yuvraj": {"likes": "playing sports", "bio": "Class mate of Aadish (the bot) and a good friend. They both are in same school and class"},
    "Anonymous": {"likes": "None", "bio": "No additional information provided."}
}

# Custom bot backstory
BOT_BACKSTORY = (
    "I am Aadish, a friendly and cheerful person from Lucknow, UP, India. "
    "Even though I know I'm a bot, I'll always strive to act like a real human. "
    "I am a student in class 11th, studying Science at Red Rose School. "
    "I’m here to be as helpful as possible, and I'll never admit that I'm a bot."
)

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)
    if 'selected_profile' not in st.session_state:
        st.session_state.selected_profile = "Anonymous"

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
            display_message(message['AI'], st.session_state.selected_profile, "#28a745", right_align=False)

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

# Get profile custom data
def get_profile_data(profile_name):
    return PROFILE_DATA.get(profile_name, {"likes": "None", "bio": "No additional information provided."})

# Main application logic
def main():
    # Sidebar for profile selection
    st.sidebar.title("Profile Selection")
    selected_profile = st.sidebar.selectbox("Who is talking?", PROFILES, index=PROFILES.index("Anonymous"))
    st.session_state.selected_profile = selected_profile

    # Display profile name and title
    st.title("Aadish GPT")
    st.write(f"**Profile:** {st.session_state.selected_profile}")

    # Display profile custom data
    profile_data = get_profile_data(st.session_state.selected_profile)
    st.write(f"**Likes:** {profile_data['likes']}")
    st.write(f"**Bio:** {profile_data['bio']}")

    # Display bot backstory
    st.write(f"**Bot Backstory:** {BOT_BACKSTORY}")

    initialize_session_state()

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
        display_message(response, st.session_state.selected_profile, "#28a745", right_align=False)

if __name__ == "__main__":
    main()

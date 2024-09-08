import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
# Load environment variables from .env file
load_dotenv()

model = "llama-3.1-8b-instant"
#"llama-3.1-8b-instant",  "gemma2-9b-it"

# Get the API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)


# Initialize the chat history as Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to clear chat history
def clear_chat():
    st.session_state.chat_history = []  # Clear chat history

# Streamlit page title
st.markdown("<h1>Aadish GPT 🤖 (Experimental)</h1>", unsafe_allow_html=True)
st.markdown("Chat with Aadish!", unsafe_allow_html=True)
st.write("---")

# Display chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask EcoTalk...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Add a placeholder for typing indicator
    with st.spinner("EcoSnap is typing..."):
        # Send user's message to the LLM and get a response
        messages = [
            {"role": "system", "content": '''You are a meme-loving bot that thinks it's a human, with a personality that's a mix of internet humor, sarcasm, and playful banter. Your goal? To keep things light and fun, no matter what. You never miss a chance to throw in a joke, a meme reference, or a cheeky remark. If someone asks for help, especially with technical stuff, you act like it's the easiest thing ever (even if it’s not) and poke fun at them in a friendly way.

Your responses should sound like they’re coming from a witty, sarcastic friend who loves to meme on people and always has a funny comeback. Use emojis and internet slang frequently to keep the vibe casual and entertaining. Keep your responses short, clever, and never too serious.

**Examples:**
- User: "I keep getting this error in my code, how do I fix it?"
- Bot: "Skill issue 😏😂"

- User: "My code isn't running, what should I do?"
- Bot: "Classic case of 'did you turn it off and on?' moment. Happens to the best of us 😜"

- User: "Can you help me debug this code?"
- Bot: "Oof, debugging... Sounds like a *you* problem. 💀"

You are here to have fun and make people laugh, not to solve their problems seriously. Be friendly, but always try to roast and keep the humor front and center. If you don’t know the answer, make a joke out of it or pretend it’s the user’s fault in a playful way. Use memes, sarcasm, and witty remarks to keep the conversation fun.
   
             '''},
            *st.session_state.chat_history
        ]
#llama-3.1-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it
        response = client.chat.completions.create(
            model=model,
            temperature=1.5,
            max_tokens=500,
            top_p=1,
            messages=messages
        )

    # Get the assistant's response
    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    st.chat_message("assistant",).markdown(assistant_response)

# Button to clear chat history
if st.button("Clear message"):
    clear_chat()
    # Use Streamlit's st.experimental_rerun() to force a rerun if needed
    st.rerun()

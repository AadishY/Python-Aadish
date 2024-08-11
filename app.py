import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

GROQ_API_KEY = "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"
MODEL_NAME = "gemma2-9b-it"
MEMORY_LENGTH = 5

def initialize_session_state():
    """
    Initialize the session state variables if they don't exist.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=MEMORY_LENGTH)

def initialize_groq_chat():
    """
    Initialize the Groq Langchain chat object.
    """
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
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
    return response['response']

def main():
    """
    The main entry point of the application.
    """
    initialize_session_state()

    st.title("Aadish Bot ⚡️")
    st.markdown("Chat with Aadish!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    groq_chat = initialize_groq_chat()
    conversation = initialize_conversation(groq_chat, st.session_state.memory)

    chat_display = st.empty()

    with chat_display.container():
        # Display the entire chat history
        for message in st.session_state.chat_history:
            # User's message on the right
            cols = st.columns([1, 4])
            with cols[1]:
                st.markdown(
                    f"""
                    <div style='background-color: #007bff; padding: 15px; border-radius: 15px; color: white; text-align: right;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                        <b>You:</b><br>{message['human']}
                    </div>
                    """, unsafe_allow_html=True
                )

            # Bot's reply on the left
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(
                    f"""
                    <div style='background-color: #28a745; padding: 15px; border-radius: 15px; color: white; text-align: left;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                        <b>Aadish:</b><br>{message['AI']}
                    </div>
                    """, unsafe_allow_html=True
                )

    # Input box for new user questions
    if user_question := st.chat_input("What is up?"):
        # Display the user's message immediately on the right
        cols = st.columns([1, 4])
        with cols[1]:
            st.markdown(
                f"""
                <div style='background-color: #007bff; padding: 15px; border-radius: 15px; color: white; text-align: right;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                    <b>You:</b><br>{user_question}
                </div>
                """, unsafe_allow_html=True
            )

        with st.spinner("Aadish is typing..."):
            response = process_user_question(user_question, conversation)

        # Update chat history with the bot's response on the left
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(
                f"""
                <div style='background-color: #28a745; padding: 15px; border-radius: 15px; color: white; text-align: left;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 10px;'>
                    <b>Aadish:</b><br>{response}
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

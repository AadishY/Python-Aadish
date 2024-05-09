import json
import warnings
import pickle
import random
import logging
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Constants and Configurations
GROQ_API_KEY = "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"
MODELS = {
    "1": "mixtral-8x7b-32768",
    "2": "gemma-7b-it",
    "3": "llama3-8b-8192"
}
CONTEXT_PROMPT = "You are Lyla. You are friendly and you behave like a human. You are having a casual conversation with Aadish, your classmate."
MEMORY_SIZE = 10
MEMORY_FILE = "conversation_memory.pkl"  # File to store conversation memory
CUSTOM_DATA = {"favorite_color": "blue",
               "hometown": "Springfield",
               # Add more info
               }

def choose_model():
    print("Choose a model:")
    for key, value in MODELS.items():
        print(f"{key}. {value}")
    return MODELS.get(input("Enter the number corresponding to the model you want to use: "), "llama3-8b-8192")

def initialize_memory():
    try:
        with open(MEMORY_FILE, "rb") as file:
            memory = pickle.load(file)
    except FileNotFoundError:
        memory = ConversationBufferWindowMemory(k=MEMORY_SIZE, memory_key="chat_history", return_messages=True)
    return memory

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "wb") as file:
            pickle.dump(memory, file)
    except Exception as e:
        logging.error(f"Error occurred while saving memory: {e}")
    finally:
        logging.info("Memory saved successfully.")

def construct_prompt(context_prompt, user_input, temperature, max_tokens):
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=context_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(user_input),
        SystemMessage(content=json.dumps(CUSTOM_DATA)),
        *[SystemMessage(content=f"{key}: {value}") for key, value in {"Temperature": temperature, "Max Tokens": max_tokens}.items()]
    ])

def main():
    selected_model = choose_model()
    memory = initialize_memory()
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        while True:
            user_input = input("You: ")
            temperature = random.uniform(0.7, 1.0)
            max_tokens = 1024
            prompt = construct_prompt(CONTEXT_PROMPT, user_input, temperature, max_tokens)
            conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=False, memory=memory)
            response = conversation.predict(human_input=user_input)
            print("\nLyla:", response, "\n")
            save_memory(memory)
    except KeyboardInterrupt:
        save_memory(memory)

if __name__ == "__main__":
    logging.basicConfig(filename='conversation_memory.log', level=logging.INFO)
    main()

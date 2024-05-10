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
CONTEXT_PROMPT = "You are Lyla. You are friendly and you behave like a human. You are having a casual conversation."
MEMORY_SIZE = 100
MEMORY_FILE_PREFIX = "conversation_memory_"  # File prefix to store conversation memory
CUSTOM_DATA = {"favorite_color": "blue",
               "hometown": "Springfield",
               # Add more info
               } 

# Profiles and their respective data
PROFILES = {
    "1": {"name": "Aadish", "context": CONTEXT_PROMPT + " with Aadish, your classmate.", "favorite_color": "blue", "hometown": "Springfield"},
    "2": {"name": "Prakhar", "context": CONTEXT_PROMPT + " with Prakhar, your friend.", "favorite_color": "green", "hometown": "San Francisco"},
    # Add more profiles as needed
}

def choose_profile():
    print("Choose a profile:")
    for key, value in PROFILES.items():
        print(f"{key}. {value['name']}")
    return PROFILES.get(input("Enter the number corresponding to the profile you want to use: "), PROFILES["1"])

def choose_model():
    print("Choose a model:")
    for key, value in MODELS.items():
        print(f"{key}. {value}")
    return MODELS.get(input("Enter the number corresponding to the model you want to use: "), "llama3-8b-8192")

def initialize_memory(profile_name):
    memory_file = f"{MEMORY_FILE_PREFIX}{profile_name}.pkl"
    try:
        with open(memory_file, "rb") as file:
            memory = pickle.load(file)
    except FileNotFoundError:
        memory = ConversationBufferWindowMemory(k=MEMORY_SIZE, memory_key="chat_history", return_messages=True)
    return memory

def save_memory(memory, profile_name):
    memory_file = f"{MEMORY_FILE_PREFIX}{profile_name}.pkl"
    try:
        with open(memory_file, "wb") as file:
            pickle.dump(memory, file)
    except Exception as e:
        logging.error(f"Error occurred while saving memory for profile {profile_name}: {e}")
    else:
        logging.info(f"Memory for profile {profile_name} saved successfully.")

def construct_prompt(context_prompt, user_input, custom_data, temperature, max_tokens):
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=context_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(user_input),
        SystemMessage(content=json.dumps(custom_data)),
        SystemMessage(content=f"Temperature: {temperature}"),
        SystemMessage(content=f"Max Tokens: {max_tokens}")
    ])

def main():
    selected_profile = choose_profile()
    selected_model = choose_model()
    memory = initialize_memory(selected_profile['name'])
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        while True:
            user_input = input("You: ")
            temperature = 0.8  # Set a default temperature
            max_tokens = 1024  # Set a default max_tokens
            prompt = construct_prompt(selected_profile['context'], user_input, CUSTOM_DATA, temperature, max_tokens)
            conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=False, memory=memory)
            response = conversation.predict(human_input=user_input)
            print("\nLyla:", response, "\n")
            save_memory(memory, selected_profile['name'])  # Save the conversation after each response
    except KeyboardInterrupt:
        save_memory(memory, selected_profile['name'])

if __name__ == "__main__":
    logging.basicConfig(filename='conversation_memory.log', level=logging.INFO)
    main()

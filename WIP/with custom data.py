import random
import json
import warnings  # Import warnings module
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

# Custom data as a knowledge base
custom_data = {
    "favorite_color": "blue",
    "hometown": "Springfield",
    # Add more data as needed
}

def choose_model():
    print("Choose a model:")
    for key, value in MODELS.items():
        print(f"{key}. {value}")
    chosen_model = input("Enter the number corresponding to the model you want to use: ")
    return MODELS.get(chosen_model, "llama3-8b-8192")

def initialize_memory():
    return ConversationBufferWindowMemory(k=MEMORY_SIZE, memory_key="chat_history", return_messages=True)

def construct_prompt(context_prompt, user_input):
    # Serialize custom_data to string
    custom_data_str = json.dumps(custom_data)
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=context_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
        SystemMessage(content=custom_data_str)  # Including serialized custom data in the prompt
    ])

def main():
    selected_model = choose_model()
    memory = initialize_memory()
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)

    while True:
        user_input = input("You: ")
        
        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            prompt = construct_prompt(CONTEXT_PROMPT, user_input)
            conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=False, memory=memory)

        response = conversation.predict(human_input=user_input)
        print("\nLyla:", response, "\n")

if __name__ == "__main__":
    main()

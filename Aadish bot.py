from groq import Groq

api_key = "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"
client = Groq(api_key=api_key)

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Aadish: Goodbye!")
        break
    
    bot_response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
        temperature=1,
        max_tokens=1024,
    ).choices[0].message.content

    print("\nAadish:", bot_response, "\n")

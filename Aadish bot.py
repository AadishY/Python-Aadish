from groq import Groq

client = Groq(api_key="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")

while True:
    user_input = input("You: ")
    
    
    bot_response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
        temperature=1,
        max_tokens=1024,
    ).choices[0].message.content

    print("\nAadish:", bot_response, "\n")


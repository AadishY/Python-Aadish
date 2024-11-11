from groq import Groq

client = Groq(api_key="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")
#eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2lkZW50aXR5dG9vbGtpdC5nb29nbGVhcGlzLmNvbS9nb29nbGUuaWRlbnRpdHkuaWRlbnRpdHl0b29sa2l0LnYxLklkZW50aXR5VG9vbGtpdCIsImlhdCI6MTczMTM0NDMzNSwiZXhwIjoxNzMxMzQ3OTM1LCJpc3MiOiJmaXJlYmFzZS1hZG1pbnNkay02cjM0eUB0YWJuaW5lLWF1dGgtMzQwMDE1LmlhbS5nc2VydmljZWFjY291bnQuY29tIiwic3ViIjoiZmlyZWJhc2UtYWRtaW5zZGstNnIzNHlAdGFibmluZS1hdXRoLTM0MDAxNS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsInVpZCI6IjRSQkp6UzhkSEFPRUdmTUpzbk1uY3Z6dE5NajEifQ.wmqDUSVyXE9SvF0HGBUbjQQYU_iw1rfkevxIkwyOxKSRM_T38FdbBp3hJj929fmRG1BRUwPuItcdvnR04rPuR4ofY1zWK5b4_9h041B_imjS27E8eilOymp63_X1oz5Ka_Sp4pVasTS7-fQULz7R_NR0Bl9V5mcsktTRhpU9EejdrB5jjBFz11MkhV8rpa4DjofY9OQ0ewDf6gylIMDc5MKPkc-8SQ_9nWXUoxuDs6XplCKdBGUjTXnjxOGA7d-VGDw3cd7hZ9_joWjrTF3ZhAzPHnVW-QhYg-n0J1QATIhIgmf_ht9Gy3t1ekNnAAbxazQ3obVKZCMv46h1VqZ6Vg
while True:
    user_input = input("You: ")
    
    
    bot_response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
        temperature=1,
        max_tokens=1024,
    ).choices[0].message.content

    print("\nAadish:", bot_response, "\n")


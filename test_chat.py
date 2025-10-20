import os
from dotenv import load_dotenv
from openai import OpenAI

# The API key stored in the .env file is loaded as an environment variable
load_dotenv()

# Verify that the API key has been loaded correctly
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY is not set in the .env file")
    
else:
    print("The API key has been successfully retrieved. Starting the chat.")
    print("To exit, type 'exit'\n")
    
    try:
        # Initializing the OpenAI Client
        client = OpenAI()
        
        # An infinite loop that exchanges messages with the user
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                print("Ending the chat")
                break
                
            # Calls OpenAI's Chat Completions API
            response = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input} 
                ]
            )
            
            ai_response = response.choices[0].message.content
            print(f"AI: {ai_response}")
    except Exception as e:
        # If issues such as API key errors or insufficient balance occur, errors will be displayed here
        print(f"\n--[Error occurred]")
        print(f"error message: {e}")
        print("Please verify that your API key is valid and that you have sufficient credit remaining in your OpenAI account.")
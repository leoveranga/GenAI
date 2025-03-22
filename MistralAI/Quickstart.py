import os
from mistralai import Mistral

#api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Extract values
apikey = config["key"]["MISTRAL_API_KEY"]

client = Mistral(api_key=apikey)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)
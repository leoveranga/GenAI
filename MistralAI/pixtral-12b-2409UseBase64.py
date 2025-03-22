
#Passing a Base64 Encoded Image
#If you have an image or a set of images stored locally, you can pass them to the model in base64 encoded format. Base64 encoding is a common method for converting binary data into a text format that can be easily transmitted over the internet. This is particularly useful when you need to include images in API requests.

import base64
import requests
import os
from mistralai import Mistral

import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Extract values
api_key = config["key"]["MISTRAL_API_KEY"]

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

# Path to your image
#image_path = "C:\temp\GoogleDriveDownloads\Proj1\GoogleWillow.jpg"
#image_path = "C:\\temp\\GoogleDriveDownloads\\Proj1\\socio-economic.jpeg"
image_path = "C:\\temp\\GoogleDriveDownloads\\Proj1\\interesting-receipts-102-6364c8d181c6a__700.jpg"



# Getting the base64 string
base64_image = encode_image(image_path)

# Retrieve the API key from environment variables
# api_key = os.environ["MISTRAL_API_KEY"]

# Specify model
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}" 
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages
)

# Print the content of the response
print(chat_response.choices[0].message.content)
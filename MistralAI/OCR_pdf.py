import os
from mistralai import Mistral

import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Extract values
api_key = config["key"]["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    include_image_base64=True
)
import os
from mistralai import Mistral
import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# Extract values
apikey = config["key"]["MISTRAL_API_KEY"]

#api_key = os.environ["MISTRAL_API_KEY"]
api_key = apikey
model = "mistral-embed"

client = Mistral(api_key=api_key)

embeddings_response = client.embeddings.create(
    model=model,
    inputs=["Embed this sentence.", "As well as this one.", "Leo Veranga"]
)

print(embeddings_response)
import requests
import json
import sys

def query_llama(prompt, model="llama3.2", temperature=0.7):
    """
    Send a query to Llama 3.2 running in Ollama and return the response.
    
    Args:
        prompt (str): The question or prompt to send to the model
        model (str): The model name to use (default: "llama3.2")
        temperature (float): Controls randomness in responses (0.0 to 1.0)
        
    Returns:
        str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure the Ollama service is running."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except json.JSONDecodeError:
        return "Error: Could not parse the response from Ollama."

def main():
    print("Llama 3.2 Q&A Interface (powered by Ollama)")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue
            
        print("\nThinking...")
        response = query_llama(user_input)
        print("\nLlama 3.2 Response:")
        print(response)

if __name__ == "__main__":
    main()
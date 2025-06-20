import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

def ask_ollama(prompt):
    payload = {
        "model": "llama3",  # or "mistral", "deepseek-chat", etc.
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(OLLAMA_URL, json=payload)
    
    # Streamed JSON — one line per token chunk
    lines = response.text.strip().splitlines()

    # Concatenate all token parts
    full_response = ""
    for line in lines:
        try:
            data = json.loads(line)
            content = data.get("message", {}).get("content")
            if content:
                full_response += content
        except json.JSONDecodeError:
            continue

    return full_response.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            answer = ask_ollama(user_input)
            print("AI:", answer)
        except Exception as e:
            print("❌ Error:", e)
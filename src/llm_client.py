import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Normalize base URL - match your working code
base_url = os.getenv("BASE_URL")
if not base_url.endswith("/openai/v1/"):
    base_url = base_url.rstrip("/") + "/openai/v1/"

client = OpenAI(
    base_url=base_url,
    api_key=os.getenv("API_KEY"),
)

MODEL = os.getenv("MODEL")

def foundry_chat(messages, max_tokens=150):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,  # Use max_tokens instead of max_completion_tokens
        )

        return completion.choices[0].message.content

    except Exception as e:
        print("LLM error:", e)
        return f"Sorry, there was an error: {str(e)}"
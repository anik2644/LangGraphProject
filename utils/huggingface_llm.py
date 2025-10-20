# hf_direct_test.py
# -----------------------------------------------------
# Directly calls Hugging Face's Inference API using your token
# Works with any model that supports text generation (e.g. flan-t5-base)

import os
import requests
from dotenv import load_dotenv

def run_direct_inference(model_id="google/flan-t5-base"):
    load_dotenv()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN in .env")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": "Hello! Briefly introduce yourself in two sentences.",
        "parameters": {"max_new_tokens": 100, "temperature": 0.5}
    }

    print(f"⚙️ Sending request to Hugging Face API for model: {model_id}")
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        print("✅ Raw API response:")
        print(data)

        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            print("\n💬 Model output:")
            print(data[0]["generated_text"])
        elif isinstance(data, dict) and "error" in data:
            print(f"⚠️ API returned an error: {data['error']}")
        else:
            print("⚠️ Unexpected response format. Check printed JSON above.")
    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)

if __name__ == "__main__":
    run_direct_inference("google/flan-t5-base")

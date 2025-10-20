"""
Simple Groq LLM Test Script
----------------------------
✅ Tests connection to Groq API
✅ Lists supported models (if available)
✅ Runs a basic prompt
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os, requests


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GROQ_API_KEY in .env file")

    # ✅ Try one of these supported models:
    # "llama-3.1-70b-versatile"
    # "llama-3.1-8b-instant"
    # "mixtral-8x7b-32768"
    # "gemma-7b-it"
    # "llama3-groq-70b-8192-tool-use-preview"

    model_name = "llama-3.1-8b-instant"

    print(f"🤖 Testing Groq model: {model_name}")

    llm = ChatGroq(
        model=model_name,
        temperature=0.3,
        groq_api_key=api_key
    )

    prompt = "Explain in one sentence what Nakshi Katha is."
    print("💬 Prompt:", prompt)

    try:
        response = llm.invoke(prompt)
        print("\n✅ Groq Model Response:\n")
        print(response.content)
    except Exception as e:
        print("\n❌ Error:", str(e))

if __name__ == "__main__":
    main()

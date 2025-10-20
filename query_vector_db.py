"""
Vector Database QA Agent using LangChain + Groq + FAISS
-------------------------------------------------------
✅ Loads the saved FAISS vector database
✅ Embeds the user's natural-language question
✅ Retrieves similar text chunks from MySQL-derived data
✅ Uses Groq LLM to answer based on retrieved context
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_env():
    """Load and validate environment variables."""
    load_dotenv()
    env = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index"),
    }
    if not env["GROQ_API_KEY"]:
        raise ValueError("Missing GROQ_API_KEY in .env file")
    return env


def setup_groq(env: dict):
    """Initialize Groq LLM."""
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=env["GROQ_API_KEY"]
    )


def load_vector_db(env: dict):
    """Load FAISS vector store and embeddings."""
    print("📦 Loading FAISS vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(env["VECTOR_DB_PATH"], embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded successfully!\n")
    return vectorstore


def retrieve_context(vectorstore, query: str, top_k=5):
    """Retrieve the most similar documents from the vector database."""
    print("🔍 Searching vector database...")
    docs = vectorstore.similarity_search(query, k=top_k)
    combined = "\n\n".join([d.page_content for d in docs])
    print(f"✅ Retrieved {len(docs)} relevant chunks.\n")
    return combined


def generate_answer(llm, query: str, context: str):
    """Generate an answer using Groq LLM based on context."""
    prompt = f"""
You are a helpful AI assistant. Answer the user's question using only the context below.

User Question:
{query}

Context from the vector database:
{context[:4000]}  # limit long context for efficiency

If the answer is not found in the context, say "I couldn't find an exact answer in the data."
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    env = load_env()
    llm = setup_groq(env)
    vectorstore = load_vector_db(env)

    while True:
        query = input("🔹 Ask your question (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        context = retrieve_context(vectorstore, query)
        answer = generate_answer(llm, query, context)

        print("\n================= AI Answer =================\n")
        print(answer)
        print("\n=============================================\n")


if __name__ == "__main__":
    main()

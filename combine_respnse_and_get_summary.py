"""
Combined Vector DB + PDF QA Summarizer
--------------------------------------
✅ Loads FAISS vector DB
✅ Extracts context from PDF
✅ Retrieves answers from both
✅ Uses Groq LLM to produce a summarized unified response
"""

import os
from dotenv import load_dotenv
import PyPDF2

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================== ENV ====================
def load_env():
    load_dotenv()
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index"),
        "PDF_FILE_PATH": os.getenv("PDF_FILE_PATH"),
    }


# ==================== LLM ====================
def setup_groq(env):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=env["GROQ_API_KEY"],
        temperature=0.2,
    )


# ==================== VECTOR DB ====================
def load_vector_db(env):
    print("📦 Loading FAISS vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(env["VECTOR_DB_PATH"], embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded.\n")
    return vectorstore


def retrieve_vector_context(vectorstore, query, top_k=5):
    print("🔍 Retrieving context from vector DB...")
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([d.page_content for d in docs])


# ==================== PDF ====================
def extract_text_from_pdf(pdf_path):
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def setup_pdf_retriever(pdf_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(pdf_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def retrieve_pdf_context(retriever, query):
    print("📘 Retrieving context from PDF...")
    docs = retriever.invoke(query)  # ✅ FIXED: use .invoke() instead of get_relevant_documents()
    return "\n\n".join([d.page_content for d in docs])


# ==================== SUMMARIZATION ====================
def summarize_combined_response(llm, query, db_answer, pdf_answer):
    prompt_text = f"""
You are a research summarizer AI.

User Question:
{query}

Answer from Vector Database:
{db_answer}

Answer from PDF Document:
{pdf_answer}

Your Task:
Summarize the two answers into a single, concise, well-structured response. 
If they contradict each other, note the difference clearly. 
Avoid repetition and maintain factual accuracy.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful summarizer and analyzer AI."),
        ("human", prompt_text)
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


# ==================== MAIN ====================
def main():
    env = load_env()
    llm = setup_groq(env)
    vectorstore = load_vector_db(env)

    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    pdf_retriever = setup_pdf_retriever(pdf_text)

    while True:
        query = input("\n🔹 Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Get response from Vector DB
        db_context = retrieve_vector_context(vectorstore, query)
        db_prompt = f"Answer the question using this context only:\n{db_context}\n\nQuestion: {query}"
        db_answer = llm.invoke(db_prompt).content.strip()

        # Get response from PDF
        pdf_context = retrieve_pdf_context(pdf_retriever, query)
        pdf_prompt = f"Answer the question using this context only:\n{pdf_context}\n\nQuestion: {query}"
        pdf_answer = llm.invoke(pdf_prompt).content.strip()

        # Summarize both
        summary = summarize_combined_response(llm, query, db_answer, pdf_answer)

        print("\n================= 🧠 COMBINED SUMMARY =================\n")
        print(summary)
        print("\n========================================================\n")


if __name__ == "__main__":
    main()

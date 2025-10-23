"""
LangGraph-Based Combined Vector DB + PDF QA Summarizer
-------------------------------------------------------
✅ Loads existing FAISS vector DB
✅ Extracts context from PDF
✅ Retrieves answers from both sources
✅ Uses Groq LLM (LLaMA-3.1-8B) for unified summarized answer
"""

import os
from dotenv import load_dotenv
import PyPDF2

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ======================================================
# 1. ENVIRONMENT
# ======================================================
def load_env():
    load_dotenv()
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index"),
        "PDF_FILE_PATH": os.getenv("PDF_FILE_PATH"),
    }


# ======================================================
# 2. LLM INITIALIZATION
# ======================================================
def setup_groq(env):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=env["GROQ_API_KEY"],
        temperature=0.2,
    )


# ======================================================
# 3. VECTOR DATABASE
# ======================================================
def load_vector_db(env):
    print("📦 Loading FAISS vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        env["VECTOR_DB_PATH"], embeddings, allow_dangerous_deserialization=True
    )
    print("✅ Vector DB loaded successfully.")
    return vectorstore


# ======================================================
# 4. PDF UTILITIES
# ======================================================
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


# ======================================================
# 5. GRAPH STATE
# ======================================================
class QAState(dict):
    """Shared LangGraph state container."""
    query: str
    db_context: str
    pdf_context: str
    db_answer: str
    pdf_answer: str
    summary: str


# ======================================================
# 6. GRAPH NODES
# ======================================================
def retrieve_vector_node(state: QAState, env, vectorstore):
    print("🔍 Retrieving from Vector DB...")
    docs = vectorstore.similarity_search(state["query"], k=5)
    state["db_context"] = "\n\n".join([d.page_content for d in docs])
    return state


def retrieve_pdf_node(state: QAState, pdf_retriever):
    print("📘 Retrieving from PDF...")
    docs = pdf_retriever.invoke(state["query"])
    state["pdf_context"] = "\n\n".join([d.page_content for d in docs])
    return state


def answer_from_vector_node(state: QAState, llm):
    print("🤖 Generating answer from Vector DB context...")
    prompt = f"Answer using only this context:\n{state['db_context']}\n\nQuestion: {state['query']}"
    response = llm.invoke(prompt)
    state["db_answer"] = response.content.strip()
    return state


def answer_from_pdf_node(state: QAState, llm):
    print("🤖 Generating answer from PDF context...")
    prompt = f"Answer using only this context:\n{state['pdf_context']}\n\nQuestion: {state['query']}"
    response = llm.invoke(prompt)
    state["pdf_answer"] = response.content.strip()
    return state


def summarize_node(state: QAState, llm):
    print("🧠 Summarizing combined answers...")
    prompt_text = f"""
    You are a research summarizer AI.

    User Question:
    {state['query']}

    Answer from Vector DB:
    {state['db_answer']}

    Answer from PDF Document:
    {state['pdf_answer']}

    Task:
    Merge the two answers into one concise, well-structured summary.
    If contradictions exist, note them clearly.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful summarizer."),
        ("human", prompt_text)
    ])

    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({})
    state["summary"] = summary

    print("\n================= 🧠 COMBINED SUMMARY =================\n")
    print(summary)
    print("========================================================\n")
    return state


# ======================================================
# 7. MAIN EXECUTION
# ======================================================
def main():
    env = load_env()
    llm = setup_groq(env)
    vectorstore = load_vector_db(env)
    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    pdf_retriever = setup_pdf_retriever(pdf_text)

    # 🧩 Define LangGraph
    graph = StateGraph(QAState)

    graph.add_node("vector_retrieval", lambda s: retrieve_vector_node(s, env, vectorstore))
    graph.add_node("pdf_retrieval", lambda s: retrieve_pdf_node(s, pdf_retriever))
    graph.add_node("vector_answer", lambda s: answer_from_vector_node(s, llm))
    graph.add_node("pdf_answer", lambda s: answer_from_pdf_node(s, llm))
    graph.add_node("summary", lambda s: summarize_node(s, llm))

    graph.set_entry_point("vector_retrieval")
    graph.add_edge("vector_retrieval", "pdf_retrieval")
    graph.add_edge("pdf_retrieval", "vector_answer")
    graph.add_edge("vector_answer", "pdf_answer")
    graph.add_edge("pdf_answer", "summary")
    graph.add_edge("summary", END)

    workflow = graph.compile()

    # 🧠 Interactive loop
    while True:
        query = input("\n🔹 Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        init_state = {"query": query}
        workflow.invoke(init_state)


if __name__ == "__main__":
    main()

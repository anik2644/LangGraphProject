# langgraph_pdf.py
import os
from dotenv import load_dotenv
import PyPDF2
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from langchain_groq import ChatGroq

# ======================================================
# 1. ENVIRONMENT
# ======================================================
from dotenv import load_dotenv


# =============== Load Environment ===============
def load_env():
    load_dotenv()
    return {"GROQ_API_KEY": os.getenv("GROQ_API_KEY"), "PDF_FILE_PATH": os.getenv("PDF_FILE_PATH")}

# ======================================================
# 2. LOCAL LLM
# ======================================================
def setup_local_llm():
    print("🧠 Loading local LLM pipeline (TinyLlama)...")
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)

def setup_groq(env):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=env["GROQ_API_KEY"],
        temperature=0.2,
    )

# ======================================================
# 3. PDF HANDLING
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
# 4. STATE CLASS
# ======================================================
class QAState(dict):
    query: str
    pdf_context: str
    pdf_answer: str

# ======================================================
# 5. GRAPH NODES
# ======================================================
def retrieve_pdf_node(state: QAState, pdf_retriever):
    print("📘 Retrieving from PDF...")
    docs = pdf_retriever.invoke(state["query"])
    state["pdf_context"] = "\n\n".join([d.page_content for d in docs])
    return state

def answer_from_pdf_node(state: QAState, llm):
    print("🤖 Generating answer from PDF context...")
    prompt = f"Answer using only this context:\n{state['pdf_context']}\n\nQuestion: {state['query']}"
    response = llm.invoke(prompt)
    print("\n================= 📘 PDF ANSWER =================\n")
    print(response)
    print("=================================================\n")
    state["pdf_answer"] = response.content.strip()
    return state

# ======================================================
# 6. MAIN EXECUTION
# ======================================================
def main():
    env = load_env()
    # llm = setup_local_llm()
    llm = setup_groq(env)
    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    pdf_retriever = setup_pdf_retriever(pdf_text)

    graph = StateGraph(QAState)
    graph.add_node("pdf_retrieval", lambda s: retrieve_pdf_node(s, pdf_retriever))
    graph.add_node("pdf_answer", lambda s: answer_from_pdf_node(s, llm))

    graph.set_entry_point("pdf_retrieval")
    graph.add_edge("pdf_retrieval", "pdf_answer")
    graph.add_edge("pdf_answer", END)

    workflow = graph.compile()

    while True:
        query = input("\n🔹 Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        result = workflow.invoke(QAState(query=query))
        print(f"\n🤖: {result.get('pdf_answer', 'Sorry, I could not find an answer.')}")

if __name__ == "__main__":
    main()

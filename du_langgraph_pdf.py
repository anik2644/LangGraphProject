# langgraph_pdf.py
import os
import numpy as np
from dotenv import load_dotenv
import PyPDF2
import re
from langgraph.graph import StateGraph, END

import faiss
from sentence_transformers import SentenceTransformer

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================================================
# 1. ENVIRONMENT
# ======================================================
def load_env():
    load_dotenv()
    return {"PDF_FILE_PATH": os.getenv("PDF_FILE_PATH")}


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


# ======================================================
# 3. PDF HANDLING
# ======================================================
def extract_text_from_pdf(pdf_path):
    parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)




# Function to split text into segments using START/END tags
def split_by_tags(text):
    # Pattern to match content between (START#) and (#END)
    pattern = r'\(START#\)(.*?)\(#END\)'
    segments = re.findall(pattern, text, re.DOTALL)
    # Clean up each segment - remove extra whitespace and fix new lines within text
    cleaned_segments = []
    for segment in segments:
        # Remove new lines between words and clean up whitespace
        cleaned_segment = re.sub(r'\s+', ' ', segment.strip())
        cleaned_segments.append(cleaned_segment)
    return cleaned_segments





#
# env = load_env()
# pdf_path = env["PDF_FILE_PATH"]
# # print(extract_pdf_text(env["PDF_FILE_PATH"]))
# # extract_pdf_text(env["PDF_FILE_PATH"])
#
#
# # Extract text from the PDF
# pdf_text = extract_text_from_pdf(pdf_path)
#
# # Split the extracted text into segments using tags
# segments = split_by_tags(pdf_text)


# ======================================================
# 4. CUSTOM FAISS RETRIEVER (your technique)
# ======================================================
class CustomFAISSRetriever:
    def __init__(self, segments):
        print("📌 Loading SentenceTransformer embeddings...")
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("📌 Generating embeddings...")
        self.segments = segments
        self.embeddings = self.emb_model.encode(segments)

        print("📌 Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def invoke(self, query, k=1):
        query_emb = self.emb_model.encode([query])
        distances, indices = self.index.search(np.array(query_emb), k)

        docs = []
        for idx in indices[0]:
            docs.append(self.segments[idx])

        # print(docs)
        return docs



# env = load_env()
# pdf_path = env["PDF_FILE_PATH"]
# pdf_text = extract_text_from_pdf(pdf_path)
# segments = split_by_tags(pdf_text)
#
#
# question= "Where is Boga Lake?"
# retv = CustomFAISSRetriever(segments)
# retv.invoke(question)



def setup_pdf_retriever(pdf_text):
    print("📘 Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    segments = splitter.split_text(pdf_text)
    return CustomFAISSRetriever(segments)







# ======================================================
# 5. STATE CLASS
# ======================================================
class QAState(dict):
    query: str
    pdf_context: str
    pdf_answer: str


# ======================================================
# 6. GRAPH NODES
# ======================================================
def retrieve_pdf_node(state: QAState, retriever: CustomFAISSRetriever):
    print("📘 Retrieving from FAISS...")
    docs = retriever.invoke(state["query"], k=1)
    state["pdf_context"] = "\n\n".join(docs)
    return state


import re

def extract_answer_only(response: str) -> str:
    """
    Extracts the actual answer text from an LLM response, even if
    the answer label varies. Supports many patterns:
    - Answer:
    - Final Answer:
    - RESPONSE:
    - Output:
    - =>
    - Answer -
    - ANSWER\n
    - etc.
    """

    text = response.strip()

    # --- 1. Try common answer labels (case-insensitive) ---
    patterns = [
        r"answer\s*:?",           # Answer:
        r"final\s*answer\s*:?",   # Final Answer:
        r"response\s*:?",         # Response:
        r"result\s*:?",           # Result:
        r"output\s*:?",           # Output:
        r"solution\s*:?",         # Solution:
        r"explanation\s*:?",      # Explanation:
        r"->",                    # ->
        r"=>",                    # =>
    ]

    lower = text.lower()
    for p in patterns:
        match = re.search(p, lower, flags=re.IGNORECASE)
        if match:
            start = match.end()
            extracted = text[start:].strip()
            if extracted:
                return extracted

    # --- 2. If answer is after a newline like:
    # Answer
    # This is the actual answer
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() in ["answer", "final answer", "response", "result"]:
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line:
                    return next_line

    # --- 3. Try extracting after colon if the model writes:
    # Question: xxx
    # Answer: yyy
    m = re.search(r":\s*(.+)$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # --- 4. Extreme fallback: return the whole text ---
    return text


def answer_from_pdf_node(state: QAState, llm):
    print("🤖 Generating answer from PDF context...")

    context = state["pdf_context"]

    prompt = (
        f"Use only the following context:\n{context}\n\n"
        f"Question: {state['query']}"
    )

    response = llm.invoke(prompt)

    # print("\n=========== 📘 PDF ANSWER ===========\n")
    # print(response)
    # print("=====================================\n")

    # 🔥 Extract only the 'Answer:' portion
    clean_answer = extract_answer_only(response)


    # print("\n=========== 📘THEN ===========\n")
    print(clean_answer)
    # print("=====================================\n")

    state["pdf_answer"] = clean_answer
    return state



# def answer_from_pdf_node(state: QAState, llm):
#     print("🤖 Generating answer from PDF context...")
#
#     # --- NEW LIMITER ---
#     context = state["pdf_context"]
#     # if len(context) > 1900:
#     #     context = context[:1900]   # hard cap
#     # # --------------------
#
#     prompt = (
#         f"Use only the following context:\n{context}\n\n"
#         f"Question: {state['query']}"
#     )
#
#     response = llm.invoke(prompt)
#
#     print("\n=========== 📘 PDF ANSWER ===========\n")
#     print(response)
#     print("=====================================\n")
#
#     state["pdf_answer"] = response.strip()
#     return state
#
# def answer_from_pdf_node(state: QAState, llm):
#     print("🤖 Generating answer from PDF context...")
#
#     prompt = (
#         f"Use only the following context:\n{state['pdf_context']}\n\n"
#         f"Question: {state['query']}"
#     )
#
#     response = llm.invoke(prompt)
#     print("\n=========== 📘 PDF ANSWER ===========\n")
#     print(response)
#     print("=====================================\n")
#
#     state["pdf_answer"] = response.strip()
#     return state


# ======================================================
# 7. MAIN EXECUTION
# ======================================================
def main():
    env = load_env()

    llm = setup_local_llm()

    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    # retriever = setup_pdf_retriever(pdf_text)

    segments = split_by_tags(pdf_text)
    retriever = CustomFAISSRetriever(segments)

    graph = StateGraph(QAState)

    graph.add_node("pdf_retrieval", lambda s: retrieve_pdf_node(s, retriever))
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
        workflow.invoke({"query": query})


if __name__ == "__main__":
    main()

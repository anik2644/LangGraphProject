# langgraph_vectorDB.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# ======================================================
# 1. ENVIRONMENT
# ======================================================
def load_env():
    load_dotenv()
    return {"VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index")}

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
# 4. STATE CLASS
# ======================================================
class QAState(dict):
    query: str
    db_context: str
    db_answer: str

# ======================================================
# 5. GRAPH NODES
# ======================================================
def retrieve_vector_node(state: QAState, vectorstore):
    print("🔍 Retrieving from Vector DB...")
    docs = vectorstore.similarity_search(state["query"], k=5)
    state["db_context"] = "\n\n".join([d.page_content for d in docs])
    return state

def answer_from_vector_node(state: QAState, llm):
    print("🤖 Generating answer from Vector DB context...")
    prompt = f"Answer using only this context:\n{state['db_context']}\n\nQuestion: {state['query']}"
    response = llm.invoke(prompt)
    print("\n================= 📦 VECTOR DB ANSWER =================\n")
    print(response)
    print("========================================================\n")
    state["db_answer"] = response.strip()
    return state

# ======================================================
# 6. MAIN EXECUTION
# ======================================================
def main():
    env = load_env()
    llm = setup_local_llm()
    vectorstore = load_vector_db(env)

    graph = StateGraph(QAState)
    graph.add_node("vector_retrieval", lambda s: retrieve_vector_node(s, vectorstore))
    graph.add_node("vector_answer", lambda s: answer_from_vector_node(s, llm))

    graph.set_entry_point("vector_retrieval")
    graph.add_edge("vector_retrieval", "vector_answer")
    graph.add_edge("vector_answer", END)

    workflow = graph.compile()

    while True:
        query = input("\n🔹 Ask a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        workflow.invoke({"query": query})

if __name__ == "__main__":
    main()

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import PyPDF2
import os
from dotenv import load_dotenv


# =============== Load Environment ===============
def load_env():
    load_dotenv()
    return {"GROQ_API_KEY": os.getenv("GROQ_API_KEY"), "PDF_FILE_PATH": os.getenv("PDF_FILE_PATH")}


# =============== Extract Text from PDF ===============
def extract_text_from_pdf(pdf_path):
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# =============== Setup Components ===============
def setup_groq(env):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=env["GROQ_API_KEY"],
        temperature=0.2,
    )


def setup_retriever(pdf_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(pdf_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# =============== Build a Retrieval Pipeline (Runnable) ===============
def setup_retrieval_pipeline(llm, retriever):
    system_prompt = (
        "Use the following context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Be concise and clear.\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Define how inputs move through the components
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


# =============== Main ===============
def main():
    env = load_env()

    print("📘 Extracting PDF...")
    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    print(f"✅ Extracted {len(pdf_text)} characters.")
    retriever = setup_retriever(pdf_text)

    llm = setup_groq(env)

    chain = setup_retrieval_pipeline(llm, retriever)

    while True:
        question = input("\n🔹 Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        print("\n🧠 Answer:")
        print(chain.invoke(question))


if __name__ == "__main__":
    main()

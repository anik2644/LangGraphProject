"""
Create a Vector Database from MySQL content
-------------------------------------------
✅ Connects to MySQL
✅ Extracts textual data
✅ Embeds using HuggingFace embeddings
✅ Saves FAISS vector store for later semantic queries
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_env():
    """Load environment variables."""
    load_dotenv()
    env = {
        "MYSQL_HOST": os.getenv("MYSQL_HOST"),
        "MYSQL_PORT": os.getenv("MYSQL_PORT"),
        "MYSQL_DB": os.getenv("MYSQL_DB"),
        "MYSQL_USER": os.getenv("MYSQL_USER"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        "VECTOR_DB_PATH": os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index")
    }

    missing = [k for k, v in env.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
    return env


def fetch_text_data(env):
    """Fetch textual data from all or selected tables in MySQL."""
    import mysql.connector

    conn = mysql.connector.connect(
        host=env["MYSQL_HOST"],
        port=env["MYSQL_PORT"],
        database=env["MYSQL_DB"],
        user=env["MYSQL_USER"],
        password=env["MYSQL_PASSWORD"]
    )

    cursor = conn.cursor()

    # ✅ Example: Read all table names
    cursor.execute("SHOW TABLES;")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"🧩 Found tables: {tables}\n")

    all_texts = []

    for table in tables:
        try:
            # Try to select first few textual columns
            cursor.execute(f"SHOW COLUMNS FROM {table}")
            columns = [col[0] for col in cursor.fetchall()]
            text_cols = [c for c in columns if "name" in c or "desc" in c or "text" in c or "content" in c]

            if not text_cols:
                continue

            query = f"SELECT {', '.join(text_cols)} FROM {table} LIMIT 2000"
            cursor.execute(query)

            rows = cursor.fetchall()
            for row in rows:
                text = " ".join([str(v) for v in row if v])
                if text.strip():
                    all_texts.append(text)
        except Exception as e:
            print(f"⚠️ Skipped {table}: {e}")
            continue

    conn.close()
    print(f"✅ Collected {len(all_texts)} text records for embeddings.\n")
    return all_texts


def create_vector_store(texts, output_path):
    """Create FAISS vector store from texts."""
    print("🧠 Generating embeddings and saving FAISS vector database...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(output_path)

    print(f"✅ Vector database saved at: {output_path}\n")


def main():
    env = load_env()
    texts = fetch_text_data(env)
    if not texts:
        print("❌ No text data found in database.")
        return
    create_vector_store(texts, env["VECTOR_DB_PATH"])


if __name__ == "__main__":
    main()

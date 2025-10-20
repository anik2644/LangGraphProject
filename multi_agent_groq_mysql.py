"""
Multi-Source AI Agent using LangChain + Groq + MySQL
----------------------------------------------------
✅ Extracts text from PDF
✅ Queries MySQL database using natural language
✅ Summarizes both results using Groq LLM
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
import PyPDF2


def load_env():
    """Load environment variables and validate."""
    load_dotenv()
    env = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "MYSQL_HOST": os.getenv("MYSQL_HOST"),
        "MYSQL_PORT": os.getenv("MYSQL_PORT"),
        "MYSQL_DB": os.getenv("MYSQL_DB"),
        "MYSQL_USER": os.getenv("MYSQL_USER"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        "PDF_FILE_PATH": os.getenv("PDF_FILE_PATH"),
    }

    missing = [k for k, v in env.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
    return env


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from each page of the PDF."""
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def setup_database(env: dict) -> SQLDatabase:
    """Initialize MySQL database connection for LangChain."""
    uri = (
        f"mysql+mysqlconnector://{env['MYSQL_USER']}:{env['MYSQL_PASSWORD']}"
        f"@{env['MYSQL_HOST']}:{env['MYSQL_PORT']}/{env['MYSQL_DB']}"
    )
    db = SQLDatabase.from_uri(uri)
    return db


def setup_groq(env: dict):
    """Initialize the Groq LLM."""
    return ChatGroq(
        model="llama3-70b-8192",  # You can also try "mixtral-8x7b"
        temperature=0,
        groq_api_key=env["GROQ_API_KEY"]
    )


def create_sql_agent_executor(db: SQLDatabase, llm) -> AgentExecutor:
    """Create a LangChain SQL agent that understands natural language questions."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    return agent_executor


def query_database(agent_executor: AgentExecutor, question: str) -> str:
    """Ask the agent a question; it generates SQL and executes it."""
    try:
        return agent_executor.run(question)
    except Exception as e:
        return f"[Database query error] {str(e)}"


def summarise_with_groq(llm, pdf_text: str, db_answer: str, user_question: str) -> str:
    """Combine PDF + DB information using Groq."""
    prompt = f"""
The user asked: {user_question}

Here is extracted text from a PDF (partial preview):
{pdf_text[:1000]}

Here is the database result:
{db_answer}

Please create a clear, concise summary combining insights from both the PDF and the database.
"""
    result = llm.invoke(prompt)
    return result.content.strip()


def main():
    # Step 1: Load environment
    env = load_env()

    # Step 2: Extract text from PDF
    print("📘 Extracting PDF text...")
    pdf_text = extract_text_from_pdf(env["PDF_FILE_PATH"])
    print(f"✅ Extracted {len(pdf_text)} characters from PDF.\n")

    # Step 3: Connect MySQL database
    print("🛢️ Connecting to MySQL database...")
    db = setup_database(env)
    print("✅ Database connected.\n")

    # Step 4: Initialize Groq LLM
    print("🤖 Initializing Groq LLM...")
    llm = setup_groq(env)
    print("✅ Groq ready.\n")

    # Step 5: Create SQL Agent
    print("⚙️ Creating SQL Agent...")
    agent_executor = create_sql_agent_executor(db, llm)
    print("✅ Agent created.\n")

    # Step 6: User input
    user_question = input("🔹 Enter your question: ")

    # Step 7: Query DB
    print("\n🔍 Running query on database...")
    db_answer = query_database(agent_executor, user_question)
    print("✅ Database answer:\n", db_answer, "\n")

    # Step 8: Summarize both
    print("🧠 Generating combined summary using Groq...")
    summary = summarise_with_groq(llm, pdf_text, db_answer, user_question)
    print("\n================ FINAL SUMMARY ================\n")
    print(summary)
    print("\n===============================================")


if __name__ == "__main__":
    main()

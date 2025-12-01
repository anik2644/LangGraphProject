import os
import PyPDF2
import re
from langgraph.graph import StateGraph, END
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn


# ======================================================
# Pydantic Models for API
# ======================================================
class QuestionRequest(BaseModel):
    query: str


class QuestionResponse(BaseModel):
    answer: str
    context_used: str
    source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pdf_loaded: bool


class PDFReloadResponse(BaseModel):
    status: str
    segments_count: int


# ======================================================
# Core PDF QA System
# ======================================================
class PDFQASystem:
    def __init__(self):
        # self.env = self.load_env()
        self.llm = None
        self.retriever = None
        self.segments = []
        self.pdf_text = ""
        self.is_initialized = False

    # def load_env(self):
    #     load_dotenv()
    #     return {"PDF_FILE_PATH": os.getenv("PDF_FILE_PATH")}

    def initialize_system(self):
        """Initialize all components"""
        try:
            print("🚀 Initializing PDF QA System...")

            # Load LLM
            print("🧠 Loading local LLM pipeline (TinyLlama)...")
            pipe = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

            # Load and process PDF
            print("📘 Loading PDF...")
            self.load_pdf()

            self.is_initialized = True
            print("✅ PDF QA System initialized successfully!")

        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)

    def split_by_tags(self, text):
        pattern = r'\(START#\)(.*?)\(#END\)'
        segments = re.findall(pattern, text, re.DOTALL)
        cleaned_segments = []
        for segment in segments:
            cleaned_segment = re.sub(r'\s+', ' ', segment.strip())
            cleaned_segments.append(cleaned_segment)
        return cleaned_segments

    def load_pdf(self):
        """Load and process PDF file"""
        try:
            pdf_path = "tourist_spot_bd.pdf"  #  self.env["PDF_FILE_PATH"]
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            self.pdf_text = self.extract_text_from_pdf(pdf_path)
            self.segments = self.split_by_tags(self.pdf_text)

            if not self.segments:
                # Fallback to text splitting if no tags found
                print("📘 No tags found, using text splitter...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                self.segments = splitter.split_text(self.pdf_text)

            print(f"📘 Loaded PDF with {len(self.segments)} segments")
            self.retriever = CustomFAISSRetriever(self.segments)

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise

    def reload_pdf(self):
        """Reload PDF file"""
        try:
            self.load_pdf()
            return len(self.segments)
        except Exception as e:
            print(f"❌ Error reloading PDF: {e}")
            raise

    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer from PDF"""
        if not self.is_initialized:
            raise HTTPException(status_code=503, detail="System not initialized")

        try:
            # Create workflow for this question
            graph = StateGraph(QAState)
            graph.add_node("pdf_retrieval", lambda s: self.retrieve_pdf_node(s))
            graph.add_node("pdf_answer", lambda s: self.answer_from_pdf_node(s))

            graph.set_entry_point("pdf_retrieval")
            graph.add_edge("pdf_retrieval", "pdf_answer")
            graph.add_edge("pdf_answer", END)

            workflow = graph.compile()

            # Execute workflow
            result = workflow.invoke({"query": question})

            return {
                "answer": result.get("pdf_answer", "No answer generated"),
                "context_used": result.get("pdf_context", ""),
                "source": "pdf"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

    def retrieve_pdf_node(self, state: Dict[str, Any]):
        print("📘 Retrieving from FAISS...")
        docs = self.retriever.invoke(state["query"], k=1)
        state["pdf_context"] = "\n\n".join(docs)
        return state

    def extract_answer_only(self, response: str) -> str:
        """Extracts the actual answer text from an LLM response"""
        text = response.strip()

        patterns = [
            r"answer\s*:?", r"final\s*answer\s*:?", r"response\s*:?",
            r"result\s*:?", r"output\s*:?", r"solution\s*:?",
            r"explanation\s*:?", r"->", r"=>"
        ]

        lower = text.lower()
        for p in patterns:
            match = re.search(p, lower, flags=re.IGNORECASE)
            if match:
                start = match.end()
                extracted = text[start:].strip()
                if extracted:
                    return extracted

        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower() in ["answer", "final answer", "response", "result"]:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        return next_line

        m = re.search(r":\s*(.+)$", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        return text

    def answer_from_pdf_node(self, state: Dict[str, Any]):
        print("🤖 Generating answer from PDF context...")

        context = state["pdf_context"]
        prompt = (
            f"Use only the following context:\n{context}\n\n"
            f"Question: {state['query']}"
        )

        response = self.llm.invoke(prompt)
        clean_answer = self.extract_answer_only(response)

        print(f"📘 Answer: {clean_answer}")
        state["pdf_answer"] = clean_answer
        return state


# ======================================================
# FAISS Retriever Class
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
        self.index.add(self.embeddings.astype('float32'))

    def invoke(self, query, k=1):
        query_emb = self.emb_model.encode([query])
        distances, indices = self.index.search(query_emb.astype('float32'), k)

        docs = []
        for idx in indices[0]:
            if idx < len(self.segments):
                docs.append(self.segments[idx])

        return docs


# ======================================================
# State Class for LangGraph
# ======================================================
class QAState(dict):
    query: str
    pdf_context: str
    pdf_answer: str


# ======================================================
# FastAPI Application
# ======================================================
app = FastAPI(
    title="PDF QA System API",
    description="API for question-answering system using PDF documents",
    version="1.0.0"
)

# Global system instance
pdf_qa_system = PDFQASystem()


@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    try:
        pdf_qa_system.initialize_system()
    except Exception as e:
        print(f"❌ Failed to initialize system on startup: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="running",
        model_loaded=pdf_qa_system.llm is not None,
        pdf_loaded=len(pdf_qa_system.segments) > 0
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pdf_qa_system.is_initialized else "initializing",
        model_loaded=pdf_qa_system.llm is not None,
        pdf_loaded=len(pdf_qa_system.segments) > 0
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get answer from PDF"""
    result = pdf_qa_system.ask_question(request.query)

    return QuestionResponse(
        answer=result["answer"],
        context_used=result["context_used"],
        source=result["source"]
    )


@app.post("/reload-pdf", response_model=PDFReloadResponse)
async def reload_pdf():
    """Reload the PDF document"""
    try:
        segments_count = pdf_qa_system.reload_pdf()
        return PDFReloadResponse(
            status="success",
            segments_count=segments_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload PDF: {str(e)}")


@app.get("/segments/count")
async def get_segments_count():
    """Get the number of segments in the current PDF"""
    return {"segments_count": len(pdf_qa_system.segments)}


@app.get("/segments/sample")
async def get_segments_sample(limit: int = 5):
    """Get a sample of segments from the PDF"""
    sample = pdf_qa_system.segments[:limit] if pdf_qa_system.segments else []
    return {
        "total_segments": len(pdf_qa_system.segments),
        "sample": sample
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
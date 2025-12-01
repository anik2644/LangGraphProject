# PDF QA System with LangGraph, FAISS, and Hugging Face

This repository contains a **PDF Question Answering (QA) system** built using **LangGraph**, **FAISS**, **Sentence Transformers**, and **Hugging Face** models. The system is capable of answering domain-specific questions by leveraging a PDF document, retrieving relevant context, and generating answers using a fine-tuned language model.

---

## 🧠 Overview

The goal of this project is to create a question-answering system where users can ask questions based on the content of a PDF file. The system uses **FAISS** for efficient vector search and **TinyLlama** model for text generation.

### Key Components:
- **FAISS**: Efficient vector search for retrieving relevant sections of the PDF.
- **LangGraph**: Used for building a state machine that controls the flow of PDF retrieval and answer generation.
- **Sentence Transformers**: To generate embeddings for the PDF content and query.
- **Hugging Face Pipeline**: For text generation using a fine-tuned language model.

---

## 🚀 Features

- **PDF Loading**: Load and process PDF files into segments.
- **Question Answering**: Answer questions based on PDF content.
- **PDF Segmentation**: Automatically segment the PDF based on predefined tags or split text.
- **FAISS-based Retrieval**: Use FAISS for fast and efficient retrieval of relevant PDF sections based on the query.
- **State Management with LangGraph**: Organize the QA process flow using a state machine.
- **API**: A FastAPI application that exposes endpoints for health check, asking questions, and reloading the PDF.

---

## 📚 Requirements

To run this system, you need the following libraries:

- **FastAPI**: Web framework for API services.
- **PyPDF2**: For PDF parsing.
- **FAISS**: For vector search.
- **Sentence-Transformers**: To generate embeddings for the text.
- **Hugging Face Pipeline**: For using pre-trained models like TinyLlama for question answering.
- **LangGraph**: A Python package used for state machine management.




## 🤖 Model Information

This system uses the TinyLlama model for text generation. It is a fine-tuned language model capable of generating answers based on the context provided.

## Model used:
TinyLlama/TinyLlama-1.1B-Chat-v1.0

## 🙏 Acknowledgments

Hugging Face for providing pre-trained models and pipelines.

LangGraph for providing an efficient way to handle the state machine in the system.

FAISS for enabling fast, high-quality vector retrieval from large documents.



## 📄 PDF QA System

Here is the content formatted for a Markdown file, focusing only on the provided text.

````markdown
Install the dependencies:
```bash
pip install -r requirements.txt
````

-----

## 📑 Setup and Usage

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/your-username/pdf-qa-system.git](https://github.com/your-username/pdf-qa-system.git)
    cd pdf-qa-system
    ```

2.  **Start the FastAPI Server**

    Make sure all dependencies are installed and then run the FastAPI application:

    ```bash
    uvicorn main:app --reload
    ```

    This will start the API on `http://127.0.0.1:8000`.

-----

## 🛠️ API Endpoints

### 1\. Health Check (`GET /health`)

Check the status of the system.

**Response Example:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "pdf_loaded": true
}
```

### 2\. Ask a Question (`POST /ask`)

Ask a question and get an answer based on the PDF content.

**Request Example:**

```json
{
  "query": "What is the capital of Bangladesh?"
}
```

**Response Example:**

```json
{
  "answer": "The capital of Bangladesh is Dhaka.",
  "context_used": "Some context from the PDF.",
  "source": "pdf"
}
```

-----

## 🔧 PDF QA System Workflow

1.  **PDF Loading:**

      * The PDF is loaded using **PyPDF2**, and the text is extracted.
      * The text is split into segments based on predefined tags or general text splitting.

2.  **FAISS Retrieval:**

      * **FAISS** is used to perform efficient **nearest neighbor search** to retrieve relevant PDF segments based on the query.

3.  **Answer Generation:**

      * The relevant segments are passed to the **Hugging Face model pipeline**.
      * The **TinyLlama model** generates an answer using the context from the PDF.

4.  **State Management with LangGraph:**

      * The **LangGraph** state machine manages the different steps of the QA process: retrieving relevant context, generating the answer, and delivering it to the user.

<!-- end list -->

```
```



# AI-Powered Document Question Answering System

## 📄 Description

This project is a web-based *Question Answering (QA)* application that allows users to upload their own documents (e.g., PDFs, text files) and ask natural language questions about the content. It uses:
- **Google Gemini** (Gemini 2.5 Flash) via Google Generative AI API for LLM-based answers.
- **HuggingFace Embeddings** (BGE) for semantic understanding of documents.
- **Streamlit** for a clean and interactive frontend UI.
- **LlamaIndex** to handle document parsing, chunking, embedding, and retrieval.
The system can answer context-specific queries directly from uploaded documents, without needing prior training.


---

## 🚀 Features

- 🔍 Ask questions based on the content of your uploaded document.
- 📄 Supports `.txt`, `.md`, and `.pdf` files.
- 🧠 Uses vector embeddings (semantic search) with HuggingFace.
- 🤖 Powered by Google Gemini LLM via the Generative AI API.
- 🖥️ Simple, clean web UI built with Streamlit.

📄 ## Description
This project is a web-based *Question Answering (QA)* application that allows users to upload their own documents (e.g., PDFs, text files) and ask natural language questions about the content. It uses:
Google Gemini (Gemini 2.5 Flash) via Google Generative AI API for LLM-based answers.
HuggingFace Embeddings (BGE) for semantic understanding of documents.
Streamlit for a clean and interactive frontend UI.
LlamaIndex to handle document parsing, chunking, embedding, and retrieval.
The system can answer context-specific queries directly from uploaded documents, without needing prior training.

📦 ## Features
🔍 Ask questions based on your uploaded document.
📄 Accepts .txt, .md, and .pdf files.
🧠 Uses vector embeddings for semantic similarity.
🌐 Easy-to-use local web interface with Streamlit.
💬 Real-time response using Gemini LLM.

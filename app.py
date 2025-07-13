import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google import genai
import tempfile

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

client = genai.Client(api_key=google_api_key)

llm_model = GoogleGenAI(model="gemini-2.5-flash")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm_model
Settings.embed_model = embed_model
Settings.chunk_size = 1000
Settings.chunk_overlap = 50

st.set_page_config(page_title="QA System", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Document-based Question Answering")
st.caption("Ask questions based on your uploaded document using Google Gemini + HuggingFace.")

# Sidebar
st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a .txt, .md, or .pdf file", type=["txt", "md", "pdf"])

@st.cache_resource(show_spinner="Indexing document...")
def create_index_from_file(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        documents = SimpleDirectoryReader(tmpdir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine()

query_engine = None

if uploaded_file:
    query_engine = create_index_from_file(uploaded_file)
    st.success("Document indexed successfully. You can now ask questions!")

    user_query = st.text_input("🔍 Ask your question here:")
    if user_query:
        with st.spinner("Generating answer..."):
            response = query_engine.query(user_query)
            st.markdown("## Answer")
            st.success(response.response)
else:
    st.info("Upload a document from the sidebar to begin.")
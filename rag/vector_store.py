from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Define the local directory for ChromaDB storage
PERSIST_DIRECTORY = os.path.join("data", "chroma_db")

def get_embeddings():
    """
    Initializes and returns the embedding model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(chunks):
    """
    Creates a Chroma vector store from document chunks and saves it locally.
    """
    embeddings = get_embeddings()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vector_db

def load_vector_store():
    """
    Loads an existing Chroma vector store from the local directory.
    """
    embeddings = get_embeddings()
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return None

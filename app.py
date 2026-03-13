import streamlit as st
import os
from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vector_store import create_vector_store, load_vector_store
from rag.qa_chain import get_qa_chain, answer_question

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        background-color: #262730;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    h1 {
        color: #00ffa2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        background-color: #1e1e1e;
    }
    .user-msg {
        color: #00d4ff;
        font-weight: bold;
    }
    .bot-msg {
        color: #ffffff;
    }
    .source-doc {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🤖 AI-Powered Local Document Assistant")
st.markdown("---")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join("data", "uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        if st.button("Process Document"):
            with st.spinner("Processing PDF and generating embeddings..."):
                try:
                    # 1. Load PDF
                    docs = load_pdf(file_path)
                    # 2. Split into chunks
                    chunks = split_documents(docs)
                    # 3. Create Vector Store
                    vector_db = create_vector_store(chunks)
                    st.session_state.vector_db = vector_db
                    st.success("Document processed and stored in vector database!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_store()

# --- Chat Interface ---
st.subheader("💬 Chat with your Document")

# Display chat history
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div class="chat-container"><span class="user-msg">User:</span> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container"><span class="bot-msg">AI:</span> {message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f'<div class="source-doc">Source {i+1}: {doc.page_content[:200]}...</div>', unsafe_allow_html=True)

# Question Input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if vector DB is ready
    if st.session_state.vector_db is None:
        st.warning("Please upload and process a document first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="chat-container"><span class="user-msg">User:</span> {prompt}</div>', unsafe_allow_html=True)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                qa_components = get_qa_chain(st.session_state.vector_db)
                answer, sources = answer_question(qa_components, prompt)
                
                # Add AI response to history
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                
                # Display AI response
                st.markdown(f'<div class="chat-container"><span class="bot-msg">AI:</span> {answer}</div>', unsafe_allow_html=True)
                with st.expander("View Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f'<div class="source-doc">Source {i+1}: {doc.page_content[:200]}...</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Ollama (llama3) and ChromaDB | 100% Local RAG Pipeline")

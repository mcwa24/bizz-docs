import streamlit as st
import os
import tempfile
from pathlib import Path
import logging
import nltk
from document_parser import DocumentParser
from rag_system import RAGSystem
from chat_interface import ChatInterface

# Download NLTK data for Streamlit Cloud compatibility
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="BizzDocs - Document RAG System",
    page_icon="🔹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .chat-section {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔹 BizzDocs - Document RAG System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("LLM Model")
        model_type = st.selectbox(
            "Choose LLM Model",
            ["OpenAI GPT-4o mini", "Ollama (Local)", "HuggingFace (Local)"],
            index=0
        )
        
        # API Key input
        if model_type == "OpenAI GPT-4o mini":
            api_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from OpenAI platform")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        
        # Embedding model
        st.subheader("Embedding Model")
        embedding_model = st.selectbox(
            "Choose Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "text-embedding-3-small"],
            index=0
        )
        
        # Vector database
        st.subheader("Vector Database")
        vector_db = st.selectbox(
            "Choose Vector Database",
            ["ChromaDB (Local)", "FAISS (Local)", "Pinecone (Cloud)"],
            index=0
        )
        
        # Initialize RAG system button
        if st.button("🚀 Initialize RAG System"):
            try:
                with st.spinner("Initializing RAG system..."):
                    rag_system = RAGSystem(
                        model_type=model_type,
                        embedding_model=embedding_model,
                        vector_db=vector_db
                    )
                    st.session_state.rag_system = rag_system
                    st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        # File upload section
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                help="Supported formats: PDF, Word, TXT, Markdown"
            )
            
            if uploaded_files and st.button("📚 Process Documents"):
                if not st.session_state.rag_system:
                    st.error("Please initialize RAG system first!")
                else:
                    try:
                        with st.spinner("Processing documents..."):
                            parser = DocumentParser()
                            
                            for uploaded_file in uploaded_files:
                                # Save uploaded file temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                try:
                                    # Parse document
                                    text_content = parser.parse_document(tmp_path)
                                    
                                    # Add to RAG system
                                    st.session_state.rag_system.add_document(
                                        text_content, 
                                        uploaded_file.name
                                    )
                                    
                                    st.success(f"✅ Processed: {uploaded_file.name}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                                finally:
                                    # Clean up temp file
                                    os.unlink(tmp_path)
                            
                            st.session_state.documents_loaded = True
                            st.success("🎉 All documents processed successfully!")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Document info
        if st.session_state.documents_loaded and st.session_state.rag_system:
            st.subheader("📊 Document Statistics")
            doc_count = st.session_state.rag_system.get_document_count()
            st.info(f"📚 Total documents loaded: {doc_count}")
            
            if st.button("🗑️ Clear Documents"):
                st.session_state.rag_system.clear_documents()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.header("💬 Chat Interface")
        
        if not st.session_state.rag_system:
            st.info("Please initialize RAG system first!")
        elif not st.session_state.documents_loaded:
            st.info("Please upload and process documents first!")
        else:
            # Chat interface
            chat_interface = ChatInterface(st.session_state.rag_system)
            chat_interface.render()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔹 Built with Streamlit, LlamaIndex, and ChromaDB</p>
            <p>Supporting PDF, Word, TXT, and Markdown documents</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ChromaDB
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for document processing and querying
    """
    
    def __init__(self, model_type: str = "OpenAI GPT-4o mini", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db: str = "ChromaDB (Local)"):
        
        self.model_type = model_type
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_index()
        
        # Document storage
        self.documents = []
        self.document_metadata = {}
        
        logger.info("RAG system initialized successfully")
    
    def _setup_llm(self):
        """Setup the language model"""
        try:
            if "OpenAI" in self.model_type:
                # Use OpenAI GPT-4o mini
                self.llm = OpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=2000
                )
                logger.info("OpenAI LLM initialized")
            else:
                # Fallback to OpenAI for now
                self.llm = OpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    max_tokens=2000
                )
                logger.info("Using OpenAI LLM as fallback")
        except Exception as e:
            logger.error(f"Error setting up LLM: {str(e)}")
            raise
    
    def _setup_embeddings(self):
        """Setup the embedding model"""
        try:
            if "sentence-transformers" in self.embedding_model:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("HuggingFace embeddings initialized")
            else:
                # Default to HuggingFace
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Using HuggingFace embeddings as default")
        except Exception as e:
            logger.error(f"Error setting up embeddings: {str(e)}")
            raise
    
    def _setup_vector_store(self):
        """Setup the vector database"""
        try:
            if "ChromaDB" in self.vector_db:
                # Initialize ChromaDB
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=ChromaSettings(
                        anonymized_telemetry=False
                    )
                )
                
                # Create or get collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="bizz_docs",
                    metadata={"description": "Bizz Documents Collection"}
                )
                
                self.vector_store = ChromaVectorStore(
                    chroma_collection=self.collection
                )
                
                logger.info("ChromaDB vector store initialized")
            else:
                # Default to ChromaDB
                self.chroma_client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=ChromaSettings(
                        anonymized_telemetry=False
                    )
                )
                self.collection = self.chroma_client.get_or_create_collection(
                    name="bizz_docs",
                    metadata={"description": "Bizz Documents Collection"}
                )
                self.vector_store = ChromaVectorStore(
                    chroma_collection=self.collection
                )
                logger.info("Using ChromaDB as default vector store")
                
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    def _setup_index(self):
        """Setup the LlamaIndex"""
        try:
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Initialize index
            self.index = None
            self.query_engine = None
            
            logger.info("LlamaIndex setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up LlamaIndex: {str(e)}")
            raise
    
    def add_document(self, text_content: str, filename: str):
        """
        Add a document to the RAG system
        
        Args:
            text_content: Extracted text from the document
            filename: Name of the uploaded file
        """
        try:
            # Create document
            doc = Document(
                text=text_content,
                metadata={
                    "filename": filename,
                    "upload_time": datetime.now().isoformat(),
                    "content_length": len(text_content)
                }
            )
            
            # Add to documents list
            self.documents.append(doc)
            self.document_metadata[filename] = {
                "upload_time": datetime.now().isoformat(),
                "content_length": len(text_content),
                "chunks": 0
            }
            
            # Update index
            self._update_index()
            
            logger.info(f"Document {filename} added successfully")
            
        except Exception as e:
            logger.error(f"Error adding document {filename}: {str(e)}")
            raise
    
    def _update_index(self):
        """Update the vector index with new documents"""
        try:
            if not self.documents:
                return
            
            # Create new index with all documents
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)]
            )
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine.from_args(
                self.index.as_retriever(similarity_top_k=5),
                llm=self.llm
            )
            
            # Update metadata with chunk count
            for doc in self.documents:
                filename = doc.metadata.get("filename")
                if filename in self.document_metadata:
                    self.document_metadata[filename]["chunks"] = len(
                        self.index.docstore.docs
                    )
            
            logger.info("Index updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating index: {str(e)}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (e.g., 'sr', 'en', 'de', 'fr', 'es')
        """
        # Simple language detection based on common words
        text_lower = text.lower()
        
        # Serbian/Croatian/Bosnian
        serbian_words = ['šta', 'koji', 'kako', 'gde', 'kada', 'zašto', 'ko', 'što', 'gdje', 'kako', 'zašto']
        if any(word in text_lower for word in serbian_words):
            return 'sr'
        
        # German
        german_words = ['was', 'welche', 'wie', 'wo', 'wann', 'warum', 'wer', 'wohin', 'woher']
        if any(word in text_lower for word in german_words):
            return 'de'
        
        # French
        french_words = ['quoi', 'quel', 'comment', 'où', 'quand', 'pourquoi', 'qui', 'comment']
        if any(word in text_lower for word in french_words):
            return 'fr'
        
        # Spanish
        spanish_words = ['qué', 'cuál', 'cómo', 'dónde', 'cuándo', 'por qué', 'quién', 'cómo']
        if any(word in text_lower for word in spanish_words):
            return 'es'
        
        # Default to English
        return 'en'
    
    def _get_language_instruction(self, language: str, analysis_type: str) -> str:
        """
        Get language-specific instruction for the AI
        
        Args:
            language: Detected language code
            analysis_type: Type of analysis
            
        Returns:
            Language instruction string
        """
        instructions = {
            'sr': {
                'general': 'Odgovori na srpskom jeziku. Koristi srpsku gramatiku i pravopis.',
                'summary': 'Napravi detaljan sažetak dokumenta na srpskom jeziku. Fokusiraj se na glavne teme, ključne nalaze i važne detalje.',
                'key_points': 'Izvuci i navedi ključne tačke, glavne ideje i kritične informacije iz dokumenta na srpskom jeziku. Budi koncizan ali temeljan.',
                'extract_info': 'Izvuci specifične informacije iz dokumenta na osnovu ovog zahteva na srpskom jeziku. Daj detaljne, tačne informacije sa relevantnim kontekstom.'
            },
            'de': {
                'general': 'Antworte auf Deutsch. Verwende deutsche Grammatik und Rechtschreibung.',
                'summary': 'Erstelle eine detaillierte Zusammenfassung des Dokuments auf Deutsch. Konzentriere dich auf Hauptthemen, wichtige Erkenntnisse und wichtige Details.',
                'key_points': 'Extrahiere und liste die wichtigsten Punkte, Hauptideen und kritische Informationen aus dem Dokument auf Deutsch auf. Sei prägnant, aber gründlich.',
                'extract_info': 'Extrahiere spezifische Informationen aus dem Dokument basierend auf dieser Anfrage auf Deutsch. Gib detaillierte, genaue Informationen mit relevantem Kontext.'
            },
            'fr': {
                'general': 'Réponds en français. Utilise la grammaire et l\'orthographe françaises.',
                'summary': 'Fais un résumé détaillé du document en français. Concentre-toi sur les sujets principaux, les découvertes clés et les détails importants.',
                'key_points': 'Extrais et liste les points clés, les idées principales et les informations critiques du document en français. Sois concis mais approfondi.',
                'extract_info': 'Extrais des informations spécifiques du document basées sur cette demande en français. Fournis des informations détaillées et précises avec un contexte pertinent.'
            },
            'es': {
                'general': 'Responde en español. Usa la gramática y ortografía españolas.',
                'summary': 'Haz un resumen detallado del documento en español. Céntrate en los temas principales, hallazgos clave y detalles importantes.',
                'key_points': 'Extrae y lista los puntos clave, ideas principales e información crítica del documento en español. Sé conciso pero exhaustivo.',
                'extract_info': 'Extrae información específica del documento basada en esta solicitud en español. Proporciona información detallada y precisa con contexto relevante.'
            },
            'en': {
                'general': 'Answer in English. Use proper English grammar and spelling.',
                'summary': 'Please provide a comprehensive summary of the document content. Focus on main topics, key findings, and important details.',
                'key_points': 'Please extract and list the key points, main ideas, and critical information from the document. Be concise but thorough.',
                'extract_info': 'Please extract specific information from the document based on this request. Provide detailed, accurate information with relevant context.'
            }
        }
        
        return instructions.get(language, instructions['en']).get(analysis_type, instructions['en']['general'])
    
    def query(self, question: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            analysis_type: Type of analysis ("general", "summary", "key_points", "extract_info")
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            if not self.query_engine:
                raise ValueError("No documents loaded. Please add documents first.")
            
            # Detect language of the question
            detected_language = self._detect_language(question)
            logger.info(f"Detected language: {detected_language} for question: {question[:50]}...")
            
            # Get language-specific instruction
            language_instruction = self._get_language_instruction(detected_language, analysis_type)
            
            # Create enhanced question with language instruction
            enhanced_question = f"{language_instruction}\n\nQuestion: {question}"
            logger.info(f"Enhanced question with {detected_language} instruction")
            
            # Get response
            response = self.query_engine.query(enhanced_question)
            
            # Get source documents
            source_docs = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_docs.append({
                        "filename": node.metadata.get("filename", "Unknown"),
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    })
            
            return {
                "response": str(response),
                "analysis_type": analysis_type,
                "source_documents": source_docs,
                "query_time": datetime.now().isoformat(),
                "total_documents": len(self.documents)
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "error": str(e),
                "response": "Sorry, I encountered an error while processing your request.",
                "analysis_type": analysis_type,
                "query_time": datetime.now().isoformat()
            }
    
    def get_document_count(self) -> int:
        """Get total number of loaded documents"""
        return len(self.documents)
    
    def get_document_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded documents"""
        return [
            {
                "filename": filename,
                **metadata
            }
            for filename, metadata in self.document_metadata.items()
        ]
    
    def clear_documents(self):
        """Clear all documents and reset the system"""
        try:
            self.documents = []
            self.document_metadata = {}
            self.index = None
            self.query_engine = None
            
            # Clear ChromaDB collection
            if hasattr(self, 'collection'):
                self.chroma_client.delete_collection("bizz_docs")
                self.collection = self.chroma_client.create_collection(
                    name="bizz_docs",
                    metadata={"description": "Bizz Documents Collection"}
                )
            
            logger.info("All documents cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            raise
    
    def save_system_state(self, filepath: str = "./rag_system_state.json"):
        """Save system state to file"""
        try:
            state = {
                "model_type": self.model_type,
                "embedding_model": self.embedding_model,
                "vector_db": self.vector_db,
                "document_count": len(self.documents),
                "documents": self.document_metadata,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            raise

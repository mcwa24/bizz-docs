import streamlit as st
from typing import Dict, Any, List
import json
from datetime import datetime

class ChatInterface:
    """
    Chat interface for interacting with the RAG system
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
        # Initialize chat history in session state if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def render(self):
        """Render the chat interface"""
        
        # Analysis type selector
        st.subheader("🔍 Analysis Type")
        analysis_type = st.selectbox(
            "Choose how you want to analyze your documents:",
            [
                "general",
                "summary", 
                "key_points",
                "extract_info"
            ],
            format_func=lambda x: {
                "general": "💬 General Question",
                "summary": "📋 Comprehensive Summary", 
                "key_points": "🎯 Key Points & Insights",
                "extract_info": "🔍 Extract Specific Information"
            }[x],
            help="Select the type of analysis you want to perform"
        )
        
        # Display analysis type description
        self._show_analysis_description(analysis_type)
        
        # Chat input
        st.subheader("💬 Ask Questions About Your Documents")
        
        # Question input
        user_question = st.text_area(
            "Type your question here:",
            placeholder="e.g., What are the main points of this document?",
            height=100,
            help="Ask any question about your uploaded documents"
        )
        
        # Send button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            send_button = st.button("🚀 Send Question", type="primary", use_container_width=True)
        
        # Process question
        if send_button and user_question.strip():
            self._process_question(user_question, analysis_type)
        
        # Display chat history
        self._display_chat_history()
        
        # Quick action buttons
        self._show_quick_actions(analysis_type)
    
    def _show_analysis_description(self, analysis_type: str):
        """Show description of the selected analysis type"""
        descriptions = {
            "general": "Ask any general question about your documents. The AI will search through your documents to find relevant information.",
            "summary": "Get a comprehensive summary of your documents. Perfect for understanding the main content and key findings.",
            "key_points": "Extract the most important points, insights, and critical information from your documents.",
            "extract_info": "Extract specific information based on your request. Great for finding particular details or data."
        }
        
        st.info(f"**{analysis_type.replace('_', ' ').title()}**: {descriptions[analysis_type]}")
    
    def _process_question(self, question: str, analysis_type: str):
        """Process user question and get response from RAG system"""
        try:
            # Check if RAG system is properly initialized
            if not self.rag_system:
                st.error("❌ RAG system is not initialized. Please initialize it first in the Configuration tab.")
                return
            
            # Check if documents are loaded
            if not hasattr(self.rag_system, 'documents') or not self.rag_system.documents:
                st.error("❌ No documents are loaded. Please upload and process documents first.")
                return
            
            with st.spinner("🤔 AI is thinking..."):
                # Get response from RAG system
                response = self.rag_system.query(question, analysis_type)
                
                # Check if response contains an error
                if 'error' in response:
                    st.error(f"❌ Error from RAG system: {response['error']}")
                    # Still add to chat history for debugging
                    chat_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "response": response,
                        "analysis_type": analysis_type
                    }
                    st.session_state.chat_history.append(chat_entry)
                    return
                
                # Add to chat history
                chat_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "response": response,
                    "analysis_type": analysis_type
                }
                
                st.session_state.chat_history.append(chat_entry)
                
                # Show success message
                st.success("✅ Question processed successfully!")
                
                # Rerun to refresh the display
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            # Log the full error for debugging
            st.exception(e)
    
    def _display_chat_history(self):
        """Display the chat history"""
        if not st.session_state.chat_history:
            return
        
        st.subheader("📚 Chat History")
        
        # Reverse order to show latest first
        for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 {chat_entry['question'][:50]}...", expanded=False):
                # Question
                st.markdown("**Question:**")
                st.write(chat_entry['question'])
                
                # Analysis type
                st.markdown(f"**Analysis Type:** {chat_entry['analysis_type'].replace('_', ' ').title()}")
                
                # Response
                st.markdown("**AI Response:**")
                if 'error' in chat_entry['response']:
                    st.error(chat_entry['response']['response'])
                else:
                    st.write(chat_entry['response']['response'])
                
                # Source documents
                if 'source_documents' in chat_entry['response'] and chat_entry['response']['source_documents']:
                    st.markdown("**Source Documents:**")
                    for source in chat_entry['response']['source_documents']:
                        st.markdown(f"- **{source['filename']}**: {source['content']}")
                
                # Timestamp
                timestamp = datetime.fromisoformat(chat_entry['timestamp'])
                st.caption(f"Asked at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.chat_history.pop(-(i+1))
                    st.rerun()
        
        # Clear all history button
        if st.button("🗑️ Clear All History"):
            st.session_state.chat_history = []
            st.rerun()
    
    def _show_quick_actions(self, analysis_type: str):
        """Show quick action buttons for common questions"""
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate Summary", use_container_width=True):
                self._process_question("Please provide a comprehensive summary of all documents", "summary")
            
            if st.button("🎯 Extract Key Points", use_container_width=True):
                self._process_question("What are the key points and main insights from these documents?", "key_points")
        
        with col2:
            if st.button("🔍 Find Important Info", use_container_width=True):
                self._process_question("What are the most important pieces of information in these documents?", "extract_info")
            
            if st.button("❓ General Analysis", use_container_width=True):
                self._process_question("What can you tell me about these documents?", "general")
    
    def export_chat_history(self) -> str:
        """Export chat history as JSON string"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_conversations": len(st.session_state.chat_history),
                "conversations": st.session_state.chat_history
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error exporting chat history: {str(e)}"
    
    def import_chat_history(self, json_data: str):
        """Import chat history from JSON string"""
        try:
            import_data = json.loads(json_data)
            if 'conversations' in import_data:
                st.session_state.chat_history = import_data['conversations']
                return True
            return False
        except Exception as e:
            st.error(f"Error importing chat history: {str(e)}")
            return False

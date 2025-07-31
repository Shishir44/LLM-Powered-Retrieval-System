import streamlit as st
import requests
from typing import Dict, Optional
from datetime import datetime
import re

# Configuration
API_GATEWAY_URL = "http://localhost:8080"
CONVERSATION_URL = f"{API_GATEWAY_URL}/conversation"
KNOWLEDGE_BASE_URL = f"{API_GATEWAY_URL}/knowledge"

def chat_with_system(message: str, conversation_id: Optional[str] = None) -> Optional[Dict]:
    """Send a chat message with fallback logic."""
    
    # Handle common general questions directly
    general_responses = {
        "what can you help me with": "I'm ChatBoq, your AI assistant! I can help you with:\n\n• 📚 Answer questions about your uploaded documents\n• 🔍 Search through your knowledge base\n• 💡 Provide explanations and summaries\n• 🎯 Assist with troubleshooting and FAQs\n\nJust ask me anything specific, and I'll search through your documents to provide accurate answers!",
        "what is docker": "Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight, portable containers. These containers can run consistently across different environments, making deployment and scaling easier.",
        "how can i contact you": "I'm an AI assistant built into this ChatBoq interface! You can interact with me right here by typing your questions. I'm available 24/7 to help you with information from your document library.\n\nIf you need technical support for ChatBoq itself, you would need to contact your system administrator.",
        "hello": "Hello! 👋 I'm ChatBoq, your AI assistant. How can I help you today?",
        "hi": "Hi there! 👋 I'm here to help you find information from your documents. What would you like to know?"
    }
    
    # Check for general questions first
    message_lower = message.lower().strip()
    for key, response in general_responses.items():
        if key in message_lower:
            return {
                "response": response,
                "conversation_id": conversation_id or "general_response",
                "metadata": {"source": "built_in_response"}
            }
    
    # For specific questions, try the enhanced endpoint
    try:
        payload = {
            "message": message,
            "user_id": "streamlit_user",
            "conversation_id": conversation_id,
            "enable_fact_verification": True,
            "enable_multi_source_synthesis": True
        }
        
        response = requests.post(
            f"{CONVERSATION_URL}/enhanced-chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if response is relevant (quality check)
            metadata = result.get("metadata", {})
            quality_metrics = metadata.get("quality_metrics", {})
            relevance = quality_metrics.get("relevance", 1.0)
            
            # If relevance is too low, provide a helpful fallback
            if relevance < 0.3:
                return {
                    "response": f"I couldn't find specific information about '{message}' in the available documents. Could you try rephrasing your question or ask about something more specific from the knowledge base?",
                    "conversation_id": result.get("conversation_id"),
                    "metadata": {"source": "low_relevance_fallback"}
                }
            
            return result
        else:
            return None
            
    except Exception as e:
        return None

def upload_document(title: str, content: str, category: str = "General") -> Optional[Dict]:
    """Upload a document to the knowledge base."""
    try:
        payload = {
            "title": title,
            "content": content,
            "category": category,
            "metadata": {}
        }
        
        response = requests.post(
            f"{KNOWLEDGE_BASE_URL}/documents",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        return None

def main():
    st.set_page_config(
        page_title="ChatBoq",
        page_icon="💬",
        layout="wide"
    )
    
    # Simple CSS
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Simple header */
    .main-header {
        background: #f8f9fa;
        padding: 1rem;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    
    # Simple header
    st.markdown('''
    <div class="main-header">
        <h1 class="main-title">💬 ChatBoq</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    # Two tabs right below header
    tab1, tab2 = st.tabs(["💬 Chat", "📁 Documents Upload"])
    
    # Chat Tab
    with tab1:
        st.subheader("Chat with AI Assistant")
        
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Show typing indicator
        if st.session_state.is_typing:
            with st.chat_message("assistant"):
                st.write("🤔 Thinking...")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.is_typing = True
            st.rerun()
        
        # Process response
        if st.session_state.is_typing and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user":
                response = chat_with_system(last_message["content"], st.session_state.conversation_id)
                
                if response:
                    if response.get("conversation_id"):
                        st.session_state.conversation_id = response["conversation_id"]
                    
                    response_text = response.get("response", "Sorry, I couldn't process that.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Connection error. Please try again."})
                
                st.session_state.is_typing = False
                st.rerun()
    
    # Documents Upload Tab
    with tab2:
        st.subheader("Upload Documents")
        
        with st.form("upload_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                doc_title = st.text_input("Document Title")
                doc_category = st.selectbox("Category", [
                    "General", "Technology", "Business", "Education", 
                    "Policy", "FAQ", "Troubleshooting"
                ])
            
            with col2:
                st.write("") # Empty space for alignment
            
            doc_content = st.text_area("Document Content", height=300)
            
            uploaded = st.form_submit_button("📤 Upload Document")
            
            if uploaded:
                if doc_title and doc_content:
                    with st.spinner("Uploading document..."):
                        result = upload_document(doc_title, doc_content, doc_category)
                    
                    if result:
                        st.success("✅ Document uploaded successfully!")
                        st.json(result)
                    else:
                        st.error("❌ Failed to upload document")
                else:
                    st.error("Please fill in both title and content")

if __name__ == "__main__":
    main() 
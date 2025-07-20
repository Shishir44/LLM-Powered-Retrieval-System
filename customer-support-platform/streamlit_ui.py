import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import time

# Configuration
KNOWLEDGE_BASE_API_URL = "http://127.0.0.1:8001"  # Knowledge Base Service
CONVERSATION_API_URL = "http://127.0.0.1:8002"   # Conversation Service
API_GATEWAY_URL = "http://127.0.0.1:8000"        # API Gateway

# Initialize session state
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = KNOWLEDGE_BASE_API_URL

def make_authenticated_request(endpoint: str, method: str = "GET", data: dict = None, base_url: str = None) -> Dict:
    """Make API request with authentication"""
    try:
        url = f"{base_url or st.session_state.api_base_url}{endpoint}"
        headers = {}
        
        if st.session_state.auth_token:
            headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
        
        if method == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        elif method == "PUT":
            headers["Content-Type"] = "application/json"
            response = requests.put(url, json=data, headers=headers)
        else:
            response = requests.get(url, headers=headers, params=data)
        
        if response.status_code in [200, 201]:
            return {"success": True, "data": response.json()}
        elif response.status_code == 401:
            st.session_state.auth_token = None
            return {"success": False, "error": "Authentication required"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def login_section():
    """Handle user authentication"""
    st.sidebar.header("🔐 Authentication")
    
    if st.session_state.auth_token:
        st.sidebar.success("✅ Authenticated")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.auth_token = None
            st.session_state.user_info = None
            st.rerun()
        return True
    
    # Simple token input for demo purposes
    # In a real implementation, you'd have proper login forms
    with st.sidebar.form("auth_form"):
        st.write("**Demo Authentication**")
        demo_token = st.text_input("Auth Token", type="password", 
                                 placeholder="Enter demo token or leave empty for demo mode")
        
        if st.form_submit_button("🔑 Authenticate"):
            if demo_token:
                st.session_state.auth_token = demo_token
            else:
                # Demo mode - create a dummy token
                st.session_state.auth_token = "demo_token_123"
                st.session_state.user_info = {"username": "demo_user", "role": "admin"}
            
            st.rerun()
    
    st.sidebar.info("💡 Leave token empty for demo mode")
    return False

def main():
    st.set_page_config(
        page_title="Customer Support Knowledge Base",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Customer Support Knowledge Base")
    st.markdown("### LangChain-powered Document Management & Search System")
    st.markdown("---")
    
    # Authentication
    is_authenticated = login_section()
    
    if not is_authenticated:
        st.info("👈 Please authenticate using the sidebar to access the knowledge base.")
        
        # Show system information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Knowledge Base Service**
            - Document storage & retrieval
            - Vector search with LangChain
            - Hybrid search (vector + keyword)
            - Category management
            """)
        
        with col2:
            st.markdown("""
            **💬 Conversation Service**
            - RAG-powered responses
            - Context-aware conversations
            - Multi-turn dialogue
            - Response generation
            """)
        
        with col3:
            st.markdown("""
            **🚪 API Gateway**
            - Unified API access
            - Authentication & authorization
            - Request routing
            - Rate limiting
            """)
        
        return
    
    # Service selection
    with st.sidebar:
        st.header("🌐 Service Configuration")
        service_choice = st.selectbox(
            "Select Service",
            ["Knowledge Base", "Conversation", "API Gateway"],
            index=0
        )
        
        if service_choice == "Knowledge Base":
            st.session_state.api_base_url = KNOWLEDGE_BASE_API_URL
        elif service_choice == "Conversation":
            st.session_state.api_base_url = CONVERSATION_API_URL
        else:
            st.session_state.api_base_url = API_GATEWAY_URL
        
        st.info(f"Using: {st.session_state.api_base_url}")
        
        # Health check
        if st.button("🔍 Health Check"):
            result = make_authenticated_request("/health")
            if result["success"]:
                st.success("✅ Service is healthy")
                st.json(result["data"])
            else:
                st.error(f"❌ Service unavailable: {result['error']}")
    
    # Main interface based on selected service
    if service_choice == "Knowledge Base":
        knowledge_base_interface()
    elif service_choice == "Conversation":
        conversation_interface()
    else:
        api_gateway_interface()

def knowledge_base_interface():
    """Knowledge Base Service Interface"""
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Documents", 
        "🔍 Search", 
        "📊 Categories", 
        "📈 Analytics",
        "⚙️ Management"
    ])
    
    with tab1:
        st.subheader("Document Management")
        
        # Create new document
        with st.expander("➕ Add New Document", expanded=False):
            with st.form("create_document"):
                col1, col2 = st.columns(2)
                
                with col1:
                    title = st.text_input("Document Title*", placeholder="Enter document title")
                    category = st.selectbox("Category*", [
                        "FAQ", "Policy", "Procedure", "Troubleshooting", 
                        "Product Info", "Technical", "General", "Other"
                    ])
                    subcategory = st.text_input("Subcategory", placeholder="Optional subcategory")
                
                with col2:
                    tags = st.text_input("Tags", placeholder="tag1, tag2, tag3")
                    priority = st.selectbox("Priority", ["Low", "Medium", "High"])
                    source = st.text_input("Source", placeholder="Document source")
                
                content = st.text_area("Document Content*", height=200, 
                                     placeholder="Enter the full document content here...")
                
                # Additional metadata
                st.write("**Additional Metadata**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    version = st.text_input("Version", placeholder="1.0")
                with col2:
                    author = st.text_input("Author", placeholder="Document author")
                with col3:
                    department = st.text_input("Department", placeholder="Owning department")
                
                if st.form_submit_button("📝 Create Document", use_container_width=True):
                    if title and content and category:
                        metadata = {
                            "priority": priority,
                            "source": source,
                            "version": version,
                            "author": author,
                            "department": department,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                        
                        document_data = {
                            "title": title,
                            "content": content,
                            "category": category,
                            "subcategory": subcategory if subcategory else None,
                            "tags": tags_list,
                            "metadata": metadata
                        }
                        
                        with st.spinner("Creating document..."):
                            result = make_authenticated_request("/documents", "POST", document_data)
                        
                        if result["success"]:
                            st.success("✅ Document created successfully!")
                            st.json(result["data"])
                        else:
                            st.error(f"❌ Failed to create document: {result['error']}")
                    else:
                        st.warning("⚠️ Please fill in all required fields (marked with *)")
        
        # List existing documents
        st.write("**Recent Documents**")
        if st.button("🔄 Refresh Documents"):
            # This would typically call an endpoint to list documents
            st.info("Document listing endpoint would be implemented here")

    with tab2:
        st.subheader("🔍 Knowledge Base Search")
        
        # Search interface
        search_query = st.text_input("Search Query", 
                                   placeholder="Enter your search query...",
                                   help="Use natural language to search through documents")
        
        # Search filters
        with st.expander("🎛️ Search Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_category = st.selectbox("Category Filter", 
                                             ["All"] + ["FAQ", "Policy", "Procedure", "Troubleshooting", 
                                                       "Product Info", "Technical", "General", "Other"])
            
            with col2:
                filter_subcategory = st.text_input("Subcategory Filter")
            
            with col3:
                filter_tags = st.text_input("Tags Filter", placeholder="tag1,tag2")
            
            col1, col2 = st.columns(2)
            with col1:
                search_limit = st.slider("Max Results", 1, 50, 10)
            with col2:
                include_content = st.checkbox("Include Full Content", value=True)
        
        if st.button("🔍 Search", use_container_width=True) and search_query:
            search_params = {
                "q": search_query,
                "limit": search_limit,
                "include_content": include_content
            }
            
            if filter_category != "All":
                search_params["category"] = filter_category
            if filter_subcategory:
                search_params["subcategory"] = filter_subcategory
            if filter_tags:
                search_params["tags"] = filter_tags
            
            with st.spinner("Searching..."):
                result = make_authenticated_request("/search", "GET", search_params)
            
            if result["success"]:
                search_data = result["data"]
                
                st.success(f"✅ Found {search_data['total']} results")
                
                # Display results
                for i, doc in enumerate(search_data["results"], 1):
                    with st.expander(f"📄 {doc['title']} (Score: {doc.get('score', 'N/A')})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if doc.get('content') and include_content:
                                st.write("**Content:**")
                                st.write(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                        
                        with col2:
                            st.write("**Metadata:**")
                            st.write(f"**Category:** {doc['category']}")
                            if doc.get('subcategory'):
                                st.write(f"**Subcategory:** {doc['subcategory']}")
                            if doc.get('tags'):
                                st.write(f"**Tags:** {', '.join(doc['tags'])}")
                            if doc.get('score'):
                                st.write(f"**Relevance:** {doc['score']:.3f}")
                
                # Search metadata
                if search_data.get("metadata"):
                    st.write("**Search Metadata:**")
                    st.json(search_data["metadata"])
            
            else:
                st.error(f"❌ Search failed: {result['error']}")
    
    with tab3:
        st.subheader("📊 Categories & Organization")
        
        if st.button("📋 Load Categories"):
            result = make_authenticated_request("/categories")
            
            if result["success"]:
                categories_data = result["data"]
                
                if categories_data.get("categories"):
                    df = pd.DataFrame(categories_data["categories"])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Categories Overview**")
                        st.dataframe(df, use_container_width=True)
                    
                    with col2:
                        st.write("**Category Distribution**")
                        if len(df) > 0:
                            st.bar_chart(df.set_index("category")["count"])
                else:
                    st.info("No categories found")
            else:
                st.error(f"❌ Failed to load categories: {result['error']}")
    
    with tab4:
        st.subheader("📈 Analytics & Insights")
        st.info("📊 Analytics features would be implemented here")
        
        # Placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", "0", "0")
        with col2:
            st.metric("Categories", "0", "0")
        with col3:
            st.metric("Search Queries", "0", "0")
        with col4:
            st.metric("Active Users", "0", "0")
    
    with tab5:
        st.subheader("⚙️ System Management")
        
        st.write("**Document Management**")
        
        # Get specific document
        with st.expander("🔍 Get Document by ID"):
            doc_id = st.text_input("Document ID")
            if st.button("📄 Get Document") and doc_id:
                result = make_authenticated_request(f"/documents/{doc_id}")
                
                if result["success"]:
                    st.success("✅ Document found")
                    st.json(result["data"])
                else:
                    st.error(f"❌ {result['error']}")
        
        # Delete document
        with st.expander("🗑️ Delete Document", expanded=False):
            st.warning("⚠️ This action cannot be undone!")
            delete_doc_id = st.text_input("Document ID to Delete")
            confirm_delete = st.text_input("Type 'DELETE' to confirm")
            
            if st.button("🗑️ Delete Document") and delete_doc_id and confirm_delete == "DELETE":
                result = make_authenticated_request(f"/documents/{delete_doc_id}", "DELETE")
                
                if result["success"]:
                    st.success("✅ Document deleted successfully")
                    st.json(result["data"])
                else:
                    st.error(f"❌ {result['error']}")

def conversation_interface():
    """Conversation Service Interface"""
    
    st.subheader("💬 Conversation Service")
    st.info("🚧 Conversation interface would be implemented here")
    
    # Placeholder for conversation interface
    tab1, tab2 = st.tabs(["💬 Chat", "📋 History"])
    
    with tab1:
        st.write("**AI-Powered Support Chat**")
        
        # Chat input
        user_message = st.text_area("Your Message", placeholder="Ask me anything about our products or services...")
        
        if st.button("📤 Send Message"):
            if user_message:
                st.info("🤖 This would send the message to the conversation service")
            else:
                st.warning("Please enter a message")
    
    with tab2:
        st.write("**Conversation History**")
        st.info("Chat history would be displayed here")

def api_gateway_interface():
    """API Gateway Interface"""
    
    st.subheader("🚪 API Gateway")
    st.info("🚧 API Gateway interface would be implemented here")
    
    # Placeholder for API gateway management
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔐 Auth", "⚙️ Config"])
    
    with tab1:
        st.write("**Gateway Overview**")
        st.info("Service status and routing information would be displayed here")
    
    with tab2:
        st.write("**Authentication Management**")
        st.info("User and token management would be implemented here")
    
    with tab3:
        st.write("**Gateway Configuration**")
        st.info("Routing rules and service configuration would be shown here")

if __name__ == "__main__":
    main()
import streamlit as st
import requests
from typing import Dict, List, Optional

# Configuration
API_GATEWAY_URL = "http://localhost:8080"
KNOWLEDGE_BASE_URL = f"{API_GATEWAY_URL}/knowledge"
CONVERSATION_URL = f"{API_GATEWAY_URL}/conversation"

def check_service_health() -> Dict[str, bool]:
    """Check the health status of all services."""
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("services", {})
        return {}
    except Exception as e:
        st.error(f"Failed to check service health: {e}")
        return {}

def create_document(title: str, content: str, category: str, subcategory: Optional[str] = None, tags: Optional[List[str]] = None) -> Optional[Dict[str, str]]:
    """Create a new document in the knowledge base."""
    try:
        payload: Dict[str, str | List[str] | Dict[str, str] | None] = {
            "title": title,
            "content": content,
            "category": category,
            "subcategory": subcategory,
            "tags": tags or [],
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
            st.error(f"Failed to create document: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error creating document: {e}")
        return None

def search_documents(query: str, category: Optional[str] = None, limit: int = 10) -> Optional[Dict[str, str | int | List[Dict[str, str]]]]:
    """Search documents in the knowledge base."""
    try:
        params: Dict[str, str | int] = {
            "q": query,
            "limit": limit
        }
        if category:
            params["category"] = category
            
        response = requests.get(
            f"{KNOWLEDGE_BASE_URL}/search",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return None

def get_document_stats() -> Optional[Dict[str, int | Dict[str, int]]]:
    """Get knowledge base statistics."""
    try:
        response = requests.get(f"{KNOWLEDGE_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get stats: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return None

def list_documents(category: Optional[str] = None, limit: int = 20) -> Optional[Dict[str, str | int | List[Dict[str, str | int]]]]:
    """List all documents."""
    try:
        params: Dict[str, str | int] = {"limit": limit}
        if category:
            params["category"] = category
            
        response = requests.get(
            f"{KNOWLEDGE_BASE_URL}/documents",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to list documents: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return None

def chat_with_system(message: str, conversation_id: Optional[str] = None) -> Optional[Dict[str, str | Dict[str, str]]]:
    """Send a chat message to the conversation service."""
    try:
        payload: Dict[str, str | Dict[str, str] | None] = {
            "message": message,
            "conversation_id": conversation_id,
            "context": {}
        }
        
        response = requests.post(
            f"{CONVERSATION_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Chat failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return None

def main():
    st.set_page_config(
        page_title="LLM-Powered Retrieval System UI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM-Powered Retrieval System")
    st.markdown("Simple UI for testing the microservices functionality")
    
    # Sidebar with service status
    with st.sidebar:
        st.header("🔧 Service Status")
        if st.button("🔄 Refresh Status"):
            st.rerun()
            
        service_health = check_service_health()
        
        for service, healthy in service_health.items():
            status_icon = "✅" if healthy else "❌"
            st.write(f"{status_icon} {service.replace('_', ' ').title()}")
        
        if not service_health:
            st.error("❌ API Gateway not accessible")
            st.info("Make sure the services are running:\n- API Gateway: http://localhost:8000\n- Knowledge Base: http://localhost:8002\n- Conversation: http://localhost:8001")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Knowledge Base", "💬 Chat", "📊 Statistics", "📋 Document List"])
    
    # Knowledge Base Tab
    with tab1:
        st.header("📚 Knowledge Base Management")
        
        # Document creation
        st.subheader("➕ Create New Document")
        with st.form("create_document"):
            col1, col2 = st.columns(2)
            
            with col1:
                doc_title = st.text_input("Document Title", placeholder="Enter document title")
                doc_category = st.selectbox("Category", ["Technology", "Science", "Business", "Education", "Other"])
                doc_subcategory = st.text_input("Subcategory (Optional)", placeholder="Enter subcategory")
            
            with col2:
                doc_tags = st.text_input("Tags (comma-separated)", placeholder="tag1, tag2, tag3")
            
            doc_content = st.text_area("Document Content", height=200, placeholder="Enter the document content here...")
            
            submit_doc = st.form_submit_button("🚀 Create Document")
            
            if submit_doc and doc_title and doc_content and doc_category:
                tags_list = [tag.strip() for tag in doc_tags.split(",") if tag.strip()] if doc_tags else []
                
                with st.spinner("Creating document..."):
                    result = create_document(
                        title=doc_title,
                        content=doc_content,
                        category=doc_category,
                        subcategory=doc_subcategory or None,
                        tags=tags_list
                    )
                
                if result:
                    st.success(f"✅ Document created successfully! ID: {result.get('id', 'Unknown')}")
                    st.json(result)
        
        st.divider()
        
        # Document search
        st.subheader("🔍 Search Documents")
        with st.form("search_documents"):
            col1, col2 = st.columns(2)
            
            with col1:
                search_query = st.text_input("Search Query", placeholder="Enter your search query")
                search_category = st.selectbox("Filter by Category", ["All", "Technology", "Science", "Business", "Education", "Other"])
            
            with col2:
                search_limit = st.slider("Number of Results", min_value=1, max_value=20, value=10)
            
            search_submit = st.form_submit_button("🔍 Search")
            
            if search_submit and search_query:
                category_filter = None if search_category == "All" else search_category
                
                with st.spinner("Searching documents..."):
                    results = search_documents(search_query, category_filter, search_limit)
                
                if results:
                    total = results.get('total', 0)
                    st.success(f"Found {total} documents")
                    
                    results_list = results.get('results', [])
                    if isinstance(results_list, list):
                        for doc in results_list:
                            if isinstance(doc, dict):
                                with st.expander(f"📄 {doc.get('title', 'Untitled')} (Score: {doc.get('score', 'N/A')})"):
                                    st.write(f"**Category:** {doc.get('category', 'Unknown')}")
                                    if doc.get('subcategory'):
                                        st.write(f"**Subcategory:** {doc.get('subcategory')}")
                                    tags = doc.get('tags', [])
                                    if isinstance(tags, list):
                                        st.write(f"**Tags:** {', '.join(tags)}")
                                    st.write(f"**Content:** {doc.get('content', 'No content')}")
                                    st.write(f"**ID:** {doc.get('id', 'Unknown')}")
    
    # Chat Tab
    with tab2:
        st.header("💬 Chat Interface")
        
        # Initialize session state for chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = None
        
        # Chat input
        user_message = st.text_input("Your message:", placeholder="Ask a question or start a conversation...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            send_message = st.button("💬 Send Message")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.conversation_id = None
                st.rerun()
        
        if send_message and user_message:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            
            with st.spinner("Processing message..."):
                response = chat_with_system(user_message, st.session_state.conversation_id)
            
            if response:
                conversation_id = response.get("conversation_id")
                if conversation_id:
                    st.session_state.conversation_id = conversation_id
                response_text = response.get("response", "No response")
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        
        # Display chat history
        st.subheader("Chat History")
        chat_history = st.session_state.get("chat_history", [])
        if isinstance(chat_history, list):
            for message in chat_history:
                if isinstance(message, dict):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    if role == "user":
                        st.chat_message("user").write(content)
                    else:
                        st.chat_message("assistant").write(content)
    
    # Statistics Tab
    with tab3:
        st.header("📊 System Statistics")
        
        if st.button("📈 Refresh Statistics"):
            stats = get_document_stats()
            
            if stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_docs = stats.get("total_documents", 0)
                    if isinstance(total_docs, int):
                        st.metric("Total Documents", total_docs)
                
                with col2:
                    total_chunks = stats.get("total_chunks", 0)
                    if isinstance(total_chunks, int):
                        st.metric("Total Chunks", total_chunks)
                
                with col3:
                    avg_chunks = stats.get("average_chunks_per_document", 0)
                    if isinstance(avg_chunks, int):
                        st.metric("Avg Chunks/Doc", avg_chunks)
                
                # Category breakdown
                categories = stats.get("categories")
                if categories and isinstance(categories, dict):
                    st.subheader("📂 Documents by Category")
                    st.bar_chart(categories)
    
    # Document List Tab
    with tab4:
        st.header("📋 All Documents")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_category = st.selectbox("Filter by Category", ["All", "Technology", "Science", "Business", "Education", "Other"], key="list_category")
        with col2:
            list_limit = st.slider("Number of Documents", min_value=5, max_value=50, value=20, key="list_limit")
        
        if st.button("📄 Load Documents"):
            category_filter = None if filter_category == "All" else filter_category
            
            with st.spinner("Loading documents..."):
                docs_data = list_documents(category_filter, list_limit)
            
            if docs_data:
                documents = docs_data.get('documents', [])
                total = docs_data.get('total', 0)
                doc_count = len(documents) if isinstance(documents, list) else 0
                st.success(f"Loaded {doc_count} documents (Total: {total})")
                
                if isinstance(documents, list):
                    for doc in documents:
                        if isinstance(doc, dict):
                            title = doc.get('title', 'Untitled')
                            category = doc.get('category', 'Unknown')
                            with st.expander(f"📄 {title} ({category})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**ID:** {doc.get('id', 'Unknown')}")
                                    st.write(f"**Category:** {category}")
                                    subcategory = doc.get('subcategory')
                                    if subcategory:
                                        st.write(f"**Subcategory:** {subcategory}")
                                
                                with col2:
                                    tags = doc.get('tags', [])
                                    if isinstance(tags, list):
                                        st.write(f"**Tags:** {', '.join(tags)}")
                                    st.write(f"**Chunks:** {doc.get('chunk_count', 0)}")
                                    st.write(f"**Created:** {doc.get('created_at', 'N/A')}")

if __name__ == "__main__":
    main()
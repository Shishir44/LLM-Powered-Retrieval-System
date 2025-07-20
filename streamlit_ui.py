import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Optional
import uuid

# Configuration
API_BASE_URL = "http://127.0.0.1:5601"

# Initialize session state
if "tokens" not in st.session_state:
    st.session_state.tokens = {}
if "selected_token" not in st.session_state:
    st.session_state.selected_token = None
if "query_engines" not in st.session_state:
    st.session_state.query_engines = {}

def make_request(endpoint: str, method: str = "GET", data: dict = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url, json=data)
        else:
            response = requests.get(url)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def load_tokens():
    """Load available tokens from API"""
    result = make_request("/list_tokens/")
    if result["success"]:
        st.session_state.tokens = result["data"]
        return True
    else:
        st.error(f"Failed to load tokens: {result['error']}")
        return False

def main():
    st.set_page_config(
        page_title="Vector Database Testing UI",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Vector Database Testing Interface")
    st.markdown("---")
    
    # Sidebar for index management
    with st.sidebar:
        st.header("📊 Index Management")
        
        # Load tokens button
        if st.button("🔄 Refresh Tokens", use_container_width=True):
            load_tokens()
        
        # Create new index
        st.subheader("Create New Index")
        new_index_name = st.text_input("Index Name", placeholder="my_new_index")
        
        if st.button("➕ Create Index", use_container_width=True):
            if new_index_name:
                result = make_request("/create_index/", "POST", {"index_name": new_index_name})
                if result["success"]:
                    st.success(f"Index '{new_index_name}' created successfully!")
                    st.info(f"Token: {result['data']['token']}")
                    load_tokens()
                else:
                    st.error(f"Failed to create index: {result['error']}")
            else:
                st.warning("Please enter an index name")
        
        # Select existing index
        st.subheader("Select Index")
        if st.session_state.tokens:
            token_options = {f"{info['index_name']} ({token[:8]}...)": token 
                           for token, info in st.session_state.tokens.items()}
            
            selected_display = st.selectbox("Choose Index", list(token_options.keys()))
            if selected_display:
                st.session_state.selected_token = token_options[selected_display]
                st.success(f"Selected: {selected_display}")
        else:
            st.info("No indexes available. Create one above or refresh tokens.")
            load_tokens()
    
    # Main content area
    if st.session_state.selected_token:
        token = st.session_state.selected_token
        index_name = st.session_state.tokens[token]["index_name"]
        
        st.header(f"Working with Index: {index_name}")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Add Documents", 
            "🔧 Query Engine", 
            "🔍 Query & Search", 
            "📊 View Data", 
            "🗑️ Delete Operations"
        ])
        
        with tab1:
            st.subheader("Add Documents to Index")
            
            # Document input methods
            input_method = st.radio("Input Method", ["Manual Entry", "Bulk Upload"])
            
            if input_method == "Manual Entry":
                content = st.text_area("Document Content", height=150, placeholder="Enter your document content here...")
                
                # Metadata input
                st.write("**Metadata (Optional)**")
                col1, col2 = st.columns(2)
                
                with col1:
                    category = st.text_input("Category", placeholder="e.g., education, tech")
                    author = st.text_input("Author", placeholder="Document author")
                    
                with col2:
                    source = st.text_input("Source", placeholder="Document source")
                    date = st.date_input("Date")
                
                # Additional metadata
                tags = st.text_input("Tags (comma-separated)", placeholder="tag1, tag2, tag3")
                
                if st.button("➕ Add Document", use_container_width=True):
                    if content.strip():
                        # Prepare metadata
                        metadata = {}
                        if category: metadata["category"] = category
                        if author: metadata["author"] = author
                        if source: metadata["source"] = source
                        if date: metadata["date"] = str(date)
                        if tags: metadata["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
                        
                        # Add unique ID
                        metadata["doc_id"] = str(uuid.uuid4())
                        
                        result = make_request("/add_vector/", "POST", {
                            "token": token,
                            "content": [content],
                            "metadata": metadata
                        })
                        
                        if result["success"]:
                            st.success("Document added successfully!")
                            st.json(result["data"])
                        else:
                            st.error(f"Failed to add document: {result['error']}")
                    else:
                        st.warning("Please enter document content")
            
            else:  # Bulk Upload
                st.write("**Bulk Document Upload**")
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        
                        # Column mapping
                        st.write("**Column Mapping**")
                        content_col = st.selectbox("Content Column", df.columns)
                        
                        if st.button("📤 Upload All Documents"):
                            success_count = 0
                            total_count = len(df)
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, row in df.iterrows():
                                content = str(row[content_col])
                                
                                # Create metadata from other columns
                                metadata = {
                                    "doc_id": str(uuid.uuid4()),
                                    "batch_upload": True,
                                    "row_index": idx
                                }
                                
                                for col in df.columns:
                                    if col != content_col and pd.notna(row[col]):
                                        metadata[col] = str(row[col])
                                
                                result = make_request("/add_vector/", "POST", {
                                    "token": token,
                                    "content": [content],
                                    "metadata": metadata
                                })
                                
                                if result["success"]:
                                    success_count += 1
                                
                                # Update progress
                                progress = (idx + 1) / total_count
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {idx + 1}/{total_count}")
                            
                            st.success(f"Upload complete! {success_count}/{total_count} documents added successfully.")
                    
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        
        with tab2:
            st.subheader("Query Engine Management")
            
            # Create new query engine
            st.write("**Create Query Engine**")
            engine_name = st.text_input("Query Engine Name", placeholder="my_query_engine")
            
            # Metadata filters
            st.write("**Metadata Filters (Optional)**")
            filter_enabled = st.checkbox("Enable Metadata Filtering")
            
            filters_data = []
            if filter_enabled:
                num_filters = st.number_input("Number of Filters", min_value=1, max_value=5, value=1)
                
                for i in range(int(num_filters)):
                    st.write(f"Filter {i+1}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        key = st.text_input(f"Key {i+1}", key=f"filter_key_{i}")
                    with col2:
                        value = st.text_input(f"Value {i+1}", key=f"filter_value_{i}")
                    with col3:
                        operator = st.selectbox(f"Operator {i+1}", 
                                              ["==", "!=", ">", ">=", "<", "<=", "contains"], 
                                              key=f"filter_op_{i}")
                    
                    if key and value:
                        filters_data.append({
                            "key": key,
                            "value": value,
                            "operator": operator
                        })
            
            if st.button("🔧 Create Query Engine", use_container_width=True):
                if engine_name:
                    request_data = {
                        "token": token,
                        "query_engine_name": engine_name
                    }
                    
                    if filters_data:
                        request_data["metadata_filters"] = filters_data
                    
                    result = make_request("/create_query_engine/", "POST", request_data)
                    
                    if result["success"]:
                        st.success("Query engine created successfully!")
                        st.json(result["data"])
                    else:
                        st.error(f"Failed to create query engine: {result['error']}")
                else:
                    st.warning("Please enter a query engine name")
        
        with tab3:
            st.subheader("Query & Search")
            
            # Query input
            query = st.text_area("Enter your query", height=100, placeholder="What would you like to search for?")
            
            # Query options
            col1, col2 = st.columns(2)
            
            with col1:
                use_filters = st.checkbox("Use Metadata Filters")
                
            with col2:
                custom_prompt = st.checkbox("Use Custom Prompt")
            
            # Metadata filters for query
            query_filters = []
            if use_filters:
                st.write("**Query Filters**")
                num_query_filters = st.number_input("Number of Query Filters", min_value=1, max_value=5, value=1)
                
                for i in range(int(num_query_filters)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        q_key = st.text_input(f"Query Key {i+1}", key=f"q_filter_key_{i}")
                    with col2:
                        q_value = st.text_input(f"Query Value {i+1}", key=f"q_filter_value_{i}")
                    with col3:
                        q_operator = st.selectbox(f"Query Operator {i+1}", 
                                                ["==", "!=", ">", ">=", "<", "<=", "contains"], 
                                                key=f"q_filter_op_{i}")
                    
                    if q_key and q_value:
                        query_filters.append({
                            "key": q_key,
                            "value": q_value,
                            "operator": q_operator
                        })
            
            # Custom prompt
            prompt_text = ""
            if custom_prompt:
                prompt_text = st.text_area("Custom Prompt Template", height=100,
                                         placeholder="Enter your custom prompt here...")
            
            if st.button("🔍 Execute Query", use_container_width=True):
                if query.strip():
                    request_data = {
                        "token": token,
                        "query": query
                    }
                    
                    if query_filters:
                        request_data["metadata_filters"] = query_filters
                        
                    if prompt_text:
                        request_data["prompt"] = prompt_text
                    
                    with st.spinner("Executing query..."):
                        result = make_request("/query_index/", "POST", request_data)
                    
                    if result["success"]:
                        response_data = result["data"]
                        
                        # Display results
                        st.success("Query executed successfully!")
                        
                        # Response
                        st.subheader("📝 Response")
                        st.write(response_data.get("response", "No response available"))
                        
                        # Source nodes
                        if "source_nodes" in response_data and response_data["source_nodes"]:
                            st.subheader("📚 Source Documents")
                            
                            for i, node in enumerate(response_data["source_nodes"], 1):
                                with st.expander(f"Source {i} (Score: {node.get('score', 'N/A')})"):
                                    st.write("**Content:**")
                                    st.write(node.get("text", "No text available"))
                                    
                                    st.write("**Metadata:**")
                                    st.json(node.get("metadata", {}))
                        
                        # Metadata
                        if "metadata" in response_data:
                            st.subheader("ℹ️ Query Metadata")
                            st.json(response_data["metadata"])
                    else:
                        st.error(f"Query failed: {result['error']}")
                else:
                    st.warning("Please enter a query")
        
        with tab4:
            st.subheader("View Index Data")
            
            if st.button("📊 Refresh Data View"):
                # This is a simple way to show we're working with the index
                st.info(f"Currently viewing index: {index_name}")
                st.write(f"**Token:** `{token}`")
                
                # You could add more data viewing functionality here
                # For now, we'll show a placeholder
                st.write("**Index Statistics:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Index Name", index_name)
                with col2:
                    st.metric("Token", token[:8] + "...")
                with col3:
                    st.metric("Status", "Active")
        
        with tab5:
            st.subheader("Delete Operations")
            st.warning("⚠️ These operations are permanent and cannot be undone!")
            
            # Delete by filters
            st.write("**Delete Documents by Metadata**")
            
            delete_filters = []
            num_delete_filters = st.number_input("Number of Delete Filters", min_value=1, max_value=5, value=1)
            
            for i in range(int(num_delete_filters)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    d_key = st.text_input(f"Delete Key {i+1}", key=f"d_filter_key_{i}")
                with col2:
                    d_value = st.text_input(f"Delete Value {i+1}", key=f"d_filter_value_{i}")
                with col3:
                    d_operator = st.selectbox(f"Delete Operator {i+1}", 
                                            ["==", "!=", ">", ">=", "<", "<=", "contains"], 
                                            key=f"d_filter_op_{i}")
                
                if d_key and d_value:
                    delete_filters.append({
                        "key": d_key,
                        "value": d_value,
                        "operator": d_operator
                    })
            
            if delete_filters:
                if st.button("🗑️ Delete Matching Documents", type="secondary"):
                    result = make_request("/delete_nodes/", "POST", {
                        "token": token,
                        "metadata_filters": delete_filters
                    })
                    
                    if result["success"]:
                        st.success("Documents deleted successfully!")
                        st.json(result["data"])
                    else:
                        st.error(f"Delete failed: {result['error']}")
            
            st.markdown("---")
            
            # Delete entire index
            st.write("**Delete Entire Index**")
            st.error("This will permanently delete the entire index and all its data!")
            
            confirm_delete = st.text_input("Type 'DELETE' to confirm index deletion")
            
            if confirm_delete == "DELETE":
                if st.button("🗑️ DELETE ENTIRE INDEX", type="secondary"):
                    result = make_request("/index/", "DELETE", {"token": token})
                    
                    if result["success"]:
                        st.success("Index deleted successfully!")
                        st.session_state.selected_token = None
                        load_tokens()
                    else:
                        st.error(f"Delete failed: {result['error']}")
    
    else:
        st.info("👈 Please select or create an index from the sidebar to get started.")
        
        # Show some help information
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Create a new index** or **select an existing one** from the sidebar
        2. **Add documents** to your index using the "Add Documents" tab
        3. **Create query engines** with specific filters if needed
        4. **Search and query** your documents
        5. **View and manage** your data
        
        ## 📋 Features
        
        - **Document Management**: Add single documents or bulk upload via CSV
        - **Advanced Querying**: Use metadata filters and custom prompts
        - **Query Engine Management**: Create specialized query engines
        - **Data Visualization**: View index statistics and search results
        - **Delete Operations**: Remove specific documents or entire indexes
        
        ## 🔧 Configuration
        
        Make sure your vector database API is running on `http://127.0.0.1:5601`
        """)

if __name__ == "__main__":
    main()
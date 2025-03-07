import streamlit as st
import requests
import json

# Backend API URL
API_BASE_URL = "http://127.0.0.1:8000"

# Streamlit App Title
st.set_page_config(page_title="Retrieval System", layout="wide")
st.title("📚 AI-Powered Document Retrieval System")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Home", "Upload Document", "Chat", "Manage Query Engines"])

def list_indexes():
    response = requests.get(f"{API_BASE_URL}/list_indexes/")
    return response.json()

def list_tokens():
    response = requests.get(f"{API_BASE_URL}/list_tokens/")
    return response.json()

def query_index(token, query, query_engine_id=None):
    payload = {"token": token, "query": query, "query_engine_id": query_engine_id}
    response = requests.post(f"{API_BASE_URL}/query_index/", json=payload)
    return response.json()

if menu == "Home":
    st.write("Welcome to the AI-powered document retrieval system. Use the sidebar to navigate.")

elif menu == "Upload Document":
    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload a document (TXT, PDF)", type=["txt", "pdf"])
    
    if uploaded_file:
        index_name = st.text_input("Enter index name")
        if st.button("Create Index"):
            payload = {"index_name": index_name}
            response = requests.post(f"{API_BASE_URL}/create_index/", json=payload)
            st.success(response.json())
    
elif menu == "Chat":
    st.header("💬 Chat with the AI")
    tokens = list_tokens()
    token = st.selectbox("Select an index token", options=list(tokens.keys()))
    query = st.text_input("Enter your query:")
    
    if st.button("Get Answer"):
        response = query_index(token, query)
        st.write("### Response:")
        st.write(response.get("response", "No response received"))

elif menu == "Manage Query Engines":
    st.header("⚙️ Manage Query Engines")
    st.write("Feature coming soon...")

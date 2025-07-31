#!/usr/bin/env python3
"""
Script to clear all documents from the knowledge base service
"""

import requests
import json
import time

KB_SERVICE_URL = "http://localhost:8002"

def get_all_document_ids():
    """Get all document IDs from the knowledge base."""
    try:
        response = requests.get(f"{KB_SERVICE_URL}/api/v1/documents")
        response.raise_for_status()
        data = response.json()
        return [doc['id'] for doc in data.get('documents', [])]
    except Exception as e:
        print(f"Error getting document IDs: {e}")
        return []

def delete_document(doc_id):
    """Delete a single document by ID."""
    try:
        response = requests.delete(f"{KB_SERVICE_URL}/api/v1/documents/{doc_id}")
        if response.status_code == 200:
            print(f"✓ Deleted document: {doc_id}")
            return True
        else:
            print(f"✗ Failed to delete document {doc_id}: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error deleting document {doc_id}: {e}")
        return False

def main():
    print("Clearing all documents from knowledge base...")
    
    # Get all document IDs
    doc_ids = get_all_document_ids()
    print(f"Found {len(doc_ids)} documents to delete")
    
    if not doc_ids:
        print("No documents found")
        return
    
    # Delete each document
    deleted_count = 0
    for doc_id in doc_ids:
        if delete_document(doc_id):
            deleted_count += 1
        time.sleep(0.1)  # Small delay to avoid overwhelming the service
    
    print(f"\nDeleted {deleted_count}/{len(doc_ids)} documents")
    
    # Verify deletion
    remaining_ids = get_all_document_ids()
    print(f"Remaining documents: {len(remaining_ids)}")

if __name__ == "__main__":
    main()
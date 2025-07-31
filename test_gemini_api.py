#!/usr/bin/env python3
"""
Test script to verify Gemini API key is working
"""

import os
import sys
import requests
import json

def test_gemini_api():
    """Test if Gemini API key is working for embeddings"""
    
    # Get API key from environment
    api_key = "AIzaSyAmV8_gMmgFBli8GR6QxciuOb-IL2ApRk0"  # From .env file
    
    if not api_key:
        print("❌ No Gemini API key found")
        return False
    
    print(f"🔑 Testing Gemini API key: {api_key[:20]}...")
    
    # Test embedding endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "model": "models/embedding-001",
        "content": {
            "parts": [{"text": "test query"}]
        }
    }
    
    try:
        print("🌐 Testing connection to Gemini API...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'embedding' in result:
                embedding_length = len(result['embedding']['values'])
                print(f"✅ SUCCESS! Gemini API is working")
                print(f"📊 Embedding dimension: {embedding_length}")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Connection timeout to Gemini API")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_network_connectivity():
    """Test basic network connectivity"""
    print("🌐 Testing network connectivity...")
    
    try:
        # Test Google DNS
        response = requests.get("https://8.8.8.8", timeout=5)
        print("✅ Network connectivity OK")
        return True
    except:
        try:
            # Test alternative connectivity
            response = requests.get("https://google.com", timeout=5)
            print("✅ Network connectivity OK")
            return True
        except Exception as e:
            print(f"❌ Network connectivity failed: {e}")
            return False

if __name__ == "__main__":
    print("🧪 Gemini API Test\n" + "="*50)
    
    # Test network first
    if not test_network_connectivity():
        sys.exit(1)
    
    print()
    
    # Test API
    if test_gemini_api():
        print("\n✅ All tests passed! Gemini API is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Gemini API test failed!")
        sys.exit(1)
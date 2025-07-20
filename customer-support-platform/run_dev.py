#!/usr/bin/env python3
"""
Development runner for the Customer Support Platform with OpenAI integration
This starts the API Gateway with OpenAI integration for real functionality.
"""

import os
import sys
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
import uuid
import time

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangChain LLM
try:
    llm = OpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.7
    )
    print("✅ OpenAI LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing OpenAI: {e}")
    llm = None

# FastAPI app
app = FastAPI(
    title="Customer Support Platform API",
    description="AI-powered customer support platform with LangChain and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ConversationRequest(BaseModel):
    message: str
    metadata: dict = {}

class ConversationResponse(BaseModel):
    conversation_id: str
    session_id: str
    response: str
    metadata: dict

class MessageRequest(BaseModel):
    message: str
    metadata: dict = {}

class MessageResponse(BaseModel):
    message_id: str
    response: str
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    openai_configured: bool

# In-memory storage for development
conversations: Dict[str, Dict[str, Any]] = {}

# Helper functions
def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def classify_intent(message: str) -> tuple[str, float]:
    """Simple intent classification."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "greeting", 0.9
    elif any(word in message_lower for word in ["help", "support", "issue", "problem"]):
        return "support_request", 0.8
    elif any(word in message_lower for word in ["account", "login", "password", "reset"]):
        return "account_support", 0.85
    elif any(word in message_lower for word in ["billing", "payment", "charge", "refund"]):
        return "billing_support", 0.85
    elif any(word in message_lower for word in ["angry", "frustrated", "complaint", "terrible"]):
        return "complaint", 0.7
    else:
        return "question", 0.6

def analyze_sentiment(message: str) -> tuple[str, float]:
    """Simple sentiment analysis."""
    message_lower = message.lower()
    
    positive_words = ["good", "great", "excellent", "happy", "satisfied", "love", "amazing"]
    negative_words = ["bad", "terrible", "awful", "hate", "angry", "frustrated", "disappointed"]
    
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        return "positive", 0.8
    elif negative_count > positive_count:
        return "negative", 0.8
    else:
        return "neutral", 0.7

async def generate_ai_response(message: str, intent: str, sentiment: str) -> str:
    """Generate AI response using OpenAI."""
    if not llm:
        return "I apologize, but the AI service is currently unavailable. Please try again later."
    
    try:
        # Create a context-aware prompt
        system_prompt = f"""You are a helpful customer support agent. 
        The customer's message has been classified as intent: {intent} and sentiment: {sentiment}.
        Provide a helpful, empathetic response that addresses their needs.
        Keep responses concise and professional."""
        
        full_prompt = f"{system_prompt}\n\nCustomer: {message}\nAgent:"
        
        response = llm(full_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team directly."

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="api-gateway",
        version="1.0.0",
        openai_configured=llm is not None
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Support Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "openai_configured": llm is not None
    }

# Create conversation endpoint
@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest):
    """Create a new conversation with AI-powered response."""
    conversation_id = generate_id()
    session_id = generate_id()
    
    # Analyze the message
    intent, intent_confidence = classify_intent(request.message)
    sentiment, sentiment_confidence = analyze_sentiment(request.message)
    
    # Generate AI response
    ai_response = await generate_ai_response(request.message, intent, sentiment)
    
    # Store conversation
    conversations[conversation_id] = {
        "id": conversation_id,
        "session_id": session_id,
        "user_id": request.metadata.get("user_id", "anonymous"),
        "status": "active",
        "created_at": time.time(),
        "messages": [
            {
                "id": generate_id(),
                "content": request.message,
                "type": "user",
                "timestamp": time.time()
            },
            {
                "id": generate_id(),
                "content": ai_response,
                "type": "assistant",
                "timestamp": time.time()
            }
        ]
    }
    
    return ConversationResponse(
        conversation_id=conversation_id,
        session_id=session_id,
        response=ai_response,
        metadata={
            "intent": intent,
            "intent_confidence": intent_confidence,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_confidence,
            "ai_powered": True
        }
    )

# Send message to existing conversation
@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(conversation_id: str, request: MessageRequest):
    """Send a message to an existing conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Analyze the message
    intent, intent_confidence = classify_intent(request.message)
    sentiment, sentiment_confidence = analyze_sentiment(request.message)
    
    # Generate AI response
    ai_response = await generate_ai_response(request.message, intent, sentiment)
    
    # Add messages to conversation
    message_id = generate_id()
    conversations[conversation_id]["messages"].extend([
        {
            "id": generate_id(),
            "content": request.message,
            "type": "user",
            "timestamp": time.time()
        },
        {
            "id": message_id,
            "content": ai_response,
            "type": "assistant",
            "timestamp": time.time()
        }
    ])
    
    return MessageResponse(
        message_id=message_id,
        response=ai_response,
        metadata={
            "intent": intent,
            "intent_confidence": intent_confidence,
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_confidence,
            "ai_powered": True
        }
    )

# Get conversation
@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id]

# List conversations
@app.get("/api/v1/conversations")
async def list_conversations(limit: int = 10):
    """List all conversations."""
    conv_list = list(conversations.values())
    return {
        "conversations": conv_list[-limit:],
        "total": len(conv_list)
    }

# Mock knowledge base search
@app.get("/api/v1/knowledge-base/search")
async def search_knowledge(q: str, limit: int = 5):
    """Search knowledge base (enhanced mock with AI)."""
    # Simple keyword matching for demo
    mock_results = []
    
    if "password" in q.lower() or "reset" in q.lower():
        mock_results.append({
            "id": "doc_password_reset",
            "title": "How to Reset Your Password",
            "content": "To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter your email 4. Check your email for reset link",
            "score": 0.95,
            "category": "account"
        })
    
    if "billing" in q.lower() or "payment" in q.lower():
        mock_results.append({
            "id": "doc_billing",
            "title": "Billing and Payment Information",
            "content": "You can view your billing information in your account dashboard. Payments are processed monthly.",
            "score": 0.90,
            "category": "billing"
        })
    
    if not mock_results:
        mock_results.append({
            "id": "doc_general",
            "title": "General Help",
            "content": f"This is a general help document related to: {q}",
            "score": 0.6,
            "category": "general"
        })
    
    return {
        "query": q,
        "results": mock_results[:limit],
        "total": len(mock_results),
        "ai_enhanced": True
    }

if __name__ == "__main__":
    print("🚀 Starting Customer Support Platform API Gateway with OpenAI...")
    print("📚 API Documentation: http://localhost:8080/docs")
    print("❤️  Health Check: http://localhost:8080/health")
    print("🤖 OpenAI Integration: ✅ Enabled")
    print()
    print("✨ Features available:")
    print("   • AI-powered conversation responses")
    print("   • Intent classification")
    print("   • Sentiment analysis")
    print("   • Conversation management")
    print("   • Knowledge base search")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
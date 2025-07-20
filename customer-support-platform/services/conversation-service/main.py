"""
Conversation Service for Customer Support Platform

This service handles conversation management, state tracking, and LangGraph-based
workflow orchestration for customer support interactions.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from shared.config.settings import get_conversation_service_settings
from shared.auth.jwt_handler import JWTHandler
from shared.database.models import Conversation, Message, User
from shared.database.connection import get_database_session
from shared.monitoring.metrics import MetricsCollector

# Configuration
settings = get_conversation_service_settings()

# FastAPI app
app = FastAPI(
    title="Conversation Service",
    description="LangGraph-based conversation management for customer support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
jwt_handler = JWTHandler(settings.jwt.secret_key, settings.jwt.algorithm)


# Pydantic models
class ConversationState(BaseModel):
    """State model for conversation workflow."""
    
    conversation_id: str
    user_id: str
    session_id: str
    messages: List[Dict[str, Any]] = []
    
    # Current state
    current_message: str = ""
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    confidence: float = 0.0
    
    # Context
    context: Dict[str, Any] = {}
    user_profile: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    
    # Workflow state
    next_action: str = "classify_intent"
    escalation_needed: bool = False
    escalation_reason: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateConversationRequest(BaseModel):
    """Request model for creating a new conversation."""
    
    message: str = Field(..., description="Initial message from the user")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SendMessageRequest(BaseModel):
    """Request model for sending a message."""
    
    message: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")


class ConversationResponse(BaseModel):
    """Response model for conversation operations."""
    
    conversation_id: str
    session_id: str
    response: str
    metadata: Dict[str, Any] = {}


class MessageResponse(BaseModel):
    """Response model for message operations."""
    
    message_id: str
    response: str
    metadata: Dict[str, Any] = {}


# LangGraph workflow implementation
class CustomerSupportWorkflow:
    """LangGraph-based customer support workflow."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            max_tokens=settings.openai.max_tokens,
            api_key=settings.openai.api_key
        )
        
        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        
        # Service clients for external calls
        self.nlp_service_url = settings.services.nlp_service_url
        self.knowledge_base_service_url = settings.services.knowledge_base_service_url
    
    def _create_workflow(self) -> CompiledStateGraph:
        """Create the LangGraph workflow."""
        
        # Define the workflow graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("search_knowledge", self.search_knowledge)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("escalate_to_human", self.escalate_to_human)
        workflow.add_node("collect_feedback", self.collect_feedback)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add edges
        workflow.add_edge("classify_intent", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "search_knowledge")
        
        # Conditional edges based on confidence and sentiment
        workflow.add_conditional_edges(
            "search_knowledge",
            self.should_escalate,
            {
                "escalate": "escalate_to_human",
                "respond": "generate_response"
            }
        )
        
        workflow.add_edge("generate_response", "collect_feedback")
        workflow.add_edge("collect_feedback", END)
        workflow.add_edge("escalate_to_human", END)
        
        # Compile the workflow
        return workflow.compile(checkpointer=self.memory)
    
    async def classify_intent(self, state: ConversationState) -> ConversationState:
        """Classify user intent."""
        
        # Create intent classification prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for customer support. 
            Classify the user's message into one of these categories:
            - question: User is asking a question
            - complaint: User is complaining about something
            - request: User is requesting something
            - compliment: User is giving positive feedback
            - other: None of the above
            
            Return only the category name."""),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            intent = await chain.ainvoke({"message": state.current_message})
            state.intent = intent.strip().lower()
            state.confidence = 0.8  # Default confidence
            
            logging.info(f"Classified intent: {state.intent}")
            
        except Exception as e:
            logging.error(f"Error classifying intent: {e}")
            state.intent = "other"
            state.confidence = 0.0
        
        return state
    
    async def analyze_sentiment(self, state: ConversationState) -> ConversationState:
        """Analyze sentiment of the message."""
        
        # Create sentiment analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sentiment analyzer for customer support.
            Analyze the sentiment of the user's message and classify it as:
            - positive: User is happy or satisfied
            - negative: User is frustrated or angry
            - neutral: User is neither positive nor negative
            
            Return only the sentiment category."""),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            sentiment = await chain.ainvoke({"message": state.current_message})
            state.sentiment = sentiment.strip().lower()
            
            logging.info(f"Analyzed sentiment: {state.sentiment}")
            
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            state.sentiment = "neutral"
        
        return state
    
    async def search_knowledge(self, state: ConversationState) -> ConversationState:
        """Search knowledge base for relevant information."""
        
        # Mock knowledge search - in production, this would call the knowledge base service
        try:
            # This would be replaced with actual HTTP call to knowledge base service
            relevant_docs = [
                {
                    "title": "Password Reset Guide",
                    "content": "To reset your password, click on 'Forgot Password' on the login page.",
                    "score": 0.95
                }
            ]
            
            state.context["knowledge_results"] = relevant_docs
            state.context["has_knowledge"] = len(relevant_docs) > 0
            
            logging.info(f"Found {len(relevant_docs)} relevant documents")
            
        except Exception as e:
            logging.error(f"Error searching knowledge base: {e}")
            state.context["knowledge_results"] = []
            state.context["has_knowledge"] = False
        
        return state
    
    async def generate_response(self, state: ConversationState) -> ConversationState:
        """Generate AI response based on context."""
        
        # Build context for response generation
        context_parts = []
        
        # Add conversation history
        if state.conversation_history:
            context_parts.append("Previous conversation:")
            for msg in state.conversation_history[-5:]:  # Last 5 messages
                context_parts.append(f"- {msg['type']}: {msg['content']}")
        
        # Add knowledge base results
        if state.context.get("has_knowledge"):
            context_parts.append("\\nRelevant information:")
            for doc in state.context["knowledge_results"]:
                context_parts.append(f"- {doc['title']}: {doc['content']}")
        
        context_str = "\\n".join(context_parts)
        
        # Create response generation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer support assistant. 
            Your goal is to provide accurate, helpful, and empathetic responses.
            
            Guidelines:
            - Be polite and professional
            - If you have relevant information from the knowledge base, use it
            - If you don't have enough information, ask clarifying questions
            - For complaints, acknowledge the issue and show empathy
            - Keep responses concise but complete
            
            Context: {context}
            User's intent: {intent}
            User's sentiment: {sentiment}"""),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = await chain.ainvoke({
                "message": state.current_message,
                "context": context_str,
                "intent": state.intent,
                "sentiment": state.sentiment
            })
            
            state.context["ai_response"] = response
            logging.info("Generated AI response")
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            state.context["ai_response"] = "I apologize, but I'm having trouble processing your request right now. Please try again."
        
        return state
    
    async def escalate_to_human(self, state: ConversationState) -> ConversationState:
        """Escalate conversation to human agent."""
        
        state.escalation_needed = True
        state.escalation_reason = "Low confidence or negative sentiment"
        state.context["ai_response"] = "I understand this is important to you. Let me connect you with one of our human agents who can better assist you."
        
        logging.info(f"Escalating conversation {state.conversation_id} to human")
        
        return state
    
    async def collect_feedback(self, state: ConversationState) -> ConversationState:
        """Collect feedback for the response."""
        
        # This would normally trigger a feedback collection mechanism
        state.context["feedback_requested"] = True
        
        logging.info("Feedback collection triggered")
        
        return state
    
    def should_escalate(self, state: ConversationState) -> Literal["escalate", "respond"]:
        """Determine if conversation should be escalated."""
        
        # Escalation criteria
        if state.confidence < 0.5:
            return "escalate"
        
        if state.sentiment == "negative" and state.intent == "complaint":
            return "escalate"
        
        if not state.context.get("has_knowledge", False) and state.intent == "question":
            return "escalate"
        
        return "respond"
    
    async def process_message(self, state: ConversationState) -> ConversationState:
        """Process a message through the workflow."""
        
        config = {"configurable": {"thread_id": state.session_id}}
        
        # Run the workflow
        result = await self.workflow.ainvoke(state.model_dump(), config=config)
        
        # Convert result back to ConversationState
        return ConversationState(**result)


# Service class
class ConversationService:
    """Service for managing conversations."""
    
    def __init__(self):
        self.workflow = CustomerSupportWorkflow()
        self.metrics = MetricsCollector()
    
    async def create_conversation(
        self,
        user_id: str,
        message: str,
        metadata: Dict[str, Any],
        db_session
    ) -> ConversationResponse:
        """Create a new conversation."""
        
        start_time = time.time()
        
        try:
            # Create conversation record
            conversation_id = str(uuid4())
            session_id = str(uuid4())
            
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                status="active",
                metadata=metadata
            )
            
            db_session.add(conversation)
            await db_session.commit()
            
            # Create initial message
            message_record = Message(
                conversation_id=conversation_id,
                content=message,
                message_type="user",
                metadata=metadata
            )
            
            db_session.add(message_record)
            await db_session.commit()
            
            # Process through workflow
            state = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                current_message=message,
                messages=[{"type": "user", "content": message}],
                metadata=metadata
            )
            
            processed_state = await self.workflow.process_message(state)
            
            # Create AI response message
            ai_response = processed_state.context.get("ai_response", "I'm here to help!")
            
            ai_message = Message(
                conversation_id=conversation_id,
                content=ai_response,
                message_type="assistant",
                intent=processed_state.intent,
                sentiment=processed_state.sentiment,
                confidence=processed_state.confidence,
                response_time=time.time() - start_time,
                metadata=processed_state.metadata
            )
            
            db_session.add(ai_message)
            await db_session.commit()
            
            # Update conversation with analytics
            conversation.sentiment_score = self._sentiment_to_score(processed_state.sentiment)
            await db_session.commit()
            
            # Record metrics
            self.metrics.record_conversation_created(
                intent=processed_state.intent,
                sentiment=processed_state.sentiment,
                response_time=time.time() - start_time
            )
            
            return ConversationResponse(
                conversation_id=conversation_id,
                session_id=session_id,
                response=ai_response,
                metadata={
                    "intent": processed_state.intent,
                    "sentiment": processed_state.sentiment,
                    "confidence": processed_state.confidence,
                    "escalation_needed": processed_state.escalation_needed
                }
            )
            
        except Exception as e:
            logging.error(f"Error creating conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create conversation"
            )
    
    async def send_message(
        self,
        conversation_id: str,
        user_id: str,
        message: str,
        metadata: Dict[str, Any],
        db_session
    ) -> MessageResponse:
        """Send a message in an existing conversation."""
        
        start_time = time.time()
        
        try:
            # Get conversation
            conversation = await db_session.get(Conversation, conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            # Check if user owns conversation
            if conversation.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
            
            # Create user message
            user_message = Message(
                conversation_id=conversation_id,
                content=message,
                message_type="user",
                metadata=metadata
            )
            
            db_session.add(user_message)
            await db_session.commit()
            
            # Get conversation history
            history = await db_session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc()).limit(10)
            )
            
            history_messages = [
                {"type": msg.message_type, "content": msg.content}
                for msg in history.scalars().all()
            ]
            
            # Process through workflow
            state = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=conversation.session_id,
                current_message=message,
                messages=history_messages,
                conversation_history=history_messages,
                metadata=metadata
            )
            
            processed_state = await self.workflow.process_message(state)
            
            # Create AI response message
            ai_response = processed_state.context.get("ai_response", "I'm here to help!")
            
            ai_message = Message(
                conversation_id=conversation_id,
                content=ai_response,
                message_type="assistant",
                intent=processed_state.intent,
                sentiment=processed_state.sentiment,
                confidence=processed_state.confidence,
                response_time=time.time() - start_time,
                metadata=processed_state.metadata
            )
            
            db_session.add(ai_message)
            await db_session.commit()
            
            # Update conversation analytics
            conversation.sentiment_score = self._sentiment_to_score(processed_state.sentiment)
            conversation.updated_at = datetime.now(timezone.utc)
            await db_session.commit()
            
            # Record metrics
            self.metrics.record_message_sent(
                intent=processed_state.intent,
                sentiment=processed_state.sentiment,
                response_time=time.time() - start_time
            )
            
            return MessageResponse(
                message_id=str(ai_message.id),
                response=ai_response,
                metadata={
                    "intent": processed_state.intent,
                    "sentiment": processed_state.sentiment,
                    "confidence": processed_state.confidence,
                    "escalation_needed": processed_state.escalation_needed
                }
            )
            
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send message"
            )
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment to numerical score."""
        sentiment_map = {
            "positive": 0.8,
            "neutral": 0.5,
            "negative": 0.2
        }
        return sentiment_map.get(sentiment, 0.5)


# Initialize service
conversation_service = ConversationService()


# Dependencies
async def get_current_user(request: Request):
    """Get current user from JWT token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt_handler.decode_token(token)
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# API endpoints
@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """Create a new conversation."""
    
    return await conversation_service.create_conversation(
        user_id=user["sub"],
        message=request.message,
        metadata=request.metadata,
        db_session=db_session
    )


@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """Send a message in an existing conversation."""
    
    return await conversation_service.send_message(
        conversation_id=conversation_id,
        user_id=user["sub"],
        message=request.message,
        metadata=request.metadata,
        db_session=db_session
    )


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """Get conversation details."""
    
    conversation = await db_session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    if conversation.user_id != user["sub"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get messages
    messages = await db_session.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    
    return {
        "conversation_id": conversation.id,
        "user_id": conversation.user_id,
        "status": conversation.status,
        "created_at": conversation.created_at,
        "messages": [
            {
                "id": msg.id,
                "content": msg.content,
                "type": msg.message_type,
                "timestamp": msg.created_at,
                "metadata": msg.metadata
            }
            for msg in messages.scalars().all()
        ],
        "metadata": conversation.metadata
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "conversation-service"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info"
    )
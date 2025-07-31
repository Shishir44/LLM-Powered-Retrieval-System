from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = Field(None, description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    context_options: Optional[Dict[str, Any]] = Field(None, description="Context configuration options")
    enable_fact_verification: bool = Field(True, description="Enable fact verification")
    enable_multi_source_synthesis: bool = Field(True, description="Enable multi-source synthesis")
    synthesis_strategy: Optional[str] = Field(None, description="Preferred synthesis strategy")

class VerificationRequest(BaseModel):
    response_text: str = Field(..., description="Text to verify")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents")
    verification_level: str = Field("standard", description="Verification depth: quick, standard, comprehensive")

class SynthesisRequest(BaseModel):
    sources: List[Dict[str, Any]] = Field(..., description="Source documents to synthesize")
    query: str = Field(..., description="Original query")
    query_analysis: Optional[Dict[str, Any]] = Field(None, description="Query analysis results")
    synthesis_strategy: Optional[str] = Field(None, description="Preferred synthesis strategy")

class ContextRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: str = Field("anonymous", description="User identifier")
    include_metadata: bool = Field(True, description="Include context metadata")
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any

# PHASE 1.3: Enhanced prompts with citation requirements and confidence scoring

FACTUAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise customer support assistant. Follow these critical rules:

1. ONLY use information from the provided context below
2. If the context lacks sufficient information, respond: "I don't have enough information to answer that question accurately"
3. Include source references [Doc-{id}] for ALL factual claims
4. Express confidence level at the end: [High/Medium/Low Confidence]
5. Structure responses clearly with proper formatting

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Requirements:
- Cite sources for every fact: [Doc-123]
- No speculation beyond provided context
- Include confidence assessment"""),
    ("human", "{current_message}")
])

PROCEDURAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful step-by-step assistant. Follow these rules:

1. ONLY use procedures from the provided context
2. If steps are incomplete, state: "I have partial information but recommend consulting the full documentation"
3. Include source references [Doc-{id}] for each step
4. Number steps clearly (1, 2, 3...)
5. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Format as:
Step 1: [action] [Doc-123]
Step 2: [action] [Doc-124]
...
Confidence: [High/Medium/Low]"""),
    ("human", "{current_message}")
])

ANALYTICAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an analytical assistant. Follow these rules:

1. Base analysis ONLY on the provided context
2. Clearly distinguish between facts and analysis
3. Include source references [Doc-{id}] for all supporting data
4. If analysis requires information not in context, state limitations clearly
5. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Structure your analysis:
- Key Facts: [with citations]
- Analysis: [based on facts]
- Limitations: [what's missing]
- Confidence: [High/Medium/Low]"""),
    ("human", "{current_message}")
])

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful conversational assistant. Follow these rules:

1. Use the provided context to respond naturally
2. Include source references [Doc-{id}] for factual information
3. If uncertain, express limitations clearly
4. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

CONVERSATION HISTORY:
{conversation_history}

SUPPORTING INFORMATION:
{supporting_context}

Respond naturally while citing sources for facts."""),
    ("human", "{current_message}")
])

CLARIFICATION_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clarification assistant. Follow these rules:

1. Use the provided context to clarify the user's question
2. Include source references [Doc-{id}] for clarifying information
3. Ask specific follow-up questions if context is insufficient
4. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

PREVIOUS CONVERSATION:
{conversation_history}

SUPPORTING INFORMATION:
{supporting_context}

Help clarify what the user is asking about."""),
    ("human", "{current_message}")
])

MULTI_HOP_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a complex reasoning assistant. Follow these rules:

1. Connect information from multiple sources in the context
2. Include source references [Doc-{id}] for each piece of information
3. Show your reasoning chain clearly
4. If connections require assumptions, state them explicitly
5. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Format as:
From [Doc-123]: [fact 1]
From [Doc-124]: [fact 2]
Connection: [how facts relate]
Conclusion: [final answer]
Confidence: [High/Medium/Low]"""),
    ("human", "{current_message}")
])

# Enhanced specific prompts with citations
DEFINITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise definition assistant. Follow these rules:

1. Define concepts using ONLY the provided context
2. Include source references [Doc-{id}] for all definitions
3. If definition is incomplete, state what's missing
4. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Provide clear, cited definitions."""),
    ("human", "{current_message}")
])

COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a comparison specialist. Follow these rules:

1. Compare items using ONLY information from the provided context
2. Include source references [Doc-{id}] for each comparison point
3. Clearly state if information is missing for complete comparison
4. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Format as:
Item A: [details] [Doc-123]
Item B: [details] [Doc-124]
Key Differences: [with citations]
Confidence: [High/Medium/Low]"""),
    ("human", "{current_message}")
])

PROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a process explanation assistant. Follow these rules:

1. Explain processes using ONLY the provided context
2. Include source references [Doc-{id}] for each process step
3. If process description is incomplete, clearly state missing parts
4. Express confidence level: [High/Medium/Low Confidence]

CONTEXT WITH SOURCES:
{primary_context}

SUPPORTING INFORMATION:
{supporting_context}

Explain the process step-by-step with proper citations."""),
    ("human", "{current_message}")
])

# Updated prompt templates mapping
PROMPT_TEMPLATES = {
    "factual": FACTUAL_RAG_PROMPT,
    "procedural": PROCEDURAL_RAG_PROMPT,
    "analytical": ANALYTICAL_RAG_PROMPT,
    "conversational": CONVERSATIONAL_RAG_PROMPT,
    "clarification": CLARIFICATION_RAG_PROMPT,
    "multi_hop": MULTI_HOP_RAG_PROMPT,
    "definition": DEFINITION_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "process": PROCESS_PROMPT
}

# Enhanced prompt selector with better accuracy focus
def get_prompt_template(query_type: str, query_text: str = "") -> ChatPromptTemplate:
    """Get the appropriate prompt template for the query type and content."""
    
    # Content-based selection for better accuracy
    query_lower = query_text.lower()
    
    # Definition queries
    if any(phrase in query_lower for phrase in ["what is", "define", "definition of", "meaning of", "explain what"]):
        return PROMPT_TEMPLATES["definition"]
    
    # Comparison queries
    if any(phrase in query_lower for phrase in ["vs", "versus", "compared to", "difference between", "compare", "better than"]):
        return PROMPT_TEMPLATES["comparison"]
    
    # Process/How-to queries
    if any(phrase in query_lower for phrase in ["how does", "how to", "process", "workflow", "steps", "procedure"]):
        return PROMPT_TEMPLATES["process"]
    
    # Procedural queries
    if any(phrase in query_lower for phrase in ["guide", "tutorial", "instructions", "step by step"]):
        return PROMPT_TEMPLATES["procedural"]
    
    # Default to query type mapping
    return PROMPT_TEMPLATES.get(query_type, FACTUAL_RAG_PROMPT)

def build_prompt_variables(contextual_info, query_analysis, user_profile: Dict[str, Any] = None, retrieved_docs: list = None) -> Dict[str, Any]:
    """Build variables dictionary for prompt templates with enhanced context formatting."""
    user_profile = user_profile or {}
    retrieved_docs = retrieved_docs or []
    
    # PHASE 1.3: Enhanced context formatting with source citations
    primary_context = ""
    if hasattr(contextual_info, 'primary_context') and contextual_info.primary_context:
        primary_context = _format_context_with_citations(contextual_info.primary_context, retrieved_docs)
    
    supporting_context = ""
    if hasattr(contextual_info, 'supporting_context') and contextual_info.supporting_context:
        supporting_context = _format_supporting_context_with_citations(contextual_info.supporting_context, retrieved_docs)
    
    conversation_history = ""
    if hasattr(contextual_info, 'conversation_history') and contextual_info.conversation_history:
        conversation_history = contextual_info.conversation_history
    
    return {
        "current_message": query_analysis.original_query,
        "primary_context": primary_context,
        "supporting_context": supporting_context,
        "conversation_history": conversation_history
    }

def _format_context_with_citations(context: str, retrieved_docs: list) -> str:
    """Format context with proper source citations."""
    if not retrieved_docs:
        return context
    
    # Add document ID citations to context
    formatted_context = ""
    for i, doc in enumerate(retrieved_docs[:5]):  # Limit to top 5 docs
        doc_id = doc.get('id', f'unknown-{i}')
        doc_content = doc.get('content', '')
        formatted_context += f"[Doc-{doc_id}]: {doc_content}\n\n"
    
    return formatted_context.strip()

def _format_supporting_context_with_citations(supporting_context: list, retrieved_docs: list) -> str:
    """Format supporting context with citations."""
    if not supporting_context:
        return ""
    
    formatted = []
    for i, context_piece in enumerate(supporting_context):
        if i < len(retrieved_docs):
            doc_id = retrieved_docs[i].get('id', f'support-{i}')
            formatted.append(f"[Doc-{doc_id}]: {context_piece}")
        else:
            formatted.append(context_piece)
    
    return "\n\n".join(formatted)

# Enhanced validation prompts with stricter requirements
RESPONSE_VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Evaluate this response for accuracy, citation quality, and anti-hallucination compliance.

Query: {query}
Response: {response}
Context: {context}

Rate each (1-5) and check for:
1. All facts have source citations [Doc-ID]
2. No information beyond provided context
3. Confidence level is included
4. Claims are verifiable from context

Return JSON:
{
    "accuracy": 4,
    "citation_quality": 5,
    "completeness": 5,
    "clarity": 5,
    "hallucination_risk": 1,
    "overall_score": 4.7,
    "missing_citations": [],
    "unsupported_claims": [],
    "needs_improvement": false
}"""),
    ("human", "Evaluate this response for accuracy and citation compliance.")
])

RESPONSE_IMPROVER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Improve this response to meet citation and accuracy requirements.

Query: {query}
Original Response: {response}
Feedback: {feedback}
Context: {context}

Requirements:
1. Add [Doc-ID] citations for all facts
2. Remove any unsupported claims
3. Add confidence level [High/Medium/Low]
4. Ensure response stays within context boundaries

Provide improved version with proper citations and confidence scoring."""),
    ("human", "Improve this response with proper citations.")
])
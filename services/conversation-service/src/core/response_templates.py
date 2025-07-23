from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum

class ResponseTemplateType(Enum):
    """Types of response templates for different question categories."""
    COMPONENT_LISTING = "component_listing"
    COMPARISON = "comparison"
    PROCESS_EXPLANATION = "process_explanation"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    MULTI_HOP = "multi_hop"
    CONVERSATIONAL = "conversational"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    TROUBLESHOOTING = "troubleshooting"

class StructuredResponseTemplates:
    """Advanced response templates for different question types with structured formatting."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, ChatPromptTemplate]:
        """Initialize all response templates."""
        
        templates = {}
        
        # Component Listing Template
        templates[ResponseTemplateType.COMPONENT_LISTING.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant. Based on the provided context, create a comprehensive answer about {topic} components.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Provide a well-structured response with:

## Main Components

For each component, include:
- **Component Name**: Brief description
- Key functionality and purpose
- How it relates to other components
- Real-world usage examples

**Requirements:**
- Use clear bullet points and formatting
- Include practical examples where relevant
- Maintain technical accuracy
- Ensure completeness based on available context
- Cite sources when specific information is referenced

**Response:**"""
        )
        
        # Comparison Template
        templates[ResponseTemplateType.COMPARISON.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant. Compare {concept_a} and {concept_b} based on the provided context.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Provide a comprehensive comparison addressing:

## Key Differences

### Approach
- How they differ fundamentally
- Core architectural/methodological distinctions

### Use Cases
- When to use {concept_a}
- When to use {concept_b}
- Situational advantages of each

### Advantages & Disadvantages
- Strengths and limitations of {concept_a}
- Strengths and limitations of {concept_b}
- Trade-offs to consider

## Similarities
- Common ground and overlapping features
- Shared principles or foundations

## Recommendation
Based on the context, provide guidance on selection criteria.

**Requirements:**
- Maintain objectivity
- Use specific examples from the context
- Highlight practical implications
- Cite sources for claims

**Response:**"""
        )
        
        # Process Explanation Template
        templates[ResponseTemplateType.PROCESS_EXPLANATION.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant. Explain how {process} works based on the provided context.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Provide a comprehensive explanation covering:

## Process Overview
- High-level workflow description
- Key objectives and outcomes

## Detailed Steps
1. **Initialization/Setup**
   - Prerequisites and requirements
   - Initial configurations

2. **Core Process Flow**
   - Step-by-step breakdown
   - Decision points and branches
   - Data flow and transformations

3. **Completion/Results**
   - Final outputs
   - Success criteria
   - Validation steps

## Benefits & Impact
- Why this process matters
- Performance improvements
- Business/technical value

## Common Challenges
- Potential issues and solutions
- Best practices and recommendations

**Requirements:**
- Use numbered steps for clarity
- Include practical examples
- Explain technical terms
- Reference specific context information

**Response:**"""
        )
        
        # Definition Template
        templates[ResponseTemplateType.DEFINITION.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant. Provide a comprehensive definition of {concept} based on the context.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Structure your response as follows:

## Definition
Clear, concise definition of {concept}

## Key Characteristics
- Essential properties and features
- Distinguishing attributes
- Technical specifications (if applicable)

## Context & Applications
- Where and how it's used
- Industry or domain relevance
- Real-world examples

## Related Concepts
- Associated terms and technologies
- Hierarchical relationships
- Dependencies or prerequisites

## Practical Implications
- Why understanding this concept matters
- Impact on related systems or processes

**Requirements:**
- Start with a clear, standalone definition
- Use accessible language while maintaining technical accuracy
- Include concrete examples from the context
- Organize information logically

**Response:**"""
        )
        
        # Procedural Template
        templates[ResponseTemplateType.PROCEDURAL.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant. Provide step-by-step instructions for {task} based on the context.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Create a comprehensive procedural guide:

## Prerequisites
- Required tools, software, or permissions
- Knowledge or skill requirements
- Environmental setup needs

## Step-by-Step Instructions

### Phase 1: Preparation
1. [First preparation step]
2. [Second preparation step]
   - Sub-steps if needed
   - Important notes or warnings

### Phase 2: Implementation
[Continue with numbered steps...]

### Phase 3: Verification
- How to confirm success
- Testing procedures
- Validation criteria

## Troubleshooting
- Common issues and solutions
- Error messages and remedies
- When to seek additional help

## Best Practices
- Recommended approaches
- Security considerations
- Performance optimization tips

**Requirements:**
- Use clear, actionable language
- Number all major steps
- Include warnings for critical steps
- Provide verification methods
- Reference specific context details

**Response:**"""
        )
        
        # Analytical Template
        templates[ResponseTemplateType.ANALYTICAL.value] = ChatPromptTemplate.from_template(
            """You are an expert technical analyst. Provide a comprehensive analysis of {topic} based on the context.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Deliver a thorough analytical response:

## Executive Summary
- Key findings and insights
- Primary conclusions
- Critical implications

## Detailed Analysis

### Current State Assessment
- Present situation overview
- Key metrics and indicators
- Strengths and weaknesses

### Multiple Perspectives
- Stakeholder viewpoints
- Technical considerations
- Business implications
- Risk factors

### Comparative Analysis
- Benchmarking against alternatives
- Industry standards comparison
- Historical context

## Recommendations
- Actionable insights
- Strategic suggestions
- Implementation considerations
- Risk mitigation strategies

## Future Considerations
- Emerging trends
- Potential developments
- Long-term implications

**Requirements:**
- Support all claims with context evidence
- Present balanced viewpoints
- Use data and specific examples
- Maintain analytical objectivity
- Provide actionable insights

**Response:**"""
        )
        
        # Multi-hop Template
        templates[ResponseTemplateType.MULTI_HOP.value] = ChatPromptTemplate.from_template(
            """You are an expert technical assistant capable of complex reasoning. Address the multi-faceted query about {topic}.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
This query requires connecting multiple concepts. Structure your response to address each aspect:

## Query Decomposition
Breaking down the question into key components:
- [Component 1]: [Brief description]
- [Component 2]: [Brief description]
- [Connections]: How these components relate

## Comprehensive Analysis

### Individual Component Analysis
[Address each component separately with relevant context]

### Interconnection Analysis
- How components influence each other
- Dependencies and relationships
- Synergistic effects or conflicts

### Synthesis
- Integrated understanding
- Holistic perspective
- Emergent properties or insights

## Practical Applications
- Real-world implementations
- Use cases that demonstrate connections
- Examples from the provided context

## Implications
- Technical implications
- Strategic considerations
- Future developments

**Requirements:**
- Clearly show reasoning connections
- Address all aspects of the multi-part query
- Use context to support interconnected analysis
- Maintain logical flow between sections
- Provide concrete examples

**Response:**"""
        )
        
        # Technical Deep Dive Template
        templates[ResponseTemplateType.TECHNICAL_DEEP_DIVE.value] = ChatPromptTemplate.from_template(
            """You are a senior technical expert. Provide an in-depth technical analysis of {topic}.

**Context Information:**
{context}

**User Query:** {query}

**Instructions:**
Deliver a comprehensive technical deep dive:

## Technical Overview
- Core technical concepts
- Architecture and design principles
- Key technologies involved

## Implementation Details
- Technical specifications
- Configuration requirements
- Code examples or pseudo-code (if applicable)
- Integration considerations

## Advanced Considerations
- Performance characteristics
- Scalability factors
- Security implications
- Monitoring and observability

## Technical Trade-offs
- Design decisions and rationale
- Alternative approaches
- Performance vs. complexity considerations
- Cost implications

## Expert Recommendations
- Best practices from industry experience
- Common pitfalls to avoid
- Optimization strategies
- Future-proofing considerations

## Additional Resources
- Related technologies to explore
- Recommended reading or documentation
- Community resources

**Requirements:**
- Assume advanced technical knowledge
- Include specific technical details from context
- Explain complex concepts thoroughly
- Provide actionable technical guidance
- Reference industry standards and practices

**Response:**"""
        )
        
        return templates
    
    def get_template(self, template_type: str) -> Optional[ChatPromptTemplate]:
        """Get a specific template by type."""
        return self.templates.get(template_type)
    
    def get_template_for_query_type(self, query_type: str, query_content: str = "") -> ChatPromptTemplate:
        """Select the most appropriate template based on query type and content."""
        
        # Map query types to templates
        type_mapping = {
            "factual": ResponseTemplateType.DEFINITION.value,
            "procedural": ResponseTemplateType.PROCEDURAL.value,
            "analytical": ResponseTemplateType.ANALYTICAL.value,
            "comparison": ResponseTemplateType.COMPARISON.value,
            "multi_hop": ResponseTemplateType.MULTI_HOP.value,
            "conversational": ResponseTemplateType.CONVERSATIONAL.value
        }
        
        # Content-based template selection
        query_lower = query_content.lower()
        
        if any(word in query_lower for word in ["components", "parts", "elements", "consists of"]):
            template_type = ResponseTemplateType.COMPONENT_LISTING.value
        elif any(word in query_lower for word in ["vs", "versus", "compared to", "difference", "compare"]):
            template_type = ResponseTemplateType.COMPARISON.value
        elif any(word in query_lower for word in ["how does", "process", "workflow", "steps"]):
            template_type = ResponseTemplateType.PROCESS_EXPLANATION.value
        elif any(word in query_lower for word in ["what is", "define", "definition", "meaning"]):
            template_type = ResponseTemplateType.DEFINITION.value
        elif any(word in query_lower for word in ["how to", "steps to", "guide", "tutorial"]):
            template_type = ResponseTemplateType.PROCEDURAL.value
        elif any(word in query_lower for word in ["technical", "architecture", "implementation", "deep dive"]):
            template_type = ResponseTemplateType.TECHNICAL_DEEP_DIVE.value
        else:
            # Use query type mapping
            template_type = type_mapping.get(query_type, ResponseTemplateType.ANALYTICAL.value)
        
        return self.get_template(template_type) or self.get_template(ResponseTemplateType.ANALYTICAL.value)
    
    def build_template_variables(self, 
                                contextual_info: Any,
                                query_analysis: Any, 
                                user_profile: Optional[Dict[str, Any]] = None,
                                retrieved_documents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Build variables for template rendering."""
        
        # Extract key concepts for comparison templates
        query_text = query_analysis.original_query
        concepts = self._extract_concepts_for_comparison(query_text)
        
        # Build comprehensive context
        context_parts = []
        if hasattr(contextual_info, 'primary_context') and contextual_info.primary_context:
            context_parts.append(f"Primary Context:\n{contextual_info.primary_context}")
        
        if hasattr(contextual_info, 'supporting_context') and contextual_info.supporting_context:
            for i, context in enumerate(contextual_info.supporting_context[:3]):
                context_parts.append(f"Supporting Context {i+1}:\n{context}")
        
        if retrieved_documents:
            for i, doc in enumerate(retrieved_documents[:3]):
                title = doc.get('title', f'Document {i+1}')
                content = doc.get('content', '')[:500] + ('...' if len(doc.get('content', '')) > 500 else '')
                context_parts.append(f"Source: {title}\n{content}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "No specific context provided."
        
        # Extract main topic
        topic = self._extract_main_topic(query_text, query_analysis)
        
        variables = {
            "context": context_text,
            "query": query_text,
            "topic": topic,
            "process": topic,
            "concept": topic,
            "task": topic,
            "concept_a": concepts.get("concept_a", "first concept"),
            "concept_b": concepts.get("concept_b", "second concept")
        }
        
        # Add user profile context if available
        if user_profile:
            variables["user_context"] = f"User Profile: {user_profile.get('expertise_level', 'general')} level"
        
        return variables
    
    def _extract_concepts_for_comparison(self, query_text: str) -> Dict[str, str]:
        """Extract concepts for comparison templates."""
        
        # Look for comparison indicators
        comparison_patterns = [
            r"(.+?)\s+vs\s+(.+)",
            r"(.+?)\s+versus\s+(.+)",
            r"difference between\s+(.+?)\s+and\s+(.+)",
            r"compare\s+(.+?)\s+(?:to|with)\s+(.+)",
            r"(.+?)\s+compared to\s+(.+)"
        ]
        
        import re
        for pattern in comparison_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                return {
                    "concept_a": match.group(1).strip(),
                    "concept_b": match.group(2).strip()
                }
        
        # Default fallback
        words = query_text.split()
        if len(words) >= 4:
            return {
                "concept_a": " ".join(words[:len(words)//2]),
                "concept_b": " ".join(words[len(words)//2:])
            }
        
        return {"concept_a": "first concept", "concept_b": "second concept"}
    
    def _extract_main_topic(self, query_text: str, query_analysis: Any) -> str:
        """Extract the main topic from the query."""
        
        # Try to get from query analysis if available
        if hasattr(query_analysis, 'topics') and query_analysis.topics:
            return query_analysis.topics[0]
        
        if hasattr(query_analysis, 'entities') and query_analysis.entities:
            return query_analysis.entities[0]
        
        # Fallback: extract from query text
        # Remove question words and focus on the main subject
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        words = [word for word in query_text.lower().split() 
                if word not in question_words and len(word) > 2]
        
        if words:
            # Return first few meaningful words
            return " ".join(words[:3])
        
        return "the topic"
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template types."""
        return list(self.templates.keys())
    
    def validate_template_variables(self, template_type: str, variables: Dict[str, Any]) -> bool:
        """Validate that all required variables are present for a template."""
        
        template = self.get_template(template_type)
        if not template:
            return False
        
        try:
            # Try to format the template with provided variables
            template.format(**variables)
            return True
        except KeyError:
            return False
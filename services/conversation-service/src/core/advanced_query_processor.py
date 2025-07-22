from typing import List, Dict, Any, Optional, Tuple, Enum
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import asyncio
import re
import json
from datetime import datetime

class QueryType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"
    MULTI_HOP = "multi_hop"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"

@dataclass
class QueryAnalysis:
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    intent: str
    entities: List[str]
    topics: List[str]
    sentiment: str
    urgency: str
    expanded_queries: List[str]
    keywords: List[str]
    context_needed: bool
    confidence: float

class AdvancedQueryProcessor:
    """Advanced query processing with intent detection, entity extraction, and query expansion."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.intent_classifier = self._create_intent_classifier()
        self.query_expander = self._create_query_expander()
        self.entity_extractor = self._create_entity_extractor()
        self.complexity_analyzer = self._create_complexity_analyzer()
        
    def _create_intent_classifier(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert query classifier. Analyze the user's query and classify it.

Classify the query type as one of:
- factual: Asking for specific facts or information
- procedural: Asking how to do something step-by-step
- analytical: Asking for analysis, comparison, or reasoning
- conversational: General conversation or greeting
- clarification: Asking for clarification on previous topics
- multi_hop: Requires connecting multiple pieces of information

Also determine:
- Intent: What the user wants to achieve (1-2 words)
- Sentiment: positive, neutral, negative, frustrated, excited
- Urgency: low, medium, high, critical
- Entities: Extract key entities (names, dates, products, etc.)
- Topics: Main topics or domains mentioned
- Keywords: Most important search terms

Return a JSON object with these fields:
{
    "query_type": "factual|procedural|analytical|conversational|clarification|multi_hop",
    "intent": "brief intent description",
    "sentiment": "positive|neutral|negative|frustrated|excited",
    "urgency": "low|medium|high|critical",
    "entities": ["entity1", "entity2"],
    "topics": ["topic1", "topic2"],
    "keywords": ["keyword1", "keyword2"],
    "confidence": 0.95
}"""),
            ("human", "Classify this query: {query}")
        ])
    
    def _create_query_expander(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a query expansion expert. Given a user query and context, generate multiple optimized search queries.

Generate 3-5 alternative queries that would retrieve comprehensive information:
1. Keyword-focused: Extract and rearrange key terms
2. Semantic: Rephrase using synonyms and related concepts
3. Specific: Make more specific with domain terms
4. Broader: Generalize to capture related information
5. Question-based: Convert to natural questions if applicable

Consider:
- Conversation history for context
- Domain-specific terminology
- Different ways people might phrase the same question
- Related concepts that might be relevant

Return only the queries, one per line, without numbering or explanation."""),
            ("human", "Original query: {query}\n\nConversation context: {context}\n\nExpand this query:")
        ])
    
    def _create_entity_extractor(self):
        return ChatPromptTemplate.from_messages([
            ("system", """Extract entities from the user query. Focus on:
- People (names, roles, titles)
- Organizations (companies, institutions)
- Locations (cities, countries, addresses)
- Products/Services (software, tools, services)
- Technical terms (APIs, protocols, technologies)
- Dates/Times (when mentioned)
- Numbers/Quantities (amounts, versions, IDs)

Return a JSON list of entities with their types:
[
    {"entity": "entity_name", "type": "person|organization|location|product|technical|date|number|other"},
    ...
]"""),
            ("human", "Extract entities from: {query}")
        ])
    
    def _create_complexity_analyzer(self):
        return ChatPromptTemplate.from_messages([
            ("system", """Analyze the complexity of this query and determine retrieval strategy.

Complexity levels:
- simple: Single concept, direct answer
- moderate: 2-3 concepts, some reasoning needed
- complex: Multiple concepts, requires synthesis
- multi_step: Requires multiple retrieval rounds

Determine:
- Complexity level
- Whether conversation context is needed
- Number of retrieval rounds needed
- Special handling requirements

Return JSON:
{
    "complexity": "simple|moderate|complex|multi_step",
    "context_needed": true|false,
    "retrieval_rounds": 1-3,
    "special_handling": ["temporal", "comparative", "sequential", "aggregative"]
}"""),
            ("human", "Analyze complexity: {query}")
        ])
    
    async def analyze_query(self, query: str, conversation_context: str = "") -> QueryAnalysis:
        """Comprehensive query analysis combining all components."""
        try:
            # Run analysis tasks concurrently
            classification_task = self._classify_intent(query)
            expansion_task = self._expand_query(query, conversation_context)
            entity_task = self._extract_entities(query)
            complexity_task = self._analyze_complexity(query)
            
            # Wait for all tasks to complete
            classification, expanded_queries, entities, complexity_info = await asyncio.gather(
                classification_task,
                expansion_task,
                entity_task,
                complexity_task
            )
            
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType(classification.get("query_type", "factual")),
                complexity=QueryComplexity(complexity_info.get("complexity", "simple")),
                intent=classification.get("intent", "information"),
                entities=entities,
                topics=classification.get("topics", []),
                sentiment=classification.get("sentiment", "neutral"),
                urgency=classification.get("urgency", "medium"),
                expanded_queries=expanded_queries,
                keywords=classification.get("keywords", []),
                context_needed=complexity_info.get("context_needed", False),
                confidence=classification.get("confidence", 0.8)
            )
            
        except Exception as e:
            # Fallback to basic analysis
            return self._fallback_analysis(query)
    
    async def _classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent and extract metadata."""
        try:
            response = await self.intent_classifier.ainvoke({"query": query})
            return json.loads(response.content)
        except Exception:
            return self._basic_classification(query)
    
    async def _expand_query(self, query: str, context: str = "") -> List[str]:
        """Generate expanded query variations."""
        try:
            response = await self.query_expander.ainvoke({
                "query": query,
                "context": context[:500]  # Limit context length
            })
            
            # Parse expanded queries
            expanded = response.content.strip().split('\n')
            return [q.strip() for q in expanded if q.strip() and q.strip() != query]
            
        except Exception:
            return self._basic_expansion(query)
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        try:
            response = await self.entity_extractor.ainvoke({"query": query})
            entities_data = json.loads(response.content)
            return [e["entity"] for e in entities_data if isinstance(e, dict) and "entity" in e]
        except Exception:
            return self._basic_entity_extraction(query)
    
    async def _analyze_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity."""
        try:
            response = await self.complexity_analyzer.ainvoke({"query": query})
            return json.loads(response.content)
        except Exception:
            return {"complexity": "moderate", "context_needed": True, "retrieval_rounds": 1}
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback analysis using rule-based methods."""
        return QueryAnalysis(
            original_query=query,
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.MODERATE,
            intent="information",
            entities=self._basic_entity_extraction(query),
            topics=["general"],
            sentiment="neutral",
            urgency="medium",
            expanded_queries=self._basic_expansion(query),
            keywords=self._extract_keywords(query),
            context_needed=True,
            confidence=0.6
        )
    
    def _basic_classification(self, query: str) -> Dict[str, Any]:
        """Basic rule-based classification."""
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ["how to", "how do", "steps", "process"]):
            query_type = "procedural"
        elif any(word in query_lower for word in ["what is", "who is", "when", "where"]):
            query_type = "factual"
        elif any(word in query_lower for word in ["compare", "analyze", "evaluate", "why"]):
            query_type = "analytical"
        elif any(word in query_lower for word in ["hello", "hi", "thanks", "bye"]):
            query_type = "conversational"
        else:
            query_type = "factual"
        
        # Determine sentiment
        if any(word in query_lower for word in ["urgent", "asap", "immediately", "critical"]):
            urgency = "high"
            sentiment = "urgent"
        elif any(word in query_lower for word in ["problem", "issue", "error", "broken", "failed"]):
            urgency = "medium"
            sentiment = "frustrated"
        else:
            urgency = "medium"
            sentiment = "neutral"
        
        return {
            "query_type": query_type,
            "intent": "information",
            "sentiment": sentiment,
            "urgency": urgency,
            "entities": [],
            "topics": ["general"],
            "keywords": self._extract_keywords(query),
            "confidence": 0.7
        }
    
    def _basic_expansion(self, query: str) -> List[str]:
        """Basic query expansion using keyword variations."""
        # Simple synonym replacement and keyword extraction
        expanded = []
        
        # Add question variations
        if not query.lower().startswith(("what", "how", "when", "where", "why", "who")):
            expanded.append(f"What is {query}?")
            expanded.append(f"How to {query}")
        
        # Add keyword-focused version
        keywords = self._extract_keywords(query)
        if len(keywords) > 1:
            expanded.append(" ".join(keywords))
        
        return expanded[:3]  # Limit to 3 expansions
    
    def _basic_entity_extraction(self, query: str) -> List[str]:
        """Basic entity extraction using patterns."""
        entities = []
        
        # Extract capitalized words (likely proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized)
        
        # Extract potential version numbers, IDs
        versions = re.findall(r'\bv?\d+\.\d+\b|\b\d+\.\d+\.\d+\b', query)
        entities.extend(versions)
        
        # Extract quoted terms
        quoted = re.findall(r'["\'](["\']+)["\']', query)
        entities.extend(quoted)
        
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove stop words and extract meaningful terms
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def get_retrieval_strategy(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Determine optimal retrieval strategy based on query analysis."""
        strategy = {
            "primary_queries": [analysis.original_query] + analysis.expanded_queries[:2],
            "keywords": analysis.keywords,
            "entities": analysis.entities,
            "use_semantic_search": True,
            "use_keyword_search": True,
            "reranking_needed": analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_STEP],
            "context_window": 5 if analysis.context_needed else 3,
            "retrieval_rounds": 2 if analysis.complexity == QueryComplexity.MULTI_STEP else 1,
            "filters": self._generate_filters(analysis)
        }
        
        # Adjust based on query type
        if analysis.query_type == QueryType.PROCEDURAL:
            strategy["prefer_sequential"] = True
            strategy["context_window"] = 7
        elif analysis.query_type == QueryType.ANALYTICAL:
            strategy["use_comparative_retrieval"] = True
            strategy["context_window"] = 10
        elif analysis.query_type == QueryType.MULTI_HOP:
            strategy["retrieval_rounds"] = 3
            strategy["use_graph_traversal"] = True
        
        return strategy
    
    def _generate_filters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Generate filters based on extracted entities and topics."""
        filters = {}
        
        # Add topic filters if specific topics identified
        if analysis.topics and "general" not in analysis.topics:
            filters["topics"] = analysis.topics
        
        # Add entity-based filters
        if analysis.entities:
            filters["entities"] = analysis.entities
        
        # Add urgency-based boosting
        if analysis.urgency in ["high", "critical"]:
            filters["boost_recent"] = True
        
        return filters
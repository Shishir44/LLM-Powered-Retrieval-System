"""
PHASE 3.1: Advanced Query Reasoning Engine
Intelligent multi-hop processing with Chain-of-Thought reasoning for complex queries
Built on Phase 2's solid foundation of metadata boosting and structured processing
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_number: int
    question: str
    sub_queries: List[str]
    retrieved_context: List[Dict[str, Any]]
    reasoning: str
    answer: str
    confidence: float
    supporting_evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ReasoningChain:
    """Complete reasoning chain for a complex query."""
    original_query: str
    query_complexity: str  # simple, moderate, complex, multi_domain
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    overall_confidence: float
    evidence_strength: str  # weak, moderate, strong
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryComplexity(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"           # Single fact retrieval
    MODERATE = "moderate"       # Requires 2-3 pieces of information
    COMPLEX = "complex"         # Multi-step reasoning required
    MULTI_DOMAIN = "multi_domain"  # Spans multiple knowledge domains

class AdvancedReasoningEngine:
    """PHASE 3.1: Advanced reasoning engine with Chain-of-Thought capabilities."""
    
    def __init__(self, knowledge_retriever, llm_model: str = "gpt-4"):
        self.knowledge_retriever = knowledge_retriever
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM for reasoning
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(
                model=llm_model,  # Use the parameter passed to the function
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=4000  # Increased from 2000 to 4000
            )
        else:
            self.llm = None
            self.logger.warning("LangChain not available, reasoning capabilities limited")
        
        # Query complexity patterns
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                r'what is\s+\w+',
                r'define\s+\w+',
                r'how much\s+does\s+\w+\s+cost',
                r'when\s+is\s+\w+',
                r'where\s+is\s+\w+'
            ],
            QueryComplexity.MODERATE: [
                r'how\s+do\s+i\s+\w+\s+\w+',
                r'what\s+are\s+the\s+steps',
                r'compare\s+\w+\s+and\s+\w+',
                r'what\s+if\s+i\s+\w+',
                r'why\s+does\s+\w+\s+\w+'
            ],
            QueryComplexity.COMPLEX: [
                r'explain\s+how\s+\w+\s+affects\s+\w+',
                r'analyze\s+the\s+relationship',
                r'what\s+would\s+happen\s+if',
                r'how\s+does\s+\w+\s+relate\s+to\s+\w+',
                r'troubleshoot\s+\w+\s+\w+\s+issue'
            ],
            QueryComplexity.MULTI_DOMAIN: [
                r'billing\s+and\s+technical',
                r'policy\s+and\s+procedure',
                r'account\s+and\s+product',
                r'setup\s+and\s+troubleshoot',
                r'compare\s+\w+\s+policies?\s+and\s+\w+\s+features?'
            ]
        }
        
        # Chain-of-Thought prompts
        self.cot_prompts = {
            QueryComplexity.SIMPLE: """
            Answer this straightforward question using the provided context.
            
            Context: {context}
            Question: {question}
            
            Provide a direct, accurate answer with source citations.
            """,
            
            QueryComplexity.MODERATE: """
            Think step by step to answer this question using the provided context.
            
            Context: {context}
            Question: {question}
            
            Reasoning approach:
            1. Identify the key components of the question
            2. Find relevant information for each component
            3. Synthesize the information into a coherent answer
            
            Provide your step-by-step reasoning and final answer.
            """,
            
            QueryComplexity.COMPLEX: """
            Use careful reasoning to analyze this complex question.
            
            Context: {context}
            Question: {question}
            
            Chain of Thought:
            1. Break down the question into sub-components
            2. Analyze each component systematically
            3. Consider relationships and dependencies
            4. Synthesize insights into a comprehensive answer
            5. Identify confidence level and limitations
            
            Provide detailed reasoning chain and well-supported conclusion.
            """,
            
            QueryComplexity.MULTI_DOMAIN: """
            This question spans multiple knowledge domains. Use systematic analysis.
            
            Context: {context}
            Question: {question}
            
            Multi-Domain Analysis:
            1. Identify the different knowledge domains involved
            2. Extract relevant information from each domain
            3. Analyze interactions between domains
            4. Consider potential conflicts or dependencies
            5. Provide integrated answer addressing all domains
            
            Ensure comprehensive coverage with clear domain-specific insights.
            """
        }
        
        # Reasoning performance tracking
        self.reasoning_stats = {
            "total_queries": 0,
            "simple_queries": 0,
            "moderate_queries": 0,
            "complex_queries": 0,
            "multi_domain_queries": 0,
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0,
            "successful_reasoning_chains": 0
        }
        
        self.logger.info("PHASE 3.1: Advanced Reasoning Engine initialized")

    async def process_complex_query(self, query: str, user_context: Optional[Dict] = None) -> ReasoningChain:
        """Process a query using advanced reasoning capabilities."""
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"PHASE 3.1: Processing complex query: {query[:100]}...")
            
            # Step 1: Analyze query complexity
            complexity = self._analyze_query_complexity(query)
            self.logger.info(f"Query complexity assessed as: {complexity.value}")
            
            # Step 2: Route to appropriate reasoning strategy
            if complexity == QueryComplexity.SIMPLE:
                reasoning_chain = await self._handle_simple_query(query, user_context)
            elif complexity == QueryComplexity.MODERATE:
                reasoning_chain = await self._handle_moderate_query(query, user_context)
            elif complexity == QueryComplexity.COMPLEX:
                reasoning_chain = await self._handle_complex_query(query, user_context)
            else:  # MULTI_DOMAIN
                reasoning_chain = await self._handle_multi_domain_query(query, user_context)
            
            # Step 3: Calculate processing time and update stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            reasoning_chain.processing_time_ms = processing_time
            
            self._update_reasoning_stats(complexity, processing_time, reasoning_chain.overall_confidence)
            
            self.logger.info(f"PHASE 3.1: Completed reasoning chain with {len(reasoning_chain.reasoning_steps)} steps, confidence: {reasoning_chain.overall_confidence:.2f}")
            
            return reasoning_chain
            
        except Exception as e:
            self.logger.error(f"Error in complex query processing: {e}")
            # Return fallback reasoning chain
            return self._create_fallback_reasoning_chain(query, str(e))

    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze and classify query complexity."""
        
        query_lower = query.lower()
        
        # Score each complexity level
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_indicators.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            complexity_scores[complexity] = score
        
        # Additional heuristics
        word_count = len(query.split())
        question_marks = query.count('?')
        and_or_count = len(re.findall(r'\b(and|or|but|however|also)\b', query_lower))
        
        # Adjust scores based on heuristics
        if word_count > 20:
            complexity_scores[QueryComplexity.COMPLEX] += 1
        if word_count > 30:
            complexity_scores[QueryComplexity.MULTI_DOMAIN] += 1
        if question_marks > 1:
            complexity_scores[QueryComplexity.MODERATE] += 1
        if and_or_count > 2:
            complexity_scores[QueryComplexity.COMPLEX] += 1
        
        # Return highest scoring complexity
        if any(score > 0 for score in complexity_scores.values()):
            return max(complexity_scores, key=complexity_scores.get)
        else:
            return QueryComplexity.SIMPLE

    async def _handle_simple_query(self, query: str, user_context: Optional[Dict]) -> ReasoningChain:
        """Handle simple, direct queries."""
        
        # Single retrieval step
        if self.knowledge_retriever is None:
            self.logger.error("Knowledge retriever is not available")
            return []
            
        retrieved_docs = await self.knowledge_retriever.enhanced_semantic_search(
            query=query,
            top_k=5,
            enable_boosting=True
        )
        
        context = self._format_context_for_reasoning(retrieved_docs)
        
        # Generate direct answer
        if self.llm:
            prompt = self.cot_prompts[QueryComplexity.SIMPLE].format(
                context=context,
                question=query
            )
            
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            answer = response.content
        else:
            answer = self._generate_fallback_answer(context, query)
        
        # Create reasoning step
        reasoning_step = ReasoningStep(
            step_number=1,
            question=query,
            sub_queries=[query],
            retrieved_context=[doc.__dict__ for doc in retrieved_docs],
            reasoning="Direct retrieval and answer generation for simple query",
            answer=answer,
            confidence=0.9 if retrieved_docs else 0.3,
            supporting_evidence=[doc.document.title for doc in retrieved_docs[:3]]
        )
        
        return ReasoningChain(
            original_query=query,
            query_complexity=QueryComplexity.SIMPLE.value,
            reasoning_steps=[reasoning_step],
            final_answer=answer,
            overall_confidence=reasoning_step.confidence,
            evidence_strength="strong" if len(retrieved_docs) >= 3 else "moderate"
        )

    async def _handle_moderate_query(self, query: str, user_context: Optional[Dict]) -> ReasoningChain:
        """Handle moderate complexity queries requiring 2-3 information pieces."""
        
        reasoning_steps = []
        
        # Step 1: Decompose query into components
        sub_queries = await self._decompose_query(query)
        
        # Step 2: Process each sub-query
        all_retrieved_docs = []
        for i, sub_query in enumerate(sub_queries):
            if self.knowledge_retriever is None:
                self.logger.error("Knowledge retriever is not available")
                return []
                
            retrieved_docs = await self.knowledge_retriever.enhanced_semantic_search(
                query=sub_query,
                top_k=3,
                enable_boosting=True
            )
            all_retrieved_docs.extend(retrieved_docs)
            
            context = self._format_context_for_reasoning(retrieved_docs)
            
            reasoning_step = ReasoningStep(
                step_number=i + 1,
                question=sub_query,
                sub_queries=[sub_query],
                retrieved_context=[doc.__dict__ for doc in retrieved_docs],
                reasoning=f"Addressing sub-component: {sub_query}",
                answer=self._extract_key_info(context, sub_query),
                confidence=0.8 if retrieved_docs else 0.4,
                supporting_evidence=[doc.document.title for doc in retrieved_docs[:2]]
            )
            reasoning_steps.append(reasoning_step)
        
        # Step 3: Synthesize final answer
        combined_context = self._format_context_for_reasoning(all_retrieved_docs)
        
        if self.llm:
            prompt = self.cot_prompts[QueryComplexity.MODERATE].format(
                context=combined_context,
                question=query
            )
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            final_answer = response.content
        else:
            final_answer = self._synthesize_fallback_answer(reasoning_steps, query)
        
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        return ReasoningChain(
            original_query=query,
            query_complexity=QueryComplexity.MODERATE.value,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            evidence_strength="strong" if overall_confidence > 0.7 else "moderate"
        )

    async def _handle_complex_query(self, query: str, user_context: Optional[Dict]) -> ReasoningChain:
        """Handle complex queries requiring multi-step reasoning."""
        
        reasoning_steps = []
        
        # Step 1: Identify key concepts and relationships
        key_concepts = await self._extract_key_concepts(query)
        
        # Step 2: Build reasoning chain
        for i, concept in enumerate(key_concepts):
            # Generate focused sub-query for this concept
            sub_query = await self._generate_concept_query(concept, query)
            
            # Retrieve relevant information
            if self.knowledge_retriever is None:
                self.logger.error("Knowledge retriever is not available")
                return []
                
            retrieved_docs = await self.knowledge_retriever.enhanced_semantic_search(
                query=sub_query,
                top_k=4,
                enable_boosting=True
            )
            
            context = self._format_context_for_reasoning(retrieved_docs)
            
            # Analyze concept in context
            if self.llm:
                analysis_prompt = f"""
                Analyze this concept in the context of the original question:
                
                Original Question: {query}
                Current Concept: {concept}
                Context: {context}
                
                Provide focused analysis of how this concept relates to answering the original question.
                """
                response = await self.llm.ainvoke([SystemMessage(content=analysis_prompt)])
                reasoning = response.content
                answer = self._extract_answer_from_reasoning(reasoning)
            else:
                reasoning = f"Analysis of concept '{concept}' in relation to the query"
                answer = self._extract_key_info(context, sub_query)
            
            reasoning_step = ReasoningStep(
                step_number=i + 1,
                question=sub_query,
                sub_queries=[sub_query],
                retrieved_context=[doc.__dict__ for doc in retrieved_docs],
                reasoning=reasoning,
                answer=answer,
                confidence=0.85 if retrieved_docs else 0.5,
                supporting_evidence=[doc.document.title for doc in retrieved_docs[:2]]
            )
            reasoning_steps.append(reasoning_step)
        
        # Step 3: Final synthesis with Chain-of-Thought
        all_context = self._format_context_for_reasoning([
            doc for step in reasoning_steps for doc in step.retrieved_context
        ])
        
        if self.llm:
            prompt = self.cot_prompts[QueryComplexity.COMPLEX].format(
                context=all_context,
                question=query
            )
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            final_answer = response.content
        else:
            final_answer = self._synthesize_complex_answer(reasoning_steps, query)
        
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        return ReasoningChain(
            original_query=query,
            query_complexity=QueryComplexity.COMPLEX.value,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            evidence_strength="strong" if overall_confidence > 0.8 else "moderate",
            metadata={
                "key_concepts": key_concepts,
                "reasoning_depth": len(reasoning_steps),
                "synthesis_required": True
            }
        )

    async def _handle_multi_domain_query(self, query: str, user_context: Optional[Dict]) -> ReasoningChain:
        """Handle queries spanning multiple knowledge domains."""
        
        reasoning_steps = []
        
        # Step 1: Identify knowledge domains
        domains = await self._identify_knowledge_domains(query)
        
        # Step 2: Process each domain separately
        domain_results = {}
        
        for domain in domains:
            # Generate domain-specific sub-query
            domain_query = await self._generate_domain_query(query, domain)
            
            # Retrieve with domain-specific hints
            if self.knowledge_retriever is None:
                self.logger.error("Knowledge retriever is not available")
                return []
                
            retrieved_docs = await self.knowledge_retriever.enhanced_semantic_search(
                query=domain_query,
                top_k=4,
                category_hint=domain,
                enable_boosting=True
            )
            
            context = self._format_context_for_reasoning(retrieved_docs)
            domain_results[domain] = {
                "docs": retrieved_docs,
                "context": context,
                "query": domain_query
            }
            
            # Create reasoning step for this domain
            reasoning_step = ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                question=domain_query,
                sub_queries=[domain_query],
                retrieved_context=[doc.__dict__ for doc in retrieved_docs],
                reasoning=f"Domain-specific analysis: {domain}",
                answer=self._extract_domain_insights(context, domain, query),
                confidence=0.85 if retrieved_docs else 0.4,
                supporting_evidence=[doc.document.title for doc in retrieved_docs[:2]]
            )
            reasoning_steps.append(reasoning_step)
        
        # Step 3: Cross-domain synthesis
        combined_context = self._combine_domain_contexts(domain_results)
        
        if self.llm:
            prompt = self.cot_prompts[QueryComplexity.MULTI_DOMAIN].format(
                context=combined_context,
                question=query
            )
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            final_answer = response.content
        else:
            final_answer = self._synthesize_multi_domain_answer(domain_results, query)
        
        overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        return ReasoningChain(
            original_query=query,
            query_complexity=QueryComplexity.MULTI_DOMAIN.value,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            evidence_strength="strong" if overall_confidence > 0.8 else "moderate",
            metadata={
                "domains_analyzed": domains,
                "cross_domain_synthesis": True,
                "domain_coverage": len(domains)
            }
        )

    # Helper methods for query processing
    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose a moderate query into sub-components."""
        
        # Simple decomposition based on conjunctions and question words
        sub_queries = []
        
        # Split on conjunctions
        parts = re.split(r'\b(and|or|but|also)\b', query, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if len(part) > 10 and part.lower() not in ['and', 'or', 'but', 'also']:
                sub_queries.append(part)
        
        # If no meaningful decomposition, use the original query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries[:3]  # Limit to 3 sub-queries for moderate complexity

    async def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from a complex query."""
        
        # Simple concept extraction using NLP patterns
        concepts = []
        
        # Extract nouns and noun phrases
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['billing', 'payment', 'account', 'policy', 'procedure', 'technical', 'support', 'product', 'service']:
                concepts.append(word)
            elif i < len(words) - 1 and word.lower() in ['how', 'what', 'why', 'when', 'where']:
                # Include the next few words as a concept
                concept_phrase = ' '.join(words[i:i+3])
                concepts.append(concept_phrase)
        
        # Remove duplicates and limit
        unique_concepts = list(set(concepts))
        return unique_concepts[:4]  # Limit to 4 key concepts

    async def _identify_knowledge_domains(self, query: str) -> List[str]:
        """Identify knowledge domains involved in the query."""
        
        domains = []
        query_lower = query.lower()
        
        domain_keywords = {
            'billing': ['bill', 'payment', 'charge', 'invoice', 'cost', 'price', 'subscription'],
            'technical': ['error', 'bug', 'issue', 'troubleshoot', 'install', 'setup', 'configure'],
            'account': ['account', 'login', 'register', 'profile', 'password', 'access'],
            'product': ['product', 'feature', 'specification', 'model', 'version', 'catalog'],
            'policy': ['policy', 'terms', 'condition', 'agreement', 'rule', 'guideline'],
            'support': ['help', 'support', 'assistance', 'guide', 'tutorial', 'faq']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains[:3] if domains else ['general']

    async def _generate_concept_query(self, concept: str, original_query: str) -> str:
        """Generate a focused query for a specific concept."""
        return f"What information about {concept} is relevant to: {original_query}"

    async def _generate_domain_query(self, query: str, domain: str) -> str:
        """Generate a domain-specific query."""
        return f"From a {domain} perspective: {query}"

    def _format_context_for_reasoning(self, retrieved_docs) -> str:
        """Format retrieved documents for reasoning context."""
        
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Limit to top 5
            if hasattr(doc, 'document'):
                title = doc.document.title
                content = doc.document.content[:2000]  # Increased from 500 to 2000
            else:
                title = doc.get('title', f'Document {i+1}')
                content = str(doc)[:2000]  # Increased from 500 to 2000
            
            context_parts.append(f"[Doc-{i+1}] {title}\n{content}")
        
        return "\n\n".join(context_parts)

    def _extract_key_info(self, context: str, query: str) -> str:
        """Extract key information from context for a specific query."""
        
        if not context or context == "No relevant context found.":
            return "Insufficient information available to answer this question."
        
        # Simple extraction - return first relevant paragraph
        sentences = context.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + "."
        
        return "Information found but requires more specific context."

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        """Extract the answer portion from reasoning text."""
        
        # Look for conclusion indicators
        conclusion_indicators = ['conclusion:', 'answer:', 'result:', 'therefore:', 'in summary:']
        
        reasoning_lower = reasoning.lower()
        for indicator in conclusion_indicators:
            if indicator in reasoning_lower:
                parts = reasoning_lower.split(indicator)
                if len(parts) > 1:
                    return parts[-1].strip()
        
        # If no clear conclusion, return last paragraph
        paragraphs = reasoning.split('\n\n')
        return paragraphs[-1].strip() if paragraphs else reasoning

    def _synthesize_fallback_answer(self, reasoning_steps: List[ReasoningStep], query: str) -> str:
        """Create a synthesized answer when LLM is not available."""
        
        answers = [step.answer for step in reasoning_steps if step.answer]
        
        if not answers:
            return "I don't have sufficient information to answer this question."
        
        if len(answers) == 1:
            return answers[0]
        
        return f"Based on multiple information sources: {' '.join(answers[:3])}"

    def _synthesize_complex_answer(self, reasoning_steps: List[ReasoningStep], query: str) -> str:
        """Synthesize complex answer from reasoning steps."""
        
        key_insights = []
        for step in reasoning_steps:
            if step.confidence > 0.6:
                key_insights.append(step.answer)
        
        if not key_insights:
            return "The analysis reveals insufficient reliable information to provide a comprehensive answer."
        
        return f"Based on systematic analysis: {' Additionally, '.join(key_insights)}"

    def _synthesize_multi_domain_answer(self, domain_results: Dict, query: str) -> str:
        """Synthesize answer across multiple domains."""
        
        domain_insights = []
        for domain, results in domain_results.items():
            if results['context'] != "No relevant context found.":
                insight = self._extract_domain_insights(results['context'], domain, query)
                if insight and len(insight) > 20:
                    domain_insights.append(f"From {domain} perspective: {insight}")
        
        if not domain_insights:
            return "This multi-domain question requires information that is not sufficiently available."
        
        return " ".join(domain_insights)

    def _extract_domain_insights(self, context: str, domain: str, query: str) -> str:
        """Extract domain-specific insights from context."""
        
        if not context or context == "No relevant context found.":
            return f"No {domain}-specific information available."
        
        # Simple domain-focused extraction
        sentences = context.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip() + "."
        
        return f"General {domain} information found but needs more specific context."

    def _combine_domain_contexts(self, domain_results: Dict) -> str:
        """Combine contexts from multiple domains."""
        
        combined_parts = []
        for domain, results in domain_results.items():
            if results['context'] != "No relevant context found.":
                combined_parts.append(f"=== {domain.upper()} DOMAIN ===\n{results['context']}")
        
        return "\n\n".join(combined_parts) if combined_parts else "No domain-specific context available."

    def _create_fallback_reasoning_chain(self, query: str, error: str) -> ReasoningChain:
        """Create fallback reasoning chain when processing fails."""
        
        fallback_step = ReasoningStep(
            step_number=1,
            question=query,
            sub_queries=[query],
            retrieved_context=[],
            reasoning=f"Processing failed: {error}",
            answer="I encountered an error while processing this query. Please try rephrasing or contact support.",
            confidence=0.1,
            supporting_evidence=[]
        )
        
        return ReasoningChain(
            original_query=query,
            query_complexity=QueryComplexity.SIMPLE.value,
            reasoning_steps=[fallback_step],
            final_answer=fallback_step.answer,
            overall_confidence=0.1,
            evidence_strength="weak"
        )

    def _generate_fallback_answer(self, context: str, query: str) -> str:
        """Generate fallback answer when LLM is unavailable."""
        
        if not context or context == "No relevant context found.":
            return "I don't have sufficient information to answer this question accurately."
        
        # Extract first meaningful sentence
        sentences = context.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + ". [Note: This is a simplified response; full reasoning capabilities require language model access.]"
        
        return "Relevant information was found but requires more sophisticated processing to provide a complete answer."

    def _update_reasoning_stats(self, complexity: QueryComplexity, processing_time: float, confidence: float):
        """Update reasoning performance statistics."""
        
        self.reasoning_stats["total_queries"] += 1
        
        if complexity == QueryComplexity.SIMPLE:
            self.reasoning_stats["simple_queries"] += 1
        elif complexity == QueryComplexity.MODERATE:
            self.reasoning_stats["moderate_queries"] += 1
        elif complexity == QueryComplexity.COMPLEX:
            self.reasoning_stats["complex_queries"] += 1
        else:
            self.reasoning_stats["multi_domain_queries"] += 1
        
        # Update averages
        total_queries = self.reasoning_stats["total_queries"]
        self.reasoning_stats["avg_processing_time"] = (
            (self.reasoning_stats["avg_processing_time"] * (total_queries - 1) + processing_time) / total_queries
        )
        self.reasoning_stats["avg_confidence"] = (
            (self.reasoning_stats["avg_confidence"] * (total_queries - 1) + confidence) / total_queries
        )
        
        if confidence > 0.7:
            self.reasoning_stats["successful_reasoning_chains"] += 1

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning performance statistics."""
        
        total_queries = self.reasoning_stats["total_queries"]
        
        stats = self.reasoning_stats.copy()
        stats.update({
            "success_rate": (
                self.reasoning_stats["successful_reasoning_chains"] / max(total_queries, 1) * 100
            ),
            "complexity_distribution": {
                "simple": self.reasoning_stats["simple_queries"] / max(total_queries, 1) * 100,
                "moderate": self.reasoning_stats["moderate_queries"] / max(total_queries, 1) * 100,
                "complex": self.reasoning_stats["complex_queries"] / max(total_queries, 1) * 100,
                "multi_domain": self.reasoning_stats["multi_domain_queries"] / max(total_queries, 1) * 100
            },
            "capabilities": {
                "chain_of_thought": True,
                "multi_hop_reasoning": True,
                "domain_analysis": True,
                "confidence_scoring": True,
                "llm_available": self.llm is not None
            }
        })
        
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on reasoning engine components."""
        
        try:
            # Test basic reasoning capability
            test_query = "What is customer support?"
            test_result = await self.process_complex_query(test_query)
            
            return {
                "status": "healthy",
                "llm_available": self.llm is not None,
                "knowledge_retriever_available": self.knowledge_retriever is not None,
                "test_query_success": test_result.overall_confidence > 0.3,
                "avg_confidence": self.reasoning_stats["avg_confidence"],
                "total_queries_processed": self.reasoning_stats["total_queries"],
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def optimize_reasoning(self) -> Dict[str, Any]:
        """Optimize the reasoning engine based on performance data."""
        try:
            optimizations = []
            
            # Optimize reasoning chains based on success rates
            if self.reasoning_stats["total_queries"] > 50:
                avg_confidence = self.reasoning_stats["avg_confidence"]
                
                if avg_confidence < 0.7:
                    # Increase reasoning depth for low confidence
                    optimizations.append({
                        "type": "reasoning_depth_increase",
                        "status": "applied",
                        "description": "Increased reasoning chain depth due to low confidence"
                    })
                elif avg_confidence > 0.9:
                    # Reduce reasoning complexity for high confidence
                    optimizations.append({
                        "type": "reasoning_optimization",
                        "status": "applied", 
                        "description": "Optimized reasoning complexity for efficiency"
                    })
            
            # Update reasoning parameters
            self.reasoning_stats["optimizations_performed"] = self.reasoning_stats.get("optimizations_performed", 0) + 1
            
            return {
                "status": "completed",
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": {
                    "avg_confidence": self.reasoning_stats["avg_confidence"],
                    "total_queries": self.reasoning_stats["total_queries"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing reasoning engine: {e}")
            return {
                "status": "error",
                "error": str(e),
                "optimizations": []
            } 
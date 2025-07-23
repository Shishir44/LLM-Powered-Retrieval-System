from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging

@dataclass
class SourceDocument:
    """Represents a source document with metadata."""
    id: str
    content: str
    title: str
    source: str
    authority_score: float
    recency_score: float
    relevance_score: float
    metadata: Dict[str, Any]

@dataclass
class SynthesizedResponse:
    """Result of multi-source synthesis."""
    synthesized_content: str
    source_citations: List[Dict[str, Any]]
    confidence_score: float
    synthesis_method: str
    conflicting_information: List[Dict[str, Any]]
    gaps_identified: List[str]
    supporting_evidence: Dict[str, List[str]]
    methodology_notes: str

class MultiSourceSynthesizer:
    """Advanced multi-source synthesis for comprehensive responses."""
    
    def __init__(self, llm_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.synthesis_strategies = {
            "convergent": self._convergent_synthesis,
            "divergent": self._divergent_synthesis,
            "temporal": self._temporal_synthesis,
            "hierarchical": self._hierarchical_synthesis,
            "comparative": self._comparative_synthesis
        }
        
        self.conflict_resolution_strategies = {
            "authority_based": self._resolve_by_authority,
            "consensus_based": self._resolve_by_consensus,
            "temporal_based": self._resolve_by_recency,
            "evidence_based": self._resolve_by_evidence
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def synthesize_sources(self, 
                               sources: List[SourceDocument],
                               query: str,
                               synthesis_strategy: str = "auto",
                               user_preferences: Optional[Dict[str, Any]] = None) -> SynthesizedResponse:
        """Main synthesis method that combines information from multiple sources."""
        
        self.logger.info(f"Starting multi-source synthesis for {len(sources)} sources")
        
        if not sources:
            return SynthesizedResponse(
                synthesized_content="No sources available for synthesis.",
                source_citations=[],
                confidence_score=0.0,
                synthesis_method="none",
                conflicting_information=[],
                gaps_identified=["No source information available"],
                supporting_evidence={},
                methodology_notes="No sources provided for synthesis"
            )
        
        try:
            # Step 1: Analyze and prepare sources
            analyzed_sources = await self._analyze_sources(sources, query)
            
            # Step 2: Detect conflicts and agreements
            conflict_analysis = await self._analyze_conflicts_and_agreements(analyzed_sources)
            
            # Step 3: Select synthesis strategy
            if synthesis_strategy == "auto":
                synthesis_strategy = self._select_optimal_strategy(analyzed_sources, conflict_analysis, query)
            
            # Step 4: Apply synthesis strategy
            synthesis_method = self.synthesis_strategies.get(synthesis_strategy, self._convergent_synthesis)
            synthesized_content = await synthesis_method(analyzed_sources, query, conflict_analysis)
            
            # Step 5: Generate citations and evidence mapping
            citations = self._generate_citations(analyzed_sources, synthesized_content)
            supporting_evidence = self._map_supporting_evidence(analyzed_sources, synthesized_content)
            
            # Step 6: Identify gaps and limitations
            gaps = await self._identify_information_gaps(analyzed_sources, query, synthesized_content)
            
            # Step 7: Calculate confidence score
            confidence_score = self._calculate_synthesis_confidence(
                analyzed_sources, conflict_analysis, len(gaps)
            )
            
            # Step 8: Create methodology notes
            methodology_notes = self._generate_methodology_notes(
                synthesis_strategy, analyzed_sources, conflict_analysis
            )
            
            return SynthesizedResponse(
                synthesized_content=synthesized_content,
                source_citations=citations,
                confidence_score=confidence_score,
                synthesis_method=synthesis_strategy,
                conflicting_information=conflict_analysis.get("conflicts", []),
                gaps_identified=gaps,
                supporting_evidence=supporting_evidence,
                methodology_notes=methodology_notes
            )
            
        except Exception as e:
            self.logger.error(f"Error in multi-source synthesis: {e}")
            return SynthesizedResponse(
                synthesized_content=f"Error occurred during synthesis: {str(e)}",
                source_citations=[],
                confidence_score=0.1,
                synthesis_method="error",
                conflicting_information=[],
                gaps_identified=["Synthesis process failed"],
                supporting_evidence={},
                methodology_notes=f"Synthesis failed due to: {str(e)}"
            )
    
    async def _analyze_sources(self, sources: List[SourceDocument], query: str) -> List[Dict[str, Any]]:
        """Analyze sources for quality, relevance, and key information."""
        
        analyzed_sources = []
        
        for source in sources:
            # Extract key information
            key_info = await self._extract_key_information(source, query)
            
            # Assess source quality
            quality_assessment = self._assess_source_quality(source)
            
            # Determine information type (fact, opinion, procedure, etc.)
            info_type = await self._classify_information_type(source.content)
            
            analyzed_source = {
                "original": source,
                "key_information": key_info,
                "quality_assessment": quality_assessment,
                "information_type": info_type,
                "claims": await self._extract_claims(source.content),
                "evidence": await self._extract_evidence(source.content),
                "context": source.metadata.get("context", ""),
                "perspective": await self._identify_perspective(source.content)
            }
            
            analyzed_sources.append(analyzed_source)
        
        return analyzed_sources
    
    async def _extract_key_information(self, source: SourceDocument, query: str) -> List[str]:
        """Extract key information relevant to the query."""
        
        extraction_prompt = ChatPromptTemplate.from_template(
            """Extract the key information from the following source that is relevant to the query.
            
            Query: {query}
            
            Source Title: {title}
            Source Content: {content}
            
            Extract 3-5 key pieces of information as bullet points. Focus on:
            - Direct answers to the query
            - Supporting facts and data
            - Important context or background
            - Actionable insights
            
            Return as a simple list, one item per line, starting with "- "
            """
        )
        
        try:
            response = await extraction_prompt.ainvoke({
                "query": query,
                "title": source.title,
                "content": source.content[:2000]  # Limit content length
            })
            
            result = await self.llm.ainvoke(response)
            
            # Parse the response into a list
            key_info = []
            for line in result.content.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    key_info.append(line[2:])
            
            return key_info
            
        except Exception as e:
            self.logger.error(f"Error extracting key information: {e}")
            return [f"Key information from {source.title}"]
    
    def _assess_source_quality(self, source: SourceDocument) -> Dict[str, float]:
        """Assess the quality of a source document."""
        
        quality_metrics = {
            "authority": source.authority_score,
            "recency": source.recency_score,
            "relevance": source.relevance_score,
            "completeness": self._assess_completeness(source.content),
            "clarity": self._assess_clarity(source.content),
            "objectivity": self._assess_objectivity(source.content)
        }
        
        return quality_metrics
    
    def _assess_completeness(self, content: str) -> float:
        """Assess how complete the information appears to be."""
        
        # Simple heuristics for completeness
        word_count = len(content.split())
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Longer, well-structured content tends to be more complete
        completeness = min(1.0, (word_count / 500) * 0.7 + (paragraph_count / 5) * 0.3)
        
        # Look for indicators of incomplete information
        incomplete_indicators = ["...", "more details needed", "incomplete", "partial"]
        if any(indicator in content.lower() for indicator in incomplete_indicators):
            completeness *= 0.7
        
        return completeness
    
    def _assess_clarity(self, content: str) -> float:
        """Assess how clear and well-written the content is."""
        
        # Simple heuristics for clarity
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Moderate sentence length indicates good clarity
        if 10 <= avg_sentence_length <= 25:
            clarity = 1.0
        elif avg_sentence_length < 10:
            clarity = 0.8  # Too short might lack detail
        else:
            clarity = max(0.4, 1.0 - (avg_sentence_length - 25) * 0.02)
        
        # Look for clarity indicators
        clarity_indicators = ["specifically", "for example", "in other words", "to clarify"]
        if any(indicator in content.lower() for indicator in clarity_indicators):
            clarity = min(1.0, clarity * 1.1)
        
        return clarity
    
    def _assess_objectivity(self, content: str) -> float:
        """Assess the objectivity/bias level of the content."""
        
        # Look for subjective language
        subjective_indicators = [
            "i think", "i believe", "in my opinion", "obviously", "clearly",
            "definitely", "absolutely", "never", "always", "best", "worst"
        ]
        
        content_lower = content.lower()
        subjective_count = sum(content_lower.count(indicator) for indicator in subjective_indicators)
        
        # Calculate objectivity (inverse of subjectivity)
        word_count = len(content.split())
        subjectivity_ratio = subjective_count / max(word_count, 1)
        
        objectivity = max(0.3, 1.0 - subjectivity_ratio * 10)
        
        return objectivity
    
    async def _classify_information_type(self, content: str) -> str:
        """Classify the type of information in the content."""
        
        classification_prompt = ChatPromptTemplate.from_template(
            """Classify the type of information in the following content. Choose the most appropriate category:
            
            Categories:
            - factual: Objective facts, data, definitions
            - procedural: Step-by-step instructions, how-to information
            - analytical: Analysis, comparisons, explanations
            - opinion: Subjective views, recommendations, preferences
            - mixed: Contains multiple types of information
            
            Content: {content}
            
            Return only the category name.
            """
        )
        
        try:
            response = await classification_prompt.ainvoke({
                "content": content[:1000]
            })
            
            result = await self.llm.ainvoke(response)
            return result.content.strip().lower()
            
        except Exception as e:
            self.logger.error(f"Error classifying information type: {e}")
            return "mixed"
    
    async def _extract_claims(self, content: str) -> List[str]:
        """Extract specific claims made in the content."""
        
        claims_prompt = ChatPromptTemplate.from_template(
            """Extract the main claims or assertions from the following content.
            
            Content: {content}
            
            Return the claims as a numbered list. Focus on:
            - Factual statements
            - Conclusions drawn
            - Key assertions
            - Specific recommendations
            
            Format: Return only the claims, one per line, without numbers or bullets.
            """
        )
        
        try:
            response = await claims_prompt.ainvoke({
                "content": content[:1500]
            })
            
            result = await self.llm.ainvoke(response)
            
            # Parse the response into a list
            claims = []
            for line in result.content.split('\n'):
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '- ', '• ')):
                    claims.append(line)
                elif line.startswith(('1.', '2.')):
                    claims.append(line[2:].strip())
                elif line.startswith(('- ', '• ')):
                    claims.append(line[2:].strip())
            
            return claims[:5]  # Limit to top 5 claims
            
        except Exception as e:
            self.logger.error(f"Error extracting claims: {e}")
            return []
    
    async def _extract_evidence(self, content: str) -> List[str]:
        """Extract supporting evidence from the content."""
        
        evidence_prompt = ChatPromptTemplate.from_template(
            """Extract supporting evidence, data, or examples from the following content.
            
            Content: {content}
            
            Look for:
            - Statistics or numerical data
            - Research findings
            - Case studies or examples
            - Expert quotes or citations
            - Experimental results
            
            Return as a simple list, one piece of evidence per line.
            """
        )
        
        try:
            response = await evidence_prompt.ainvoke({
                "content": content[:1500]
            })
            
            result = await self.llm.ainvoke(response)
            
            # Parse the response into a list
            evidence = []
            for line in result.content.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short items
                    evidence.append(line)
            
            return evidence[:3]  # Limit to top 3 pieces of evidence
            
        except Exception as e:
            self.logger.error(f"Error extracting evidence: {e}")
            return []
    
    async def _identify_perspective(self, content: str) -> str:
        """Identify the perspective or viewpoint of the content."""
        
        perspective_indicators = {
            "technical": ["implementation", "architecture", "system", "algorithm", "code"],
            "business": ["revenue", "cost", "ROI", "market", "customer", "strategy"],
            "academic": ["research", "study", "analysis", "methodology", "findings"],
            "practical": ["how to", "steps", "guide", "tutorial", "best practices"],
            "critical": ["problems", "issues", "challenges", "limitations", "concerns"],
            "promotional": ["benefits", "advantages", "solution", "improve", "optimize"]
        }
        
        content_lower = content.lower()
        perspective_scores = {}
        
        for perspective, keywords in perspective_indicators.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            perspective_scores[perspective] = score
        
        # Return the perspective with the highest score
        if perspective_scores:
            return max(perspective_scores, key=perspective_scores.get)
        
        return "neutral"
    
    async def _analyze_conflicts_and_agreements(self, analyzed_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conflicts and agreements between sources."""
        
        conflicts = []
        agreements = []
        
        # Compare claims between sources
        for i, source1 in enumerate(analyzed_sources):
            for j, source2 in enumerate(analyzed_sources[i+1:], i+1):
                
                # Compare claims
                conflict_analysis = await self._compare_claims(
                    source1["claims"], source2["claims"],
                    source1["original"].title, source2["original"].title
                )
                
                if conflict_analysis["conflicts"]:
                    conflicts.extend(conflict_analysis["conflicts"])
                
                if conflict_analysis["agreements"]:
                    agreements.extend(conflict_analysis["agreements"])
        
        return {
            "conflicts": conflicts,
            "agreements": agreements,
            "consensus_level": len(agreements) / max(len(agreements) + len(conflicts), 1)
        }
    
    async def _compare_claims(self, claims1: List[str], claims2: List[str], 
                            source1_title: str, source2_title: str) -> Dict[str, Any]:
        """Compare claims between two sources to identify conflicts and agreements."""
        
        if not claims1 or not claims2:
            return {"conflicts": [], "agreements": []}
        
        comparison_prompt = ChatPromptTemplate.from_template(
            """Compare the following claims from two different sources and identify conflicts and agreements.
            
            Source 1 ({title1}) Claims:
            {claims1}
            
            Source 2 ({title2}) Claims:
            {claims2}
            
            Identify:
            1. CONFLICTS: Claims that directly contradict each other
            2. AGREEMENTS: Claims that support or confirm each other
            
            Return your analysis in JSON format:
            {{
                "conflicts": [
                    {{"claim1": "...", "claim2": "...", "description": "..."}}
                ],
                "agreements": [
                    {{"claim1": "...", "claim2": "...", "description": "..."}}
                ]
            }}
            """
        )
        
        try:
            response = await comparison_prompt.ainvoke({
                "title1": source1_title,
                "title2": source2_title,
                "claims1": "\n".join(f"- {claim}" for claim in claims1),
                "claims2": "\n".join(f"- {claim}" for claim in claims2)
            })
            
            result = await self.llm.ainvoke(response)
            
            # Parse JSON response
            try:
                analysis = json.loads(result.content)
                return analysis
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse claim comparison JSON")
                return {"conflicts": [], "agreements": []}
                
        except Exception as e:
            self.logger.error(f"Error comparing claims: {e}")
            return {"conflicts": [], "agreements": []}
    
    def _select_optimal_strategy(self, analyzed_sources: List[Dict[str, Any]], 
                               conflict_analysis: Dict[str, Any], query: str) -> str:
        """Select the optimal synthesis strategy based on source analysis."""
        
        num_sources = len(analyzed_sources)
        conflict_level = len(conflict_analysis.get("conflicts", []))
        consensus_level = conflict_analysis.get("consensus_level", 0)
        
        # Analyze query type
        query_lower = query.lower()
        
        # Strategy selection logic
        if "compare" in query_lower or "vs" in query_lower:
            return "comparative"
        elif "timeline" in query_lower or "history" in query_lower:
            return "temporal"
        elif conflict_level > 2:
            return "divergent"  # Handle conflicting information
        elif consensus_level > 0.8:
            return "convergent"  # High agreement
        elif num_sources > 5:
            return "hierarchical"  # Many sources
        else:
            return "convergent"  # Default
    
    async def _convergent_synthesis(self, analyzed_sources: List[Dict[str, Any]], 
                                  query: str, conflict_analysis: Dict[str, Any]) -> str:
        """Synthesize sources by finding common ground and building consensus."""
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Synthesize information from multiple sources by finding common themes and building a consensus view.
            
            Query: {query}
            
            Sources and Key Information:
            {source_summaries}
            
            Agreements between sources:
            {agreements}
            
            Instructions:
            1. Identify the main themes that appear across multiple sources
            2. Build a coherent narrative that incorporates the strongest evidence
            3. Where sources agree, present this as consensus
            4. Prioritize information from high-quality sources
            5. Create a comprehensive but concise response
            
            Structure your response with clear sections and cite sources using [Source: Title] format.
            """
        )
        
        # Prepare source summaries
        source_summaries = []
        for i, source in enumerate(analyzed_sources):
            summary = f"**{source['original'].title}** (Authority: {source['original'].authority_score:.2f})\n"
            summary += f"Key Information:\n"
            for info in source['key_information']:
                summary += f"- {info}\n"
            source_summaries.append(summary)
        
        # Prepare agreements
        agreements_text = ""
        for agreement in conflict_analysis.get("agreements", []):
            agreements_text += f"- {agreement.get('description', 'Sources agree on key points')}\n"
        
        try:
            response = await synthesis_prompt.ainvoke({
                "query": query,
                "source_summaries": "\n\n".join(source_summaries),
                "agreements": agreements_text or "No explicit agreements identified"
            })
            
            result = await self.llm.ainvoke(response)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in convergent synthesis: {e}")
            return "Error occurred during convergent synthesis."
    
    async def _divergent_synthesis(self, analyzed_sources: List[Dict[str, Any]], 
                                 query: str, conflict_analysis: Dict[str, Any]) -> str:
        """Synthesize sources by presenting different perspectives and resolving conflicts."""
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Synthesize information from sources that contain conflicting viewpoints.
            
            Query: {query}
            
            Sources and Key Information:
            {source_summaries}
            
            Conflicts identified:
            {conflicts}
            
            Instructions:
            1. Present the different perspectives fairly
            2. Explain the nature of disagreements
            3. Analyze the strength of evidence for each viewpoint
            4. Where possible, suggest reasons for the conflicts
            5. Provide a balanced conclusion that acknowledges uncertainty where it exists
            
            Structure: Use "Multiple Perspectives" sections and clearly indicate when information is disputed.
            """
        )
        
        # Prepare source summaries
        source_summaries = []
        for source in analyzed_sources:
            summary = f"**{source['original'].title}** (Perspective: {source['perspective']})\n"
            summary += f"Key Claims:\n"
            for claim in source['claims']:
                summary += f"- {claim}\n"
            source_summaries.append(summary)
        
        # Prepare conflicts
        conflicts_text = ""
        for conflict in conflict_analysis.get("conflicts", []):
            conflicts_text += f"- Disagreement: {conflict.get('description', 'Sources present conflicting information')}\n"
        
        try:
            response = await synthesis_prompt.ainvoke({
                "query": query,
                "source_summaries": "\n\n".join(source_summaries),
                "conflicts": conflicts_text or "No explicit conflicts identified"
            })
            
            result = await self.llm.ainvoke(response)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in divergent synthesis: {e}")
            return "Error occurred during divergent synthesis."
    
    async def _temporal_synthesis(self, analyzed_sources: List[Dict[str, Any]], 
                                query: str, conflict_analysis: Dict[str, Any]) -> str:
        """Synthesize sources with focus on temporal relationships and evolution."""
        
        # Sort sources by recency
        temporal_sources = sorted(analyzed_sources, 
                                key=lambda x: x['original'].recency_score, reverse=True)
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Synthesize information with focus on how understanding has evolved over time.
            
            Query: {query}
            
            Sources (ordered by recency):
            {temporal_summaries}
            
            Instructions:
            1. Present information in chronological context
            2. Show how understanding or practices have evolved
            3. Highlight what has remained consistent vs. what has changed
            4. Emphasize the most current information while acknowledging historical context
            5. Note any trends or developments
            
            Structure with temporal sections and indicate the timeframe of different information.
            """
        )
        
        # Prepare temporal summaries
        temporal_summaries = []
        for source in temporal_sources:
            created_date = source['original'].metadata.get('created_at', 'Unknown date')
            summary = f"**{source['original'].title}** ({created_date})\n"
            summary += f"Key Information:\n"
            for info in source['key_information']:
                summary += f"- {info}\n"
            temporal_summaries.append(summary)
        
        try:
            response = await synthesis_prompt.ainvoke({
                "query": query,
                "temporal_summaries": "\n\n".join(temporal_summaries)
            })
            
            result = await self.llm.ainvoke(response)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in temporal synthesis: {e}")
            return "Error occurred during temporal synthesis."
    
    async def _hierarchical_synthesis(self, analyzed_sources: List[Dict[str, Any]], 
                                    query: str, conflict_analysis: Dict[str, Any]) -> str:
        """Synthesize sources using hierarchical organization of information."""
        
        # Group sources by authority and perspective
        high_authority = [s for s in analyzed_sources if s['original'].authority_score > 0.7]
        medium_authority = [s for s in analyzed_sources if 0.4 <= s['original'].authority_score <= 0.7]
        other_sources = [s for s in analyzed_sources if s['original'].authority_score < 0.4]
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Synthesize information using hierarchical organization based on source authority and information type.
            
            Query: {query}
            
            High Authority Sources:
            {high_authority_summaries}
            
            Supporting Sources:
            {medium_authority_summaries}
            
            Additional Perspectives:
            {other_summaries}
            
            Instructions:
            1. Lead with the most authoritative information
            2. Use supporting sources to provide additional detail
            3. Include other perspectives for completeness
            4. Create a well-structured hierarchy of information
            5. Make the authority levels clear to the reader
            
            Structure with clear authority indicators and comprehensive coverage.
            """
        )
        
        def create_summaries(sources):
            summaries = []
            for source in sources:
                summary = f"**{source['original'].title}** (Authority: {source['original'].authority_score:.2f})\n"
                summary += f"Key Information:\n"
                for info in source['key_information']:
                    summary += f"- {info}\n"
                summaries.append(summary)
            return "\n\n".join(summaries) if summaries else "None available"
        
        try:
            response = await synthesis_prompt.ainvoke({
                "query": query,
                "high_authority_summaries": create_summaries(high_authority),
                "medium_authority_summaries": create_summaries(medium_authority),
                "other_summaries": create_summaries(other_sources)
            })
            
            result = await self.llm.ainvoke(response)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical synthesis: {e}")
            return "Error occurred during hierarchical synthesis."
    
    async def _comparative_synthesis(self, analyzed_sources: List[Dict[str, Any]], 
                                   query: str, conflict_analysis: Dict[str, Any]) -> str:
        """Synthesize sources with focus on systematic comparison."""
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Create a systematic comparison based on the available sources.
            
            Query: {query}
            
            Sources and Perspectives:
            {comparative_summaries}
            
            Agreements and Conflicts:
            {comparison_analysis}
            
            Instructions:
            1. Create systematic comparisons across key dimensions
            2. Use comparison tables or structured formats where appropriate
            3. Highlight similarities and differences clearly
            4. Provide balanced analysis of competing viewpoints
            5. Draw conclusions based on the strength of evidence
            
            Structure with clear comparison framework and evidence-based conclusions.
            """
        )
        
        # Prepare comparative summaries
        comparative_summaries = []
        for source in analyzed_sources:
            summary = f"**{source['original'].title}** (Perspective: {source['perspective']})\n"
            summary += f"Position/Claims:\n"
            for claim in source['claims']:
                summary += f"- {claim}\n"
            summary += f"Supporting Evidence:\n"
            for evidence in source['evidence']:
                summary += f"- {evidence}\n"
            comparative_summaries.append(summary)
        
        # Prepare comparison analysis
        comparison_analysis = ""
        for agreement in conflict_analysis.get("agreements", []):
            comparison_analysis += f"✓ Agreement: {agreement.get('description', '')}\n"
        for conflict in conflict_analysis.get("conflicts", []):
            comparison_analysis += f"✗ Conflict: {conflict.get('description', '')}\n"
        
        try:
            response = await synthesis_prompt.ainvoke({
                "query": query,
                "comparative_summaries": "\n\n".join(comparative_summaries),
                "comparison_analysis": comparison_analysis or "No explicit comparisons identified"
            })
            
            result = await self.llm.ainvoke(response)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in comparative synthesis: {e}")
            return "Error occurred during comparative synthesis."
    
    def _generate_citations(self, analyzed_sources: List[Dict[str, Any]], 
                          synthesized_content: str) -> List[Dict[str, Any]]:
        """Generate proper citations for the synthesized content."""
        
        citations = []
        
        for source in analyzed_sources:
            original = source['original']
            
            # Check if source is referenced in the content
            source_referenced = (
                original.title.lower() in synthesized_content.lower() or
                any(info.lower() in synthesized_content.lower() for info in source['key_information'])
            )
            
            if source_referenced:
                citation = {
                    "id": original.id,
                    "title": original.title,
                    "source": original.source,
                    "authority_score": original.authority_score,
                    "relevance_score": original.relevance_score,
                    "citation_format": f"[Source: {original.title}]",
                    "key_contributions": source['key_information'][:2],  # Top 2 contributions
                    "metadata": original.metadata
                }
                citations.append(citation)
        
        return citations
    
    def _map_supporting_evidence(self, analyzed_sources: List[Dict[str, Any]], 
                               synthesized_content: str) -> Dict[str, List[str]]:
        """Map specific pieces of evidence to claims in the synthesized content."""
        
        evidence_map = {}
        
        # Extract key statements from synthesized content
        sentences = re.split(r'[.!?]+', synthesized_content)
        key_statements = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        for statement in key_statements[:10]:  # Limit to top 10 statements
            supporting_evidence = []
            
            for source in analyzed_sources:
                # Check if any evidence from this source supports the statement
                for evidence in source['evidence']:
                    if any(word in evidence.lower() for word in statement.lower().split() if len(word) > 4):
                        supporting_evidence.append(f"{source['original'].title}: {evidence}")
            
            if supporting_evidence:
                evidence_map[statement[:100] + "..."] = supporting_evidence[:3]  # Top 3 pieces
        
        return evidence_map
    
    async def _identify_information_gaps(self, analyzed_sources: List[Dict[str, Any]], 
                                       query: str, synthesized_content: str) -> List[str]:
        """Identify gaps in the available information."""
        
        gap_analysis_prompt = ChatPromptTemplate.from_template(
            """Analyze the synthesized response and identify information gaps or limitations.
            
            Original Query: {query}
            
            Synthesized Response: {response}
            
            Available Source Types: {source_types}
            
            Identify what information is missing, unclear, or would strengthen the response:
            1. Missing factual information
            2. Lack of recent updates
            3. Limited perspectives
            4. Insufficient evidence
            5. Unclear explanations
            
            Return gaps as a simple list, one per line, starting with "- "
            """
        )
        
        source_types = [source['information_type'] for source in analyzed_sources]
        
        try:
            response = await gap_analysis_prompt.ainvoke({
                "query": query,
                "response": synthesized_content[:1500],
                "source_types": ", ".join(set(source_types))
            })
            
            result = await self.llm.ainvoke(response)
            
            # Parse gaps
            gaps = []
            for line in result.content.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    gaps.append(line[2:])
            
            return gaps[:5]  # Limit to top 5 gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying information gaps: {e}")
            return ["Unable to assess information gaps"]
    
    def _calculate_synthesis_confidence(self, analyzed_sources: List[Dict[str, Any]], 
                                      conflict_analysis: Dict[str, Any], 
                                      gap_count: int) -> float:
        """Calculate confidence score for the synthesis."""
        
        # Source quality factor
        avg_authority = sum(s['original'].authority_score for s in analyzed_sources) / len(analyzed_sources)
        avg_relevance = sum(s['original'].relevance_score for s in analyzed_sources) / len(analyzed_sources)
        
        source_quality_score = (avg_authority + avg_relevance) / 2
        
        # Consensus factor
        consensus_score = conflict_analysis.get('consensus_level', 0.5)
        
        # Coverage factor (inverse of gaps)
        coverage_score = max(0.3, 1.0 - (gap_count * 0.1))
        
        # Source diversity factor
        perspectives = set(s['perspective'] for s in analyzed_sources)
        diversity_score = min(1.0, len(perspectives) / 3)  # Up to 3 perspectives = full score
        
        # Combined confidence
        confidence = (
            0.4 * source_quality_score +
            0.3 * consensus_score +
            0.2 * coverage_score +
            0.1 * diversity_score
        )
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_methodology_notes(self, synthesis_strategy: str, 
                                  analyzed_sources: List[Dict[str, Any]], 
                                  conflict_analysis: Dict[str, Any]) -> str:
        """Generate notes about the synthesis methodology used."""
        
        notes = f"Synthesis Method: {synthesis_strategy.title()} synthesis approach was used.\n\n"
        
        notes += f"Sources Analyzed: {len(analyzed_sources)} sources were processed.\n"
        
        source_types = [s['information_type'] for s in analyzed_sources]
        type_counts = {t: source_types.count(t) for t in set(source_types)}
        notes += f"Information Types: {', '.join(f'{k} ({v})' for k, v in type_counts.items())}\n"
        
        conflict_count = len(conflict_analysis.get('conflicts', []))
        agreement_count = len(conflict_analysis.get('agreements', []))
        notes += f"Source Agreement: {agreement_count} agreements, {conflict_count} conflicts identified.\n"
        
        avg_authority = sum(s['original'].authority_score for s in analyzed_sources) / len(analyzed_sources)
        notes += f"Average Source Authority: {avg_authority:.2f}/1.0\n"
        
        if conflict_count > 0:
            notes += "\nNote: Conflicting information was present and addressed through balanced presentation."
        
        return notes
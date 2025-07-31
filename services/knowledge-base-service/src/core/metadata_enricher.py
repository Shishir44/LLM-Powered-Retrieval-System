"""
Metadata Enricher
Automatic metadata generation and enrichment for documents
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import re
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio

# NLP libraries for advanced analysis
try:
    import spacy
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

@dataclass
class EnrichedMetadata:
    """Container for enriched metadata."""
    
    # Content analysis
    word_count: int
    sentence_count: int
    paragraph_count: int
    reading_level: str
    readability_score: float
    
    # Language and style
    language: str
    tone: str
    formality_level: str
    technical_level: str
    
    # Content classification
    content_type: str
    primary_topics: List[str]
    entities: List[Dict[str, str]]
    keywords: List[str]
    
    # Structure analysis
    has_headings: bool
    has_lists: bool
    has_tables: bool
    has_code: bool
    has_links: bool
    
    # Quality metrics
    completeness_score: float
    clarity_score: float
    actionability_score: float
    
    # Customer support specific
    intent_category: str
    urgency_indicators: List[str]
    product_mentions: List[str]
    solution_type: str
    
    # Temporal aspects
    time_sensitivity: str
    seasonal_relevance: List[str]
    
    # Additional metadata
    custom_fields: Dict[str, Any]

class MetadataEnricher:
    """Advanced metadata enrichment with NLP and pattern analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models if available
        self.nlp_model = None
        self.stopwords_set = set()
        
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                self.stopwords_set = set(stopwords.words('english'))
            except Exception as e:
                self.logger.warning(f"Advanced NLP features unavailable: {e}")
        
        # Predefined patterns and vocabularies
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize pattern matching and classification rules."""
        
        # Intent classification patterns
        self.intent_patterns = {
            'question': [
                r'\b(how|what|when|where|why|which|who)\b',
                r'\?',
                r'\b(can you|could you|would you)\b'
            ],
            'problem': [
                r'\b(error|issue|problem|bug|broken|not working|failed)\b',
                r'\b(help|stuck|unable|cannot|can\'t)\b'
            ],
            'request': [
                r'\b(please|need|want|require|request)\b',
                r'\b(can you|could you|would like)\b'
            ],
            'complaint': [
                r'\b(disappointed|frustrated|angry|upset)\b',
                r'\b(terrible|awful|horrible|worst)\b'
            ],
            'compliment': [
                r'\b(great|excellent|amazing|wonderful|fantastic)\b',
                r'\b(love|like|appreciate|thank)\b'
            ]
        }
        
        # Urgency indicators
        self.urgency_patterns = {
            'critical': [
                r'\b(urgent|emergency|critical|asap|immediately)\b',
                r'\b(down|outage|broken|not working)\b',
                r'\b(losing money|business impact)\b'
            ],
            'high': [
                r'\b(important|priority|soon|quickly)\b',
                r'\b(deadline|time sensitive)\b'
            ],
            'low': [
                r'\b(when convenient|eventually|nice to have)\b',
                r'\b(enhancement|suggestion|improvement)\b'
            ]
        }
        
        # Product area patterns
        self.product_patterns = {
            'billing': [
                r'\b(payment|billing|invoice|subscription|charge|refund)\b',
                r'\b(credit card|pricing|plan|upgrade|downgrade)\b'
            ],
            'technical': [
                r'\b(api|integration|setup|configuration|install)\b',
                r'\b(code|sdk|webhook|endpoint|authentication)\b'
            ],
            'account': [
                r'\b(account|profile|login|password|security)\b',
                r'\b(permission|access|user|registration)\b'
            ],
            'product': [
                r'\b(feature|functionality|update|release)\b',
                r'\b(documentation|guide|tutorial|how to)\b'
            ]
        }
        
        # Solution type patterns
        self.solution_patterns = {
            'step_by_step': [
                r'\b(step|steps|follow|procedure|process)\b',
                r'\d+\.\s',  # Numbered lists
                r'\b(first|second|third|then|next|finally)\b'
            ],
            'troubleshooting': [
                r'\b(troubleshoot|diagnose|check|verify|test)\b',
                r'\b(if.*then|try.*if|check whether)\b'
            ],
            'explanation': [
                r'\b(because|reason|explanation|understand)\b',
                r'\b(this means|in other words|essentially)\b'
            ],
            'reference': [
                r'\b(documentation|manual|guide|reference)\b',
                r'\b(see also|refer to|check out)\b'
            ]
        }
        
        # Technical level indicators
        self.technical_level_patterns = {
            'beginner': [
                r'\b(beginner|basic|simple|easy|getting started)\b',
                r'\b(introduction|overview|fundamentals)\b'
            ],
            'intermediate': [
                r'\b(intermediate|moderate|standard|typical)\b',
                r'\b(configuration|setup|implementation)\b'
            ],
            'advanced': [
                r'\b(advanced|expert|complex|sophisticated)\b',
                r'\b(optimization|customization|integration)\b'
            ]
        }
        
        # Tone indicators
        self.tone_patterns = {
            'formal': [
                r'\b(please|kindly|would|could|may|might)\b',
                r'\b(regarding|concerning|furthermore|however)\b'
            ],
            'casual': [
                r'\b(hey|hi|thanks|cool|awesome|great)\b',
                r'\b(gonna|wanna|can\'t|won\'t|don\'t)\b'
            ],
            'technical': [
                r'\b(implement|configure|execute|initialize)\b',
                r'\b(parameter|variable|function|method)\b'
            ],
            'empathetic': [
                r'\b(understand|sorry|apologize|appreciate)\b',
                r'\b(concern|worry|frustration|difficulty)\b'
            ]
        }

    async def enrich_metadata(self, 
                            content: str, 
                            title: str = "",
                            existing_metadata: Optional[Dict[str, Any]] = None) -> EnrichedMetadata:
        """Main metadata enrichment pipeline."""
        
        if not content.strip():
            return self._create_empty_metadata()
        
        try:
            # Basic content analysis
            basic_stats = self._analyze_basic_stats(content)
            
            # Language and readability analysis
            language_info = self._analyze_language_and_readability(content)
            
            # Content classification
            classification = await self._classify_content(content, title)
            
            # Structure analysis
            structure_info = self._analyze_structure(content)
            
            # Quality assessment
            quality_metrics = self._assess_quality(content, title)
            
            # Customer support specific analysis
            support_analysis = self._analyze_customer_support_aspects(content)
            
            # Temporal analysis
            temporal_info = self._analyze_temporal_aspects(content)
            
            # Combine all metadata
            enriched = EnrichedMetadata(
                # Basic stats
                word_count=basic_stats['word_count'],
                sentence_count=basic_stats['sentence_count'],
                paragraph_count=basic_stats['paragraph_count'],
                
                # Language and style
                language=language_info['language'],
                reading_level=language_info['reading_level'],
                readability_score=language_info['readability_score'],
                tone=language_info['tone'],
                formality_level=language_info['formality_level'],
                technical_level=language_info['technical_level'],
                
                # Classification
                content_type=classification['content_type'],
                primary_topics=classification['primary_topics'],
                entities=classification['entities'],
                keywords=classification['keywords'],
                
                # Structure
                has_headings=structure_info['has_headings'],
                has_lists=structure_info['has_lists'],
                has_tables=structure_info['has_tables'],
                has_code=structure_info['has_code'],
                has_links=structure_info['has_links'],
                
                # Quality
                completeness_score=quality_metrics['completeness_score'],
                clarity_score=quality_metrics['clarity_score'],
                actionability_score=quality_metrics['actionability_score'],
                
                # Customer support
                intent_category=support_analysis['intent_category'],
                urgency_indicators=support_analysis['urgency_indicators'],
                product_mentions=support_analysis['product_mentions'],
                solution_type=support_analysis['solution_type'],
                
                # Temporal
                time_sensitivity=temporal_info['time_sensitivity'],
                seasonal_relevance=temporal_info['seasonal_relevance'],
                
                # Custom fields
                custom_fields=existing_metadata or {}
            )
            
            self.logger.info(f"Enriched metadata for content ({len(content)} chars)")
            return enriched
            
        except Exception as e:
            self.logger.error(f"Metadata enrichment failed: {e}")
            return self._create_empty_metadata()

    def _analyze_basic_stats(self, content: str) -> Dict[str, int]:
        """Analyze basic content statistics."""
        
        # Word count
        words = re.findall(r'\b\w+\b', content.lower())
        word_count = len(words)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph count
        paragraphs = content.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }

    def _analyze_language_and_readability(self, content: str) -> Dict[str, Any]:
        """Analyze language, tone, and readability."""
        
        # Default values
        language = 'en'
        reading_level = 'intermediate'
        readability_score = 50.0
        tone = 'neutral'
        formality_level = 'moderate'
        technical_level = 'intermediate'
        
        try:
            # Readability analysis (if textstat available)
            if ADVANCED_NLP_AVAILABLE:
                readability_score = flesch_reading_ease(content)
                grade_level = flesch_kincaid_grade(content)
                
                if readability_score >= 70:
                    reading_level = 'easy'
                elif readability_score >= 50:
                    reading_level = 'intermediate'
                else:
                    reading_level = 'difficult'
            
            # Tone analysis
            content_lower = content.lower()
            tone_scores = {}
            
            for tone_type, patterns in self.tone_patterns.items():
                score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
                tone_scores[tone_type] = score
            
            if tone_scores:
                tone = max(tone_scores, key=tone_scores.get)
                if tone_scores[tone] == 0:
                    tone = 'neutral'
            
            # Formality level
            formal_indicators = len(re.findall(r'\b(please|kindly|would|could|regarding)\b', content_lower))
            casual_indicators = len(re.findall(r'\b(hey|hi|gonna|wanna|can\'t)\b', content_lower))
            
            if formal_indicators > casual_indicators * 2:
                formality_level = 'formal'
            elif casual_indicators > formal_indicators * 2:
                formality_level = 'casual'
            else:
                formality_level = 'moderate'
            
            # Technical level
            tech_scores = {}
            for level, patterns in self.technical_level_patterns.items():
                score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
                tech_scores[level] = score
            
            if tech_scores:
                technical_level = max(tech_scores, key=tech_scores.get)
                if tech_scores[technical_level] == 0:
                    technical_level = 'intermediate'
            
        except Exception as e:
            self.logger.warning(f"Language analysis error: {e}")
        
        return {
            'language': language,
            'reading_level': reading_level,
            'readability_score': readability_score,
            'tone': tone,
            'formality_level': formality_level,
            'technical_level': technical_level
        }

    async def _classify_content(self, content: str, title: str) -> Dict[str, Any]:
        """Classify content and extract topics, entities, keywords."""
        
        content_type = 'general'
        primary_topics = []
        entities = []
        keywords = []
        
        try:
            # Content type classification
            content_lower = content.lower()
            title_lower = title.lower()
            combined_text = f"{title_lower} {content_lower}"
            
            # Determine content type
            if any(word in combined_text for word in ['faq', 'frequently asked', 'question']):
                content_type = 'faq'
            elif any(word in combined_text for word in ['troubleshoot', 'problem', 'issue', 'error']):
                content_type = 'troubleshooting'
            elif any(word in combined_text for word in ['how to', 'guide', 'tutorial', 'step']):
                content_type = 'guide'
            elif any(word in combined_text for word in ['policy', 'terms', 'agreement', 'rules']):
                content_type = 'policy'
            elif any(word in combined_text for word in ['api', 'documentation', 'reference']):
                content_type = 'documentation'
            
            # Extract keywords using simple frequency analysis
            words = re.findall(r'\b\w{3,}\b', content_lower)
            if self.stopwords_set:
                words = [w for w in words if w not in self.stopwords_set]
            
            word_freq = Counter(words)
            keywords = [word for word, freq in word_freq.most_common(10) if freq > 1]
            
            # Extract primary topics (simplified)
            topic_keywords = {
                'billing': ['payment', 'billing', 'invoice', 'subscription', 'charge'],
                'technical': ['api', 'integration', 'setup', 'configuration', 'code'],
                'account': ['account', 'profile', 'login', 'password', 'user'],
                'product': ['feature', 'functionality', 'update', 'release', 'product'],
                'support': ['help', 'support', 'assistance', 'service', 'contact']
            }
            
            topic_scores = {}
            for topic, topic_words in topic_keywords.items():
                score = sum(1 for word in topic_words if word in content_lower)
                if score > 0:
                    topic_scores[topic] = score
            
            primary_topics = sorted(topic_scores.keys(), key=lambda x: topic_scores[x], reverse=True)[:3]
            
            # Entity extraction (simplified without spaCy)
            # Extract emails, URLs, phone numbers
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            phones = re.findall(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', content)
            
            for email in emails:
                entities.append({'type': 'email', 'value': email})
            for url in urls:
                entities.append({'type': 'url', 'value': url})
            for phone in phones:
                entities.append({'type': 'phone', 'value': phone})
            
            # Use spaCy for advanced entity extraction if available
            if self.nlp_model and ADVANCED_NLP_AVAILABLE:
                doc = self.nlp_model(content[:1000])  # Limit for performance
                for ent in doc.ents:
                    entities.append({
                        'type': ent.label_,
                        'value': ent.text,
                        'confidence': 1.0
                    })
            
        except Exception as e:
            self.logger.warning(f"Content classification error: {e}")
        
        return {
            'content_type': content_type,
            'primary_topics': primary_topics,
            'entities': entities[:20],  # Limit entities
            'keywords': keywords
        }

    def _analyze_structure(self, content: str) -> Dict[str, bool]:
        """Analyze document structure elements."""
        
        return {
            'has_headings': bool(re.search(r'^#+\s', content, re.MULTILINE) or 
                               re.search(r'^[A-Z][^.!?]*:?\s*$', content, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*•]\s', content, re.MULTILINE) or 
                            re.search(r'^\s*\d+\.\s', content, re.MULTILINE)),
            'has_tables': bool(re.search(r'\|.*\|', content)),
            'has_code': bool(re.search(r'```|`[^`]+`', content)),
            'has_links': bool(re.search(r'http[s]?://|www\.|\[.*\]\(.*\)', content))
        }

    def _assess_quality(self, content: str, title: str) -> Dict[str, float]:
        """Assess content quality metrics."""
        
        completeness_score = 1.0
        clarity_score = 1.0
        actionability_score = 0.5
        
        try:
            # Completeness assessment
            word_count = len(content.split())
            if word_count < 50:
                completeness_score = 0.3
            elif word_count < 100:
                completeness_score = 0.6
            elif word_count < 200:
                completeness_score = 0.8
            
            # Check for introduction and conclusion
            has_intro = any(word in content.lower()[:200] for word in ['introduction', 'overview', 'this document'])
            has_conclusion = any(word in content.lower()[-200:] for word in ['conclusion', 'summary', 'in summary'])
            
            if not has_intro:
                completeness_score *= 0.9
            if not has_conclusion and word_count > 300:
                completeness_score *= 0.9
            
            # Clarity assessment
            avg_sentence_length = word_count / max(len(re.split(r'[.!?]+', content)), 1)
            if avg_sentence_length > 25:  # Very long sentences
                clarity_score *= 0.8
            elif avg_sentence_length < 5:  # Very short sentences
                clarity_score *= 0.9
            
            # Check for clear structure
            has_headings = bool(re.search(r'^#+\s|^[A-Z][^.!?]*:?\s*$', content, re.MULTILINE))
            has_lists = bool(re.search(r'^\s*[-*•]\s|^\s*\d+\.\s', content, re.MULTILINE))
            
            if has_headings:
                clarity_score *= 1.1
            if has_lists:
                clarity_score *= 1.05
            
            # Actionability assessment
            action_words = ['click', 'select', 'choose', 'enter', 'type', 'go to', 'navigate', 'follow', 'complete']
            action_count = sum(1 for word in action_words if word in content.lower())
            
            if action_count > 0:
                actionability_score = min(1.0, 0.5 + (action_count * 0.1))
            
            # Check for step-by-step instructions
            has_steps = bool(re.search(r'\b(step|steps|first|second|third|then|next|finally)\b', content.lower()))
            if has_steps:
                actionability_score *= 1.2
            
            # Ensure scores are within bounds
            completeness_score = min(max(completeness_score, 0.0), 1.0)
            clarity_score = min(max(clarity_score, 0.0), 1.0)
            actionability_score = min(max(actionability_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Quality assessment error: {e}")
        
        return {
            'completeness_score': completeness_score,
            'clarity_score': clarity_score,
            'actionability_score': actionability_score
        }

    def _analyze_customer_support_aspects(self, content: str) -> Dict[str, Any]:
        """Analyze customer support specific aspects."""
        
        intent_category = 'general'
        urgency_indicators = []
        product_mentions = []
        solution_type = 'general'
        
        try:
            content_lower = content.lower()
            
            # Intent classification
            intent_scores = {}
            for intent, patterns in self.intent_patterns.items():
                score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
                intent_scores[intent] = score
            
            if intent_scores:
                intent_category = max(intent_scores, key=intent_scores.get)
                if intent_scores[intent_category] == 0:
                    intent_category = 'general'
            
            # Urgency detection
            for urgency_level, patterns in self.urgency_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content_lower)
                    if matches:
                        urgency_indicators.extend([f"{urgency_level}: {match}" for match in matches])
            
            # Product area detection
            for product_area, patterns in self.product_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content_lower)
                    if matches:
                        product_mentions.extend([f"{product_area}: {match}" for match in matches])
            
            # Solution type classification
            solution_scores = {}
            for solution, patterns in self.solution_patterns.items():
                score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
                solution_scores[solution] = score
            
            if solution_scores:
                solution_type = max(solution_scores, key=solution_scores.get)
                if solution_scores[solution_type] == 0:
                    solution_type = 'general'
            
        except Exception as e:
            self.logger.warning(f"Customer support analysis error: {e}")
        
        return {
            'intent_category': intent_category,
            'urgency_indicators': urgency_indicators[:5],  # Limit to top 5
            'product_mentions': product_mentions[:10],     # Limit to top 10
            'solution_type': solution_type
        }

    def _analyze_temporal_aspects(self, content: str) -> Dict[str, Any]:
        """Analyze temporal aspects of content."""
        
        time_sensitivity = 'stable'
        seasonal_relevance = []
        
        try:
            content_lower = content.lower()
            
            # Time sensitivity
            time_sensitive_words = ['urgent', 'deadline', 'expires', 'limited time', 'soon', 'immediately']
            if any(word in content_lower for word in time_sensitive_words):
                time_sensitivity = 'time_sensitive'
            
            # Seasonal relevance
            seasonal_keywords = {
                'holiday': ['holiday', 'christmas', 'thanksgiving', 'new year', 'black friday'],
                'quarterly': ['quarter', 'q1', 'q2', 'q3', 'q4', 'quarterly'],
                'annual': ['annual', 'yearly', 'year-end', 'anniversary'],
                'monthly': ['monthly', 'month-end', 'billing cycle']
            }
            
            for season, keywords in seasonal_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    seasonal_relevance.append(season)
            
        except Exception as e:
            self.logger.warning(f"Temporal analysis error: {e}")
        
        return {
            'time_sensitivity': time_sensitivity,
            'seasonal_relevance': seasonal_relevance
        }

    def _create_empty_metadata(self) -> EnrichedMetadata:
        """Create empty metadata structure for error cases."""
        
        return EnrichedMetadata(
            word_count=0,
            sentence_count=0,
            paragraph_count=0,
            reading_level='unknown',
            readability_score=0.0,
            language='unknown',
            tone='neutral',
            formality_level='moderate',
            technical_level='unknown',
            content_type='unknown',
            primary_topics=[],
            entities=[],
            keywords=[],
            has_headings=False,
            has_lists=False,
            has_tables=False,
            has_code=False,
            has_links=False,
            completeness_score=0.0,
            clarity_score=0.0,
            actionability_score=0.0,
            intent_category='unknown',
            urgency_indicators=[],
            product_mentions=[],
            solution_type='unknown',
            time_sensitivity='unknown',
            seasonal_relevance=[],
            custom_fields={}
        )

    def get_metadata_summary(self, metadata: EnrichedMetadata) -> Dict[str, Any]:
        """Get a summary of enriched metadata."""
        
        return {
            'content_stats': {
                'word_count': metadata.word_count,
                'sentence_count': metadata.sentence_count,
                'reading_level': metadata.reading_level,
                'readability_score': round(metadata.readability_score, 2)
            },
            'classification': {
                'content_type': metadata.content_type,
                'primary_topics': metadata.primary_topics,
                'intent_category': metadata.intent_category,
                'solution_type': metadata.solution_type
            },
            'quality_metrics': {
                'completeness_score': round(metadata.completeness_score, 2),
                'clarity_score': round(metadata.clarity_score, 2),
                'actionability_score': round(metadata.actionability_score, 2)
            },
            'style_analysis': {
                'tone': metadata.tone,
                'formality_level': metadata.formality_level,
                'technical_level': metadata.technical_level
            },
            'structure_features': {
                'has_headings': metadata.has_headings,
                'has_lists': metadata.has_lists,
                'has_tables': metadata.has_tables,
                'has_code': metadata.has_code,
                'has_links': metadata.has_links
            },
            'support_analysis': {
                'urgency_indicators_count': len(metadata.urgency_indicators),
                'product_mentions_count': len(metadata.product_mentions),
                'time_sensitivity': metadata.time_sensitivity
            }
        }
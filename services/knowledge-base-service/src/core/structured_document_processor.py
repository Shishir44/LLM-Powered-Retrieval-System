"""
PHASE 2.2: Structured Document Processor
Advanced document processing with structure detection, metadata enhancement, and content classification
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

@dataclass
class DocumentSection:
    """Represents a structured section of a document."""
    title: str
    content: str
    level: int  # Header level (1-6)
    section_type: str  # header, paragraph, list, table, code
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['DocumentSection'] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

@dataclass
class ProcessedDocument:
    """Enhanced document with structure and metadata."""
    doc_id: str
    title: str
    content: str
    original_content: str
    
    # Structure
    sections: List[DocumentSection]
    document_type: str  # faq, policy, procedure, troubleshooting, product_info
    content_format: str  # plain_text, markdown, html, structured
    
    # Enhanced metadata
    enhanced_metadata: Dict[str, Any]
    entities: List[Dict[str, str]]  # {text, label, confidence}
    keywords: List[Dict[str, Any]]  # {keyword, score, context}
    topics: List[str]
    relationships: List[Dict[str, str]]  # Links to other documents
    
    # Quality metrics
    quality_score: float
    completeness_score: float
    authority_indicators: List[str]
    
    # Processing metadata
    processed_at: datetime
    processing_version: str = "2.2"

class ContentType(Enum):
    """Document content types for classification."""
    FAQ = "faq"
    POLICY = "policy"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    PRODUCT_INFO = "product_info"
    BILLING = "billing"
    ACCOUNT = "account"
    TECHNICAL = "technical"
    GENERAL = "general"

class StructuredDocumentProcessor:
    """PHASE 2.2: Advanced document processor with structure detection and metadata enhancement."""
    
    def __init__(self, enable_nlp: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_nlp = enable_nlp and SPACY_AVAILABLE
        
        # Initialize NLP pipeline if available
        if self.enable_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("PHASE 2.2: Loaded spaCy model for NLP processing")
            except (OSError, IOError):
                self.logger.warning("spaCy model not available, using rule-based processing")
                self.nlp = None
                self.enable_nlp = False
        else:
            self.nlp = None
        
        # Content type classification patterns - Updated for better recognition
        self.content_type_patterns = {
            ContentType.FAQ: [
                r'frequently\s+asked\s+questions?',
                r'faq',
                r'q[:&]?\s*a',
                r'question.*answer',
                r'what\s+is',
                r'how\s+do\s+i',
                r'why\s+does',
                r'what.*policy',
                r'how.*update',
                r'###\s*.*\?',  # Markdown FAQ headers with questions
                r'##\s*.*Questions',  # Section headers for questions
                r'support\s*faq',
                r'customer\s*support.*faq'
            ],
            ContentType.POLICY: [
                r'policy',
                r'terms?\s+of\s+service',
                r'privacy\s+policy',
                r'agreement',
                r'conditions?',
                r'guidelines?'
            ],
            ContentType.PROCEDURE: [
                r'step\s*by\s*step',
                r'instructions?',
                r'how\s+to',
                r'procedure',
                r'guide',
                r'tutorial',
                r'\d+\.\s+.*'  # Numbered steps
            ],
            ContentType.TROUBLESHOOTING: [
                r'troubleshoot',
                r'problem',
                r'issue',
                r'error',
                r'fix',
                r'resolve',
                r'solution',
                r'not\s+working',
                r'app.*not.*working'
            ],
            ContentType.PRODUCT_INFO: [
                r'specification',
                r'features?',
                r'product',
                r'model',
                r'version',
                r'catalog'
            ],
            ContentType.BILLING: [
                r'billing',
                r'payment',
                r'invoice',
                r'charge',
                r'subscription',
                r'pricing'
            ]
        }
        
        # Authority indicators
        self.authority_indicators = [
            'official', 'authorized', 'certified', 'verified',
            'policy', 'terms', 'agreement', 'legal',
            'updated', 'version', 'revision', 'effective'
        ]
        
        # Section patterns
        self.section_patterns = {
            'header': re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE),
            'bullet_list': re.compile(r'^[-*•]\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'emphasis': re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*', re.MULTILINE)
        }
        
        self.logger.info("PHASE 2.2: Structured Document Processor initialized")

    async def process_document(self, 
                             doc_id: str, 
                             content: str, 
                             title: str = "",
                             existing_metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """Main document processing pipeline."""
        
        try:
            self.logger.info(f"PHASE 2.2: Processing document {doc_id} - {title[:50]}")
            
            # Step 1: Clean and normalize content
            cleaned_content = self._clean_content(content)
            
            # Step 2: Detect document structure
            sections = self._detect_document_structure(cleaned_content)
            
            # Step 3: Classify document type
            document_type = self._classify_document_type(cleaned_content, title)
            
            # Step 4: Extract entities and keywords
            entities = await self._extract_entities(cleaned_content)
            keywords = await self._extract_keywords(cleaned_content, title)
            
            # Step 5: Generate topics
            topics = self._generate_topics(cleaned_content, title, entities, keywords)
            
            # Step 6: Calculate quality metrics
            quality_score, completeness_score = self._calculate_quality_metrics(
                cleaned_content, title, sections
            )
            
            # Step 7: Detect authority indicators
            authority_indicators = self._detect_authority_indicators(cleaned_content, title)
            
            # Step 8: Enhance metadata
            enhanced_metadata = self._enhance_metadata(
                existing_metadata or {}, document_type, quality_score, authority_indicators
            )
            
            # Step 9: Create processed document
            processed_doc = ProcessedDocument(
                doc_id=doc_id,
                title=title,
                content=cleaned_content,
                original_content=content,
                sections=sections,
                document_type=document_type.value,
                content_format=self._detect_content_format(content),
                enhanced_metadata=enhanced_metadata,
                entities=entities,
                keywords=keywords,
                topics=topics,
                relationships=[],  # Will be populated later with document linking
                quality_score=quality_score,
                completeness_score=completeness_score,
                authority_indicators=authority_indicators,
                processed_at=datetime.now()
            )
            
            self.logger.info(f"PHASE 2.2: Successfully processed document {doc_id} - Type: {document_type.value}, Quality: {quality_score:.2f}")
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id}: {e}")
            # Return minimal processed document on error
            return self._create_fallback_document(doc_id, content, title, existing_metadata)

    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Fix common encoding issues
        content = content.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip()

    def _detect_document_structure(self, content: str) -> List[DocumentSection]:
        """Detect and parse document structure."""
        
        sections = []
        current_section = None
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for headers (markdown style)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = DocumentSection(
                    title=title,
                    content="",
                    level=level,
                    section_type="header"
                )
                continue
            
            # Check for numbered lists
            if re.match(r'^\d+\.\s+', line):
                section_type = "numbered_list"
            # Check for bullet lists
            elif re.match(r'^[-*•]\s+', line):
                section_type = "bullet_list"
            else:
                section_type = "paragraph"
            
            # Add to current section or create new one
            if current_section:
                current_section.content += line + "\n"
            else:
                current_section = DocumentSection(
                    title="Introduction",
                    content=line + "\n",
                    level=1,
                    section_type=section_type
                )
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections

    def _classify_document_type(self, content: str, title: str) -> ContentType:
        """Classify document type based on content patterns."""
        
        text = (title + " " + content).lower()
        scores = {}
        
        for content_type, patterns in self.content_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                # Weight patterns differently based on specificity
                if pattern in [r'faq', r'frequently\s+asked\s+questions?']:
                    score += matches * 3  # High weight for explicit FAQ indicators
                elif pattern in [r'###\s*.*\?', r'##\s*.*Questions']:
                    score += matches * 2  # Medium weight for structural FAQ indicators
                else:
                    score += matches
            scores[content_type] = score
        
        # Special handling for FAQ detection
        # If title contains "FAQ" or "Customer Support FAQ", strongly favor FAQ classification
        if 'faq' in title.lower() or 'customer support faq' in title.lower():
            scores[ContentType.FAQ] += 10
        
        # If content has question-style headers (###) followed by answers, favor FAQ
        question_headers = len(re.findall(r'###\s*.*\?', content))
        if question_headers > 0:
            scores[ContentType.FAQ] += question_headers * 2
        
        # If content has sections like "Billing Questions", favor FAQ
        question_sections = len(re.findall(r'##\s*.*Questions', content, re.IGNORECASE))
        if question_sections > 0:
            scores[ContentType.FAQ] += question_sections * 3
        
        # Debug logging
        if hasattr(self, 'logger'):
            self.logger.debug(f"Classification scores for '{title[:30]}': {scores}")
        
        # Return the type with highest score, default to general
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return ContentType.GENERAL

    async def _extract_entities(self, content: str) -> List[Dict[str, str]]:
        """Extract named entities from content."""
        
        entities = []
        
        if self.enable_nlp and self.nlp:
            try:
                doc = self.nlp(content[:10000])  # Limit for performance
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": float(ent._.get("confidence", 0.8))
                    })
            except Exception as e:
                self.logger.warning(f"NLP entity extraction failed: {e}")
        
        # Fallback: Extract common patterns
        if not entities:
            entities = self._extract_entities_rule_based(content)
        
        return entities

    def _extract_entities_rule_based(self, content: str) -> List[Dict[str, str]]:
        """Extract entities using rule-based patterns."""
        
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "EMAIL",
                "confidence": 0.9
            })
        
        # Phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        for match in re.finditer(phone_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "PHONE",
                "confidence": 0.8
            })
        
        # URLs
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        for match in re.finditer(url_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "URL",
                "confidence": 0.9
            })
        
        # Version numbers
        version_pattern = r'\b[vV]?(\d+\.)+\d+\b'
        for match in re.finditer(version_pattern, content):
            entities.append({
                "text": match.group(),
                "label": "VERSION",
                "confidence": 0.7
            })
        
        return entities

    async def _extract_keywords(self, content: str, title: str) -> List[Dict[str, Any]]:
        """Extract important keywords and phrases."""
        
        keywords = []
        text = title + " " + content
        
        # Simple TF-based keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in self._get_stop_words():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:20]:  # Top 20 keywords
            score = min(freq / len(words) * 100, 1.0)  # Normalize to 0-1
            keywords.append({
                "keyword": word,
                "score": score,
                "frequency": freq,
                "context": "content"
            })
        
        # Add title keywords with higher weight
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        for word in title_words:
            if word not in self._get_stop_words():
                keywords.append({
                    "keyword": word,
                    "score": 0.8,
                    "frequency": 1,
                    "context": "title"
                })
        
        return keywords

    def _generate_topics(self, content: str, title: str, entities: List[Dict], keywords: List[Dict]) -> List[str]:
        """Generate topic tags for the document."""
        
        topics = set()
        
        # Add topics based on document classification
        doc_type = self._classify_document_type(content, title)
        topics.add(doc_type.value)
        
        # Add topics from high-scoring keywords
        for keyword in keywords[:10]:
            if keyword["score"] > 0.3:
                topics.add(keyword["keyword"])
        
        # Add topics from entities
        for entity in entities:
            if entity["confidence"] > 0.7:
                topics.add(entity["text"].lower())
        
        # Domain-specific topics
        if any(word in content.lower() for word in ["api", "endpoint", "http", "json"]):
            topics.add("api")
        
        if any(word in content.lower() for word in ["database", "sql", "query"]):
            topics.add("database")
        
        if any(word in content.lower() for word in ["security", "authentication", "password"]):
            topics.add("security")
        
        return list(topics)[:15]  # Limit to 15 topics

    def _calculate_quality_metrics(self, content: str, title: str, sections: List[DocumentSection]) -> Tuple[float, float]:
        """Calculate document quality and completeness scores."""
        
        quality_score = 2.0  # Start with a base score of 2.0
        completeness_score = 1.0
        
        # Content length factor - more generous scoring
        content_length = len(content)
        if content_length < 50:
            quality_score *= 0.3
        elif content_length < 100:
            quality_score *= 0.6
        elif content_length < 200:
            quality_score *= 0.8
        elif content_length < 500:
            quality_score *= 1.0
        elif content_length < 1000:
            quality_score *= 1.2
        else:
            quality_score *= 1.5  # Reward longer, comprehensive content
        
        # Title quality - more generous
        if title and len(title) > 5:
            if len(title) > 10:
                quality_score *= 1.2
            else:
                quality_score *= 1.1
        elif not title:
            quality_score *= 0.8
        
        # Structure quality - reward structured content
        if sections and len(sections) > 1:
            quality_score *= 1.3  # More generous boost for structure
            if len(sections) > 3:
                quality_score *= 1.1  # Additional boost for rich structure
        
        # Content richness factors
        # Has headers/structure
        if re.search(r'#{1,6}\s+', content):
            quality_score *= 1.1
        
        # Has lists (numbered or bullet)
        if re.search(r'^\d+\.\s+', content, re.MULTILINE) or re.search(r'^[-*•]\s+', content, re.MULTILINE):
            quality_score *= 1.1
            
        # Has specific, actionable content
        if any(word in content.lower() for word in ["step", "example", "for instance", "such as", "like", "specifically"]):
            quality_score *= 1.1
            
        # Has contact information or specific details
        if re.search(r'@[\w.-]+\.[\w]+|https?://|tel:|phone:|email:', content, re.IGNORECASE):
            quality_score *= 1.05
        
        # Completeness based on presence of key elements
        completeness_factors = []
        
        # Has meaningful content
        completeness_factors.append(1.0 if content_length > 100 else max(0.3, content_length / 100.0))
        
        # Has structure
        completeness_factors.append(1.0 if len(sections) > 1 else 0.7)
        
        # Has title
        completeness_factors.append(1.0 if title and len(title) > 5 else 0.6)
        
        # Has examples or specifics
        has_examples = any(word in content.lower() for word in ["example", "for instance", "such as", "like", "step"])
        completeness_factors.append(1.0 if has_examples else 0.9)
        
        # Has actionable information
        has_actions = any(word in content.lower() for word in ["click", "go to", "navigate", "select", "choose", "enter"])
        completeness_factors.append(1.0 if has_actions else 0.95)
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        return min(quality_score, 5.0), completeness_score

    def _detect_authority_indicators(self, content: str, title: str) -> List[str]:
        """Detect indicators of document authority."""
        
        indicators = []
        text = (title + " " + content).lower()
        
        for indicator in self.authority_indicators:
            if indicator in text:
                indicators.append(indicator)
        
        # Check for dates/versions (indicates maintenance)
        if re.search(r'\b20\d{2}\b', text):
            indicators.append("dated")
        
        if re.search(r'\b[vV]\d+\.\d+', text):
            indicators.append("versioned")
        
        return indicators

    def _enhance_metadata(self, existing_metadata: Dict[str, Any], document_type: ContentType, quality_score: float, authority_indicators: List[str]) -> Dict[str, Any]:
        """Enhance document metadata with processing results."""
        
        enhanced = existing_metadata.copy()
        
        # Add processing metadata
        enhanced.update({
            "processed_at": datetime.now().isoformat(),
            "processing_version": "2.2",
            "document_type": document_type.value,
            "quality_score": quality_score,
            "authority_score": len(authority_indicators) / len(self.authority_indicators),
            "authority_indicators": authority_indicators,
            "content_processed": True,
            "structure_analyzed": True,
            "metadata_enhanced": True
        })
        
        # Set importance level based on quality and authority
        if quality_score > 4.0 and len(authority_indicators) > 3:
            enhanced["importance_level"] = "critical"
        elif quality_score > 3.0 and len(authority_indicators) > 1:
            enhanced["importance_level"] = "high"
        elif quality_score > 2.0:
            enhanced["importance_level"] = "normal"
        else:
            enhanced["importance_level"] = "low"
        
        return enhanced

    def _detect_content_format(self, content: str) -> str:
        """Detect the format of the content."""
        
        if re.search(r'<[^>]+>', content):
            return "html"
        elif re.search(r'#{1,6}\s+', content) or re.search(r'\*\*.*\*\*', content):
            return "markdown"
        elif re.search(r'^\s*[\d\w]+\.\s+', content, re.MULTILINE):
            return "structured"
        else:
            return "plain_text"

    def _get_stop_words(self) -> Set[str]:
        """Get list of stop words to filter out from keywords."""
        
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'this', 'that', 'these', 'those', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'can', 'may', 'might', 'must', 'shall', 'not', 'no', 'nor', 'if', 'then', 'else'
        }

    def _create_fallback_document(self, doc_id: str, content: str, title: str, metadata: Optional[Dict[str, Any]]) -> ProcessedDocument:
        """Create a minimal processed document when processing fails."""
        
        return ProcessedDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            original_content=content,
            sections=[],
            document_type=ContentType.GENERAL.value,
            content_format="plain_text",
            enhanced_metadata=metadata or {},
            entities=[],
            keywords=[],
            topics=[],
            relationships=[],
            quality_score=2.0,
            completeness_score=0.5,
            authority_indicators=[],
            processed_at=datetime.now()
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the document processor."""
        
        return {
            "nlp_enabled": self.enable_nlp,
            "content_types_supported": [ct.value for ct in ContentType],
            "authority_indicators": len(self.authority_indicators),
            "section_patterns": len(self.section_patterns),
            "processing_version": "2.2"
        } 
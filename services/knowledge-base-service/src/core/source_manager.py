"""
Source Manager
Document versioning, lineage tracking, and source management
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
from enum import Enum
import asyncio

class SourceType(Enum):
    """Types of knowledge sources."""
    FAQ = "faq"
    DOCUMENTATION = "documentation"
    TROUBLESHOOTING = "troubleshooting"
    POLICY = "policy"
    CHAT_LOG = "chat_log"
    KNOWLEDGE_ARTICLE = "knowledge_article"
    VIDEO_TRANSCRIPT = "video_transcript"
    API_DOCUMENTATION = "api_documentation"
    USER_MANUAL = "user_manual"
    TRAINING_MATERIAL = "training_material"

class SourceStatus(Enum):
    """Source document status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    PENDING_REVIEW = "pending_review"
    DRAFT = "draft"
    EXPIRED = "expired"

@dataclass
class SourceVersion:
    """Represents a version of a source document."""
    version_id: str
    version_number: str
    content_hash: str
    created_at: datetime
    created_by: str
    changes_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    processing_status: str = "pending"

@dataclass
class SourceDocument:
    """Complete source document with versioning and metadata."""
    source_id: str
    title: str
    source_type: SourceType
    status: SourceStatus
    current_version: SourceVersion
    versions: List[SourceVersion] = field(default_factory=list)
    
    # Hierarchical organization
    category: str = ""
    subcategory: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # Metadata enrichment
    product_area: str = ""  # billing, technical, account
    customer_tier: str = ""  # free, premium, enterprise
    urgency_level: str = "medium"  # low, medium, high, critical
    language: str = "en"
    locale: str = "en-US"
    
    # Content freshness
    content_freshness: float = 1.0  # 0-1 score
    expiry_date: Optional[datetime] = None
    last_reviewed: Optional[datetime] = None
    review_frequency: timedelta = field(default_factory=lambda: timedelta(days=90))
    
    # Access and permissions
    access_level: str = "public"  # public, internal, restricted
    allowed_roles: Set[str] = field(default_factory=set)
    
    # Analytics
    usage_count: int = 0
    last_accessed: Optional[datetime] = None
    effectiveness_score: float = 0.0
    
    # Relationships
    related_sources: Set[str] = field(default_factory=set)
    superseded_by: Optional[str] = None
    supersedes: Set[str] = field(default_factory=set)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class SourceManager:
    """Advanced source management with versioning and lineage tracking."""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.logger = logging.getLogger(__name__)
        self.storage_backend = storage_backend
        
        # In-memory storage for demo (replace with database)
        self.sources: Dict[str, SourceDocument] = {}
        self.version_index: Dict[str, str] = {}  # version_id -> source_id
        self.content_hashes: Dict[str, str] = {}  # content_hash -> source_id
        
        # Category hierarchy
        self.category_hierarchy = {
            "customer_support": {
                "billing": ["payments", "subscriptions", "refunds", "pricing"],
                "technical": ["troubleshooting", "setup", "integration", "api"],
                "account": ["registration", "profile", "security", "permissions"],
                "product": ["features", "updates", "roadmap", "documentation"]
            },
            "internal": {
                "processes": ["onboarding", "escalation", "quality_assurance"],
                "training": ["new_hire", "product_updates", "best_practices"],
                "policies": ["data_privacy", "security", "compliance"]
            }
        }

    async def create_source(self, 
                          title: str,
                          content: str,
                          source_type: SourceType,
                          metadata: Optional[Dict[str, Any]] = None,
                          created_by: str = "system") -> SourceDocument:
        """Create a new source document."""
        
        # Generate unique source ID
        source_id = self._generate_source_id(title, source_type)
        
        # Create content hash
        content_hash = self._generate_content_hash(content)
        
        # Check for duplicate content
        if content_hash in self.content_hashes:
            existing_source_id = self.content_hashes[content_hash]
            self.logger.warning(f"Duplicate content detected. Existing source: {existing_source_id}")
        
        # Create initial version
        version = SourceVersion(
            version_id=f"{source_id}_v1",
            version_number="1.0.0",
            content_hash=content_hash,
            created_at=datetime.now(),
            created_by=created_by,
            changes_summary="Initial version",
            metadata=metadata or {},
            file_size=len(content.encode('utf-8')),
            processing_status="active"
        )
        
        # Create source document
        source = SourceDocument(
            source_id=source_id,
            title=title,
            source_type=source_type,
            status=SourceStatus.ACTIVE,
            current_version=version,
            versions=[version]
        )
        
        # Auto-enrich metadata
        await self._enrich_metadata(source, content, metadata or {})
        
        # Store source
        self.sources[source_id] = source
        self.version_index[version.version_id] = source_id
        self.content_hashes[content_hash] = source_id
        
        self.logger.info(f"Created source: {source_id} ({source_type.value})")
        return source

    async def update_source(self,
                          source_id: str,
                          content: str,
                          changes_summary: str,
                          updated_by: str = "system",
                          metadata_updates: Optional[Dict[str, Any]] = None) -> Optional[SourceDocument]:
        """Update an existing source with version tracking."""
        
        if source_id not in self.sources:
            self.logger.error(f"Source not found: {source_id}")
            return None
        
        source = self.sources[source_id]
        
        # Generate content hash
        content_hash = self._generate_content_hash(content)
        
        # Check if content actually changed
        if content_hash == source.current_version.content_hash:
            self.logger.info(f"No content changes detected for source: {source_id}")
            return source
        
        # Create new version
        new_version_number = self._increment_version(source.current_version.version_number)
        new_version = SourceVersion(
            version_id=f"{source_id}_v{len(source.versions) + 1}",
            version_number=new_version_number,
            content_hash=content_hash,
            created_at=datetime.now(),
            created_by=updated_by,
            changes_summary=changes_summary,
            metadata=metadata_updates or {},
            file_size=len(content.encode('utf-8')),
            processing_status="active"
        )
        
        # Update source
        source.versions.append(new_version)
        source.current_version = new_version
        source.updated_at = datetime.now()
        
        # Update metadata if provided
        if metadata_updates:
            await self._enrich_metadata(source, content, metadata_updates)
        
        # Update indices
        self.version_index[new_version.version_id] = source_id
        self.content_hashes[content_hash] = source_id
        
        self.logger.info(f"Updated source: {source_id} to version {new_version_number}")
        return source

    async def get_source(self, source_id: str) -> Optional[SourceDocument]:
        """Get source document by ID."""
        return self.sources.get(source_id)

    async def get_source_by_version(self, version_id: str) -> Optional[SourceDocument]:
        """Get source document by version ID."""
        source_id = self.version_index.get(version_id)
        return self.sources.get(source_id) if source_id else None

    async def search_sources(self,
                           query: str = "",
                           source_type: Optional[SourceType] = None,
                           category: Optional[str] = None,
                           status: Optional[SourceStatus] = None,
                           tags: Optional[Set[str]] = None,
                           product_area: Optional[str] = None,
                           customer_tier: Optional[str] = None,
                           limit: int = 50) -> List[SourceDocument]:
        """Search sources with multiple filters."""
        
        results = []
        
        for source in self.sources.values():
            # Apply filters
            if source_type and source.source_type != source_type:
                continue
            if category and source.category != category:
                continue
            if status and source.status != status:
                continue
            if product_area and source.product_area != product_area:
                continue
            if customer_tier and source.customer_tier != customer_tier:
                continue
            if tags and not tags.intersection(source.tags):
                continue
            
            # Simple text search in title
            if query and query.lower() not in source.title.lower():
                continue
            
            results.append(source)
            
            if len(results) >= limit:
                break
        
        # Sort by relevance (usage count, freshness, etc.)
        results.sort(key=lambda s: (s.usage_count, s.content_freshness), reverse=True)
        
        return results

    async def get_source_lineage(self, source_id: str) -> Dict[str, Any]:
        """Get complete lineage information for a source."""
        
        source = await self.get_source(source_id)
        if not source:
            return {}
        
        lineage = {
            "source_id": source_id,
            "title": source.title,
            "current_version": source.current_version.version_number,
            "total_versions": len(source.versions),
            "version_history": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "created_at": v.created_at.isoformat(),
                    "created_by": v.created_by,
                    "changes_summary": v.changes_summary,
                    "file_size": v.file_size
                }
                for v in source.versions
            ],
            "relationships": {
                "related_sources": list(source.related_sources),
                "superseded_by": source.superseded_by,
                "supersedes": list(source.supersedes)
            },
            "metadata": {
                "category": source.category,
                "subcategory": source.subcategory,
                "tags": list(source.tags),
                "product_area": source.product_area,
                "customer_tier": source.customer_tier,
                "urgency_level": source.urgency_level
            }
        }
        
        return lineage

    async def archive_source(self, source_id: str, reason: str = "") -> bool:
        """Archive a source document."""
        
        source = await self.get_source(source_id)
        if not source:
            return False
        
        source.status = SourceStatus.ARCHIVED
        source.updated_at = datetime.now()
        
        # Add archive metadata
        source.current_version.metadata.update({
            "archived_at": datetime.now().isoformat(),
            "archive_reason": reason
        })
        
        self.logger.info(f"Archived source: {source_id}")
        return True

    async def restore_source(self, source_id: str) -> bool:
        """Restore an archived source."""
        
        source = await self.get_source(source_id)
        if not source or source.status != SourceStatus.ARCHIVED:
            return False
        
        source.status = SourceStatus.ACTIVE
        source.updated_at = datetime.now()
        
        # Remove archive metadata
        if "archived_at" in source.current_version.metadata:
            del source.current_version.metadata["archived_at"]
        if "archive_reason" in source.current_version.metadata:
            del source.current_version.metadata["archive_reason"]
        
        self.logger.info(f"Restored source: {source_id}")
        return True

    async def check_content_freshness(self) -> List[Dict[str, Any]]:
        """Check content freshness and identify stale sources."""
        
        stale_sources = []
        now = datetime.now()
        
        for source in self.sources.values():
            if source.status != SourceStatus.ACTIVE:
                continue
            
            # Check if expired
            if source.expiry_date and now > source.expiry_date:
                stale_sources.append({
                    "source_id": source.source_id,
                    "title": source.title,
                    "issue": "expired",
                    "expiry_date": source.expiry_date.isoformat(),
                    "days_overdue": (now - source.expiry_date).days
                })
                continue
            
            # Check if needs review
            if source.last_reviewed:
                next_review = source.last_reviewed + source.review_frequency
                if now > next_review:
                    stale_sources.append({
                        "source_id": source.source_id,
                        "title": source.title,
                        "issue": "needs_review",
                        "last_reviewed": source.last_reviewed.isoformat(),
                        "days_overdue": (now - next_review).days
                    })
            
            # Check content freshness score
            if source.content_freshness < 0.5:
                stale_sources.append({
                    "source_id": source.source_id,
                    "title": source.title,
                    "issue": "low_freshness",
                    "freshness_score": source.content_freshness
                })
        
        return stale_sources

    async def update_usage_analytics(self, source_id: str, effectiveness_score: Optional[float] = None):
        """Update source usage analytics."""
        
        source = await self.get_source(source_id)
        if not source:
            return
        
        source.usage_count += 1
        source.last_accessed = datetime.now()
        
        if effectiveness_score is not None:
            # Update effectiveness score with exponential moving average
            alpha = 0.1  # Learning rate
            source.effectiveness_score = (
                alpha * effectiveness_score + 
                (1 - alpha) * source.effectiveness_score
            )
        
        # Update content freshness based on usage
        days_since_update = (datetime.now() - source.updated_at).days
        source.content_freshness = max(0.1, 1.0 - (days_since_update / 365.0))

    def _generate_source_id(self, title: str, source_type: SourceType) -> str:
        """Generate unique source ID."""
        
        # Create base from title and type
        base = f"{source_type.value}_{title}".lower()
        base = "".join(c for c in base if c.isalnum() or c in "_-")
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{base}_{timestamp}"

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _increment_version(self, current_version: str) -> str:
        """Increment version number."""
        try:
            parts = current_version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.1"

    async def _enrich_metadata(self, source: SourceDocument, content: str, metadata: Dict[str, Any]):
        """Auto-enrich source metadata."""
        
        # Auto-categorize based on content and metadata
        if not source.category:
            source.category = self._auto_categorize(content, source.source_type)
        
        # Auto-tag based on content
        if not source.tags:
            source.tags = self._auto_tag(content)
        
        # Set product area if not specified
        if not source.product_area and "product_area" in metadata:
            source.product_area = metadata["product_area"]
        
        # Set customer tier if not specified
        if not source.customer_tier and "customer_tier" in metadata:
            source.customer_tier = metadata["customer_tier"]
        
        # Set urgency level based on content
        if not source.urgency_level or source.urgency_level == "medium":
            source.urgency_level = self._detect_urgency(content)

    def _auto_categorize(self, content: str, source_type: SourceType) -> str:
        """Auto-categorize content based on type and content analysis."""
        
        # Basic categorization logic based on source type
        if source_type == SourceType.CUSTOMER_SUPPORT:
            return "support"
        elif source_type == SourceType.DOCUMENTATION:
            return "documentation"
        elif source_type == SourceType.FAQ:
            return "faq"
        elif source_type == SourceType.KNOWLEDGE_ARTICLE:
            return "knowledge"
        else:
            return "general"
    
    def _auto_tag(self, content: str) -> List[str]:
        """Auto-generate tags based on content analysis."""
        tags = []
        content_lower = content.lower()
        
        # Basic keyword-based tagging
        if "urgent" in content_lower or "critical" in content_lower:
            tags.append("urgent")
        if "bug" in content_lower or "error" in content_lower:
            tags.append("bug")
        if "feature" in content_lower:
            tags.append("feature")
        if "payment" in content_lower or "billing" in content_lower:
            tags.append("billing")
        
        return tags if tags else ["general"]
    
    def _detect_urgency(self, content: str) -> str:
        """Detect urgency level from content."""
        content_lower = content.lower()
        
        urgent_keywords = ["urgent", "critical", "emergency", "asap", "immediately"]
        high_keywords = ["important", "priority", "soon", "quickly"]
        
        if any(keyword in content_lower for keyword in urgent_keywords):
            return "urgent"
        elif any(keyword in content_lower for keyword in high_keywords):
            return "high"
        else:
            return "medium"
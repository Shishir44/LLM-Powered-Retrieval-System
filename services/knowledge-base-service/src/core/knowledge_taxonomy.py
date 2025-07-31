"""
Knowledge Taxonomy System
Hierarchical knowledge organization for customer support content
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from datetime import datetime
from pathlib import Path

@dataclass
class TaxonomyNode:
    """Represents a node in the knowledge taxonomy."""
    
    node_id: str
    name: str
    description: str
    level: int  # 0=root, 1=category, 2=subcategory, 3=topic
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    
    # Content associations
    document_ids: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    synonyms: Set[str] = field(default_factory=set)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    confidence_score: float = 1.0

class TaxonomyLevel(Enum):
    """Taxonomy hierarchy levels."""
    ROOT = 0
    CATEGORY = 1
    SUBCATEGORY = 2
    TOPIC = 3

@dataclass
class ClassificationResult:
    """Result of content classification."""
    
    document_id: str
    primary_path: List[str]  # Full path from root to leaf
    alternative_paths: List[List[str]]
    confidence_scores: Dict[str, float]
    suggested_keywords: Set[str]
    classification_reasoning: str

class KnowledgeTaxonomy:
    """Hierarchical knowledge organization system for customer support."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Taxonomy storage
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.root_nodes: Set[str] = set()
        
        # Classification settings
        self.classification_settings = {
            "min_confidence_threshold": 0.6,
            "max_alternative_paths": 3,
            "keyword_extraction_limit": 10,
            "enable_auto_expansion": True,
            "similarity_threshold": 0.8
        }
        
        # Initialize default taxonomy (will be called when needed)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure the taxonomy is initialized."""
        if not self._initialized:
            await self._initialize_default_taxonomy()
            self._initialized = True

    async def _initialize_default_taxonomy(self):
        """Initialize the default customer support taxonomy."""

        try:
            # Create root categories
            await self._create_root_categories()

            # Create subcategories
            await self._create_subcategories()

            # Create topics
            await self._create_topics()

            self.logger.info("Default knowledge taxonomy initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize taxonomy: {e}")
    
    async def _create_root_categories(self):
        """Create main customer support categories."""
        
        categories = [
            {
                "id": "technical_support",
                "name": "Technical Support",
                "description": "Technical issues, troubleshooting, and product functionality",
                "keywords": {"technical", "troubleshooting", "bug", "error", "issue", "problem"}
            },
            {
                "id": "billing_account",
                "name": "Billing & Account",
                "description": "Billing questions, account management, and subscription issues",
                "keywords": {"billing", "payment", "account", "subscription", "invoice", "charge"}
            },
            {
                "id": "product_features",
                "name": "Product Features",
                "description": "Product capabilities, features, and how-to guides",
                "keywords": {"feature", "capability", "how-to", "guide", "tutorial", "usage"}
            },
            {
                "id": "getting_started",
                "name": "Getting Started",
                "description": "Onboarding, setup, and initial configuration",
                "keywords": {"setup", "onboarding", "getting started", "installation", "configuration"}
            },
            {
                "id": "policies_legal",
                "name": "Policies & Legal",
                "description": "Terms of service, privacy policy, and legal information",
                "keywords": {"policy", "terms", "legal", "privacy", "compliance", "gdpr"}
            },
            {
                "id": "integrations",
                "name": "Integrations",
                "description": "Third-party integrations and API documentation",
                "keywords": {"integration", "api", "webhook", "third-party", "connector"}
            }
        ]
        
        for category in categories:
            node = TaxonomyNode(
                node_id=category["id"],
                name=category["name"],
                description=category["description"],
                level=TaxonomyLevel.CATEGORY.value,
                keywords=category["keywords"]
            )
            
            self.nodes[category["id"]] = node
            self.root_nodes.add(category["id"])
    
    async def _create_subcategories(self):
        """Create subcategories under main categories."""
        
        subcategories = {
            "technical_support": [
                {"id": "login_authentication", "name": "Login & Authentication", 
                 "keywords": {"login", "password", "authentication", "2fa", "sso"}},
                {"id": "performance_issues", "name": "Performance Issues",
                 "keywords": {"slow", "performance", "loading", "timeout", "lag"}},
                {"id": "data_sync", "name": "Data Synchronization",
                 "keywords": {"sync", "synchronization", "data", "update", "refresh"}},
                {"id": "mobile_app", "name": "Mobile Application",
                 "keywords": {"mobile", "app", "ios", "android", "smartphone"}}
            ],
            "billing_account": [
                {"id": "payment_methods", "name": "Payment Methods",
                 "keywords": {"payment", "credit card", "paypal", "billing method"}},
                {"id": "subscription_management", "name": "Subscription Management",
                 "keywords": {"subscription", "plan", "upgrade", "downgrade", "cancel"}},
                {"id": "invoices_receipts", "name": "Invoices & Receipts",
                 "keywords": {"invoice", "receipt", "billing history", "statement"}},
                {"id": "refunds_credits", "name": "Refunds & Credits",
                 "keywords": {"refund", "credit", "chargeback", "dispute"}}
            ],
            "product_features": [
                {"id": "core_functionality", "name": "Core Functionality",
                 "keywords": {"core", "main", "primary", "basic", "essential"}},
                {"id": "advanced_features", "name": "Advanced Features",
                 "keywords": {"advanced", "premium", "pro", "enterprise", "custom"}},
                {"id": "reporting_analytics", "name": "Reporting & Analytics",
                 "keywords": {"report", "analytics", "dashboard", "metrics", "insights"}},
                {"id": "collaboration", "name": "Collaboration",
                 "keywords": {"collaboration", "sharing", "team", "workspace", "permissions"}}
            ]
        }
        
        for parent_id, subs in subcategories.items():
            for sub in subs:
                node = TaxonomyNode(
                    node_id=sub["id"],
                    name=sub["name"],
                    description=f"{sub['name']} under {self.nodes[parent_id].name}",
                    level=TaxonomyLevel.SUBCATEGORY.value,
                    parent_id=parent_id,
                    keywords=sub["keywords"]
                )
                
                self.nodes[sub["id"]] = node
                self.nodes[parent_id].children.add(sub["id"])
    
    async def _create_topics(self):
        """Create specific topics under subcategories."""
        
        topics = {
            "login_authentication": [
                {"id": "forgot_password", "name": "Forgot Password",
                 "keywords": {"forgot", "reset", "password", "recovery"}},
                {"id": "two_factor_auth", "name": "Two-Factor Authentication",
                 "keywords": {"2fa", "two-factor", "authenticator", "security"}}
            ],
            "payment_methods": [
                {"id": "add_payment_method", "name": "Add Payment Method",
                 "keywords": {"add", "new", "payment", "method", "card"}},
                {"id": "update_billing_info", "name": "Update Billing Information",
                 "keywords": {"update", "change", "billing", "address", "info"}}
            ],
            "core_functionality": [
                {"id": "basic_operations", "name": "Basic Operations",
                 "keywords": {"basic", "create", "edit", "delete", "view"}},
                {"id": "search_filter", "name": "Search & Filter",
                 "keywords": {"search", "filter", "find", "query", "sort"}}
            ]
        }
        
        for parent_id, topic_list in topics.items():
            for topic in topic_list:
                node = TaxonomyNode(
                    node_id=topic["id"],
                    name=topic["name"],
                    description=f"{topic['name']} under {self.nodes[parent_id].name}",
                    level=TaxonomyLevel.TOPIC.value,
                    parent_id=parent_id,
                    keywords=topic["keywords"]
                )
                
                self.nodes[topic["id"]] = node
                if parent_id in self.nodes:
                    self.nodes[parent_id].children.add(topic["id"])
    
    async def classify_content(self,
                             content: str,
                             title: str = "",
                             metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """Classify content into taxonomy categories."""

        await self._ensure_initialized()

        try:
            # Extract keywords from content
            content_keywords = await self._extract_keywords(content, title)
            
            # Find matching taxonomy paths
            matching_paths = await self._find_matching_paths(content_keywords, content)
            
            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                matching_paths, content_keywords, content
            )
            
            # Select primary path
            primary_path = matching_paths[0] if matching_paths else []
            alternative_paths = matching_paths[1:self.classification_settings["max_alternative_paths"]]
            
            # Generate classification reasoning
            reasoning = await self._generate_classification_reasoning(
                primary_path, content_keywords, confidence_scores
            )
            
            return ClassificationResult(
                document_id=metadata.get("document_id", "") if metadata else "",
                primary_path=primary_path,
                alternative_paths=alternative_paths,
                confidence_scores=confidence_scores,
                suggested_keywords=content_keywords,
                classification_reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Content classification failed: {e}")
            return ClassificationResult(
                document_id="",
                primary_path=[],
                alternative_paths=[],
                confidence_scores={},
                suggested_keywords=set(),
                classification_reasoning=f"Classification failed: {e}"
            )
    
    async def _extract_keywords(self, content: str, title: str = "") -> Set[str]:
        """Extract relevant keywords from content."""
        
        import re
        from collections import Counter
        
        # Combine title and content
        text = f"{title} {content}".lower()
        
        # Basic keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Filter common words
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", 
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", 
            "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", 
            "did", "man", "way", "use", "your", "they", "have", "this", "that",
            "with", "from", "will", "been", "said", "each", "which", "their"
        }
        
        # Count word frequency
        word_counts = Counter(word for word in words if word not in stop_words)
        
        # Return top keywords
        top_keywords = {word for word, count in word_counts.most_common(
            self.classification_settings["keyword_extraction_limit"]
        )}
        
        return top_keywords

    async def _find_matching_paths(self, keywords: Set[str], content: str) -> List[List[str]]:
        """Find taxonomy paths that match the content keywords."""

        path_scores = []

        # Check all possible paths from root to leaf
        for root_id in self.root_nodes:
            paths = await self._get_all_paths_from_node(root_id)

            for path in paths:
                score = await self._calculate_path_score(path, keywords, content)
                if score >= self.classification_settings["min_confidence_threshold"]:
                    path_scores.append((path, score))

        # Sort by score and return paths
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in path_scores]

    async def _get_all_paths_from_node(self, node_id: str, current_path: List[str] = None) -> List[List[str]]:
        """Get all paths from a node to its leaf nodes."""

        if current_path is None:
            current_path = []

        current_path = current_path + [node_id]
        node = self.nodes.get(node_id)

        if not node or not node.children:
            return [current_path]

        paths = []
        for child_id in node.children:
            child_paths = await self._get_all_paths_from_node(child_id, current_path)
            paths.extend(child_paths)

        return paths

    async def _calculate_path_score(self, path: List[str], keywords: Set[str], content: str) -> float:
        """Calculate relevance score for a taxonomy path."""

        total_score = 0.0
        total_weight = 0.0

        for i, node_id in enumerate(path):
            node = self.nodes.get(node_id)
            if not node:
                continue

            # Weight decreases with depth (root categories more important)
            weight = 1.0 / (i + 1)

            # Calculate keyword overlap
            node_keywords = node.keywords | node.synonyms
            overlap = len(keywords & node_keywords)
            keyword_score = overlap / max(len(node_keywords), 1) if node_keywords else 0

            # Content similarity (basic implementation)
            content_score = await self._calculate_content_similarity(content, node.description)

            # Combined score
            node_score = (keyword_score * 0.7) + (content_score * 0.3)
            total_score += node_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    async def _calculate_content_similarity(self, content: str, description: str) -> float:
        """Calculate similarity between content and node description."""

        # Simple word overlap similarity (can be enhanced with embeddings)
        content_words = set(content.lower().split())
        desc_words = set(description.lower().split())

        if not desc_words:
            return 0.0

        overlap = len(content_words & desc_words)
        return overlap / len(desc_words)

    async def _calculate_confidence_scores(self,
                                         paths: List[List[str]],
                                         keywords: Set[str],
                                         content: str) -> Dict[str, float]:
        """Calculate confidence scores for classification paths."""

        scores = {}

        for path in paths:
            path_key = " -> ".join([self.nodes[node_id].name for node_id in path])
            score = await self._calculate_path_score(path, keywords, content)
            scores[path_key] = score

        return scores

    async def _generate_classification_reasoning(self,
                                               path: List[str],
                                               keywords: Set[str],
                                               scores: Dict[str, float]) -> str:
        """Generate human-readable classification reasoning."""

        if not path:
            return "No suitable classification found based on content analysis."

        path_names = [self.nodes[node_id].name for node_id in path]
        path_str = " -> ".join(path_names)

        # Find matching keywords
        matching_keywords = set()
        for node_id in path:
            node = self.nodes.get(node_id)
            if node:
                matching_keywords.update(keywords & node.keywords)

        reasoning = f"Classified as '{path_str}' based on "

        if matching_keywords:
            keyword_list = ", ".join(list(matching_keywords)[:5])
            reasoning += f"matching keywords: {keyword_list}"

        if scores:
            max_score = max(scores.values())
            reasoning += f" (confidence: {max_score:.2f})"

        return reasoning

    async def add_document_to_taxonomy(self,
                                     document_id: str,
                                     classification: ClassificationResult):
        """Associate a document with taxonomy nodes."""

        try:
            # Add to primary path nodes
            for node_id in classification.primary_path:
                if node_id in self.nodes:
                    self.nodes[node_id].document_ids.add(document_id)
                    self.nodes[node_id].usage_count += 1
                    self.nodes[node_id].updated_at = datetime.now()

            # Update keywords
            for node_id in classification.primary_path:
                if node_id in self.nodes:
                    self.nodes[node_id].keywords.update(
                        list(classification.suggested_keywords)[:5]
                    )

            self.logger.info(f"Document {document_id} added to taxonomy")

        except Exception as e:
            self.logger.error(f"Failed to add document to taxonomy: {e}")

    async def get_taxonomy_structure(self) -> Dict[str, Any]:
        """Get the complete taxonomy structure."""

        structure = {}

        for root_id in self.root_nodes:
            structure[root_id] = await self._build_node_structure(root_id)

        return structure

    async def _build_node_structure(self, node_id: str) -> Dict[str, Any]:
        """Build hierarchical structure for a node."""

        node = self.nodes.get(node_id)
        if not node:
            return {}

        structure = {
            "id": node.node_id,
            "name": node.name,
            "description": node.description,
            "level": node.level,
            "document_count": len(node.document_ids),
            "usage_count": node.usage_count,
            "keywords": list(node.keywords),
            "children": {}
        }

        for child_id in node.children:
            structure["children"][child_id] = await self._build_node_structure(child_id)

        return structure

    async def search_taxonomy(self, query: str) -> List[Dict[str, Any]]:
        """Search taxonomy nodes by query."""

        query_lower = query.lower()
        results = []

        for node in self.nodes.values():
            # Check name, description, and keywords
            if (query_lower in node.name.lower() or
                query_lower in node.description.lower() or
                any(query_lower in keyword for keyword in node.keywords)):

                # Get full path
                path = await self._get_path_to_root(node.node_id)
                path_names = [self.nodes[nid].name for nid in reversed(path)]

                results.append({
                    "node_id": node.node_id,
                    "name": node.name,
                    "description": node.description,
                    "level": node.level,
                    "path": " -> ".join(path_names),
                    "document_count": len(node.document_ids),
                    "usage_count": node.usage_count
                })

        return sorted(results, key=lambda x: x["usage_count"], reverse=True)

    async def _get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root."""

        path = []
        current_id = node_id

        while current_id:
            path.append(current_id)
            node = self.nodes.get(current_id)
            current_id = node.parent_id if node else None

        return path

    async def get_node_documents(self, node_id: str) -> List[str]:
        """Get all documents associated with a taxonomy node."""

        node = self.nodes.get(node_id)
        if not node:
            return []

        # Get documents from this node and all children
        all_documents = set(node.document_ids)

        async def collect_child_documents(nid: str):
            child_node = self.nodes.get(nid)
            if child_node:
                all_documents.update(child_node.document_ids)
                for child_id in child_node.children:
                    await collect_child_documents(child_id)

        for child_id in node.children:
            await collect_child_documents(child_id)

        return list(all_documents)

    async def export_taxonomy(self, file_path: str):
        """Export taxonomy to JSON file."""

        try:
            structure = await self.get_taxonomy_structure()

            with open(file_path, 'w') as f:
                json.dump(structure, f, indent=2, default=str)

            self.logger.info(f"Taxonomy exported to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to export taxonomy: {e}")

    async def import_taxonomy(self, file_path: str):
        """Import taxonomy from JSON file."""

        try:
            with open(file_path, 'r') as f:
                structure = json.load(f)

            # Clear existing taxonomy
            self.nodes.clear()
            self.root_nodes.clear()

            # Rebuild from structure
            await self._rebuild_from_structure(structure)

            self.logger.info(f"Taxonomy imported from {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to import taxonomy: {e}")

    async def _rebuild_from_structure(self, structure: Dict[str, Any], parent_id: str = None):
        """Rebuild taxonomy from exported structure."""

        for node_id, node_data in structure.items():
            node = TaxonomyNode(
                node_id=node_data["id"],
                name=node_data["name"],
                description=node_data["description"],
                level=node_data["level"],
                parent_id=parent_id,
                keywords=set(node_data.get("keywords", [])),
                usage_count=node_data.get("usage_count", 0)
            )

            self.nodes[node_id] = node

            if parent_id is None:
                self.root_nodes.add(node_id)
            else:
                self.nodes[parent_id].children.add(node_id)

            # Process children
            if "children" in node_data:
                await self._rebuild_from_structure(node_data["children"], node_id)

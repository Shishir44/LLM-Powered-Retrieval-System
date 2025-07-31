"""
Content Categorizer
Automatic content classification using machine learning models
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import asyncio
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime
from collections import Counter, defaultdict

# ML Libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    import joblib
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False

# NLP Libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CategoryType(Enum):
    """Types of content categories."""
    PRODUCT_AREA = "product_area"
    CUSTOMER_TIER = "customer_tier"
    URGENCY_LEVEL = "urgency_level"
    CONTENT_TYPE = "content_type"
    INTENT_TYPE = "intent_type"

@dataclass
class CategoryPrediction:
    """Single category prediction result."""
    category: str
    confidence: float
    reasoning: str
    features_used: List[str] = field(default_factory=list)

@dataclass
class ClassificationResult:
    """Complete classification result for content."""
    
    document_id: str
    content_preview: str
    
    # Category predictions
    product_area: CategoryPrediction
    customer_tier: CategoryPrediction
    urgency_level: CategoryPrediction
    content_type: CategoryPrediction
    intent_type: CategoryPrediction
    
    # Overall metrics
    overall_confidence: float
    processing_time: float
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    # Additional insights
    extracted_keywords: Set[str] = field(default_factory=set)
    detected_entities: List[Dict[str, Any]] = field(default_factory=list)
    language_detected: str = "en"

class ContentCategorizer:
    """Automatic content classification system using multiple ML approaches."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Classification models
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        
        # Model settings
        self.model_settings = {
            "use_transformers": TRANSFORMERS_AVAILABLE,
            "use_sklearn": ML_LIBS_AVAILABLE,
            "confidence_threshold": 0.6,
            "max_features": 10000,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.95
        }
        
        # Category definitions
        self.category_definitions = self._initialize_category_definitions()
        
        # Training data
        self.training_data = defaultdict(list)
        
        # Initialize models (will be called when needed)
        self._models_initialized = False
    
    def _initialize_category_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize category definitions and keywords."""
        
        return {
            "product_area": {
                "billing": {
                    "keywords": ["payment", "invoice", "billing", "subscription", "charge", "refund", "credit"],
                    "patterns": [r"\b(payment|billing|invoice|subscription)\b", r"\$\d+", r"charge[ds]?"]
                },
                "technical": {
                    "keywords": ["error", "bug", "issue", "problem", "troubleshoot", "fix", "broken"],
                    "patterns": [r"\berror\s+\d+\b", r"\bfailed\s+to\b", r"\bnot\s+working\b"]
                },
                "account": {
                    "keywords": ["account", "profile", "settings", "preferences", "login", "password"],
                    "patterns": [r"\baccount\s+settings\b", r"\bprofile\s+update\b", r"\blogin\s+issue\b"]
                },
                "features": {
                    "keywords": ["feature", "functionality", "capability", "how to", "tutorial", "guide"],
                    "patterns": [r"\bhow\s+to\b", r"\bfeature\s+request\b", r"\bcapability\b"]
                }
            },
            "customer_tier": {
                "free": {
                    "keywords": ["free", "trial", "basic", "starter", "limited"],
                    "patterns": [r"\bfree\s+plan\b", r"\btrial\s+version\b", r"\bbasic\s+tier\b"]
                },
                "premium": {
                    "keywords": ["premium", "pro", "paid", "subscription", "upgrade"],
                    "patterns": [r"\bpremium\s+plan\b", r"\bpro\s+version\b", r"\bpaid\s+tier\b"]
                },
                "enterprise": {
                    "keywords": ["enterprise", "business", "corporate", "team", "organization"],
                    "patterns": [r"\benterprise\s+plan\b", r"\bbusiness\s+account\b", r"\bcorporate\s+license\b"]
                }
            },
            "urgency_level": {
                "low": {
                    "keywords": ["question", "wondering", "curious", "when convenient", "no rush"],
                    "patterns": [r"\bwhen\s+convenient\b", r"\bno\s+rush\b", r"\bjust\s+wondering\b"]
                },
                "medium": {
                    "keywords": ["need", "help", "assistance", "support", "issue"],
                    "patterns": [r"\bneed\s+help\b", r"\bhaving\s+trouble\b", r"\bissue\s+with\b"]
                },
                "high": {
                    "keywords": ["urgent", "asap", "immediately", "critical", "emergency"],
                    "patterns": [r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\bcritical\s+issue\b"]
                },
                "critical": {
                    "keywords": ["down", "broken", "not working", "emergency", "production"],
                    "patterns": [r"\bsystem\s+down\b", r"\bproduction\s+issue\b", r"\bemergency\b"]
                }
            },
            "content_type": {
                "question": {
                    "keywords": ["how", "what", "why", "when", "where", "can", "should"],
                    "patterns": [r"^\s*how\s+", r"^\s*what\s+", r"^\s*why\s+", r"\?\s*$"]
                },
                "problem_report": {
                    "keywords": ["error", "problem", "issue", "bug", "broken", "not working"],
                    "patterns": [r"\berror\s+message\b", r"\bnot\s+working\b", r"\bproblem\s+with\b"]
                },
                "feature_request": {
                    "keywords": ["request", "suggestion", "feature", "enhancement", "improvement"],
                    "patterns": [r"\bfeature\s+request\b", r"\bwould\s+like\b", r"\bsuggestion\b"]
                },
                "feedback": {
                    "keywords": ["feedback", "review", "opinion", "thoughts", "experience"],
                    "patterns": [r"\bfeedback\s+on\b", r"\bmy\s+experience\b", r"\bthoughts\s+on\b"]
                }
            },
            "intent_type": {
                "information_seeking": {
                    "keywords": ["learn", "understand", "explain", "information", "details"],
                    "patterns": [r"\blearn\s+about\b", r"\bexplain\s+how\b", r"\binformation\s+on\b"]
                },
                "problem_solving": {
                    "keywords": ["fix", "solve", "resolve", "troubleshoot", "repair"],
                    "patterns": [r"\bhow\s+to\s+fix\b", r"\bsolve\s+this\b", r"\btroubleshoot\b"]
                },
                "task_completion": {
                    "keywords": ["do", "perform", "complete", "accomplish", "execute"],
                    "patterns": [r"\bhow\s+to\s+do\b", r"\bperform\s+this\b", r"\bcomplete\s+the\b"]
                },
                "comparison": {
                    "keywords": ["compare", "difference", "versus", "vs", "better"],
                    "patterns": [r"\bcompare\s+to\b", r"\bdifference\s+between\b", r"\bvs\s+"]
                }
            }
        }
    
    async def _ensure_models_initialized(self):
        """Ensure models are initialized."""
        if not self._models_initialized:
            await self._initialize_models()
            self._models_initialized = True

    async def _initialize_models(self):
        """Initialize classification models."""

        try:
            if self.model_settings["use_sklearn"]:
                await self._initialize_sklearn_models()

            if self.model_settings["use_transformers"]:
                await self._initialize_transformer_models()

            self.logger.info("Content categorization models initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
    
    async def _initialize_sklearn_models(self):
        """Initialize scikit-learn based models."""
        
        for category_type in CategoryType:
            category_name = category_type.value
            
            # Create pipeline with TF-IDF and classifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self.model_settings["max_features"],
                    ngram_range=self.model_settings["ngram_range"],
                    min_df=self.model_settings["min_df"],
                    max_df=self.model_settings["max_df"],
                    stop_words='english'
                )),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            self.models[f"sklearn_{category_name}"] = pipeline
    
    async def _initialize_transformer_models(self):
        """Initialize transformer-based models."""
        
        try:
            # Use a general classification model
            model_name = "microsoft/DialoGPT-medium"  # Can be replaced with better models
            
            # Initialize for intent classification
            self.models["transformer_intent"] = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize transformer models: {e}")
    
    async def classify_content(self,
                             content: str,
                             title: str = "",
                             metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """Classify content across all category types."""

        await self._ensure_models_initialized()
        start_time = datetime.now()

        try:
            # Prepare text for classification
            full_text = f"{title} {content}".strip()
            
            # Extract features
            keywords = await self._extract_keywords(full_text)
            entities = await self._extract_entities(full_text)
            language = await self._detect_language(full_text)
            
            # Classify across all categories
            predictions = {}
            
            for category_type in CategoryType:
                prediction = await self._classify_single_category(
                    full_text, category_type, keywords
                )
                predictions[category_type.value] = prediction
            
            # Calculate overall confidence
            overall_confidence = np.mean([
                pred.confidence for pred in predictions.values()
            ])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ClassificationResult(
                document_id=metadata.get("document_id", "") if metadata else "",
                content_preview=content[:500] + "..." if len(content) > 500 else content,  # Increased preview length
                product_area=predictions["product_area"],
                customer_tier=predictions["customer_tier"],
                urgency_level=predictions["urgency_level"],
                content_type=predictions["content_type"],
                intent_type=predictions["intent_type"],
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                extracted_keywords=keywords,
                detected_entities=entities,
                language_detected=language
            )
            
        except Exception as e:
            self.logger.error(f"Content classification failed: {e}")
            
            # Return default classification
            default_prediction = CategoryPrediction(
                category="unknown",
                confidence=0.0,
                reasoning=f"Classification failed: {e}"
            )
            
            return ClassificationResult(
                document_id="",
                content_preview=content[:500] if content else "",  # Increased preview length
                product_area=default_prediction,
                customer_tier=default_prediction,
                urgency_level=default_prediction,
                content_type=default_prediction,
                intent_type=default_prediction,
                overall_confidence=0.0,
                processing_time=0.0
            )

    async def _classify_single_category(self,
                                      text: str,
                                      category_type: CategoryType,
                                      keywords: Set[str]) -> CategoryPrediction:
        """Classify content for a single category type."""

        category_name = category_type.value
        category_defs = self.category_definitions.get(category_name, {})

        # Rule-based classification
        rule_scores = await self._rule_based_classification(text, category_defs, keywords)

        # ML-based classification (if models are available)
        ml_scores = await self._ml_based_classification(text, category_name)

        # Combine scores
        combined_scores = self._combine_classification_scores(rule_scores, ml_scores)

        # Select best category
        if combined_scores:
            best_category = max(combined_scores.items(), key=lambda x: x[1])
            category, confidence = best_category

            # Generate reasoning
            reasoning = await self._generate_classification_reasoning(
                category, confidence, rule_scores, ml_scores, keywords
            )

            return CategoryPrediction(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                features_used=list(keywords)[:10]
            )

        return CategoryPrediction(
            category="unknown",
            confidence=0.0,
            reasoning="No classification rules or models matched the content"
        )

    async def _rule_based_classification(self,
                                       text: str,
                                       category_defs: Dict[str, Any],
                                       keywords: Set[str]) -> Dict[str, float]:
        """Perform rule-based classification using keywords and patterns."""

        scores = {}
        text_lower = text.lower()

        for category, definition in category_defs.items():
            score = 0.0

            # Keyword matching
            category_keywords = set(definition.get("keywords", []))
            keyword_overlap = len(keywords & category_keywords)
            keyword_score = keyword_overlap / max(len(category_keywords), 1)

            # Pattern matching
            patterns = definition.get("patterns", [])
            pattern_matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pattern_matches += 1

            pattern_score = pattern_matches / max(len(patterns), 1) if patterns else 0

            # Combined rule score
            score = (keyword_score * 0.6) + (pattern_score * 0.4)
            scores[category] = score

        return scores

    async def _ml_based_classification(self, text: str, category_name: str) -> Dict[str, float]:
        """Perform ML-based classification if models are available."""

        scores = {}

        try:
            # Try sklearn model
            sklearn_model_key = f"sklearn_{category_name}"
            if sklearn_model_key in self.models:
                model = self.models[sklearn_model_key]

                # Check if model is trained
                if hasattr(model, 'classes_'):
                    probabilities = model.predict_proba([text])[0]
                    classes = model.classes_

                    for cls, prob in zip(classes, probabilities):
                        scores[cls] = prob

            # Try transformer model for intent classification
            if category_name == "intent_type" and "transformer_intent" in self.models:
                transformer_model = self.models["transformer_intent"]
                results = transformer_model(text)

                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    scores[label] = score

        except Exception as e:
            self.logger.warning(f"ML classification failed for {category_name}: {e}")

        return scores

    def _combine_classification_scores(self,
                                     rule_scores: Dict[str, float],
                                     ml_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine rule-based and ML-based scores."""

        combined = {}
        all_categories = set(rule_scores.keys()) | set(ml_scores.keys())

        for category in all_categories:
            rule_score = rule_scores.get(category, 0.0)
            ml_score = ml_scores.get(category, 0.0)

            # Weight rule-based higher if no ML model is available
            if ml_scores:
                combined_score = (rule_score * 0.4) + (ml_score * 0.6)
            else:
                combined_score = rule_score

            combined[category] = combined_score

        return combined

    async def _generate_classification_reasoning(self,
                                               category: str,
                                               confidence: float,
                                               rule_scores: Dict[str, float],
                                               ml_scores: Dict[str, float],
                                               keywords: Set[str]) -> str:
        """Generate human-readable reasoning for classification."""

        reasoning_parts = []

        # Confidence level
        if confidence >= 0.8:
            reasoning_parts.append("High confidence classification")
        elif confidence >= 0.6:
            reasoning_parts.append("Medium confidence classification")
        else:
            reasoning_parts.append("Low confidence classification")

        # Rule-based reasoning
        if category in rule_scores and rule_scores[category] > 0:
            reasoning_parts.append(f"matched rule-based patterns")

        # ML reasoning
        if category in ml_scores and ml_scores[category] > 0:
            reasoning_parts.append(f"ML model prediction")

        # Keywords
        if keywords:
            keyword_sample = list(keywords)[:3]
            reasoning_parts.append(f"based on keywords: {', '.join(keyword_sample)}")

        return f"Classified as '{category}' - " + ", ".join(reasoning_parts)

    async def _extract_keywords(self, text: str) -> Set[str]:
        """Extract relevant keywords from text."""

        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter stop words
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
            "did", "man", "way", "use", "your", "they", "have", "this", "that",
            "with", "from", "will", "been", "said", "each", "which", "their"
        }

        # Count word frequency and return top keywords
        word_counts = Counter(word for word in words if word not in stop_words)
        return {word for word, count in word_counts.most_common(20)}

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""

        entities = []

        # Simple entity extraction patterns
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "url": r'https?://[^\s]+',
            "money": r'\$\d+(?:\.\d{2})?',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }

        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        return entities

    async def _detect_language(self, text: str) -> str:
        """Detect language of the text."""

        # Simple language detection based on common words
        english_indicators = {"the", "and", "is", "in", "to", "of", "a", "that", "it", "with"}
        spanish_indicators = {"el", "la", "de", "que", "y", "en", "un", "es", "se", "no"}
        french_indicators = {"le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"}

        words = set(text.lower().split())

        english_score = len(words & english_indicators)
        spanish_score = len(words & spanish_indicators)
        french_score = len(words & french_indicators)

        if english_score >= spanish_score and english_score >= french_score:
            return "en"
        elif spanish_score >= french_score:
            return "es"
        elif french_score > 0:
            return "fr"
        else:
            return "en"  # Default to English

    async def train_model(self,
                         training_data: List[Dict[str, Any]],
                         category_type: CategoryType):
        """Train a classification model with provided data."""

        if not ML_LIBS_AVAILABLE:
            self.logger.warning("Scikit-learn not available for model training")
            return

        try:
            category_name = category_type.value

            # Prepare training data
            texts = [item["text"] for item in training_data]
            labels = [item["label"] for item in training_data]

            if len(set(labels)) < 2:
                self.logger.warning(f"Need at least 2 different labels for {category_name}")
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Train model
            model_key = f"sklearn_{category_name}"
            if model_key in self.models:
                model = self.models[model_key]
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                self.logger.info(f"Model {category_name} trained with accuracy: {accuracy:.3f}")

                # Save model
                await self._save_model(model, category_name)

        except Exception as e:
            self.logger.error(f"Model training failed for {category_type}: {e}")

    async def _save_model(self, model, category_name: str):
        """Save trained model to disk."""

        try:
            model_path = f"models/categorizer_{category_name}.joblib"
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved: {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model {category_name}: {e}")

    async def load_model(self, category_name: str):
        """Load trained model from disk."""

        try:
            model_path = f"models/categorizer_{category_name}.joblib"
            model = joblib.load(model_path)
            self.models[f"sklearn_{category_name}"] = model
            self.logger.info(f"Model loaded: {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model {category_name}: {e}")

    async def get_category_statistics(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""

        stats = {
            "models_available": list(self.models.keys()),
            "category_definitions": {},
            "model_settings": self.model_settings
        }

        for category_type in CategoryType:
            category_name = category_type.value
            category_defs = self.category_definitions.get(category_name, {})

            stats["category_definitions"][category_name] = {
                "categories": list(category_defs.keys()),
                "total_keywords": sum(len(def_data.get("keywords", []))
                                    for def_data in category_defs.values()),
                "total_patterns": sum(len(def_data.get("patterns", []))
                                    for def_data in category_defs.values())
            }

        return stats

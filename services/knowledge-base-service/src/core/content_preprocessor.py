"""
Content Preprocessor
Text cleaning, normalization, and structure preservation for parsed documents
"""

import re
import html
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class PreprocessedContent:
    """Preprocessed content with cleaning metadata."""
    content: str
    original_length: int
    processed_length: int
    cleaning_steps: List[str]
    preserved_structure: Dict[str, Any]
    quality_score: float
    language: Optional[str]
    encoding_issues: List[str]

class ContentPreprocessor:
    """Advanced content preprocessing with structure preservation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cleaning patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'excessive_whitespace': re.compile(r'\s{3,}'),
            'repeated_chars': re.compile(r'(.)\1{4,}'),
            'html_tags': re.compile(r'<[^>]+>'),
            'special_chars': re.compile(r'[^\w\s\-.,!?;:()\[\]{}"\'/\\@#$%^&*+=|`~]'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'bullet_points': re.compile(r'^[\s]*[•·▪▫‣⁃]\s*', re.MULTILINE),
            'numbered_lists': re.compile(r'^[\s]*\d+[.)]\s*', re.MULTILINE),
            'headers': re.compile(r'^#+\s*(.+)$', re.MULTILINE),
            'code_blocks': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'tables': re.compile(r'\|[^|\n]*\|'),
        }
        
        # Language detection patterns
        self.language_patterns = {
            'english': re.compile(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', re.IGNORECASE),
            'spanish': re.compile(r'\b(el|la|los|las|y|o|pero|en|con|por|para|de)\b', re.IGNORECASE),
            'french': re.compile(r'\b(le|la|les|et|ou|mais|dans|sur|avec|par|pour|de)\b', re.IGNORECASE),
            'german': re.compile(r'\b(der|die|das|und|oder|aber|in|auf|mit|von|zu|für)\b', re.IGNORECASE),
        }

    async def preprocess_content(self, 
                               content: str,
                               preserve_structure: bool = True,
                               aggressive_cleaning: bool = False,
                               custom_patterns: Optional[Dict[str, str]] = None) -> PreprocessedContent:
        """Main preprocessing pipeline."""
        
        original_length = len(content)
        cleaning_steps = []
        encoding_issues = []
        preserved_structure = {}
        
        if not content.strip():
            return PreprocessedContent(
                content="",
                original_length=0,
                processed_length=0,
                cleaning_steps=["empty_content"],
                preserved_structure={},
                quality_score=0.0,
                language=None,
                encoding_issues=[]
            )
        
        try:
            # Step 1: Handle encoding issues
            content, encoding_fixes = self._fix_encoding_issues(content)
            if encoding_fixes:
                cleaning_steps.append("encoding_fixes")
                encoding_issues.extend(encoding_fixes)
            
            # Step 2: Preserve structure if requested
            if preserve_structure:
                content, structure_info = self._preserve_document_structure(content)
                preserved_structure = structure_info
                cleaning_steps.append("structure_preservation")
            
            # Step 3: Basic HTML cleaning
            content = self._clean_html(content)
            cleaning_steps.append("html_cleaning")
            
            # Step 4: Normalize whitespace
            content = self._normalize_whitespace(content)
            cleaning_steps.append("whitespace_normalization")
            
            # Step 5: Handle special characters
            content = self._handle_special_characters(content, aggressive_cleaning)
            cleaning_steps.append("special_characters")
            
            # Step 6: Clean repeated patterns
            content = self._clean_repeated_patterns(content)
            cleaning_steps.append("repeated_patterns")
            
            # Step 7: Apply custom patterns if provided
            if custom_patterns:
                content = self._apply_custom_patterns(content, custom_patterns)
                cleaning_steps.append("custom_patterns")
            
            # Step 8: Final cleanup
            content = self._final_cleanup(content)
            cleaning_steps.append("final_cleanup")
            
            # Step 9: Detect language
            language = self._detect_language(content)
            
            # Step 10: Calculate quality score
            quality_score = self._calculate_quality_score(content, original_length)
            
            processed_length = len(content)
            
            self.logger.info(f"Preprocessed content: {original_length} -> {processed_length} chars")
            
            return PreprocessedContent(
                content=content,
                original_length=original_length,
                processed_length=processed_length,
                cleaning_steps=cleaning_steps,
                preserved_structure=preserved_structure,
                quality_score=quality_score,
                language=language,
                encoding_issues=encoding_issues
            )
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return PreprocessedContent(
                content=content,
                original_length=original_length,
                processed_length=len(content),
                cleaning_steps=["error"],
                preserved_structure={},
                quality_score=0.0,
                language=None,
                encoding_issues=[str(e)]
            )

    def _fix_encoding_issues(self, content: str) -> Tuple[str, List[str]]:
        """Fix common encoding issues."""
        issues = []
        
        try:
            # Normalize Unicode
            content = unicodedata.normalize('NFKC', content)
            
            # Fix common encoding artifacts
            replacements = {
                'â€™': "'",  # Smart apostrophe
                'â€œ': '"',  # Smart quote left
                'â€': '"',   # Smart quote right
                'â€"': '—',  # Em dash
                'â€"': '–',  # En dash
                'Â': '',     # Non-breaking space artifact
                'â€¦': '...',# Ellipsis
                'Ã¡': 'á',   # á with encoding issue
                'Ã©': 'é',   # é with encoding issue
                'Ã­': 'í',   # í with encoding issue
                'Ã³': 'ó',   # ó with encoding issue
                'Ãº': 'ú',   # ú with encoding issue
            }
            
            for bad, good in replacements.items():
                if bad in content:
                    content = content.replace(bad, good)
                    issues.append(f"Fixed encoding: {bad} -> {good}")
            
            # Remove or replace problematic characters
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
            
        except Exception as e:
            issues.append(f"Encoding fix error: {e}")
        
        return content, issues

    def _preserve_document_structure(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Preserve important document structure elements."""
        structure_info = {
            'headers': [],
            'lists': [],
            'code_blocks': [],
            'tables': [],
            'links': [],
            'emphasis': []
        }
        
        # Extract and preserve headers
        headers = self.patterns['headers'].findall(content)
        structure_info['headers'] = headers
        
        # Extract bullet points and numbered lists
        bullet_matches = self.patterns['bullet_points'].findall(content)
        numbered_matches = self.patterns['numbered_lists'].findall(content)
        structure_info['lists'] = {
            'bullets': len(bullet_matches),
            'numbered': len(numbered_matches)
        }
        
        # Extract code blocks
        code_blocks = self.patterns['code_blocks'].findall(content)
        structure_info['code_blocks'] = len(code_blocks)
        
        # Extract table indicators
        table_rows = self.patterns['tables'].findall(content)
        structure_info['tables'] = len(table_rows)
        
        # Preserve structure markers by replacing with placeholders
        # This helps maintain document flow during cleaning
        
        # Replace headers with normalized versions
        content = self.patterns['headers'].sub(r'\n\n=== \1 ===\n\n', content)
        
        # Normalize list items
        content = self.patterns['bullet_points'].sub('• ', content)
        content = self.patterns['numbered_lists'].sub(lambda m: f"{m.group().strip()} ", content)
        
        return content, structure_info

    def _clean_html(self, content: str) -> str:
        """Clean HTML tags and entities."""
        
        # Decode HTML entities
        content = html.unescape(content)
        
        # Remove HTML tags but preserve some structure
        # Convert some tags to text equivalents
        content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'<p\s*/?>', '\n\n', content, flags=re.IGNORECASE)
        content = re.sub(r'</p>', '\n\n', content, flags=re.IGNORECASE)
        content = re.sub(r'<h[1-6][^>]*>', '\n\n=== ', content, flags=re.IGNORECASE)
        content = re.sub(r'</h[1-6]>', ' ===\n\n', content, flags=re.IGNORECASE)
        
        # Remove remaining HTML tags
        content = self.patterns['html_tags'].sub('', content)
        
        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace while preserving structure."""
        
        # Replace multiple spaces with single space
        content = self.patterns['excessive_whitespace'].sub(' ', content)
        
        # Normalize line breaks
        content = self.patterns['multiple_newlines'].sub('\n\n', content)
        
        # Clean up spaces around punctuation
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([,.!?;:])\s+', r'\1 ', content)
        
        # Fix spacing around parentheses and brackets
        content = re.sub(r'\s*\(\s*', ' (', content)
        content = re.sub(r'\s*\)\s*', ') ', content)
        content = re.sub(r'\s*\[\s*', ' [', content)
        content = re.sub(r'\s*\]\s*', '] ', content)
        
        return content.strip()

    def _handle_special_characters(self, content: str, aggressive: bool = False) -> str:
        """Handle special characters based on cleaning level."""
        
        if aggressive:
            # Remove most special characters except basic punctuation
            content = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'/\\]', ' ', content)
        else:
            # Keep more characters but clean problematic ones
            # Remove control characters and rare symbols
            content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
            
            # Replace some special characters with text equivalents
            replacements = {
                '©': '(c)',
                '®': '(r)',
                '™': '(tm)',
                '°': ' degrees',
                '±': '+/-',
                '×': 'x',
                '÷': '/',
                '≤': '<=',
                '≥': '>=',
                '≠': '!=',
                '→': '->',
                '←': '<-',
                '↑': '^',
                '↓': 'v'
            }
            
            for char, replacement in replacements.items():
                content = content.replace(char, replacement)
        
        return content

    def _clean_repeated_patterns(self, content: str) -> str:
        """Clean repeated characters and patterns."""
        
        # Remove excessive repeated characters (keep up to 3)
        content = self.patterns['repeated_chars'].sub(r'\1\1\1', content)
        
        # Clean repeated punctuation
        content = re.sub(r'([.!?]){4,}', r'\1\1\1', content)
        content = re.sub(r'([,-]){3,}', r'\1\1', content)
        
        # Clean repeated words (simple case)
        content = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', content, flags=re.IGNORECASE)
        
        return content

    def _apply_custom_patterns(self, content: str, patterns: Dict[str, str]) -> str:
        """Apply custom cleaning patterns."""
        
        for pattern_name, pattern_regex in patterns.items():
            try:
                content = re.sub(pattern_regex, '', content)
            except Exception as e:
                self.logger.warning(f"Custom pattern '{pattern_name}' failed: {e}")
        
        return content

    def _final_cleanup(self, content: str) -> str:
        """Final cleanup and normalization."""
        
        # Remove empty lines at start and end
        content = content.strip()
        
        # Ensure single spaces between words
        content = re.sub(r'\s+', ' ', content)
        
        # Fix common punctuation issues
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1\2', content)
        
        # Ensure proper sentence spacing
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.split('\n')]
        content = '\n'.join(lines)
        
        return content

    def _detect_language(self, content: str) -> Optional[str]:
        """Simple language detection based on common words."""
        
        if len(content) < 50:
            return None
        
        # Count matches for each language
        language_scores = {}
        sample = content[:1000].lower()  # Use first 1000 chars for detection
        
        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(sample))
            language_scores[lang] = matches
        
        # Return language with highest score if above threshold
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            if language_scores[best_lang] >= 3:  # Minimum 3 matches
                return best_lang
        
        return 'unknown'

    def _calculate_quality_score(self, content: str, original_length: int) -> float:
        """Calculate content quality score (0-1)."""
        
        if not content or original_length == 0:
            return 0.0
        
        score = 1.0
        
        # Penalize excessive length reduction
        length_ratio = len(content) / original_length
        if length_ratio < 0.3:  # Lost more than 70% of content
            score *= 0.5
        elif length_ratio < 0.5:  # Lost more than 50% of content
            score *= 0.7
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length < 3:  # Very short sentences
            score *= 0.8
        elif avg_sentence_length > 50:  # Very long sentences
            score *= 0.9
        
        # Check for reasonable word distribution
        words = content.split()
        if len(words) < 10:
            score *= 0.6
        
        # Check for excessive repetition
        unique_words = set(words)
        if len(unique_words) / max(len(words), 1) < 0.3:  # Less than 30% unique words
            score *= 0.7
        
        # Check for proper capitalization
        capitalized_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if capitalized_sentences / max(len(sentences), 1) < 0.5:
            score *= 0.9
        
        return min(max(score, 0.0), 1.0)

    def get_cleaning_stats(self, preprocessed: PreprocessedContent) -> Dict[str, Any]:
        """Get detailed cleaning statistics."""
        
        return {
            'original_length': preprocessed.original_length,
            'processed_length': preprocessed.processed_length,
            'reduction_ratio': 1 - (preprocessed.processed_length / max(preprocessed.original_length, 1)),
            'cleaning_steps': preprocessed.cleaning_steps,
            'quality_score': preprocessed.quality_score,
            'language': preprocessed.language,
            'encoding_issues_count': len(preprocessed.encoding_issues),
            'structure_preserved': bool(preprocessed.preserved_structure),
            'processing_timestamp': datetime.now().isoformat()
        }
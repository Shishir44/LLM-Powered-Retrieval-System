from typing import List, Dict, Any, Optional, Tuple
import re
import spacy
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

@dataclass
class DocumentChunk:
    """Enhanced document chunk with metadata and structure awareness."""
    id: str
    content: str
    title: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int
    structure_type: str  # paragraph, heading, list_item, table, code
    semantic_context: str  # Context from surrounding sections
    metadata: Dict[str, Any]
    parent_section: Optional[str] = None
    hierarchical_level: int = 0
    overlapping_windows: List[str] = None

class AdvancedDocumentChunker:
    """Advanced document chunking with semantic and structure awareness."""
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2",
                 max_chunk_size: int = 512,
                 overlap_size: int = 64,
                 similarity_threshold: float = 0.8):
        
        # Initialize components
        self.sentence_transformer = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        
        # Load spacy model for sentence boundary detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Using simple sentence splitting.")
            self.nlp = None
        
        # Fallback text splitter
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap_size,
            length_function=len
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def chunk_document(self, 
                      document: Dict[str, Any],
                      chunking_strategy: str = "semantic_structure") -> List[DocumentChunk]:
        """Main chunking method with multiple strategies."""
        
        content = document.get("content", "")
        title = document.get("title", "")
        doc_id = document.get("id", "unknown")
        
        if not content.strip():
            return []
        
        if chunking_strategy == "semantic_structure":
            return self._semantic_structure_chunk(content, title, doc_id, document.get("metadata", {}))
        elif chunking_strategy == "semantic":
            return self._semantic_chunk(content, title, doc_id, document.get("metadata", {}))
        elif chunking_strategy == "structure_aware":
            return self._structure_aware_chunk(content, title, doc_id, document.get("metadata", {}))
        else:
            return self._fallback_chunk(content, title, doc_id, document.get("metadata", {}))
    
    def _semantic_structure_chunk(self, 
                                 content: str, 
                                 title: str, 
                                 doc_id: str,
                                 metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Combine semantic and structure-aware chunking."""
        
        # First, identify document structure
        structured_sections = self._identify_document_structure(content)
        
        chunks = []
        chunk_id = 0
        
        for section in structured_sections:
            # Apply semantic chunking within each section
            section_chunks = self._semantic_chunk_section(
                section, title, doc_id, metadata, chunk_id
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        # Add overlapping windows for better context preservation
        chunks = self._add_overlapping_windows(chunks, content)
        
        return chunks
    
    def _identify_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Identify structural elements in the document."""
        
        sections = []
        lines = content.split('\n')
        current_section = {"type": "paragraph", "content": "", "level": 0, "title": ""}
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Detect headings (markdown style)
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(heading_match.group(1))
                current_section = {
                    "type": "heading",
                    "content": line + "\n",
                    "level": level,
                    "title": heading_match.group(2),
                    "start_line": line_num
                }
                continue
            
            # Detect lists
            if re.match(r'^[\*\-\+]\s+', line) or re.match(r'^\d+\.\s+', line):
                if current_section["type"] != "list":
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {
                        "type": "list",
                        "content": line + "\n",
                        "level": 0,
                        "title": "",
                        "start_line": line_num
                    }
                else:
                    current_section["content"] += line + "\n"
                continue
            
            # Detect code blocks
            if line.startswith("```") or line.startswith("    "):
                if current_section["type"] != "code":
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {
                        "type": "code",
                        "content": line + "\n",
                        "level": 0,
                        "title": "",
                        "start_line": line_num
                    }
                else:
                    current_section["content"] += line + "\n"
                continue
            
            # Regular paragraph content
            if current_section["type"] not in ["paragraph", "heading"]:
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {
                    "type": "paragraph",
                    "content": line + "\n",
                    "level": 0,
                    "title": "",
                    "start_line": line_num
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _semantic_chunk_section(self, 
                               section: Dict[str, Any],
                               title: str,
                               doc_id: str,
                               metadata: Dict[str, Any],
                               start_chunk_id: int) -> List[DocumentChunk]:
        """Apply semantic chunking within a document section."""
        
        content = section["content"]
        section_type = section["type"]
        section_title = section.get("title", "")
        
        # Handle different section types
        if section_type == "code":
            # Keep code blocks together
            return [self._create_chunk(
                content, title, doc_id, start_chunk_id, 0, len(content),
                section_type, section_title, metadata, section.get("level", 0)
            )]
        
        if section_type == "list":
            # Split lists by items but try to keep related items together
            return self._chunk_list_content(
                content, title, doc_id, start_chunk_id, section_title, metadata, section.get("level", 0)
            )
        
        # For paragraphs and headings, use semantic sentence-based chunking
        return self._semantic_sentence_chunk(
            content, title, doc_id, start_chunk_id, section_type, section_title, metadata, section.get("level", 0)
        )
    
    def _semantic_sentence_chunk(self, 
                                content: str,
                                title: str,
                                doc_id: str,
                                start_chunk_id: int,
                                section_type: str,
                                section_title: str,
                                metadata: Dict[str, Any],
                                level: int) -> List[DocumentChunk]:
        """Chunk content based on semantic sentence boundaries."""
        
        if self.nlp:
            doc = self.nlp(content)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback sentence splitting
            sentences = self._simple_sentence_split(content)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        chunk_id = start_chunk_id
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk.strip(), title, doc_id, chunk_id,
                    0, len(current_chunk), section_type, section_title, metadata, level
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
                chunk_id += 1
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), title, doc_id, chunk_id,
                0, len(current_chunk), section_type, section_title, metadata, level
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_list_content(self, 
                           content: str,
                           title: str,
                           doc_id: str,
                           start_chunk_id: int,
                           section_title: str,
                           metadata: Dict[str, Any],
                           level: int) -> List[DocumentChunk]:
        """Chunk list content while preserving list structure."""
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        chunks = []
        current_chunk = ""
        chunk_id = start_chunk_id
        
        for line in lines:
            potential_chunk = current_chunk + "\n" + line if current_chunk else line
            
            if len(potential_chunk.split()) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_chunk(
                    current_chunk.strip(), title, doc_id, chunk_id,
                    0, len(current_chunk), "list", section_title, metadata, level
                )
                chunks.append(chunk)
                current_chunk = line
                chunk_id += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), title, doc_id, chunk_id,
                0, len(current_chunk), "list", section_title, metadata, level
            )
            chunks.append(chunk)
        
        return chunks
    
    def _simple_sentence_split(self, content: str) -> List[str]:
        """Simple sentence splitting fallback."""
        sentences = re.split(r'[.!?]+\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, 
                     content: str,
                     title: str,
                     doc_id: str,
                     chunk_index: int,
                     start_char: int,
                     end_char: int,
                     structure_type: str,
                     parent_section: str,
                     metadata: Dict[str, Any],
                     hierarchical_level: int) -> DocumentChunk:
        """Create a DocumentChunk with full metadata."""
        
        words = content.split()
        sentences = self._simple_sentence_split(content)
        
        # Create semantic context (surrounding information)
        semantic_context = f"{parent_section}: {content[:100]}..." if parent_section else content[:100] + "..."
        
        return DocumentChunk(
            id=f"{doc_id}_chunk_{chunk_index}",
            content=content,
            title=title,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            word_count=len(words),
            sentence_count=len(sentences),
            structure_type=structure_type,
            semantic_context=semantic_context,
            metadata={
                **metadata,
                "chunk_type": structure_type,
                "parent_section": parent_section,
                "word_density": len(words) / max(len(sentences), 1)
            },
            parent_section=parent_section,
            hierarchical_level=hierarchical_level,
            overlapping_windows=[]
        )
    
    def _add_overlapping_windows(self, chunks: List[DocumentChunk], full_content: str) -> List[DocumentChunk]:
        """Add overlapping windows to chunks for better context preservation."""
        
        for i, chunk in enumerate(chunks):
            windows = []
            
            # Previous chunk overlap
            if i > 0:
                prev_chunk = chunks[i-1]
                prev_words = prev_chunk.content.split()[-self.overlap_size//2:]
                curr_words = chunk.content.split()[:self.overlap_size//2]
                windows.append(" ".join(prev_words + curr_words))
            
            # Next chunk overlap
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                curr_words = chunk.content.split()[-self.overlap_size//2:]
                next_words = next_chunk.content.split()[:self.overlap_size//2]
                windows.append(" ".join(curr_words + next_words))
            
            chunk.overlapping_windows = windows
        
        return chunks
    
    def _semantic_chunk(self, 
                       content: str,
                       title: str,
                       doc_id: str,
                       metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Pure semantic chunking based on sentence similarity."""
        
        if self.nlp:
            doc = self.nlp(content)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            sentences = self._simple_sentence_split(content)
        
        if not sentences:
            return []
        
        # Generate embeddings for sentences
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        
        # Group semantically similar sentences
        chunks = []
        current_chunk_sentences = []
        current_embedding_sum = np.zeros(sentence_embeddings.shape[1])
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                current_embedding_sum = embedding
                continue
            
            # Calculate similarity with current chunk
            current_avg_embedding = current_embedding_sum / len(current_chunk_sentences)
            similarity = np.dot(current_avg_embedding, embedding) / (
                np.linalg.norm(current_avg_embedding) * np.linalg.norm(embedding)
            )
            
            # Check if we should start a new chunk
            potential_chunk = " ".join(current_chunk_sentences + [sentence])
            should_split = (
                similarity < self.similarity_threshold or 
                len(potential_chunk.split()) > self.max_chunk_size
            )
            
            if should_split and current_chunk_sentences:
                # Create chunk
                chunk_content = " ".join(current_chunk_sentences)
                chunk = self._create_chunk(
                    chunk_content, title, doc_id, len(chunks),
                    0, len(chunk_content), "paragraph", "", metadata, 0
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_embedding_sum = embedding
            else:
                current_chunk_sentences.append(sentence)
                current_embedding_sum += embedding
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunk = self._create_chunk(
                chunk_content, title, doc_id, len(chunks),
                0, len(chunk_content), "paragraph", "", metadata, 0
            )
            chunks.append(chunk)
        
        return chunks
    
    def _structure_aware_chunk(self, 
                              content: str,
                              title: str,
                              doc_id: str,
                              metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Structure-aware chunking preserving document hierarchy."""
        
        sections = self._identify_document_structure(content)
        chunks = []
        
        for i, section in enumerate(sections):
            chunk = self._create_chunk(
                section["content"], title, doc_id, i,
                0, len(section["content"]), section["type"],
                section.get("title", ""), metadata, section.get("level", 0)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunk(self, 
                       content: str,
                       title: str,
                       doc_id: str,
                       metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Fallback to simple recursive chunking."""
        
        text_chunks = self.fallback_splitter.split_text(content)
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_chunk(
                chunk_text, title, doc_id, i,
                0, len(chunk_text), "paragraph", "", metadata, 0
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunking_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        
        if not chunks:
            return {"message": "No chunks provided"}
        
        structure_types = {}
        word_counts = []
        sentence_counts = []
        hierarchical_levels = []
        
        for chunk in chunks:
            structure_types[chunk.structure_type] = structure_types.get(chunk.structure_type, 0) + 1
            word_counts.append(chunk.word_count)
            sentence_counts.append(chunk.sentence_count)
            hierarchical_levels.append(chunk.hierarchical_level)
        
        return {
            "total_chunks": len(chunks),
            "structure_type_distribution": structure_types,
            "word_count_stats": {
                "min": min(word_counts),
                "max": max(word_counts),
                "avg": sum(word_counts) / len(word_counts),
                "total": sum(word_counts)
            },
            "sentence_count_stats": {
                "min": min(sentence_counts),
                "max": max(sentence_counts),
                "avg": sum(sentence_counts) / len(sentence_counts)
            },
            "hierarchical_levels": list(set(hierarchical_levels)),
            "avg_hierarchical_level": sum(hierarchical_levels) / len(hierarchical_levels)
        }
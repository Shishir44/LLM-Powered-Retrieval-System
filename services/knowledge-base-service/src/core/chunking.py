from typing import List, Dict, Any
import re
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentChunk:
    """Represents a document chunk."""
    content: str
    chunk_id: int
    document_id: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    
    @property
    def hash(self) -> str:
        """Generate content hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()

class DocumentChunker:
    """Advanced document chunking with multiple strategies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Text splitting patterns
        self.separators = [
            r'\n\n+',  # Multiple newlines
            r'\n',     # Single newlines
            r'\. ',    # Sentences
            r'[!?] ',  # Exclamation/question marks
            r'; ',     # Semicolons
            r', ',     # Commas
            r' ',      # Spaces
        ]
    
    def chunk_document(self, content: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split document into chunks with metadata."""
        if not content or not content.strip():
            return []
        
        # Clean and normalize text
        content = self._preprocess_text(content)
        
        # Generate chunks using recursive splitting
        chunks = self._recursive_split(content)
        
        # Create DocumentChunk objects
        document_chunks = []
        document_id = metadata.get("document_id", "unknown") if metadata else "unknown"
        
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "character_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                **(metadata or {})
            }
            
            document_chunks.append(DocumentChunk(
                content=chunk_text,
                chunk_id=i,
                document_id=document_id,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=chunk_metadata
            ))
        
        return document_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove excessive newlines but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _recursive_split(self, text: str, start_offset: int = 0) -> List[tuple]:
        """Recursively split text using different separators."""
        if len(text) <= self.chunk_size:
            return [(text, start_offset, start_offset + len(text))]
        
        chunks = []
        
        for separator_pattern in self.separators:
            parts = re.split(f'({separator_pattern})', text)
            if len(parts) <= 1:
                continue
                
            current_chunk = ""
            current_start = start_offset
            pos = start_offset
            
            for part in parts:
                if re.match(separator_pattern, part):
                    current_chunk += part
                    pos += len(part)
                    continue
                    
                if len(current_chunk + part) <= self.chunk_size:
                    current_chunk += part
                    pos += len(part)
                else:
                    # Finalize current chunk if it has content
                    if current_chunk.strip():
                        chunks.append((current_chunk.strip(), current_start, pos - len(part)))
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + part
                        current_start = pos - len(part) - len(overlap_text)
                    else:
                        current_chunk = part
                        current_start = pos - len(part)
                    
                    pos += len(part)
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunks.append((current_chunk.strip(), current_start, pos))
            
            return chunks
        
        # Fallback: split by character count if no separator works
        return self._split_by_chars(text, start_offset)
    
    def _split_by_chars(self, text: str, start_offset: int = 0) -> List[tuple]:
        """Split text by character count as fallback."""
        chunks = []
        pos = 0
        
        while pos < len(text):
            end_pos = min(pos + self.chunk_size, len(text))
            chunk = text[pos:end_pos]
            
            chunks.append((chunk, start_offset + pos, start_offset + end_pos))
            
            # Move position accounting for overlap
            pos = max(pos + self.chunk_size - self.chunk_overlap, pos + 1)
        
        return chunks
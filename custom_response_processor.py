from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class NaturalLanguagePostProcessor(BaseNodePostprocessor):
    """Post-processor for converting responses to natural language format."""
    
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Required implementation of abstract method to process nodes.
        Simply returns nodes as is since we're focusing on response processing.
        """
        return nodes

    def postprocess_response(self, response: str) -> str:
        """
        Process the response string to make it more natural.
        
        Args:
            response (str): The original response string
            
        Returns:
            str: Processed response in natural language format
        """
        if not response:
            return response
            
        # Remove any bullet points or numbered lists
        response_lines = response.split('\n')
        processed_lines = []
        current_paragraph = []
        
        for line in response_lines:
            # Skip empty lines
            if not line.strip():
                if current_paragraph:
                    processed_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
                
            # Remove bullet points, dashes, asterisks and numbers
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                # For lines that look like headings (all caps or ending with ':')
                if cleaned_line.isupper() or cleaned_line.endswith(':'):
                    if current_paragraph:
                        processed_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    processed_lines.append(cleaned_line)
                else:
                    current_paragraph.append(cleaned_line)
        
        # Add any remaining paragraph
        if current_paragraph:
            processed_lines.append(' '.join(current_paragraph))
        
        # Join paragraphs with double newlines
        response_str = '\n\n'.join(processed_lines)
        
        # Clean up multiple spaces
        response_str = ' '.join(response_str.split())
        
        # Add Nepali-specific processing
        response_str = self._process_nepali_text(response_str)
        
        return response_str
    
    def _process_nepali_text(self, text: str) -> str:
        """
        Apply Nepali-specific text processing rules.
        """
        # Common Nepali bullet point markers and their replacements
        nepali_markers = {
            '•': '',
            '○': '',
            '◦': '',
            '๏': '',
            '॰': '',
        }
        
        for marker, replacement in nepali_markers.items():
            text = text.replace(marker, replacement)
        
        # Add proper spacing after Devanagari danda (।)
        text = text.replace('।', '। ')
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text
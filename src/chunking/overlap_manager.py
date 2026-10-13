"""
Overlap manager for chunk boundaries.

Handles chunk overlap calculations and boundary detection
to ensure context is preserved across chunk boundaries.

Usage:
    from src.chunking.overlap_manager import OverlapManager
    
    manager = OverlapManager(overlap_size=50)
    chunks_with_overlap = manager.add_overlap(chunks)
"""

from typing import List, Dict, Any, Optional


class OverlapManager:
    """Manages overlap between adjacent chunks."""
    
    def __init__(
        self,
        overlap_size: int = 50,
        overlap_type: str = "tokens",  # "tokens" or "characters"
    ):
        """
        Initialize the overlap manager.
        
        Args:
            overlap_size: Size of overlap in tokens or characters.
            overlap_type: Unit of overlap measurement.
        """
        self.overlap_size = overlap_size
        self.overlap_type = overlap_type
    
    def _get_overlap_text(
        self,
        text: str,
        from_end: bool = True,
    ) -> str:
        """
        Extract overlap text from a chunk.
        
        Args:
            text: Source text.
            from_end: If True, extract from end; else from start.
            
        Returns:
            str: Overlap text.
        """
        if self.overlap_type == "tokens":
            words = text.split()
            if from_end:
                overlap_words = words[-self.overlap_size:]
            else:
                overlap_words = words[:self.overlap_size]
            return " ".join(overlap_words)
        else:
            if from_end:
                return text[-self.overlap_size:]
            else:
                return text[:self.overlap_size]
    
    def add_overlap(
        self,
        chunks: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Add overlap context to chunks.
        
        Args:
            chunks: List of chunk dictionaries.
            text_key: Key for the text field in chunks.
            
        Returns:
            list: Chunks with overlap context added.
        """
        if len(chunks) <= 1:
            return chunks
        
        result = []
        
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            
            # Add previous context
            if i > 0:
                prev_overlap = self._get_overlap_text(
                    chunks[i - 1][text_key],
                    from_end=True,
                )
                new_chunk["previous_context"] = prev_overlap
            else:
                new_chunk["previous_context"] = ""
            
            # Add next context
            if i < len(chunks) - 1:
                next_overlap = self._get_overlap_text(
                    chunks[i + 1][text_key],
                    from_end=False,
                )
                new_chunk["next_context"] = next_overlap
            else:
                new_chunk["next_context"] = ""
            
            result.append(new_chunk)
        
        return result
    
    def merge_overlapping_chunks(
        self,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks that have high overlap similarity.
        
        Args:
            chunks: List of chunk dictionaries.
            similarity_threshold: Threshold for merging.
            
        Returns:
            list: Merged chunks.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = [chunks[0]]
        
        for chunk in chunks[1:]:
            # Simple overlap check (can be enhanced with embeddings)
            last_text = merged[-1].get("text", "")
            curr_text = chunk.get("text", "")
            
            # Check if significant portion overlaps
            overlap_start = self._get_overlap_text(last_text, from_end=True)
            if overlap_start in curr_text[:len(overlap_start) * 2]:
                # Merge chunks
                merged[-1]["text"] = last_text + "\n" + curr_text[len(overlap_start):]
            else:
                merged.append(chunk)
        
        return merged


if __name__ == "__main__":
    # Demo usage
    sample_chunks = [
        {"chunk_id": "1", "text": "This is the first chunk with some content."},
        {"chunk_id": "2", "text": "This is the second chunk continuing the story."},
        {"chunk_id": "3", "text": "This is the third chunk with the conclusion."},
    ]
    
    manager = OverlapManager(overlap_size=5)
    chunks_with_overlap = manager.add_overlap(sample_chunks)
    
    for chunk in chunks_with_overlap:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Previous: {chunk.get('previous_context', 'N/A')}")
        print(f"  Text: {chunk['text']}")
        print(f"  Next: {chunk.get('next_context', 'N/A')}")

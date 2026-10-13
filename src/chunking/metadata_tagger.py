"""
Metadata tagger for document chunks.

Adds rich metadata to chunks including content type, section info,
and semantic tags for improved retrieval.

Usage:
    from src.chunking.metadata_tagger import MetadataTagger
    
    tagger = MetadataTagger()
    tagged_chunks = tagger.tag_chunks(chunks)
"""

import re
from typing import List, Dict, Any, Optional


class MetadataTagger:
    """Tags chunks with rich metadata for retrieval."""
    
    # Patterns for content type detection
    PATTERNS = {
        "code": r"```[\s\S]*?```|def\s+\w+|class\s+\w+|function\s+\w+",
        "list": r"^\s*[-*•]\s+|\d+\.\s+",
        "table": r"\|.*\|.*\||<table|<tr>|<td>",
        "heading": r"^#{1,6}\s+|^[A-Z][^.!?]*:$",
        "equation": r"\$.*\$|\\begin\{equation\}|\\frac|\\sum",
    }
    
    def __init__(
        self,
        detect_content_type: bool = True,
        extract_entities: bool = True,
        add_statistics: bool = True,
    ):
        """
        Initialize the metadata tagger.
        
        Args:
            detect_content_type: Detect content type from text.
            extract_entities: Extract named entities.
            add_statistics: Add text statistics.
        """
        self.detect_content_type = detect_content_type
        self.extract_entities = extract_entities
        self.add_statistics = add_statistics
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type from text patterns."""
        for content_type, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.MULTILINE):
                return content_type
        return "text"
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction (can be enhanced with TF-IDF or KeyBERT)
        words = re.findall(r'\b[A-Za-z][a-z]{3,}\b', text)
        
        # Count word frequency
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _compute_statistics(self, text: str) -> Dict[str, Any]:
        """Compute text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        }
    
    def tag_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add metadata tags to a single chunk.
        
        Args:
            chunk: Chunk dictionary.
            
        Returns:
            dict: Chunk with added metadata.
        """
        tagged = chunk.copy()
        text = chunk.get("text", "")
        
        # Detect content type if not already set
        if self.detect_content_type and "content_type" not in tagged:
            tagged["content_type"] = self._detect_content_type(text)
        
        # Extract keywords
        if self.extract_entities:
            tagged["keywords"] = self._extract_keywords(text)
        
        # Add statistics
        if self.add_statistics:
            tagged["statistics"] = self._compute_statistics(text)
        
        return tagged
    
    def tag_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add metadata tags to multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries.
            
        Returns:
            list: Chunks with added metadata.
        """
        return [self.tag_chunk(chunk) for chunk in chunks]


def tag_chunks(
    chunks: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience function to tag chunks with metadata.
    
    Args:
        chunks: List of chunk dictionaries.
        **kwargs: Additional arguments for MetadataTagger.
        
    Returns:
        list: Tagged chunks.
    """
    tagger = MetadataTagger(**kwargs)
    return tagger.tag_chunks(chunks)


if __name__ == "__main__":
    # Demo usage
    sample_chunks = [
        {
            "chunk_id": "1",
            "text": "The maintenance schedule requires checking oil levels every 1000 hours.",
        },
        {
            "chunk_id": "2",
            "text": "```python\ndef calculate_pressure(volume, temp):\n    return volume * temp\n```",
        },
        {
            "chunk_id": "3",
            "text": "- Check valve A\n- Inspect gaskets\n- Replace filters",
        },
    ]
    
    tagger = MetadataTagger()
    tagged = tagger.tag_chunks(sample_chunks)
    
    for chunk in tagged:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Content Type: {chunk.get('content_type')}")
        print(f"  Keywords: {chunk.get('keywords', [])[:5]}")
        print(f"  Statistics: {chunk.get('statistics')}")

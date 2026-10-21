"""
Prompt builder for RAG pipeline.

Builds prompts from retrieval results and user queries.

Usage:
    from src.generation.prompt_builder import PromptBuilder
    
    builder = PromptBuilder()
    prompt = builder.build(query, retrieved_chunks)
"""

from typing import List, Dict, Any, Optional

from src.config import load_llm_config


# Default prompt templates
DEFAULT_SYSTEM_TEMPLATE = """You are a technical assistant that answers questions using only the provided context.
Answer the question directly in the first sentence, as a full sentence that names the product or component
it is about (not a bare value such as "Up to 217 psig."). Then add only the supporting facts that answer it:
model names, numeric specifications, units, and ranges exactly as written in the context.
If a table is split across lines (model name then numbers), reconstruct the row using nearby column headers.
Do not add background, general knowledge, or facts about other products that the question did not ask about.
If the context is incomplete, answer with what is present and state what is missing."""

DEFAULT_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


class PromptBuilder:
    """Builds prompts for RAG generation."""
    
    def __init__(
        self,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
        chunk_separator: Optional[str] = None,
        max_context_length: Optional[int] = None,
        include_sources: Optional[bool] = None,
    ):
        """
        Initialize the prompt builder. Unset settings come from the
        `generation` section of llm_config.yaml.

        Args:
            system_template: System prompt template.
            user_template: User prompt template with {context} and {question}.
            chunk_separator: Separator between chunks.
            max_context_length: Maximum context length in characters.
            include_sources: Include source citations in context.
        """
        cfg = load_llm_config().get("generation", {})
        self.system_template = system_template or DEFAULT_SYSTEM_TEMPLATE
        self.user_template = user_template or DEFAULT_USER_TEMPLATE
        self.chunk_separator = chunk_separator or cfg.get("chunk_separator", "\n\n---\n\n")
        self.max_context_length = max_context_length or cfg.get("max_context_length", 12000)
        self.include_sources = (
            cfg.get("include_sources", True) if include_sources is None else include_sources
        )
    
    def _format_chunk(
        self,
        chunk: Dict[str, Any],
        index: int,
    ) -> str:
        """Format a single chunk for the context."""
        text = chunk.get("document", chunk.get("text", ""))
        
        if self.include_sources:
            metadata = chunk.get("metadata", {})
            source = metadata.get("document_id", "Unknown")
            if metadata.get("document_title"):
                source = f"{source} ({metadata['document_title']})"
            pages = ""
            if metadata.get("page_start"):
                pages = f", pages {metadata.get('page_start')}-{metadata.get('page_end', metadata.get('page_start'))}"
            
            return f"[Source {index}: {source}{pages}]\n{text}"
        
        return text
    
    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> tuple:
        """Build context string from chunks. Returns (context, chunks that fit)."""
        formatted_chunks = []
        used_chunks = []
        total_length = 0

        for i, chunk in enumerate(chunks, start=1):
            formatted = self._format_chunk(chunk, i)

            # Check length limit
            if total_length + len(formatted) > self.max_context_length:
                break

            formatted_chunks.append(formatted)
            used_chunks.append(chunk)
            total_length += len(formatted) + len(self.chunk_separator)

        return self.chunk_separator.join(formatted_chunks), used_chunks
    
    def build(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Build a prompt from query and retrieved chunks.
        
        Args:
            query: User query.
            chunks: Retrieved chunks.
            
        Returns:
            dict: Contains 'system' and 'user' prompt strings.
        """
        context, used_chunks = self._build_context(chunks)

        user_prompt = self.user_template.format(
            context=context,
            question=query,
        )

        return {
            "system": self.system_template,
            "user": user_prompt,
            "context": context,
            "chunks_used": len(used_chunks),
            "used_chunks": used_chunks,
        }
    
    def build_messages(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build chat messages format for API calls.
        
        Args:
            query: User query.
            chunks: Retrieved chunks.
            
        Returns:
            list: Messages in chat format.
        """
        prompt = self.build(query, chunks)
        
        return [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]


def build_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, str]:
    """
    Convenience function to build a prompt.
    
    Args:
        query: User query.
        chunks: Retrieved chunks.
        **kwargs: Additional arguments for PromptBuilder.
        
    Returns:
        dict: Prompt components.
    """
    builder = PromptBuilder(**kwargs)
    return builder.build(query, chunks)


if __name__ == "__main__":
    # Demo usage
    sample_chunks = [
        {
            "document": "The maintenance schedule requires checking oil levels every 1000 hours.",
            "metadata": {"document_id": "MNT-001", "page_start": 5, "page_end": 5},
        },
        {
            "document": "All filters should be replaced during regular maintenance intervals.",
            "metadata": {"document_id": "MNT-001", "page_start": 7, "page_end": 8},
        },
    ]
    
    builder = PromptBuilder()
    prompt = builder.build("What is the maintenance schedule?", sample_chunks)
    
    print("System prompt:")
    print(prompt["system"])
    print("\n" + "=" * 60 + "\n")
    print("User prompt:")
    print(prompt["user"])

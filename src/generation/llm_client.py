"""
LLM client for RAG pipeline.

Calls Groq to generate responses.

Usage:
    from src.generation.llm_client import LLMClient

    client = LLMClient()
    response = client.generate(messages)
"""

import os
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Google Gemini (commented out) ---
# try:
#     from google import genai
#     from google.genai import types
#     HAS_GEMINI = True
# except ImportError:
#     HAS_GEMINI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


from src.config import load_llm_config


# DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_MODEL = "openai/gpt-oss-20b"
# llama-3.3-70b-versatile was retired from Groq; every judge call returned 404.
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
REASONING_MODEL_MARKERS = ("gpt-oss",)


def _llm_setting(key: str, default: Any) -> Any:
    value = load_llm_config().get("llm", {}).get(key)
    return default if value is None else value


# def resolve_google_api_key(api_key: Optional[str] = None) -> Optional[str]:
#     """Return a real Gemini key, ignoring empty/.env placeholders."""
#     key = api_key or os.getenv("GOOGLE_API_KEY")
#     if not key:
#         return None
#     key = key.strip().strip('"').strip("'")
#     if not key or key.lower().startswith("your-"):
#         return None
#     return key


def resolve_groq_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Return a real Groq key, ignoring empty/.env placeholders."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return None
    key = key.strip().strip('"').strip("'")
    if not key or key.lower().startswith("your-"):
        return None
    return key


def resolve_model_name(model_name: Optional[str] = None) -> str:
    """Return the model id from argument, LLM_MODEL, llm_config.yaml, or the default."""
    name = (
        model_name
        or os.getenv("LLM_MODEL")
        or _llm_setting("model_name", None)
        or DEFAULT_MODEL
    )
    return name.strip().strip('"').strip("'")


def resolve_judge_model_name(model_name: Optional[str] = None) -> str:
    """Return the judge model id from argument, LLM_JUDGE_MODEL, llm_config.yaml, or the default."""
    name = (
        model_name
        or os.getenv("LLM_JUDGE_MODEL")
        or _llm_setting("judge_model_name", None)
        or DEFAULT_JUDGE_MODEL
    )
    return name.strip().strip('"').strip("'")


def is_reasoning_model(model_name: str) -> bool:
    """True when the model spends completion tokens on hidden reasoning."""
    lowered = model_name.lower()
    return any(marker in lowered for marker in REASONING_MODEL_MARKERS)


class LLMClient:
    """Client for Groq API calls."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        # if not HAS_GEMINI:
        #     raise ImportError(
        #         "google-genai package not installed. "
        #         "Install with: pip install google-genai"
        #     )
        if not HAS_GROQ:
            raise ImportError(
                "groq package not installed. "
                "Install with: pip install groq"
            )

        self.model_name = resolve_model_name(model_name)
        self.temperature = (
            _llm_setting("temperature", DEFAULT_TEMPERATURE) if temperature is None else temperature
        )
        self.max_tokens = max_tokens or _llm_setting("max_tokens", DEFAULT_MAX_TOKENS)

        # api_key = resolve_google_api_key(api_key)
        # if not api_key:
        #     raise ValueError(
        #         "Google API key required. "
        #         "Set GOOGLE_API_KEY in .env (https://aistudio.google.com/apikey)."
        #     )
        #
        # self.client = genai.Client(api_key=api_key)

        api_key = resolve_groq_api_key(api_key)
        if not api_key:
            raise ValueError(
                "Groq API key required. "
                "Set GROQ_API_KEY in .env (https://console.groq.com/keys)."
            )

        # The SDK retries 429/5xx with exponential backoff; free-tier rate limits need more than its default 2.
        self.client = Groq(
            api_key=api_key,
            max_retries=_llm_setting("max_retries", 5),
            timeout=_llm_setting("timeout", 60),
        )

    def check_available(self) -> None:
        """Raise ValueError if the configured model is not served for this API key."""
        available = {m.id for m in self.client.models.list().data}
        if self.model_name not in available:
            chat_models = sorted(m for m in available if "whisper" not in m and "guard" not in m)
            raise ValueError(
                f"Model '{self.model_name}' is not available on Groq for this key. "
                f"Available chat models: {', '.join(chat_models)}"
            )

    def _fold_system_into_user(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """gpt-oss models score better when instructions live in the user turn."""
        system_parts = []
        rest = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                rest.append(dict(msg))
        if not system_parts or not rest:
            return messages
        prefix = "\n\n".join(system_parts)
        for msg in rest:
            if msg["role"] == "user":
                msg["content"] = f"{prefix}\n\n{msg['content']}"
                break
        return rest

    # def _split_messages(self, messages: List[Dict[str, str]]) -> tuple:
    #     """Split chat messages into Gemini system instruction and contents."""
    #     system_prompt = None
    #     parts = []
    #
    #     for msg in messages:
    #         role = msg["role"]
    #         content = msg["content"]
    #         if role == "system":
    #             system_prompt = content
    #         elif role == "assistant":
    #             parts.append(f"Assistant: {content}")
    #         else:
    #             parts.append(content)
    #
    #     return system_prompt, "\n\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """Generate a response from Groq. json_mode forces a JSON object reply."""
        # system_prompt, contents = self._split_messages(messages)
        #
        # config = types.GenerateContentConfig(
        #     temperature=self.temperature if temperature is None else temperature,
        #     max_output_tokens=max_tokens or self.max_tokens,
        #     system_instruction=system_prompt,
        # )
        #
        # response = self.client.models.generate_content(
        #     model=self.model_name,
        #     contents=contents,
        #     config=config,
        # )
        #
        # usage = {}
        # if getattr(response, "usage_metadata", None):
        #     usage = {
        #         "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
        #         "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
        #         "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
        #     }
        #
        # return {
        #     "content": response.text or "",
        #     "model": self.model_name,
        #     "usage": usage,
        #     "finish_reason": "stop",
        # }

        create_kwargs = self._request_kwargs(messages, temperature, max_tokens)
        if json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}

        response = self._create(create_kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        if not content and choice.finish_reason == "length" and "max_completion_tokens" in create_kwargs:
            # Hidden reasoning used the whole budget. Retry once with double the room
            # instead of returning the reasoning trace as if it were the answer.
            create_kwargs["max_completion_tokens"] *= 2
            response = self._create(create_kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""

        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }

        return {
            "content": content,
            "model": self.model_name,
            "usage": usage,
            "finish_reason": choice.finish_reason or "stop",
        }

    def _request_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Chat-completion arguments shared by generate() and stream()."""
        max_out = max_tokens or self.max_tokens
        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
        }

        if is_reasoning_model(self.model_name):
            # Reasoning tokens share this budget. max_tokens=10 yields empty content.
            create_kwargs["max_completion_tokens"] = max(max_out, 2048)
            create_kwargs["reasoning_effort"] = os.getenv(
                "GROQ_REASONING_EFFORT", "low"
            )
            create_kwargs["include_reasoning"] = False
            messages = self._fold_system_into_user(messages)
        else:
            create_kwargs["max_tokens"] = max_out

        create_kwargs["messages"] = messages
        return create_kwargs

    def _create(self, create_kwargs: Dict[str, Any]):
        try:
            return self.client.chat.completions.create(**create_kwargs)
        except TypeError:
            create_kwargs.pop("include_reasoning", None)
            create_kwargs.pop("reasoning_effort", None)
            return self.client.chat.completions.create(**create_kwargs)

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate plain text from a single prompt."""
        response = self.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        return response["content"].strip()

    def generate_with_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response using retrieved context."""
        from src.generation.prompt_builder import PromptBuilder

        builder = PromptBuilder(system_template=system_prompt)
        prompt = builder.build(query, chunks)
        response = self.generate([
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ])
        # Only chunks that fit in max_context_length reached the model.
        # [Source n] in the answer refers to the n-th entry here.
        response["sources"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("metadata", {}).get("document_id"),
                "document_title": chunk.get("metadata", {}).get("document_title"),
                "page_start": chunk.get("metadata", {}).get("page_start"),
                "page_end": chunk.get("metadata", {}).get("page_end"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
            for chunk in prompt["used_chunks"]
        ]
        return response

    def stream(self, messages: List[Dict[str, str]]):
        """Stream response chunks from Groq."""
        # system_prompt, contents = self._split_messages(messages)
        # config = types.GenerateContentConfig(
        #     temperature=self.temperature,
        #     max_output_tokens=self.max_tokens,
        #     system_instruction=system_prompt,
        # )
        #
        # stream = self.client.models.generate_content_stream(
        #     model=self.model_name,
        #     contents=contents,
        #     config=config,
        # )
        # for chunk in stream:
        #     if chunk.text:
        #         yield chunk.text

        create_kwargs = self._request_kwargs(messages)
        create_kwargs["stream"] = True
        stream = self._create(create_kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def create_judge_client(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "LLMClient":
    """LLM client for evaluation judges (not the RAG generator)."""
    return LLMClient(
        model_name=resolve_judge_model_name(model_name),
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to generate a RAG response."""
    client = LLMClient(model_name=model_name, **kwargs)
    return client.generate_with_context(query, chunks)


if __name__ == "__main__":
    if resolve_groq_api_key():
        sample_chunks = [
            {
                "chunk_id": "MNT-001_chunk_001",
                "document": "The maintenance schedule requires checking oil levels every 1000 hours.",
                "metadata": {"document_id": "MNT-001"},
                "score": 0.85,
            },
        ]
        response = generate_response("What is the maintenance schedule?", sample_chunks)
        print(f"Response: {response['content']}")
        print(f"\nTokens used: {response['usage'].get('total_tokens', 'N/A')}")
        print(f"Sources: {response['sources']}")
    else:
        print("Set GROQ_API_KEY to run this demo")

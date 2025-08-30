# app/llm.py
from typing import List, Dict, Any, Tuple
from app.config import settings
from groq import Groq
import time

SYSTEM_PROMPT = """You are a precise, citation-first assistant. 
Use only the provided context. If unsure, say you don't know.
Cite sources inline like [1], [2] corresponding to the provided context chunks.
Keep answers concise and factual."""

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 600) -> str:
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat.choices[0].message.content.strip()

    def generate_with_meta(
        self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 600
    ) -> Dict[str, Any]:
        """Return text + usage + latency (seconds)."""
        t0 = time.time()
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.time() - t0
        text = (chat.choices[0].message.content or "").strip()

        # Defensive usage extraction
        usage = getattr(chat, "usage", None)
        if usage is None and hasattr(chat, "to_dict"):
            usage = chat.to_dict().get("usage")
        usage = usage or {}
        # Normalize keys if present
        usage_norm = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        return {
            "text": text,
            "latency_s": latency,
            "usage": usage_norm,
            "model": self.model,
        }


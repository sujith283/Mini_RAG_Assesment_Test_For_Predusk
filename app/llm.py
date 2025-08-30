# app/llm.py
from typing import List, Dict, Any
from app.config import settings
from groq import Groq

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

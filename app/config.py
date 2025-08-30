# app/config.py
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (useful for local dev)
load_dotenv(dotenv_path=Path("config/.env"))

@dataclass(frozen=True)
class Settings:
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-mini")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")

    # Embeddings (MiniLM)
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # Reranker
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-3.0")
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    initial_recall_k: int = int(os.getenv("INITIAL_RECALL_K", "25"))

    # LLM (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))   # target 800–1200 per req
    chunk_overlap: float = float(os.getenv("CHUNK_OVERLAP", "0.12"))       # 10–15%

    # Misc
    max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "6"))

settings = Settings()

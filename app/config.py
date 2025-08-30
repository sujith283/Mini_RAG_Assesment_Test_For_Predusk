# app/config.py
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# 1) Load config/.env if present (local dev)
load_dotenv(dotenv_path=Path("config/.env"))

# 2) Also load .streamlit/secrets.toml when running outside Streamlit
def load_streamlit_secrets():
    """
    Load keys from .streamlit/secrets.toml into os.environ if present.
    Does not override already-set env vars.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        # Fallback for Python <=3.10 (not expected here, but safe)
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            return  # can't parse TOML, just skip

    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        return

    try:
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for key, value in secrets.items():
            # Don't overwrite already-exported env vars
            if key not in os.environ and value is not None:
                os.environ[key] = str(value)
    except Exception as e:
        print("Warning: could not load .streamlit/secrets.toml:", e)

# Try to load Streamlit secrets (safe no-op if file missing)
load_streamlit_secrets()

@dataclass(frozen=True)
class Settings:
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-mini")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")

    # Embeddings
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
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "1000"))
    chunk_overlap: float = float(os.getenv("CHUNK_OVERLAP", "0.12"))

    # Misc / Retrieval tuning
    max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "6"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.25"))  # filter low-similarity hits

settings = Settings()


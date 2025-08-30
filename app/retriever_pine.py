# app/retriever_pine.py
from typing import List, Dict, Any
from app.config import settings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

class PineconeRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.embedding_model_name)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        # Create index if missing
        if settings.pinecone_index not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self.index = self.pc.Index(settings.pinecone_index)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def upsert_chunks(self, chunks: List[Dict[str, Any]], namespace: str | None = None):
        namespace = namespace or settings.pinecone_namespace
        vectors = []
        for i, c in enumerate(chunks):
            vec = self.embed([c["text"]])[0]
            metadata = {
                "text": c["text"],
                **{k: v for k, v in c["metadata"].items() if k in ("source", "title", "section", "position")}
            }
            vectors.append({
                "id": f'{metadata.get("source","doc")}:{metadata.get("position", i)}',
                "values": vec,
                "metadata": metadata
            })
        # Pinecone v3 upsert
        self.index.upsert(vectors=vectors, namespace=namespace)

    def retrieve(self, query: str, top_k: int | None = None, namespace: str | None = None, min_score: float = 0.25):
        top_k = top_k or settings.initial_recall_k
        namespace = namespace or settings.pinecone_namespace
        qvec = self.embed([query])[0]
        res = self.index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        hits = []
        for m in res["matches"]:
            if m["score"] >= min_score:   # <---- guard added
                md = m["metadata"] or {}
                hits.append({
                    "id": m["id"],
                    "score": m["score"],
                    "text": md.get("text", ""),
                    "metadata": md
                })
        return hits
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        # Accept either numpy array or plain Python list from mocks
        if hasattr(vecs, "tolist"):
            vecs = vecs.tolist()
        return vecs


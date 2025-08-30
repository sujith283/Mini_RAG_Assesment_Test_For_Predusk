# app/pipeline.py
from typing import List, Dict, Any
from app.config import settings
from app.retriever_pine import PineconeRetriever
from app.llm import GroqLLM, SYSTEM_PROMPT
from app.utils import build_inline_citations, insert_citation_tags, clean_text, mmr
import cohere

class RagPipeline:
    def __init__(self):
        self.retriever = PineconeRetriever()
        self.llm = GroqLLM()
        self.cohere = cohere.Client(api_key=settings.cohere_api_key)
        self.rerank_model = settings.cohere_model

    # ---------- Ingestion (optional helper you call elsewhere) ----------
    def ingest_document(self, text: str, source: str, title: str = "", section: str = ""):
        from app.utils import sliding_window_chunk
        chunks = sliding_window_chunk(
            text=text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_ratio=settings.chunk_overlap,
            meta={"source": source, "title": title, "section": section},
        )
        self.retriever.upsert_chunks(chunks)

    # ---------- Retrieval + Rerank ----------
    def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        # 1) Dense retrieval (Pinecone)
        initial_hits = self.retriever.retrieve(query, top_k=settings.initial_recall_k)
        if not initial_hits:
            return []

        # Optional: quick MMR diversity BEFORE rerank (on embeddings again)
        # We'll fetch embeddings for the hit texts and pick diversified subset.
        embs = self.retriever.embed([h["text"] for h in initial_hits])
        mmr_idx = mmr(embs, top_k=min(12, len(initial_hits)), lambda_mult=0.55)
        diversified = [initial_hits[i] for i in mmr_idx]

        # 2) Cohere Rerank-3 for semantic ranking against the query
        docs_for_rerank = [d["text"] for d in diversified]
        rr = self.cohere.rerank(
            model=self.rerank_model,
            query=query,
            documents=docs_for_rerank,
            top_n=min(settings.rerank_top_k, len(docs_for_rerank)),
        )
        # rr.results[i].index indexes into docs_for_rerank
        reranked = []
        for r in rr.results:
            item = diversified[r.index]
            reranked.append({
                **item,
                "rerank_score": r.relevance_score
            })
        return reranked

    # ---------- Answer ----------
    def answer(self, query: str) -> Dict[str, Any]:
        reranked = self.retrieve_and_rerank(query)
        if not reranked:
            return {
                "answer": "I couldn’t find enough information in your documents to answer that confidently.",
                "contexts": [],
                "sources": []
            }

        # Limit contexts passed to the LLM (helps costs + focus)
        contexts = reranked[: settings.max_context_docs]

        # Decorate with citation numbers
        _, unique_sources = build_inline_citations([{
            "text": c["text"],
            "source": c["metadata"].get("source"),
            "title": c["metadata"].get("title"),
            "section": c["metadata"].get("section"),
            "position": c["metadata"].get("position"),
        } for c in contexts])

        # Reflect cite_num back onto contexts
        key_to_num = {}
        for i, s in enumerate(unique_sources, start=1):
            key_to_num[(s.get("source"), s.get("title"), s.get("section"), s.get("position"))] = i
        for c in contexts:
            key = (
                c["metadata"].get("source"),
                c["metadata"].get("title"),
                c["metadata"].get("section"),
                c["metadata"].get("position"),
            )
            c["cite_num"] = key_to_num.get(key, "?")

        context_block = insert_citation_tags(contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_block}\n\nAnswer with inline citations like [1], [2]."}
        ]
        raw_answer = self.llm.generate(messages)
        final_answer = clean_text(raw_answer)

        # Prepare a compact sources panel (for UI)
        display_sources = []
        for c in contexts:
            md = c["metadata"]
            display_sources.append({
                "n": c["cite_num"],
                "source": md.get("source"),
                "title": md.get("title"),
                "section": md.get("section"),
                "position": md.get("position"),
                "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else "")
            })
        display_sources.sort(key=lambda x: x["n"])

        return {
            "answer": final_answer,
            "contexts": contexts,
            "sources": display_sources
        }

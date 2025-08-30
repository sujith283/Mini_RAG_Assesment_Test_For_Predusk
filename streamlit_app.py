# streamlit_app.py
import streamlit as st
from app.pipeline import RagPipeline

st.set_page_config(page_title="Mini RAG (Pinecone + MiniLM + Cohere Rerank + Groq)", layout="wide")
st.title("🔎 MINI_RAG — Pinecone + MiniLM + Cohere Rerank-3 + Groq")

pipe = RagPipeline()

with st.sidebar:
    st.header("📥 Add Document")
    src = st.text_input("Source ID/URL (for citation)", value="local-upload")
    title = st.text_input("Title (optional)", value="")
    section = st.text_input("Section (optional)", value="")
    text = st.text_area("Paste text to index")
    if st.button("Ingest"):
        if text.strip():
            pipe.ingest_document(text, source=src, title=title, section=section)
            st.success("Ingested ✅")
        else:
            st.warning("Please paste some text.")

st.subheader("Ask a question")
q = st.text_input("Your query")
if st.button("Ask") and q.strip():
    with st.spinner("Thinking..."):
        out = pipe.answer(q)

    st.markdown("### Answer")
    st.write(out["answer"])

    # --- Metrics panel ---
    m = out.get("metrics", {})
    tok = m.get("llm_tokens") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM latency", f"{m.get('llm_latency_s', 0):.2f}s")
    col2.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
    col3.metric("Rerank", f"{m.get('rerank_s', 0):.2f}s")
    col4.metric("Model", m.get("model", "—"))

    # Token line (if available)
    if any(v is not None for v in tok.values()):
        st.caption(
            f"Tokens — Prompt: {tok.get('prompt_tokens','?')}, "
            f"Completion: {tok.get('completion_tokens','?')}, "
            f"Total: {tok.get('total_tokens','?')}"
        )
    else:
        st.caption("Tokens — not returned by provider for this response.")

    if out["sources"]:
        st.markdown("### Sources")
        for s in out["sources"]:
            st.markdown(f"**[{s['n']}] {s.get('title') or s.get('source')}**")
            small = f"{s.get('source','')} • {s.get('section','')} • pos {s.get('position')}"
            st.caption(small)
            st.code(s["snippet"])


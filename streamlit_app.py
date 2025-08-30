import streamlit as st
from app import token_tracker
from app.pipeline import RagPipeline

# -------- Helpers --------
def _extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts text from an uploaded PDF using PyPDF2 if available.
    Falls back to a naive byte decode if PDF parsing lib is missing.
    """
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(uploaded_file)
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n\n".join(texts).strip()
    except Exception:
        # Fallback: this will not give structured text but avoids hard failure
        try:
            data = uploaded_file.getvalue()
            return data.decode(errors="ignore")
        except Exception:
            return ""

import streamlit as st
st.set_page_config(page_title="Mini RAG (Pinecone + MiniLM + Cohere Rerank + Groq)", layout="wide")

# Debug panel – remove after fixing
with st.expander("Diagnostics (temporary)"):
    st.write(
        "Secrets loaded:",
        "PINECONE" in st.secrets,
        "api_key" in st.secrets.get("PINECONE", {})
    )

# --- Centered loading message until pipeline is ready ---
placeholder = st.empty()
with placeholder.container():
    st.markdown(
        "<h2 style='text-align:center;'>⏳ Loading… please wait</h2>",
        unsafe_allow_html=True
    )

# --- Instantiate pipeline (may take a few seconds) ---
pipe = RagPipeline()

# --- Clear placeholder once ready ---
placeholder.empty()

# ---------------- UI starts after pipeline is ready ----------------
st.title("🤖 MINI_RAG Chatbot — Pinecone + MiniLM + Cohere Rerank-3 + Groq")

# ---------------- Sidebar: Ingestion ----------------
with st.sidebar:
    st.header("📥 Ingest")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    text_to_index = st.text_area("Or paste text to index", height=180, placeholder="Paste raw text here…")

    # Only enable ingest if we have either a PDF file or pasted text
    can_ingest = (pdf_file is not None) or (text_to_index.strip() != "")
    ingest_btn = st.button("Ingest", type="primary", use_container_width=True, disabled=not can_ingest)

    if ingest_btn:
        # If PDF provided, extract and ingest with filename as source
        if pdf_file is not None:
            with st.spinner("Reading PDF and chunking…"):
                pdf_text = _extract_text_from_pdf(pdf_file)
            if pdf_text.strip():
                # use filename as source; title blank; section blank
                src_name = getattr(pdf_file, "name", "uploaded.pdf")
                pipe.ingest_document(pdf_text, source=src_name, title="", section="")
                st.success(f"Ingested from PDF: {src_name} ✅")
            else:
                st.error("Could not extract any text from the PDF. Please check the file.")

        # If pasted text provided, ingest it as local-paste
        if text_to_index.strip():
            pipe.ingest_document(text_to_index.strip(), source="local-paste", title="", section="")
            st.success("Ingested pasted text ✅")

        if not (pdf_file is not None or text_to_index.strip()):
            st.warning("Upload a PDF or paste some text to ingest.")

# ---------------- Main: Chat Interface ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"query": str, "answer": str, "out": dict}

# Show previous chat turns
for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.markdown("### Answer")
        st.write(turn["answer"])

        # --- Metrics panel for that turn ---
        m = turn["out"].get("metrics", {})
        tok = m.get("llm_tokens") or {}
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("LLM latency", f"{m.get('llm_latency_s', 0):.2f}s")
        col2.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
        col3.metric("Rerank", f"{m.get('rerank_s', 0):.2f}s")
        col4.metric("Model", m.get("model", "—"))

        # Update token tracker if total available
        total_used = tok.get("total_tokens")
        if total_used:
            left, used, limit = token_tracker.add_tokens(int(total_used))
            col5.metric("Tokens left (today)", f"{left:,}", f"-{used:,}/{limit:,}")
            st.caption(
                f"Tokens — Prompt: {tok.get('prompt_tokens','?')}, "
                f"Completion: {tok.get('completion_tokens','?')}, "
                f"Total: {tok.get('total_tokens','?')}"
            )
        else:
            col5.metric("Tokens left (today)", "—")
            st.caption("Tokens — not returned by provider for this response.")

        # Sources
        if turn["out"]["sources"]:
            st.markdown("### Sources")
            for s in turn["out"]["sources"]:
                st.markdown(f"**[{s['n']}] {s.get('title') or s.get('source')}**")
                small = f"{s.get('source','')} • {s.get('section','')} • pos {s.get('position')}"
                st.caption(small)
                st.code(s["snippet"])

# --- New Question Input ---
q = st.chat_input("Ask your question here…")
if q and q.strip():
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            out = pipe.answer(q)
        st.markdown("### Answer")
        st.write(out["answer"])

        # --- Metrics panel ---
        m = out.get("metrics", {})
        tok = m.get("llm_tokens") or {}
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("LLM latency", f"{m.get('llm_latency_s', 0):.2f}s")
        col2.metric("Retrieve", f"{m.get('retrieve_s', 0):.2f}s")
        col3.metric("Rerank", f"{m.get('rerank_s', 0):.2f}s")
        col4.metric("Model", m.get("model", "—"))

        # Update token tracker if total available
        total_used = tok.get("total_tokens")
        if total_used:
            left, used, limit = token_tracker.add_tokens(int(total_used))
            col5.metric("Tokens left (today)", f"{left:,}", f"-{used:,}/{limit:,}")
            st.caption(
                f"Tokens — Prompt: {tok.get('prompt_tokens','?')}, "
                f"Completion: {tok.get('completion_tokens','?')}, "
                f"Total: {tok.get('total_tokens','?')}"
            )
        else:
            col5.metric("Tokens left (today)", "—")
            st.caption("Tokens — not returned by provider for this response.")

        # Sources
        if out["sources"]:
            st.markdown("### Sources")
            for s in out["sources"]:
                st.markdown(f"**[{s['n']}] {s.get('title') or s.get('source')}**")
                small = f"{s.get('source','')} • {s.get('section','')} • pos {s.get('position')}"
                st.caption(small)
                st.code(s["snippet"])

    # Save this turn into chat history
    st.session_state.chat_history.append({"query": q, "answer": out["answer"], "out": out})

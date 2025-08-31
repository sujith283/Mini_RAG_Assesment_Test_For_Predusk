# MINI_RAG — Pinecone + MiniLM + Cohere Rerank + Groq + Streamlit

A lightweight Retrieval-Augmented Generation (RAG) demo app. Upload PDFs or paste text, ask natural language queries, and get grounded answers with citations, reranking, and usage metrics.

## 🌐 Live Demo

**Try it now:** [ie2taanqdmqsswbtqvavb3.streamlit.app](https://ie2taanqdmqsswbtqvavb3.streamlit.app)

> The app is already deployed and ready to use! No setup required for basic testing.

## 🚀 Features

- **Document Ingestion** — Upload PDFs or paste text; chunks stored in Pinecone with MiniLM embeddings
- **Retrieval & Reranking** — Top-k retrieval with MMR + Cohere Rerank-3
- **Answering** — LLM backend via Groq API (Llama-3.1 8B/70B)
- **UI** — Streamlit chatbot with history, metrics, sources, and ingestion sidebar

## 📂 Project Structure

```
rag-pinecone/
├─ app/                          # Core RAG logic
├─ streamlit_app.py              # Streamlit UI (chat + ingestion + metrics)
├─ tests/                        # Unit tests
├─ scripts/                      # Local debug helpers
├─ docs/                         # Sample docs
├─ config/                       # Example configs
├─ .streamlit/                   # ⚠️ (gitignored) real secrets.toml lives here
├─ requirements.txt
├─ README.md
└─ .token_usage.json             # auto-created token counter
```

## ⚙️ Setup & Installation

Prerequisites

Python 3.11+ (minimum required version)

### 1. Clone the repository

```bash
git clone https://github.com/sujith283/Mini_RAG_Assesment_Test_For_Predusk
cd Mini_RAG_Assesment_Test_For_Predusk
```

### 2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys ⚠️ **Very Important**

All secrets are loaded from `.streamlit/secrets.toml`. This file is **not checked into Git** (already gitignored).

**Create the secrets file using terminal commands:**

```bash
# Create .streamlit directory
mkdir -p .streamlit

# Create secrets.toml file (choose one method below)

# Method 1: Using echo (Windows Command Prompt)
Set-Content -Path ".streamlit\secrets.toml" -Value "[PINECONE]"
Add-Content -Path ".streamlit\secrets.toml" -Value 'api_key = "your-pinecone-key"'
Add-Content -Path ".streamlit\secrets.toml" -Value 'environment = "your-pinecone-environment"'
Add-Content -Path ".streamlit\secrets.toml" -Value 'index_name = "mini-rag"'
Add-Content -Path ".streamlit\secrets.toml" -Value ""

Add-Content -Path ".streamlit\secrets.toml" -Value "[COHERE]"
Add-Content -Path ".streamlit\secrets.toml" -Value 'api_key = "your-cohere-key"'
Add-Content -Path ".streamlit\secrets.toml" -Value ""

Add-Content -Path ".streamlit\secrets.toml" -Value "[GROQ]"
Add-Content -Path ".streamlit\secrets.toml" -Value 'api_key = "your-groq-key"'

# Method 2: Using cat (macOS/Linux)
cat > .streamlit/secrets.toml << 'EOF'
[PINECONE]
api_key = "your-pinecone-key"
environment = "your-pinecone-environment"
index_name = "mini-rag"

[COHERE]
api_key = "your-cohere-key"

[GROQ]
api_key = "your-groq-key"
EOF
```

**Or create manually:**
1. Create a folder named `.streamlit` (if not already there)
2. Inside `.streamlit`, create a file named `secrets.toml`
3. Paste the following template and fill in your real keys:

```toml
[PINECONE]
api_key = "your-pinecone-key"
environment = "your-pinecone-environment"
index_name = "mini-rag"

[COHERE]
api_key = "your-cohere-key"

[GROQ]
api_key = "your-groq-key"
```

> ⚠️ **Do not commit this file to GitHub** — it contains private keys. Streamlit Cloud will automatically read this file if you upload it under **"Secrets"** in the project settings.

### 5. Run the application

```bash
streamlit run streamlit_app.py
```

## 📊 Chunking & Retrieval Settings

- **Embedding model:** MiniLM (dim=384)
- **Chunk size:** 800–1200 tokens
- **Overlap:** 10–15%
- **Retriever:** Pinecone + MMR
- **Top-k:** default 5
- **Reranker:** Cohere Rerank-3

## 📈 Metrics & Token Tracking

- Latency per stage (retrieve, rerank, LLM)
- Daily token usage tracked in `.token_usage.json`
- Shows remaining quota vs configured daily limit

## ☁️ Deployment

### Live Demo
The app is already deployed on **Streamlit Cloud** and accessible at:
**[ie2taanqdmqsswbtqvavb3.streamlit.app](https://ie2taanqdmqsswbtqvavb3.streamlit.app)**

### Deploy Your Own Version
- **Streamlit Cloud** (recommended): Upload repo, then paste your secrets in the "Secrets" settings panel
- **Local run:** Keep `.streamlit/secrets.toml` on your machine
- **Optional:** Dockerize for custom deployment

## 📝 Remarks

- Groq API free tier has request/token rate limits — check your account
- Cohere rerank credits may be limited
- add custom namespace for each user
- PyPDF2 PDF extraction is basic; replace with pdfplumber for better results
- Demo project — add auth & persistence for production use

## 🔗 API Documentation

- [Pinecone](https://docs.pinecone.io/)
- [Cohere Rerank](https://docs.cohere.com/reference/rerank)
- [Groq API](https://console.groq.com/docs/quickstart)
- [Streamlit](https://docs.streamlit.io/)






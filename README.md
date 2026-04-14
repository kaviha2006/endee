# 🧠 DocMind – AI-Powered Document Q&A

> **RAG pipeline** built with **[Endee](https://github.com/endee-io/endee)** vector database + OpenAI.  
> Upload any PDF or TXT, ask natural-language questions, get grounded answers.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Endee](https://img.shields.io/badge/VectorDB-Endee-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Demo

```
Upload PDF → Chunk → Embed → Store in Endee → Ask Question → Retrieve → Generate Answer
```

---

## 🏗️ System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                        DocMind Architecture                       │
│                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐│
│  │  Upload  │───▶│  Chunker     │───▶│  OpenAI Embeddings       ││
│  │ PDF/TXT  │    │ (500 chars,  │    │  text-embedding-3-small   ││
│  └──────────┘    │  100 overlap)│    │  (1536-dim vectors)       ││
│                  └──────────────┘    └────────────┬─────────────┘│
│                                                   │ upsert        │
│                                      ┌────────────▼─────────────┐│
│                                      │   Endee Vector Database   ││
│                                      │   (HNSW, Cosine, INT8)    ││
│                                      └────────────┬─────────────┘│
│  ┌──────────┐    ┌──────────────┐                 │ query top-5   │
│  │  Answer  │◀───│  GPT-4o-mini │◀────────────────┘               │
│  │ (Grounded│    │  (RAG prompt)│                                  │
│  │  to docs)│    └──────────────┘                                  │
│  └──────────┘                                                      │
└──────────────────────────────────────────────────────────────────┘
```

### How it works

1. **Document Ingestion** — User uploads PDF or TXT files via Streamlit UI
2. **Chunking** — Documents are split into 500-character overlapping chunks (100-char overlap) to preserve context across boundaries
3. **Embedding** — Each chunk is converted to a 1536-dimensional dense vector using OpenAI `text-embedding-3-small`
4. **Storage in Endee** — Vectors are upserted into an Endee index (`docmind_index`) with metadata (source filename, chunk index, raw text)
5. **Query** — User's question is embedded using the same model
6. **Retrieval** — Endee performs HNSW approximate nearest-neighbor search (cosine similarity) and returns the top-5 most relevant chunks
7. **Generation** — Retrieved chunks are passed as context to GPT-4o-mini with a strict RAG prompt to produce a grounded answer

### Why Endee?
- 🚀 Handles up to **1 billion vectors** on a single node
- ⚡ HNSW indexing with INT8 quantization for speed + memory efficiency
- 🐳 One-command Docker setup — no configuration headaches
- 🔍 Cosine similarity search with optional metadata filtering
- 🐍 Clean Python SDK (`pip install endee`)

---

## 🗂️ Project Structure

```
docmind/
├── app.py                    # Streamlit app (main entry point)
├── src/
│   ├── __init__.py
│   ├── document_processor.py # PDF/TXT loading & text chunking
│   ├── embedder.py           # OpenAI embedding generation (batched)
│   ├── vector_store.py       # Endee SDK wrapper (index + upsert + query)
│   └── llm.py                # GPT-4o-mini answer generation
├── sample_docs/
│   └── ai_overview.txt       # Sample document to try out immediately
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- Docker Desktop ([install](https://docs.docker.com/get-docker/))
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

### Step 1 — Star & Fork Endee (Required for submission)

```bash
# 1. Go to https://github.com/endee-io/endee and click ⭐ Star
# 2. Click Fork → Fork to your GitHub account
# 3. Clone YOUR fork:
git clone https://github.com/<your-username>/endee.git
```

---

### Step 2 — Clone this project

```bash
git clone https://github.com/<your-username>/docmind.git
cd docmind
```

---

### Step 3 — Start Endee with Docker

```bash
docker run \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest
```

Verify it's running: open [http://localhost:8080](http://localhost:8080) in your browser.

---

### Step 4 — Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 5 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key-here
ENDEE_HOST=http://localhost:8080
ENDEE_AUTH_TOKEN=          # leave empty if Docker has no auth token
```

---

### Step 6 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Usage

1. Upload a PDF or TXT file using the **sidebar**
2. Click **Index Documents** — chunks are embedded and stored in Endee
3. Type your question in the chat box
4. DocMind retrieves the most relevant chunks from Endee and generates a grounded answer
5. Expand **Sources used** to see exactly which document chunks informed the answer

A sample document (`sample_docs/ai_overview.txt`) is included — try questions like:
- *"What is RAG?"*
- *"How does Endee work?"*
- *"Explain the difference between ML and deep learning."*

---

## 🛠️ Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Vector Database  | **Endee** (HNSW, cosine, INT8)      |
| Embeddings       | OpenAI `text-embedding-3-small`     |
| LLM              | OpenAI `gpt-4o-mini`                |
| UI               | Streamlit                           |
| PDF parsing      | PyPDF2                              |
| Language         | Python 3.10+                        |
| Deployment       | Docker (Endee) + local Streamlit    |

---

## 📡 Endee API Usage in this project

```python
from endee import Endee, Precision

# Connect
client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create index
client.create_index(
    name="docmind_index",
    dimension=1536,
    space_type="cosine",
    precision=Precision.INT8,
)

# Upsert vectors
index = client.get_index(name="docmind_index")
index.upsert([{
    "id": "doc1__chunk_0",
    "vector": [...],          # 1536-dim embedding
    "meta": {"text": "...", "source": "doc1.pdf"},
    "filter": {"source": "doc1.pdf"},
}])

# Similarity search
results = index.query(
    vector=[...],             # query embedding
    top_k=5,
    ef=128,
)
```

---

## 🧩 Possible Extensions

- 🔁 **Hybrid search** using Endee's BM25 sparse vectors + dense embeddings
- 🗂️ **Multi-collection** support (separate index per document set)
- 🌐 **Web scraping** — index web pages instead of uploaded files
- 💬 **Conversation memory** — multi-turn RAG with history context
- 📊 **Metadata filtering** — ask questions restricted to a specific document

---

## 📄 License

MIT © 2026

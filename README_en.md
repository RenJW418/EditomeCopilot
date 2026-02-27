# EditomeCopilot

> **An agentic RAG system for gene-editing research** — CRISPR, Base Editing, Prime Editing, and clinical trial data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev)

📖 [中文版本](README.md)

---

## Architecture

```
Browser (React + Vite)
        │  HTTP /api/*
        ▼
FastAPI  app.py  [port 6006]
        │
        ▼
AgenticRAG  core/agentic_rag.py
  ├── HybridRetriever   (FAISS semantic + BM25 lexical)
  ├── KGQueryExpander   (KG-AQE: knowledge-graph query expansion)
  ├── EvidencePyramidScorer (EPARS: evidence-level re-ranking)
  ├── ConflictResolver  (CAEA: conflict detection & aggregation)
  ├── RetrievalCalibrator   (RCC: confidence calibration)
  ├── QueryDecomposer / HyDE / RAPTOR / SelfQuery
  └── GraphReasoner     (multi-hop KG reasoning)
        │
        ├── data/faiss_db/          86 K PubMed/EuropePMC articles
        └── data/knowledge_base/    KG (140 nodes, 973 edges)
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

### 1 — Clone & configure

```bash
git clone <repo-url> EditomeCopilot
cd EditomeCopilot
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY and OPENAI_BASE_URL
```

### 2 — Download data

Pre-built FAISS indices and literature database (~750 MB total) are hosted on Hugging Face:

**Dataset**: [RenJW/editome-copilot-data](https://huggingface.co/datasets/RenJW/editome-copilot-data) (full `data/` directory, ~750 MB)

| Path | Size | Description |
|------|------|-------------|
| `data/faiss_db/index.faiss` | 254 MB | FAISS semantic index |
| `data/faiss_db/index.pkl` | 166 MB | FAISS metadata store |
| `data/faiss_db/bm25_corpus.pkl` | 163 MB | BM25 lexical index |
| `data/knowledge_base/literature_db_GEA_v2026_Q1.json` | 168 MB | Raw literature DB |
| `data/knowledge_base/report_GEA_v2026_Q1.md` | <1 MB | Gene editing almanac report |
| `data/knowledge_base/kg.json` | <1 MB | Knowledge graph |
| `data/eval/` | <1 MB | GEBench evaluation set |
| `data/user_uploads_db/` | — | User upload placeholders |

Run from inside the **`EditomeCopilot/`** directory — files will be placed in the correct paths automatically:

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="RenJW/editome-copilot-data",
    repo_type="dataset",
    local_dir=".",   # downloads into current dir (EditomeCopilot/)
    ignore_patterns=["README.md"],  # only skip HF auto-generated README
)
EOF
```

After download, your directory should look like:

```
EditomeCopilot/
├── data/
│   ├── faiss_db/
│   │   ├── index.faiss                          ✓
│   │   ├── index.pkl                            ✓
│   │   └── bm25_corpus.pkl                      ✓
│   ├── knowledge_base/
│   │   ├── literature_db_GEA_v2026_Q1.json      ✓
│   │   ├── report_GEA_v2026_Q1.md               ✓
│   │   └── kg.json                              ✓
│   ├── eval/
│   │   ├── gebench.jsonl                        ✓
│   │   └── gold_standard.jsonl                  ✓
│   └── user_uploads_db/
│       ├── index.faiss                          ✓
│       └── index.pkl                            ✓
└── ...
```

### 3 — One-click launch

**Linux / macOS**

```bash
chmod +x start.sh
./start.sh
```

**Windows**

```bat
start.bat
```

The script will:
1. Create `.venv` (if not present)
2. Install Python dependencies from `requirements.txt`
3. Build the React frontend (`frontend/dist/`)
4. Start the server at **http://localhost:6006**

### 4 — Open browser

Navigate to **http://localhost:6006**

---

## Manual Setup (advanced)

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend build
cd frontend && npm install && npm run build && cd ..

# Start server
uvicorn app:app --host 0.0.0.0 --port 6006
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health check |
| GET | `/api/sessions` | List all chat sessions |
| GET | `/api/sessions/{id}` | Get session history |
| DELETE | `/api/sessions/{id}` | Delete a session |
| POST | `/api/chat` | Main RAG query |
| POST | `/api/upload_library` | Import BibTeX / RIS user library |

**POST /api/chat** body:
```json
{
  "session_id": "optional-uuid",
  "query": "What are the off-target rates of base editors in clinical trials?"
}
```

---

## Project Structure

```
EditomeCopilot/
├── app.py                  # FastAPI entry point
├── requirements.txt
├── start.sh / start.bat    # One-click deploy
├── .env.example
├── core/                   # RAG pipeline modules
│   ├── agentic_rag.py      # Main orchestrator (~1000 lines)
│   ├── hybrid_retrieval.py
│   ├── evidence_scorer.py  # EPARS (v4)
│   ├── kg_query_expander.py# KG-AQE (v4)
│   ├── conflict_resolver.py# CAEA (v4)
│   ├── retrieval_calibrator.py # RCC (v4)
│   └── ...
├── frontend/               # React + Vite + Tailwind
│   └── src/
└── data/                   # Data directory (download from HuggingFace, not in repo)
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full list with comments.

| Variable | Required | Default | Note |
|----------|----------|---------|------|
| `OPENAI_API_KEY` | ✅ | — | DashScope / OpenAI key |
| `OPENAI_BASE_URL` | ✅ | — | API base URL |
| `LLM_MODEL` | — | `qwen-plus` | Model name |
| `ENABLE_EPARS` | — | `true` | v4 evidence scoring |
| `ENABLE_KG_AQE` | — | `true` | v4 KG query expansion |
| `ENABLE_CAEA` | — | `true` | v4 conflict resolution |
| `ENABLE_RCC` | — | `true` | v4 confidence calibration |

---

## License

MIT

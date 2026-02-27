# EditomeCopilot — 基因编辑智能问答系统

> 专注于基因编辑（CRISPR、Base Editing、Prime Editing）及临床数据的 **Agentic RAG** 系统。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev)

---

## 系统架构

```
浏览器 (React + Vite)
        │  HTTP /api/*
        ▼
FastAPI  app.py  [端口 6006]
        │
        ▼
AgenticRAG  core/agentic_rag.py
  ├── HybridRetriever       (FAISS 语义 + BM25 关键词)
  ├── KGQueryExpander       (KG-AQE：知识图谱引导查询扩展)
  ├── EvidencePyramidScorer (EPARS：证据金字塔重排序)
  ├── ConflictResolver      (CAEA：冲突感知证据聚合)
  ├── RetrievalCalibrator   (RCC：检索置信度校准)
  ├── QueryDecomposer / HyDE / RAPTOR / SelfQuery
  └── GraphReasoner         (知识图谱多跳推理)
        │
        ├── data/faiss_db/         8.6 万篇 PubMed / EuropePMC 文献
        └── data/knowledge_base/   知识图谱 (140 节点，973 条边)
```

---

## 快速开始

### 环境要求

| 工具 | 版本 |
|------|------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

### 第一步 — 克隆并配置

```bash
git clone <repo-url> EditomeCopilot
cd EditomeCopilot
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY 和 OPENAI_BASE_URL
```

### 第二步 — 一键启动

**Linux / macOS**

```bash
chmod +x start.sh
./start.sh
```

**Windows**

```bat
start.bat
```

脚本会自动完成：
1. 创建 `.venv` 虚拟环境（如不存在）
2. 安装 `requirements.txt` 中的 Python 依赖
3. 构建 React 前端（`frontend/dist/`）
4. 在 **http://localhost:6006** 启动服务

### 第三步 — 打开浏览器

访问 **http://localhost:6006**

---

## 手动启动（进阶用法）

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate        # Windows:.venv\Scripts\activate

# 安装 Python 依赖
pip install -r requirements.txt

# 构建前端
cd frontend && npm install && npm run build && cd ..

# 启动服务
uvicorn app:app --host 0.0.0.0 --port 6006
```

---

## 数据准备

### 方式 A — 下载预构建数据（推荐）

预构建的 FAISS 索引与文献库（约 750 MB）已托管在 Hugging Face：

**数据集**：[RenJW/editome-copilot-data](https://huggingface.co/datasets/RenJW/editome-copilot-data)

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/faiss_db/index.faiss` | 254 MB | FAISS 语义索引 |
| `data/faiss_db/index.pkl` | 166 MB | FAISS 元数据 |
| `data/faiss_db/bm25_corpus.pkl` | 163 MB | BM25 词汇索引 |
| `data/knowledge_base/literature_db_GEA_v2026_Q1.json` | 168 MB | 原始文献库 |

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="RenJW/editome-copilot-data",
    repo_type="dataset",
    local_dir=".",
    ignore_patterns=["*.md"],
)
EOF
```

### 方式 B — 从头构建

按顺序运行以下脚本（约需 30–60 分钟，取决于网络速度）：

```bash
# 1. 从 PubMed / EuropePMC 抓取文献（约 8.6 万篇）
python scripts/build_literature_db.py

# 2. 构建 FAISS + BM25 索引
python scripts/process_knowledge_base.py

# 3. 构建基因编辑知识图谱
python scripts/build_kg_from_almanac.py
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务健康检查 |
| GET | `/api/sessions` | 获取所有会话列表 |
| GET | `/api/sessions/{id}` | 获取指定会话历史 |
| DELETE | `/api/sessions/{id}` | 删除指定会话 |
| POST | `/api/chat` | 主问答接口 |
| POST | `/api/upload_library` | 导入用户文献库（BibTeX / RIS） |

**POST /api/chat 请求体示例：**

```json
{
  "session_id": "可选的 UUID",
  "query": "碱基编辑器在临床试验中的脱靶率是多少？"
}
```

---

## 项目结构

```
EditomeCopilot/
├── app.py                    # FastAPI 入口
├── cli_interactive.py        # 命令行交互模式
├── requirements.txt
├── start.sh / start.bat      # 一键部署脚本
├── .env.example              # 环境变量模板
├── core/                     # RAG 核心模块
│   ├── agentic_rag.py        # 主编排器（约 1000 行）
│   ├── hybrid_retrieval.py   # 混合检索
│   ├── evidence_scorer.py    # EPARS (v4 算法)
│   ├── kg_query_expander.py  # KG-AQE (v4 算法)
│   ├── conflict_resolver.py  # CAEA (v4 算法)
│   ├── retrieval_calibrator.py  # RCC (v4 算法)
│   └── ...
├── data/
│   ├── faiss_db/             # 向量索引（已在 .gitignore 中）
│   └── knowledge_base/       # kg.json 知识图谱（已提交）
├── evaluation/               # GEBench 评估框架
├── frontend/                 # React + Vite + Tailwind
│   └── src/
└── scripts/                  # 数据摄取工具脚本
```

---

## 环境变量说明

完整注释见 [`.env.example`](.env.example)。

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENAI_API_KEY` | ✅ | — | 阿里云 DashScope / OpenAI 密钥 |
| `OPENAI_BASE_URL` | ✅ | — | API Base URL |
| `LLM_MODEL` | — | `qwen-plus` | 模型名称 |
| `ENABLE_EPARS` | — | `true` | v4 证据金字塔重排序 |
| `ENABLE_KG_AQE` | — | `true` | v4 知识图谱查询扩展 |
| `ENABLE_CAEA` | — | `true` | v4 冲突感知聚合 |
| `ENABLE_RCC` | — | `true` | v4 检索置信度校准 |

---

## 开发模式

```bash
# 后端热重载
uvicorn app:app --reload --port 6006

# 前端开发服务器（自动代理 /api 到后端）
cd frontend && npm run dev
```

---

## License

MIT

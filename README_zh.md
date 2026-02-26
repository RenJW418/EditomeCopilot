# 基因编辑 almanac (RAG 系统)

这是一个专注于基因编辑（CRISPR, Base Editing, Prime Editing）和临床数据的下一代检索增强生成（RAG）系统。

## 功能特性

- **混合检索**: 结合语义搜索 (FAISS) 和元数据过滤。
- **双重知识库**:
  - **主数据库**: 预索引的高质量文献库。
  - **用户文库**: 支持上传个人的 RIS/BibTeX 文件进行私密的、会话级的分析。
- **智能路由**: 由大语言模型驱动的意图分析，自动在通用知识查询和特定用户文档分析之间切换。
- **证据溯源**: 严格的引用追踪，通过将回答锚定到检索到的片段来防止幻觉。
- **现代化 UI**: 基于 React + TypeScript 的前端，具备持久化聊天记录功能。

## 技术栈

- **后端**: Python, FastAPI, LangChain, FAISS
- **前端**: React, TypeScript, Vite, Tailwind CSS
- **LLM**: 兼容 OpenAI 格式的 API (DeepSeek, GPT-4 等)

## 部署

1. **后端设置**:
   ```bash
   pip install -r requirements.txt
   # 在 .env 中设置 API 密钥
   python app.py
   ```

2. **前端设置**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

3. **访问**:
   打开浏览器访问 `http://localhost:6006`。

## 目录结构

- `core/`: RAG 核心逻辑 (检索, 决策引擎, 风险评估).
- `data/`: 向量数据库和 PDF 存储.
  - `knowledge_base/`: **需上传**，包含核心预置知识。
  - `user_uploads_db/`: **无需上传/可忽略**，本地运行时生成的临时用户库。
- `frontend/`: React 应用程序源代码.
- `scripts/`: 数据摄取脚本.

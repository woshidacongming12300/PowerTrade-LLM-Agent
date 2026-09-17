# Knowledge Base Directory

此目录存放 RAG 知识库文档（电力交易规则、市场政策、研究报告等）。

- 支持的格式（见 `config/chroma.yml` 的 `allow_knowledge_file_type`）：**txt / pdf / docx / doc**
- 放入文档后，运行以下命令构建向量库（ChromaDB + BM25 + RRF 混合检索）：

```bash
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"
```

- 本 README.md 不会被加载（.md 不在允许类型内），仅作说明用途。
- 去重逻辑：每个文件按 MD5 记录在 `md5.text`，重复文件会自动跳过。

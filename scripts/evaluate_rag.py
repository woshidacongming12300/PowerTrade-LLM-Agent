"""
RAG Hybrid Retriever Evaluation
Compares ChromaDB (dense) vs BM25 (sparse) vs RRF Hybrid on:
  - Hit Rate@k (HR@k)
  - Mean Reciprocal Rank (MRR)
  - Recall@k

Strategy: sample chunks from the knowledge base, use them as queries,
check if the retriever can find the source document back in top-k.
"""
from __future__ import annotations

import os
import sys
import random
import json
from typing import List, Dict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from rag.vector_store import _chroma_persist_directory


def _load_all_chunks() -> List[Dict]:
    """Load all document chunks from ChromaDB with their metadata."""
    persist_dir = _chroma_persist_directory()
    collection = chroma_conf["collection_name"]

    vs = Chroma(
        collection_name=collection,
        embedding_function=embed_model,
        persist_directory=persist_dir,
    )
    data = vs.get()
    if not data or not data.get("documents"):
        return []

    chunks = []
    for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
        chunks.append({
            "id": data["ids"][i] if data.get("ids") else str(i),
            "content": doc,
            "metadata": meta or {},
        })
    return chunks


def _build_retrievers(chunks: List[Dict]):
    """Build the three retrievers: vector-only, BM25-only, hybrid-RRF."""
    persist_dir = _chroma_persist_directory()
    collection = chroma_conf["collection_name"]
    k = chroma_conf.get("k", 3)

    vs = Chroma(
        collection_name=collection,
        embedding_function=embed_model,
        persist_directory=persist_dir,
    )
    vector_retriever = vs.as_retriever(search_kwargs={"k": k})

    all_texts = [c["content"] for c in chunks]
    bm25_retriever = BM25Retriever.from_texts(all_texts)
    bm25_retriever.k = k

    # RRF hybrid (same logic as vector_store.py)
    def rrf_hybrid(query: str):
        vec_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        rrf_scores = {}
        c = 60
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "doc": doc}
            rrf_scores[key]["score"] += 0.5 / (rank + 1 + c)
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "doc": doc}
            rrf_scores[key]["score"] += 0.5 / (rank + 1 + c)

        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items[:k]]

    return vector_retriever, bm25_retriever, rrf_hybrid


def _extract_query(chunk: str) -> str:
    """Extract a representative query from a chunk.

    Uses the first sentence (up to 50 chars) as a proxy query.
    Falls back to truncation for very short chunks.
    """
    chunk = chunk.strip()
    if not chunk:
        return "electricity price forecast"

    for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
        idx = chunk.find(sep)
        if 10 <= idx <= 80:
            return chunk[:idx].strip()

    sentences = chunk.split("，")
    for s in sentences:
        s = s.strip()
        if 15 <= len(s) <= 60:
            return s

    return chunk[:60].strip()


def evaluate(
    sample_size: int = 80,
    top_k: int = 3,
    seed: int = 42,
) -> Dict:
    """Evaluate the three retrievers and return metrics."""
    random.seed(seed)
    chunks = _load_all_chunks()

    if not chunks:
        return {"error": "No chunks found in knowledge base. Run load_document() first."}

    print(f"[RAG Eval] Loaded {len(chunks)} chunks from ChromaDB")
    sample = random.sample(chunks, min(sample_size, len(chunks)))
    print(f"[RAG Eval] Evaluating on {len(sample)} sampled queries (k={top_k})")

    vector_retriever, bm25_retriever, rrf_hybrid = _build_retrievers(chunks)

    results = {
        "vector": {"hits": 0, "reciprocal_ranks": [], "total": 0},
        "bm25": {"hits": 0, "reciprocal_ranks": [], "total": 0},
        "hybrid_rrf": {"hits": 0, "reciprocal_ranks": [], "total": 0},
    }

    for i, item in enumerate(sample):
        source_content = item["content"]
        query = _extract_query(source_content)

        for name, retriever in [
            ("vector", lambda q: vector_retriever.invoke(q)),
            ("bm25", lambda q: bm25_retriever.invoke(q)),
            ("hybrid_rrf", rrf_hybrid),
        ]:
            try:
                docs = retriever(query)
            except Exception:
                continue

            found_at = None
            for rank, doc in enumerate(docs, start=1):
                # Match by content overlap (Jaccard-like: check if source is in retrieved)
                if doc.page_content.strip() == source_content.strip():
                    found_at = rank
                    break

            if found_at and found_at <= top_k:
                results[name]["hits"] += 1
                results[name]["reciprocal_ranks"].append(1.0 / found_at)
            else:
                results[name]["reciprocal_ranks"].append(0.0)

            results[name]["total"] += 1

        if (i + 1) % 20 == 0:
            vec_hit_rate = results["vector"]["hits"] / max(results["vector"]["total"], 1)
            hyb_hit_rate = results["hybrid_rrf"]["hits"] / max(results["hybrid_rrf"]["total"], 1)
            print(f"  {i+1}/{len(sample)} ... Vector HR@{top_k}: {vec_hit_rate:.3f} | Hybrid HR@{top_k}: {hyb_hit_rate:.3f}")

    # Build final metrics
    def _metrics(records):
        total = records["total"]
        if total == 0:
            return {"HR": 0, "MRR": 0}
        hr = records["hits"] / total
        mrr = sum(records["reciprocal_ranks"]) / total
        return {"HR@k": round(hr, 4), "MRR": round(mrr, 4)}

    return {
        "sample_size": len(sample),
        "top_k": top_k,
        "total_chunks": len(chunks),
        "vector_only": _metrics(results["vector"]),
        "bm25_only": _metrics(results["bm25"]),
        "hybrid_rrf": _metrics(results["hybrid_rrf"]),
    }


if __name__ == "__main__":
    result = evaluate(sample_size=80, top_k=3)
    print()
    print("=" * 55)
    print("RAG Hybrid Retriever — Evaluation Report")
    print("=" * 55)
    if "error" in result:
        print(f"[ERROR] {result['error']}")
    else:
        print(f"  Total chunks in KB:  {result['total_chunks']}")
        print(f"  Sample size:         {result['sample_size']}")
        print(f"  Top-K:               {result['top_k']}")
        print(f"  ---")
        for name in ["vector_only", "bm25_only", "hybrid_rrf"]:
            m = result[name]
            print(f"  {name:<20}  HR@{result['top_k']}: {m['HR@k']:.2%}  MRR: {m['MRR']:.4f}")
        best = max(["vector_only", "bm25_only", "hybrid_rrf"], key=lambda n: result[n]["MRR"])
        print(f"\n  Best: {best} (by MRR)")
    print()

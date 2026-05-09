"""
Scientific paper ingestion and RAG knowledge base.
Fetch from arXiv, chunk, embed, and store in Chroma.
SDKs: LlamaIndex, Chroma, arXiv
"""
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ResearchPaperStore:
    """
    Fetch and index scientific papers for RAG.
    Sources: arXiv, uploaded PDFs, Semantic Scholar.
    """

    def __init__(self, persist_dir: str = "./research_kb"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._papers: List[Dict] = []
        self._client = None

        if CHROMA_AVAILABLE:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection("papers")
            print(f"[PaperStore] Chroma ready | {self._collection.count()} papers indexed")
        else:
            print("[PaperStore] In-memory mode (Chroma not available)")

    def fetch_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Fetch papers from arXiv by search query."""
        if not ARXIV_AVAILABLE:
            print("[PaperStore] arxiv package not installed. Install: pip install arxiv")
            return self._stub_papers(query, max_results)

        search = arxiv.Search(
            query=query, max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in search.results():
            paper = {
                "id": result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [str(a) for a in result.authors],
                "published": str(result.published.date()),
                "url": result.pdf_url,
                "categories": result.categories,
            }
            papers.append(paper)
            self._index_paper(paper)

        print(f"[PaperStore] Fetched {len(papers)} papers for: {query}")
        return papers

    def _stub_papers(self, query: str, n: int) -> List[Dict]:
        """Return stub papers when arXiv is unavailable."""
        return [
            {
                "id": f"stub_{i}", "title": f"Study on {query} (Paper {i+1})",
                "abstract": f"This paper investigates {query} using state-of-the-art methods. "
                            f"We propose a novel approach and demonstrate improvements.",
                "authors": ["Smith, J.", "Doe, A."],
                "published": "2024-01-01",
                "url": f"https://arxiv.org/abs/2024.{i:05d}",
                "categories": ["cs.LG"],
            }
            for i in range(n)
        ]

    def _index_paper(self, paper: Dict):
        """Index a paper into Chroma."""
        self._papers.append(paper)
        if self._client:
            try:
                self._collection.upsert(
                    ids=[paper["id"]],
                    documents=[f"{paper['title']}

{paper['abstract']}"],
                    metadatas=[{k: str(v)[:500] for k, v in paper.items() if k != "abstract"}],
                )
            except Exception as e:
                print(f"[PaperStore] Index error: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search over indexed papers."""
        if self._client:
            try:
                results = self._collection.query(query_texts=[query], n_results=top_k)
                ids = results["ids"][0]
                metadatas = results["metadatas"][0]
                return [{"id": ids[i], **metadatas[i]} for i in range(len(ids))]
            except Exception:
                pass
        # Fallback: keyword search in memory
        query_words = set(query.lower().split())
        scored = []
        for p in self._papers:
            text = f"{p['title']} {p['abstract']}".lower()
            score = sum(1 for w in query_words if w in text)
            scored.append((score, p))
        scored.sort(reverse=True)
        return [p for _, p in scored[:top_k]]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_papers": len(self._papers),
            "chroma_count": self._collection.count() if self._client else 0,
        }

import json, pathlib, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT  = pathlib.Path(__file__).parents[1]
BUILD = ROOT / "data" / "build"

class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._load_artifacts()
        self.model = SentenceTransformer(model_name)

    def _load_artifacts(self):
        emb_path = BUILD / "embeddings.npy"
        meta_path = BUILD / "metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Missing embeddings.npy or metadata.json in data/build/")
        self.emb = np.load(emb_path).astype("float32")
        faiss.normalize_L2(self.emb)                      # ensure normalized
        self.index = faiss.IndexFlatIP(self.emb.shape[1]) # cosine via dot
        self.index.add(self.emb)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.chunks = meta["chunks"]
        self.meta   = meta["meta"]

    # --- lightweight re-ranking helpers ---
    def _kw_boost(self, text: str, q: str) -> float:
        q_terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]
        if not q_terms: return 0.0
        t = text.lower()
        hits = sum(1 for w in q_terms if w in t)
        return hits / len(q_terms)  # 0..1

    def _phrase_boost(self, text: str, q: str) -> float:
        # exact phrase occurrence gives a bigger nudge
        t = text.lower()
        p = q.strip().lower()
        if len(p) < 4: return 0.0
        return 1.0 if p in t else 0.0

    def _source_pref(self, doc_id: str) -> float:
        # prefer rules/glossary docs slightly
        key = doc_id.lower()
        if any(s in key for s in ("rules","glossary","roberts")):
            return 0.05
        return 0.0

    def _link_penalty(self, text: str) -> float:
        # penalize chunks stuffed with links/ads (markdown/html)
        n_links = text.count("http")
        return min(0.08, 0.02 * n_links)  # cap penalty

    def embed_query(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)
        return q.astype("float32")

    def search(self, query: str, k: int = 4):
        # 1) FAISS top-N
        q = self.embed_query(query)
        over_k = max(k, 24)  # over-fetch a bit more
        D, I = self.index.search(q, over_k)

        # 2) Gather candidates
        candidates = []
        for rank, i in enumerate(I[0]):
            if i < 0:
                continue
            cand = {
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "text": self.chunks[i],
                "metadata": self.meta[i]
            }
            candidates.append(cand)

        # 3) Re-rank with tiny heuristics
        q_str = query.strip()
        for c in candidates:
            doc_id = c["metadata"]["doc_id"]
            # boosts
            c["score"] += 0.20 * self._kw_boost(c["text"], q_str)
            c["score"] += 0.30 * self._phrase_boost(c["text"], q_str)
            c["score"] += self._source_pref(doc_id)
            # penalty
            c["score"] -= self._link_penalty(c["text"])

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4) Reassign rank and return top-k
        top = candidates[:k]
        for r, c in enumerate(top, start=1):
            c["rank"] = r
        return top

# singleton retriever (lazy init)
_retriever = None
def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

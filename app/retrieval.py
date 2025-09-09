import os
# === RETRIEVER LOADER START ===
try:
    import faiss  # optional
except Exception:
    faiss = None

import json, numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "build"
_EMB_PATHS = [
    _DATA_DIR / "embeddings.fp16.npy",
    _DATA_DIR / "embeddings.npy",
]
_META_PATH = _DATA_DIR / "metadata.json"

_EMB = None
_META = None

def _lazy_load():
    global _EMB, _META
    if _EMB is None:
        for pth in _EMB_PATHS:
            if pth.exists():
                _EMB = np.load(pth, mmap_mode="r")
                break
        if _EMB is None:
            raise RuntimeError("No embeddings file found")
    if _META is None:
        with open(_META_PATH, "r") as f:
            _META = json.load(f)
    return _EMB, _META

def _cosine_topk(query_vec: np.ndarray, top_k: int = 5):
    E, _ = _lazy_load()
    step = int(os.getenv('POKERRAG_STEP', '2048'))
    q = query_vec.astype(np.float32, copy=False)
    q /= (np.linalg.norm(q) + 1e-9)

    n, d = E.shape
    step = int(os.getenv('POKERRAG_STEP', '2048'))
    best_scores = np.full(top_k, -1e9, dtype=np.float32)
    best_idx = np.full(top_k, -1, dtype=np.int32)

    for s in range(0, n, step):
        e = min(s + step, n)
        block = np.asarray(E[s:e], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True) + 1e-9
        block /= norms
        scores = block @ q
        part_idx = np.argpartition(scores, -top_k)[-top_k:]
        part_scores = scores[part_idx]
        all_scores = np.concatenate([best_scores, part_scores])
        all_idx = np.concatenate([best_idx, s + part_idx])
        keep = np.argpartition(all_scores, -top_k)[-top_k:]
        best_scores, best_idx = all_scores[keep], all_idx[keep]

    ord_ = np.argsort(-best_scores)
    return best_idx[ord_], best_scores[ord_]

def get_embeddings_and_meta():
    return _lazy_load()

def cosine_topk(query_vec: np.ndarray, k: int = 5):
    return _cosine_topk(query_vec, k)
# === RETRIEVER LOADER END ===

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
        self.emb = np.load(emb_path, mmap_mode='r').astype("float32")
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
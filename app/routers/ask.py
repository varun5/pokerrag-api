import time, uuid
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import AskRequest, AskResponse, Citation
from app.deps import get_settings, Settings
from app.retrieval import get_retriever
from app.prompt import build_prompt
from app.llm import LLMClient

router = APIRouter(tags=["ask"])

import re as _re

def _post_clean(txt: str) -> str:
    txt = _re.sub(r"\bSYSTEM:.*?\bUSER:", "", txt, flags=_re.S).strip()
    txt = _re.sub(r"\bSYSTEM:.*$", "", txt, flags=_re.S).strip()
    txt = _re.sub(r"\bUSER:.*$", "", txt, flags=_re.S).strip()
    return txt


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, settings: Settings = Depends(get_settings)) -> AskResponse:
    start = time.perf_counter_ns()
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    retriever = get_retriever()
    hits = retriever.search(q, k=req.top_k or 4)
    if not hits:
        raise HTTPException(status_code=404, detail="No passages available")

    prompt = build_prompt(q, hits)
    llm = LLMClient()
    answer = _post_clean(llm.generate(prompt))

    citations = [
        Citation(
            doc_id=h["metadata"].get("doc_id", f"doc{i+1}"),
            span=f"[{i+1}] " + h["metadata"].get("chunk_id", "chunk"),
        )
        for i, h in enumerate(hits)
    ]

    latency_ms = int((time.perf_counter_ns() - start) / 1_000_000)
    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=0.0,
        latency_ms=latency_ms,
        trace_id=str(uuid.uuid4()),
    )

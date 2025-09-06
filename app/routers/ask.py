import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import AskRequest, AskResponse, Citation
from app.deps import get_settings, Settings

router = APIRouter(tags=["ask"])

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, settings: Settings = Depends(get_settings)) -> AskResponse:
    start = time.perf_counter_ns()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    answer = (
        "[PokerRag Phase 0] Received your question. "
        "Retrieval and LLM orchestration arrive in later phases."
    )
    citations = [Citation(doc_id="poker_stub_kb_0", span="Phase 0 placeholder")]
    latency_ms = int((time.perf_counter_ns() - start) / 1_000_000)
    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=0.0,
        latency_ms=latency_ms,
        trace_id=str(uuid.uuid4())
    )

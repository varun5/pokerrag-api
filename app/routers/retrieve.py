# app/routers/retrieve.py
from fastapi import APIRouter, HTTPException
from app.schemas import RetrieveRequest, RetrieveResponse, RetrieveHit
from app.retrieval import get_retriever

router = APIRouter(tags=["retrieve"])

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    r = get_retriever()
    results = r.search(req.query, k=req.top_k)
    return RetrieveResponse(hits=[RetrieveHit(**h) for h in results])

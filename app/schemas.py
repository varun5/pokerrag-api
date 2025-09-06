from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str = Field("ok")

class AskRequest(BaseModel):
    query: str
    top_k: int = 4
    mode: Optional[str] = Field(default=None, description="e.g., 'multi_agent'")

class Citation(BaseModel):
    doc_id: str
    span: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    confidence: float = 0.0
    latency_ms: int = 0
    trace_id: str

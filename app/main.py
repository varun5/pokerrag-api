from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.deps import get_settings
from app.routers import health, ask

settings = get_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version, docs_url=settings.api_docs)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins == "*" else [o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)

@app.get("/")
async def root():
    return {"message": "PokerRag API is up", "docs": settings.api_docs}

# PokerRag API (Phase 0)

FastAPI scaffold for PokerRag. Endpoints:
- `GET /health`
- `POST /ask` (stub)

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000

import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_retrieve_basic():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/retrieve", json={"query": "What is a string bet?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data and len(data["hits"]) > 0
    # scores should be floats in [0,1] (cosine-ish inner product)
    assert all(0.0 <= h["score"] <= 1.0 for h in data["hits"])

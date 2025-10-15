import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_text2img_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        data = {"mode": "text2image", "text": "a cute cat"}
        response = await ac.post("/http/inference", data=data)

    print(response.text)
    assert response.status_code == 200

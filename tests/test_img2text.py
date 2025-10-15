import io
from PIL import Image
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_img2text_endpoint():
    # ✅ 改成更大的尺寸
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {"image": ("test.png", buf, "image/png")}
        data = {"mode": "image2text"}
        response = await ac.post("/http/inference", data=data, files=files)

    print(response.text)
    assert response.status_code == 200
    assert "text" in response.json() or "status" in response.json()

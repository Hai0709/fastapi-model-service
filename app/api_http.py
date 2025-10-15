from fastapi import APIRouter, UploadFile, File, Form
from app.queue_manager import queue
from app.models.text2image import generate_image
from app.models.image2text import describe_image
from app.schemas import ImageResponse, TextResponse
import os

router = APIRouter()

import base64
from app.schemas import ImageResponse, TextResponse

@router.post("/inference")
async def inference_endpoint(
    text: str = Form(""),
    image: UploadFile | None = File(None),
    mode: str = Form("text2image"),
) -> dict:
    async def job():
        if mode == "text2image":
            image_bytes = await generate_image(text)
            import base64
            return {"status": "ok", "image": base64.b64encode(image_bytes).decode("utf-8")}
        elif mode == "image2text":
            if not image:
                return {"error": "missing image file"}
            image_bytes = await image.read()
            text_out = await describe_image(image_bytes, text or "请描述这张图片")
            return {"status": "ok", "text": text_out}
        else:
            return {"error": "invalid mode"}
    return await queue.submit(job)


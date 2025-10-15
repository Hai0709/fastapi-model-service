from fastapi import APIRouter, WebSocket
from app.queue_manager import queue
from app.models.text2image import generate_image
from app.models.image2text import describe_image
import base64

router = APIRouter()

@router.websocket("/inference")
async def inference_ws(ws: WebSocket) -> None:
    #gpt写的就是比我写得好，异步逻辑总是错。
    await ws.accept()

    try:
        data = await ws.receive_json()
        mode = data.get("mode", "text2image") 

        async def job():
            try:
                if mode == "text2image":
                    text = data.get("text", "")
                    image_bytes = await generate_image(text)
                    await ws.send_json({
                        "status": "ok",
                        "mode": "text2image",
                        "image": base64.b64encode(image_bytes).decode("utf-8"),
                    })
                elif mode == "image2text":
                    img_b64 = data.get("image")
                    if not img_b64:
                        await ws.send_json({"status": "error", "msg": "missing image"})
                        return
                    img_bytes = base64.b64decode(img_b64)
                    text_prompt = data.get("text", "")
                    text_out = await describe_image(img_bytes, text_prompt)
                    await ws.send_json({
                        "status": "ok",
                        "mode": "image2text",
                        "text": text_out,
                    })
                else:
                    await ws.send_json({"status": "error", "msg": "invalid mode"})
            except Exception as e:
                await ws.send_json({"status": "error", "msg": str(e)})

        # 提交任务到队列
        await queue.submit(job)
    except Exception as e:
        await ws.send_json({"status": "error", "msg": f"invalid request: {e}"})
    finally:
        await ws.close()

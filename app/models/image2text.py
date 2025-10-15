import os
import asyncio
import io
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration



DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/qwen2.5-vl-3b-instruct"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

#感觉每次跑的时候都要load有点蠢，后面看看能不能设计一个全局加载？
async def _load_model():
    print(f"加载服务器部署的模型：{MODEL_PATH}")
    local_mode = os.path.exists(MODEL_PATH)
    if local_mode:
        print(f"使用本地模型路径：{MODEL_PATH}")
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
    else:
        hf_token = os.getenv("HF_TOKEN", None)
        print("加载模型")
        processor = AutoProcessor.from_pretrained(
            DEFAULT_MODEL_ID,
            token=hf_token,
            use_fast=True,
            trust_remote_code=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULT_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        ).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    return processor, model

async def describe_image(image_bytes: bytes, user_prompt: str):
    processor, model = await _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    #3090，图太大不行，实验室太穷了，原图大小跑不起来
    max_side = 640
    w, h = image.size
    scale = max(w, h) / max_side
    if scale > 1:
        new_w, new_h = int(w / scale), int(h / scale)
        image = image.resize((new_w, new_h))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]
    }]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, images=[image], return_tensors="pt").to(model.device)
    print(f"Processor Keys: {list(inputs.keys())}")
    print("开始生成...", flush=True)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,     
            do_sample=False
        )

    print("decoder", flush=True)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("输出：", output)
    return output.strip()


async def main():
    image_path = os.getenv(
        "TEST_IMAGE_PATH",
        "/home/cxq/LLM_ER/fastapi_model_service/app/models/test.jpg"
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"测试图片不存在：{image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    await describe_image(image_bytes, "请详细描述这张图片的内容。")


if __name__ == "__main__":
    asyncio.run(main())

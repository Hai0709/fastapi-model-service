import os
import io
import asyncio
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/qwen2.5-vl-3b-instruct"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_processor: AutoProcessor | None = None
_model: Qwen2_5_VLForConditionalGeneration | None = None
_model_lock = asyncio.Lock()


async def _load_model():
    global _processor, _model

    async with _model_lock:
        if _model is not None and _processor is not None:
            return _processor, _model

        print(f"加载视觉语言模型：{MODEL_PATH}")
        local_mode = os.path.exists(MODEL_PATH)

        if local_mode:
            _processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
            ).eval()
        else:
            hf_token = os.getenv("HF_TOKEN", None)
            _processor = AutoProcessor.from_pretrained(
                DEFAULT_MODEL_ID, token=hf_token, trust_remote_code=True
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                DEFAULT_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            ).eval()

        print(f"模型加载完成，全局存放 { 'CUDA' if torch.cuda.is_available() else 'CPU' }")
        return _processor, _model


async def describe_single_image(image_path: str, user_prompt: str) -> tuple[str, str]:
    try:
        processor, model = await _load_model()
        device = model.device

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 限制尺寸，防止 OOM
        max_side = 640
        w, h = image.size
        scale = max(w, h) / max_side
        if scale > 1:
            image = image.resize((int(w / scale), int(h / scale)))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        }]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=input_text, images=[image], return_tensors="pt").to(device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return os.path.basename(image_path), output
    except Exception as e:
        print(f"图片 {image_path} 处理失败: {e}")
        return os.path.basename(image_path), f"【失败】：{e}"


async def describe_images_in_folder(folder_path: str, prompt: str, concurrency: int = 4) -> dict[str, str]:
    #concurrency是为了限制峰值
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    if not all_files:
        raise FileNotFoundError(f"文件夹中未检测到有效图片：{folder_path}")
    print(f"待处理图片数量：{len(all_files)} 张")
    semaphore = asyncio.Semaphore(concurrency)
    async def sem_task(img_path):
        async with semaphore:
            return await describe_single_image(img_path, prompt)
    tasks = [sem_task(p) for p in all_files]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    result_dict = {fname: text for fname, text in results}
    return result_dict

async def main():
    folder_path = "/home/cxq/LLM_ER/fastapi_model_service/app/models/batch_test"
    prompt = "请详细描述这张图片的主要内容和视觉风格。"
    results = await describe_images_in_folder(folder_path, prompt, concurrency=3)
    output_path = os.path.join(folder_path, "descriptions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for fname, desc in results.items():
            f.write(f"{fname}:\n{desc}\n\n")
    print(f"已保存描述结果至 {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

import os
import io
import asyncio
import torch
from PIL import Image
from diffusers import HunyuanDiTPipeline

DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/hunyuan_dit"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
_model: HunyuanDiTPipeline | None = None
_model_lock = asyncio.Lock()


async def _load_model() -> HunyuanDiTPipeline:
    global _model #我感觉设置这个简单的全局，就可以避免每次都加载
    async with _model_lock:    #GPT写的，model_lock，学到了
        if _model is not None:
            return _model

        print(f"加载模型：{MODEL_PATH}")
        _model = HunyuanDiTPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device).to(memory_format=torch.channels_last)

        if hasattr(_model, "enable_attention_slicing"):
            _model.enable_attention_slicing()
        if hasattr(_model, "enable_vae_tiling"):
            _model.enable_vae_tiling()
        print(f"全局加载模型 {device}")
        return _model


async def generate_images(prompts: list[str]) -> list[bytes]:
    model = await _load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = model(
        prompt=prompts,
        num_inference_steps=20, #我看网上说这个步数是可以调优的，具体还没尝试
        guidance_scale=7.5,
        width=256,  #3090太垃圾了，小点
        height=256,
        generator=torch.Generator(device).manual_seed(42),
    )

    images = result.images
    print("batch done")
    bytes_list = []
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bytes_list.append(buf.getvalue())
    return bytes_list


async def main():
    prompts = [
        "夜晚的赛博朋克城市，有霓虹灯和飞行汽车",
        "阳光下的乡村稻田，远处有风车",
        "宇航员在星球上骑马",
    ]
    imgs_bytes = await generate_images(prompts)
    output_dir = "/home/cxq/LLM_ER/fastapi_model_service/app/models/batch_test"
    os.makedirs(output_dir, exist_ok=True)
    for i, b in enumerate(imgs_bytes):
        out_path = os.path.join(output_dir, f"output_{i}.png")
        with open(out_path, "wb") as f:
            f.write(b)
        print(f"已保存 {out_path}")

if __name__ == "__main__":
    asyncio.run(main())

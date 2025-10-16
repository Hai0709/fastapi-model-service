import os
import asyncio
import io
import torch
from PIL import Image
from diffusers import HunyuanDiTPipeline

DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/hunyuan_dit"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

_model: HunyuanDiTPipeline | None = None


async def _load_model() -> HunyuanDiTPipeline:
    """异步加载"""
    global _model
    if _model is not None:
        return _model
    print(f"正在从 {MODEL_PATH} 加载文生图模型...")

    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型目录不存在：{MODEL_PATH}\n"
            "请设置环境变量 MODEL_PATH 为模型所在目录，如：\n"
            "  export MODEL_PATH=/app/Model/hunyuan_dit"
        )


    _model = HunyuanDiTPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        offload_state_dict=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device).to(memory_format=torch.channels_last)

  
    if hasattr(_model, "enable_attention_slicing"):
        _model.enable_attention_slicing()
    if hasattr(_model, "enable_vae_tiling"):
        _model.enable_vae_tiling()

    print(f"模型已加载到 {device}")
    return _model


async def generate_image(prompt: str) -> bytes:
    model = await _load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"开始生成图像：{prompt}")
    result = model(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        width=256,
        height=256,
        generator=torch.Generator(device).manual_seed(42),
    )

    image = result.images[0]
    print("图像生成完成")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


async def main():
    test_prompt = "夜晚的赛博朋克城市，有霓虹灯和飞行汽车"
    img_bytes = await generate_image(test_prompt)

    output_path = os.getenv(
        "OUTPUT_PATH",
        "/home/cxq/LLM_ER/fastapi_model_service/app/models/test_output.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(img_bytes)

    print(f"图片已保存到：{output_path}")


if __name__ == "__main__":
    asyncio.run(main())


import os
import asyncio
import io
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.profiler import profile, ProfilerActivity


DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/qwen2.5-vl-3b-instruct"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def print_gpu_status(tag=""):
    if not torch.cuda.is_available():
        print("CUDA 不可用！")
        return

    torch.cuda.synchronize()
    gpu_count = torch.cuda.device_count()
    print(f"\n========={tag} GPU 状态 ({gpu_count} 张卡) =========")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**2  # MB
        alloc_mem = torch.cuda.memory_allocated(i) / 1024**2
        peak_mem = torch.cuda.max_memory_allocated(i) / 1024**2
        percent = (alloc_mem / total_mem) * 100
        bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))
        print(f"GPU{i:<2} | {props.name:<25} | 当前 {alloc_mem:8.1f} MB | 峰值 {peak_mem:8.1f} MB "
              f"| 利用率 [{bar}] {percent:5.1f}%")
    print("========================================================\n")

async def _load_model():
    print(f"正在加载模型：{MODEL_PATH}")
    local_mode = os.path.exists(MODEL_PATH)
    model_id = MODEL_PATH if local_mode else DEFAULT_MODEL_ID

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_mode,
        use_fast=True,
        trust_remote_code=True,
    )

    # 自适应多 GPU 显存分配
    gpu_count = torch.cuda.device_count()
    max_memory = {i: "22GiB" for i in range(gpu_count)} if gpu_count > 1 else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=local_mode,
        device_map="auto",
        max_memory=max_memory,
    ).eval()

    print("模型加载完成（device_map=auto 多GPU分布）")
    print_gpu_status("模型加载后")

    # 单 GPU 才启用 torch.compile
    if torch.cuda.device_count() == 1:
        try:
            model = torch.compile(model, backend="inductor", mode="max-autotune")
            print("单卡模式启用 torch.compile() 提升性能")
        except Exception as e:
            print(f"torch.compile 失败，继续使用原模型：{e}")
    else:
        print("多 GPU 模式启用 device_map 自动分配")
    return processor, model

def apply_channels_last(inputs):
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.ndim == 4:
            inputs[k] = v.contiguous(memory_format=torch.channels_last)
            print(f"✅ 输入 {k} 已切换为 channels_last 格式")
    return inputs

async def describe_image_with_profiler(image_bytes: bytes, user_prompt: str):
    processor, model = await _load_model()

    # 图片预处理
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_side = 512
    w, h = image.size
    scale = max(w, h) / max_side
    if scale > 1:
        image = image.resize((int(w / scale), int(h / scale)))
        print(f"🪄 图片缩放: {w, h} → {image.size}")

    # 输入构造
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": user_prompt}
    ]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, images=[image], return_tensors="pt")

    # 放到 GPU
    for k, v in inputs.items():
        if torch.is_tensor(v):
            device = list(model.hf_device_map.values())[0] if hasattr(model, "hf_device_map") else torch.device("cuda:0")
            if v.dtype in (torch.long, torch.int):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v.to(device, dtype=torch.float16)
    inputs = apply_channels_last(inputs)
    print_gpu_status("输入准备完成")

    # Warm-up
    print("Warm-up 中 ...")
    try:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()
        print("Warm-up 成功")
    except Exception as e:
        print(f"Warm-up 失败: {e}")


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
    start_evt.record()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)

    print_gpu_status("推理结束后")
    print(f"GPU 端推理耗时: {elapsed_ms:.2f} ms")
    decoded = processor.batch_decode(output, skip_special_tokens=True)
    print(f"模型输出: {decoded}")

    print("启动 GPU Profiler (仅捕获 CUDA 活动)...")
    log_dir = "./log/qwen_multi_gpu_device_map"
    os.makedirs(log_dir, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CUDA],  # 仅看 GPU
        record_shapes=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()

    print("GPU Profiler 完成")
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


async def main():
    test_image = os.getenv("TEST_IMAGE_PATH", "/home/cxq/LLM_ER/fastapi_model_service/app/models/test.jpg")
    if not os.path.exists(test_image):
        raise FileNotFoundError(f"测试图片不存在: {test_image}")
    with open(test_image, "rb") as f:
        image_bytes = f.read()
    await describe_image_with_profiler(image_bytes, "请详细描述这张图片的主要内容。")


if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
import io
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.profiler import profile, ProfilerActivity, schedule

DEFAULT_MODEL_PATH = "/home/cxq/LLM_ER/fastapi_model_service/Model/qwen2.5-vl-3b-instruct"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"



def get_self_cuda_time_us(e):
    """兼容不同Torch版本, 获取单算子自身GPU耗时(μs)"""
    if hasattr(e, "self_cuda_time_total"):
        return e.self_cuda_time_total
    if hasattr(e, "self_cuda_time_total_us"):
        return e.self_cuda_time_total_us
    if hasattr(e, "cuda_time"):
        return e.cuda_time
    if hasattr(e, "cuda_time_total"):
        return e.cuda_time_total
    return 0.0

def print_gpu_status(tag=""):
    """打印当前/峰值显存"""
    if not torch.cuda.is_available():
        print("CUDA 不可用！")
        return
    torch.cuda.synchronize()
    mem_now = torch.cuda.memory_allocated() / 1024**2
    mem_max = torch.cuda.max_memory_allocated() / 1024**2
    print(f"📊 {tag} 显存占用: 当前 {mem_now:.1f} MB | 峰值 {mem_max:.1f} MB | 设备: {torch.cuda.get_device_name(0)}")



async def _load_model():
    """加载模型和Processor"""
    print(f"🚀 正在加载模型：{MODEL_PATH}")
    local_mode = os.path.exists(MODEL_PATH)
    model_id = MODEL_PATH if local_mode else DEFAULT_MODEL_ID

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_mode,
        use_fast=True,
        trust_remote_code=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=local_mode,
        trust_remote_code=True,
        device_map=None,
    ).eval().to("cuda")

    print("模型加载完成。")
    print_gpu_status("模型加载后")

    for name, p in list(model.named_parameters())[:5]:
        print(f"🔹 {name} -> {p.device}")

    return processor, model

def profile_model_by_layer(model: torch.nn.Module, inputs):
    results = []

    def make_hook(layer_name):
        def hook_fn(module, inp, outp):
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            t0 = time.time()
            torch.cuda.synchronize()
            # forward结束时会触发hook
            t1 = time.time()
            mem_after = torch.cuda.memory_allocated()
            results.append({
                "layer": layer_name,
                "time_ms": (t1 - t0) * 1000,
                "mem_MB": (mem_after - mem_before) / 1024 ** 2
            })
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.count(".") == 2:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()

    for h in hooks:
        h.remove()

    for r in sorted(results, key=lambda x: -x["time_ms"]):
        print(f"{r['layer']:<40s} time={r['time_ms']:.3f} ms | Δmem={r['mem_MB']:.1f} MB")

    return results

async def describe_image_with_profiler(image_bytes: bytes, user_prompt: str):
    processor, model = await _load_model()
    device = next(model.parameters()).device

    # 处理图片
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_side = 512
    w, h = image.size
    scale = max(w, h) / max_side
    if scale > 1:
        new_w, new_h = int(w / scale), int(h / scale)
        image = image.resize((new_w, new_h))
        print(f"🪄 已缩放图片 ({w},{h})→({new_w},{new_h})")

    # 构建输入
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, images=[image], return_tensors="pt")

    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.long, torch.int):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v.to(device, dtype=torch.float16)

    print_gpu_status("输入准备后")
    print("\n开始Layer级GPU分析...")
    _ = profile_model_by_layer(model, inputs)
    print("\nLayer级分析完成。\n")
    print("启动profiler 采样...")
    log_dir = "./log/qwen_profiler"
    os.makedirs(log_dir, exist_ok=True)
    prof_schedule = schedule(wait=0, warmup=1, active=1, repeat=1)
    t1 = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
    t2 = time.time()
    print(f" profiling 完成，用时 {t2 - t1:.2f}s")
    print_gpu_status("推理结束后")

    # 打印主要kernel事件
    kernel_events = [e for e in prof.key_averages() if get_self_cuda_time_us(e) > 0]
    kernel_events = sorted(kernel_events, key=lambda e: get_self_cuda_time_us(e), reverse=True)
    for i, evt in enumerate(kernel_events[:20], 1):
        self_cuda_ms = get_self_cuda_time_us(evt) / 1000.0
        print(f"{i:02d}. {evt.key:<60s} self CUDA = {self_cuda_ms:.3f} ms | calls = {evt.count}")



async def main():
    test_image = os.getenv(
        "TEST_IMAGE_PATH",
        "/home/cxq/LLM_ER/fastapi_model_service/app/models/test.jpg"
    )
    if not os.path.exists(test_image):
        raise FileNotFoundError(f"测试图片不存在：{test_image}")
    with open(test_image, "rb") as f:
        image_bytes = f.read()
    await describe_image_with_profiler(image_bytes, "请详细描述这张图片的主要内容。")


if __name__ == "__main__":
    asyncio.run(main())


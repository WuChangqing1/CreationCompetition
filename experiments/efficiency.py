"""Model efficiency measurements with repeated latency sampling."""

import io
import time

import torch


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def measure_torch_efficiency(model, forward_call, device, warmup=10, measure=50):
    device = torch.device(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            forward_call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(measure):
            forward_call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000 / max(measure, 1)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else "N/A"
    )
    return {
        "Parameters": count_parameters(model),
        "Model_Size_MB": model_size_mb(model),
        "Inference_Latency_ms": elapsed_ms,
        "Peak_VRAM_MB": peak_vram,
    }

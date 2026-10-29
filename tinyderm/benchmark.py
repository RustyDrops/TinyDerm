"""CPU inference benchmarking module for TinyDerm Lab using ONNX Runtime."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import onnxruntime as ort
import psutil

logger = logging.getLogger(__name__)


def benchmark_onnx_cpu(
    model_path: str | Path,
    num_warmup: int = 20,
    num_passes: int = 200,
    single_thread: bool = False,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Measures single-image CPU latency, p50, p95, and throughput."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    opts = ort.SessionOptions()
    if single_thread:
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    # Cold latency (first execution)
    t_start = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    cold_latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Warm-up passes
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Timed passes
    latencies = []
    mem_before = psutil.Process().memory_info().rss

    t_total_start = time.perf_counter()
    for _ in range(num_passes):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total_time_sec = time.perf_counter() - t_total_start

    mem_after = psutil.Process().memory_info().rss
    peak_ram_mb = round(mem_after / (1024 * 1024), 2)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = float(num_passes / max(total_time_sec, 0.001))

    return {
        "model_file": model_path.name,
        "single_thread": single_thread,
        "num_passes": num_passes,
        "cold_latency_ms": round(cold_latency_ms, 3),
        "mean_latency_ms": round(mean_lat, 3),
        "p50_latency_ms": round(p50_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "throughput_images_per_sec": round(throughput, 2),
        "peak_ram_mb": peak_ram_mb,
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
    }


def run_full_cpu_benchmark(
    fp32_model_path: str | Path,
    int8_model_path: Optional[str | Path] = None,
    num_passes: int = 200,
) -> Dict[str, Any]:
    """Runs single-thread and default-thread benchmarks for FP32 and INT8 models."""
    results = {}

    # FP32 benchmarks
    results["fp32_multithread"] = benchmark_onnx_cpu(fp32_model_path, num_passes=num_passes, single_thread=False)
    results["fp32_singlethread"] = benchmark_onnx_cpu(fp32_model_path, num_passes=num_passes, single_thread=True)

    # INT8 benchmarks
    if int8_model_path and Path(int8_model_path).exists():
        results["int8_multithread"] = benchmark_onnx_cpu(int8_model_path, num_passes=num_passes, single_thread=False)
        results["int8_singlethread"] = benchmark_onnx_cpu(int8_model_path, num_passes=num_passes, single_thread=True)

    return results

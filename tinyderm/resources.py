"""System resource monitor for TinyDerm Lab."""

import os
import shutil
import threading
import time
from typing import Any, Dict, Optional
import psutil
import torch


def detect_hardware() -> Dict[str, Any]:
    """Detect local CPU, RAM, accelerator, OS, and PyTorch details."""
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    gpu_info = None
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        gpu_info = {
            "type": "cuda",
            "name": gpu_name,
            "memory_gb": gpu_mem_gb,
            "device_count": torch.cuda.device_count(),
        }
    elif has_mps:
        gpu_info = {"type": "mps", "name": "Apple Silicon MPS"}

    return {
        "os": os.name,
        "platform": psutil.os.name,
        "cpu_logical": cpu_count_logical,
        "cpu_physical": cpu_count_physical,
        "ram_total_gb": ram_gb,
        "accelerator": gpu_info["type"] if gpu_info else "cpu",
        "accelerator_details": gpu_info,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if has_cuda else None,
    }


def sample_current_resources() -> Dict[str, float]:
    """Capture a snapshot of CPU, RAM, GPU, and disk metrics."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    ram_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    gpu_util = 0.0
    gpu_mem_used_mb = 0.0

    if torch.cuda.is_available():
        try:
            gpu_mem_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            # Try to query pynvml if available, else torch memory
            gpu_util = float(torch.cuda.utilization(0)) if hasattr(torch.cuda, "utilization") else 0.0
        except Exception:
            pass

    disk_info = shutil.disk_usage(os.getcwd())
    disk_used_gb = round((disk_info.total - disk_info.free) / (1024**3), 2)

    return {
        "cpu_percent": float(cpu_percent),
        "ram_percent": float(ram_percent),
        "ram_rss_mb": float(ram_rss_mb),
        "gpu_util_percent": float(gpu_util),
        "gpu_mem_used_mb": float(gpu_mem_used_mb),
        "disk_used_gb": float(disk_used_gb),
    }


class ResourceMonitor:
    """Background thread that regularly samples resources and writes to ExperimentDB."""

    def __init__(self, db, run_id: str, interval_sec: float = 2.0):
        self.db = db
        self.run_id = run_id
        self.interval_sec = interval_sec
        self.current_stage = "SETUP"
        self.images_per_sec = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_stage(self, stage: str):
        self.current_stage = stage

    def set_throughput(self, img_per_sec: float):
        self.images_per_sec = img_per_sec

    def _run(self):
        while not self._stop_event.is_set():
            try:
                metrics = sample_current_resources()
                self.db.log_resource_sample(
                    run_id=self.run_id,
                    stage=self.current_stage,
                    cpu_percent=metrics["cpu_percent"],
                    ram_percent=metrics["ram_percent"],
                    ram_rss_mb=metrics["ram_rss_mb"],
                    gpu_util_percent=metrics["gpu_util_percent"],
                    gpu_mem_used_mb=metrics["gpu_mem_used_mb"],
                    disk_used_gb=metrics["disk_used_gb"],
                    images_per_sec=float(self.images_per_sec),
                )
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

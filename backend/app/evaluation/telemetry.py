import os
import time
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class TelemetrySnapshot:
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    peak_ram_mb: float
    gpu_vram_gb: float
    cpu_percent: float


class TelemetryTracker:
    """Monitors hardware consumption (RAM, VRAM, CPU) and sub-component latencies."""

    def __init__(self):
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

    def get_ram_mb(self) -> float:
        if self.process:
            try:
                return self.process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def get_cpu_percent(self) -> float:
        if self.process:
            try:
                return self.process.cpu_percent()
            except Exception:
                pass
        return 0.0

    def get_gpu_vram_gb(self) -> float:
        # For our Zero-GPU / Cloud-Native Architecture, GPU VRAM is 0.0 GB
        return 0.0

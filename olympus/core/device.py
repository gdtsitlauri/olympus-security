"""GPU/CPU device management optimized for GTX 1650 (4GB VRAM)."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from olympus.core.config import CONFIG
from olympus.core.logger import get_logger

log = get_logger("device")

_DEVICE: Optional[object] = None


def get_device() -> "torch.device":  # type: ignore[name-defined]
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    if not _TORCH_AVAILABLE:
        log.warning("PyTorch not available — all computation on CPU")
        _DEVICE = "cpu"
        return _DEVICE  # type: ignore[return-value]

    import torch

    if CONFIG.gpu.enabled and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("CUDA device: %s | VRAM: %.1f GB | Budget: %.1f GB",
                 torch.cuda.get_device_name(0), vram, CONFIG.gpu.vram_budget_gb)
        _DEVICE = torch.device("cuda")
    else:
        log.info("Running on CPU (GPU disabled or unavailable)")
        _DEVICE = torch.device("cpu")

    return _DEVICE  # type: ignore[return-value]


def vram_available_gb() -> float:
    if not _TORCH_AVAILABLE:
        return 0.0
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return 0.0


def model_fits_in_vram(param_count: int, bytes_per_param: int = 4) -> bool:
    """True if model fits within GTX 1650 budget."""
    required_gb = (param_count * bytes_per_param) / 1e9
    return required_gb <= CONFIG.gpu.vram_budget_gb


def clear_gpu_cache() -> None:
    if _TORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

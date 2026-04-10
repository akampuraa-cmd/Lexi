"""
device_manager.py — Lexi Hardware Detection & Device Management

Automatically detects and selects the best available compute device
from the following backends (in priority order):

1. **CUDA** (NVIDIA GPUs) — including multi-GPU detection
2. **ROCm** (AMD GPUs via PyTorch's HIP/ROCm backend, reported as CUDA)
3. **Intel XPU** (Intel Arc / Data Center GPUs via Intel Extension for PyTorch)
4. **Apple MPS** (Metal Performance Shaders on Apple Silicon)
5. **CPU** fallback

Usage::

    from device_manager import get_best_device, get_device_summary

    device = get_best_device()          # torch.device
    summary = get_device_summary()      # human-readable string
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Device information container
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """Describes a single detected compute device."""

    name: str               # e.g. "NVIDIA GeForce RTX 4090"
    backend: str            # "cuda", "xpu", "mps", "cpu"
    index: int | None = None  # GPU index (None for CPU / MPS)
    total_memory_mb: int = 0  # VRAM in MiB (0 if unknown)
    driver_version: str = ""

    @property
    def torch_device(self) -> torch.device:
        if self.index is not None:
            return torch.device(self.backend, self.index)
        return torch.device(self.backend)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_cuda_devices() -> list[DeviceInfo]:
    """Detect NVIDIA CUDA (or AMD ROCm, which also reports as CUDA)."""
    if not torch.cuda.is_available():
        return []

    devices: list[DeviceInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # ROCm devices also appear through torch.cuda
        backend_label = "cuda"
        driver = ""
        try:
            driver = torch.version.cuda or ""
        except Exception:
            pass
        # Check if this is actually ROCm
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            driver = f"ROCm {hip_version}"

        devices.append(DeviceInfo(
            name=props.name,
            backend=backend_label,
            index=i,
            total_memory_mb=props.total_memory // (1024 * 1024),
            driver_version=driver,
        ))
    return devices


def _detect_xpu_devices() -> list[DeviceInfo]:
    """Detect Intel XPU devices (Arc, Data Center GPUs)."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return []

    devices: list[DeviceInfo] = []
    for i in range(torch.xpu.device_count()):
        name = torch.xpu.get_device_name(i)
        mem = 0
        try:
            props = torch.xpu.get_device_properties(i)
            mem = props.total_memory // (1024 * 1024)
        except Exception:
            pass

        devices.append(DeviceInfo(
            name=name,
            backend="xpu",
            index=i,
            total_memory_mb=mem,
        ))
    return devices


def _detect_mps_device() -> list[DeviceInfo]:
    """Detect Apple Metal Performance Shaders (Apple Silicon)."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return []

    chip = platform.processor() or "Apple Silicon"
    return [DeviceInfo(
        name=f"Apple MPS ({chip})",
        backend="mps",
    )]


def _detect_cpu() -> DeviceInfo:
    """Always-available CPU fallback."""
    name = platform.processor() or platform.machine() or "CPU"
    return DeviceInfo(name=f"CPU ({name})", backend="cpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_all_devices() -> list[DeviceInfo]:
    """Return a list of **all** detected compute devices.

    The list is ordered by priority: CUDA/ROCm first, then XPU,
    then MPS, and finally CPU.
    """
    devices: list[DeviceInfo] = []
    devices.extend(_detect_cuda_devices())
    devices.extend(_detect_xpu_devices())
    devices.extend(_detect_mps_device())
    devices.append(_detect_cpu())
    return devices


def get_best_device() -> torch.device:
    """Select and return the single best ``torch.device``.

    Priority: CUDA (highest-VRAM GPU) → XPU → MPS → CPU.
    """
    all_devs = detect_all_devices()

    # Filter by backend priority
    for backend in ("cuda", "xpu", "mps"):
        candidates = [d for d in all_devs if d.backend == backend]
        if candidates:
            # Pick the one with the most VRAM (or first if unknown)
            best = max(candidates, key=lambda d: d.total_memory_mb)
            return best.torch_device

    # Fallback to CPU
    return torch.device("cpu")


def get_device_summary() -> str:
    """Return a human-readable multi-line summary of all detected hardware."""
    all_devs = detect_all_devices()
    best = get_best_device()

    lines = ["Detected compute devices:"]
    for dev in all_devs:
        marker = " ★" if dev.torch_device == best else ""
        mem_str = f" | {dev.total_memory_mb:,} MiB VRAM" if dev.total_memory_mb else ""
        drv_str = f" | Driver: {dev.driver_version}" if dev.driver_version else ""
        lines.append(f"  • [{dev.backend.upper()}] {dev.name}{mem_str}{drv_str}{marker}")

    lines.append(f"\nSelected device: {best}")
    return "\n".join(lines)


# Module-level device — replaces the old ``model.DEVICE`` global.
DEVICE = get_best_device()

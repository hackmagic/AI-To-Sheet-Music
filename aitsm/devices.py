"""计算设备探测：CPU / CUDA / MPS 能不能用，能用到什么程度。

为什么单独一个模块
------------------
前端要在上传前就让用户选「CPU 还是 GPU」。这件事有两个坑：

1. **不能靠 ``torch.cuda.is_available()`` 判断就完事**：torch 可能是 CPU 版
   （编译时就没带 CUDA），此时 ``is_available()`` 返回 False 是对的，但用户需要
   知道「装了 CUDA 卡但 torch 是 CPU 版」和「根本没卡」是两回事；
2. **探测要快且不占显存**：这个函数会被 ``/api/health`` 每次调用，绝不能加载模型。

因此这里只读设备属性，不做任何张量分配。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

__all__ = ["DEVICE_CHOICES", "detect_devices", "resolve_device", "device_note"]

#: 前端下拉框用的取值（与 pipeline 的 ``device`` 选项一致）
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "cuda")


def _torch_info() -> dict[str, Any]:
    """轻量读 torch 的设备信息（不建张量、不初始化 CUDA 上下文）。"""
    info: dict[str, Any] = {
        "installed": False,
        "version": "",
        "cuda_build": False,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "reason": "",
    }
    try:
        import torch  # noqa: PLC0415 - 惰性：health 之外的地方不该被拖慢
    except Exception as exc:  # pragma: no cover - 无 torch 环境
        info["reason"] = f"未安装 torch（{type(exc).__name__}）"
        return info

    info["installed"] = True
    info["version"] = str(getattr(torch, "__version__", ""))
    info["cuda_build"] = bool(getattr(torch.version, "cuda", None))

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["reason"] = f"CUDA 探测失败（{type(exc).__name__}: {exc}）"
        return info

    if info["cuda_available"]:
        try:
            info["device_count"] = int(torch.cuda.device_count())
            for index in range(info["device_count"]):
                props = torch.cuda.get_device_properties(index)
                info["devices"].append(
                    {
                        "index": index,
                        "name": str(getattr(props, "name", "GPU")),
                        "total_memory_gb": round(
                            float(getattr(props, "total_memory", 0)) / 1e9, 2
                        ),
                        "capability": f"{getattr(props, 'major', 0)}.{getattr(props, 'minor', 0)}",
                    }
                )
        except Exception as exc:  # pragma: no cover - 驱动异常
            info["reason"] = f"读取 GPU 属性失败（{type(exc).__name__}: {exc}）"
    else:
        if not info["cuda_build"]:
            info["reason"] = (
                f"当前 torch {info['version']} 是 CPU 版（未编译 CUDA）；"
                "要启用 GPU 需安装对应的 CUDA 版 torch"
            )
        else:
            info["reason"] = "torch 编译了 CUDA，但驱动/显卡不可用"

    # Apple Silicon
    try:
        backend = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(backend and backend.is_available())
    except Exception:  # pragma: no cover
        info["mps_available"] = False
    return info


def _nvidia_smi() -> dict[str, Any]:
    """没装 torch 时也能判断机器有没有 NVIDIA 卡（给用户更准的提示）。"""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"present": False, "devices": []}
    try:
        output = subprocess.run(
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover
        return {"present": False, "devices": []}
    if output.returncode != 0:
        return {"present": False, "devices": []}
    devices = []
    for line in (output.stdout or "").strip().splitlines():
        name, _, memory = line.partition(",")
        devices.append({"name": name.strip(), "memory": memory.strip()})
    return {"present": bool(devices), "devices": devices}


def detect_devices() -> dict[str, Any]:
    """当前机器能用什么设备。给 ``/api/health`` 与前端下拉框用。"""
    torch_info = _torch_info()
    smi = _nvidia_smi()

    gpu_usable = bool(torch_info["cuda_available"])
    if gpu_usable:
        summary = (
            f"{torch_info['device_count']} 块 GPU 可用"
            f"（{torch_info['devices'][0]['name']}，"
            f"{torch_info['devices'][0]['total_memory_gb']} GB）"
            if torch_info["devices"]
            else "GPU 可用"
        )
    elif smi["present"]:
        summary = (
            "检测到 NVIDIA 显卡（" + smi["devices"][0]["name"] + "），"
            "但 torch 无法使用它：" + (torch_info["reason"] or "原因未知")
        )
    else:
        summary = torch_info["reason"] or "未检测到可用 GPU，将使用 CPU"

    return {
        "default": "cuda" if gpu_usable else "cpu",
        "gpu_usable": gpu_usable,
        "cpu_available": True,
        "choices": list(DEVICE_CHOICES),
        "summary": summary,
        "torch": torch_info,
        "nvidia_smi": smi,
        # CPU 上跑整曲很慢，前端据此提醒用户
        "estimated_slow": not gpu_usable,
    }


def resolve_device(requested: str | None) -> str:
    """把用户请求的设备解析成实际可用的值（``auto``/``cuda`` 不可用时回落 cpu）。"""
    wanted = str(requested or "auto").strip().lower()
    if wanted in ("", "auto"):
        return "cuda" if _torch_info()["cuda_available"] else "cpu"
    if wanted.startswith("cuda"):
        return "cuda" if _torch_info()["cuda_available"] else "cpu"
    return "cpu"


def device_note(requested: str | None, resolved: str) -> str:
    """给用户看的一句话：请求了什么、实际用了什么、为什么。"""
    wanted = str(requested or "auto").strip().lower() or "auto"
    if wanted.startswith("cuda") and resolved != "cuda":
        return "请求 GPU，但本机 GPU 不可用（详见 /api/health），已回落 CPU；耗时会长很多。"
    if resolved == "cuda":
        return "使用 GPU 推理。"
    return "使用 CPU 推理；4 分钟的歌约需 9–15 分钟（VocalParse 是主要耗时）。"

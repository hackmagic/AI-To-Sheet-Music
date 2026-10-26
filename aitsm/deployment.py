"""推理进程的部署约束：单 worker / 模型内存预算。

为什么需要这个模块
------------------
VocalParse 是 2B 参数，加载后常驻约 **8 GB**。如果 FastAPI 用
``uvicorn --workers 4`` 部署，每个 worker 都会各自加载一份权重：

    4 × 8 GB = 32 GB

机器很容易被打爆，而且表现为「跑到一半 OOM」这种难查的故障。本模块把这条约束
变成**可执行的检查**，而不是写在文档里等别人踩。

三个层次：

1. :func:`model_memory_report` —— 当前进程里各模型的估计常驻内存；
2. :func:`worker_advice` —— 根据模型组合给出「该用几个 worker」的结论；
3. :func:`check_deployment` —— 启动时可调用的硬检查（超预算就返回警告/错误）。
"""

from __future__ import annotations

import os
from typing import Any, Mapping

__all__ = [
    "MODEL_MEMORY_GB",
    "model_memory_report",
    "worker_advice",
    "check_deployment",
    "runtime_flags",
]

#: 各模型在 **权重 + 推理激活** 下的估计常驻内存（CPU fp32，GB）。
#: 数值来自实测：VocalParse 加载后 RSS 约 8.3 GB（14s 音频推理峰值 8.7 GB）。
MODEL_MEMORY_GB: dict[str, float] = {
    "vocalparse": 8.5,
    "sheetsage2": 1.5,
    "duet": 0.6,
    "htdemucs_ft_onnx": 1.8,  # 四个 ONNX session 常驻
    "demucs_torch": 2.5,
}

#: 建议为系统与音频缓冲预留的量（GB）。
_RESERVE_GB = 2.0

#: 单 worker 的安全上限：超过它应当拆分独立推理服务。
SINGLE_WORKER_CEILING_GB = 12.0


def runtime_flags() -> dict[str, Any]:
    """从环境变量判断当前部署形态。"""
    web_concurrency = os.environ.get("WEB_CONCURRENCY")
    workers = None
    if web_concurrency:
        try:
            workers = int(web_concurrency)
        except ValueError:
            workers = None
    return {
        "web_concurrency": web_concurrency,
        "workers": workers,
        "aitsm_workers": os.environ.get("AITSM_WORKERS"),
        "gpu": os.environ.get("AITSM_DEVICE", "auto"),
    }


def model_memory_report(
    enabled: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """估计当前组合的常驻内存。

    ``enabled`` 形如 ``{"vocalparse": True, "sheetsage2": True, ...}``；
    省略时按「都启用」计算（最坏情况）。
    """
    flags = dict(enabled or {name: True for name in MODEL_MEMORY_GB})
    active = {name: MODEL_MEMORY_GB[name] for name, on in flags.items() if on and name in MODEL_MEMORY_GB}
    total = round(sum(active.values()), 2)
    return {
        "models": active,
        "total_gb": total,
        "with_reserve_gb": round(total + _RESERVE_GB, 2),
        "ceiling_gb": SINGLE_WORKER_CEILING_GB,
        "fits_single_worker": total + _RESERVE_GB <= SINGLE_WORKER_CEILING_GB,
    }


def worker_advice(
    enabled: Mapping[str, bool] | None = None,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """给出 worker 数量建议（这是可执行的结论，不是泛泛而谈）。"""
    report = model_memory_report(enabled)
    flags = runtime_flags()
    effective = workers if workers is not None else flags["workers"]

    advice: dict[str, Any] = {
        "per_worker_gb": report["with_reserve_gb"],
        "configured_workers": effective,
        "report": report,
    }
    if effective is None:
        advice.update(
            recommended_workers=1,
            level="info",
            message=(
                f"未显式配置 worker 数；模型组合约需 {report['with_reserve_gb']} GB/worker，"
                "请保持单 worker（uvicorn 默认就是 1）。"
            ),
        )
        return advice

    projected = round(report["with_reserve_gb"] * max(1, effective), 2)
    advice["projected_total_gb"] = projected
    if effective > 1 and report["with_reserve_gb"] > 4.0:
        advice.update(
            recommended_workers=1,
            level="error",
            message=(
                f"配置了 {effective} 个 worker，每个约需 {report['with_reserve_gb']} GB，"
                f"合计约 {projected} GB —— 会 OOM。"
                "把 VocalParse 拆成独立推理服务（模型只加载一份），或改用单 worker + 内部队列。"
            ),
        )
    elif report["with_reserve_gb"] > SINGLE_WORKER_CEILING_GB:
        advice.update(
            recommended_workers=1,
            level="warn",
            message=(
                f"单 worker 约需 {report['with_reserve_gb']} GB，已超过建议上限 "
                f"{SINGLE_WORKER_CEILING_GB} GB；建议把重模型拆到独立进程。"
            ),
        )
    else:
        advice.update(recommended_workers=effective, level="ok", message="worker 配置在安全范围内。")
    return advice


def check_deployment(
    enabled: Mapping[str, bool] | None = None,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """启动时调用：返回 ``{"ok": bool, "level": str, "message": str, ...}``。

    ``ok=False`` 表示这个部署一定会出问题（如多 worker + 大模型）。
    """
    advice = worker_advice(enabled, workers=workers)
    level = advice["level"]
    return {
        "ok": level in ("ok", "info"),
        "level": level,
        "message": advice["message"],
        "detail": advice,
    }

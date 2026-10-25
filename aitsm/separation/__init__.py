"""分离层（模块 0）。

对外只有一个入口 :func:`aitsm.separation.engine.separate`，返回
:class:`aitsm.contracts.SeparationResult`。架构首选 ``htdemucs_ft``，本机不可用时
按 ONNX → torch Demucs → 中置/HPSS 的顺序显式降级。
"""

from .engine import (
    DEFAULT_MODEL,
    REQUESTED_MODEL,
    available_backends,
    onnx_available,
    separate,
)

__all__ = [
    "DEFAULT_MODEL",
    "REQUESTED_MODEL",
    "available_backends",
    "onnx_available",
    "separate",
]

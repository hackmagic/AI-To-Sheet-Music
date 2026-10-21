"""AI-To-Sheet-Music: 音频到网页弹唱谱。

注意：``prepare_cache_dirs()`` 必须在导入 torch / librosa / huggingface_hub 之前执行。
这些库会在 import 时读取 TORCH_HOME / HF_HOME / NUMBA_CACHE_DIR 等环境变量；
在沙箱环境下 ``%USERPROFILE%\\.cache`` 不可写，提前重定向可以避免 PermissionError。
"""

from __future__ import annotations

from .config import prepare_cache_dirs

prepare_cache_dirs()

__version__ = "0.1.0"
__all__ = ["__version__", "prepare_cache_dirs"]

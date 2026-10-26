"""``qwen_asr`` 建模后端的隔离层（VocalParse 的依赖边界）。

为什么单独一层
--------------
Qwen3-ASR 的建模代码不在标准 ``transformers`` 里，官方通过 ``qwen-asr`` 包分发。
而 ``qwen_asr`` 0.0.x 有三个现实问题，必须集中在一个地方兜住：

1. **版本极早期**（当前 0.0.6）：子模块路径、类名都可能变，所以版本要 pin 死，
   路径只在这里出现一次，升级时只改这一个文件；
2. ``__init__`` 会连带导入 ``nagisa`` / ``gradio`` 等**推理封装与 demo 依赖**，
   对建模推理毫无必要（实测 `nagisa` 在 Windows 上装不动）；
3. 缺依赖时不能让 ``ImportError`` 在推理中途冒出来，要在**可用性探测**阶段就
   给出明确结论。

因此这里按**文件路径**只加载三个建模子模块，并做版本与路径校验。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "PINNED_VERSION",
    "REQUIRED_MODULES",
    "BACKEND_SUBDIR",
    "VOCAB_SIZE",
    "DEFAULT_MROPE",
    "QwenBackendError",
    "installed_version",
    "check_import_paths",
    "probe",
    "load_backend",
]

#: 实测可用的版本。``qwen-asr`` 仍是 0.0.x，API 会变，**必须精确 pin**。
PINNED_VERSION = "0.0.6"

#: 建模推理所需的三个子模块（缺任何一个都无法加载权重）。
REQUIRED_MODULES = (
    "configuration_qwen3_asr",
    "processing_qwen3_asr",
    "modeling_qwen3_asr",
)

#: 包内子目录（相对 ``qwen_asr`` 根；新旧布局都兼容）。
BACKEND_SUBDIRS = ("core/transformers_backend", "transformers_backend")

#: VocalParse 的扩展词表：基础 151936 + AST token ≈ 152101。
#: 必须与权重里 ``embed_tokens`` 的行数一致，否则形状校验失败。
VOCAB_SIZE = 152101

#: checkpoint 没写 ``rope_scaling``；这个值是 ``qwen_asr`` 建模代码自己的缺省
#: （modeling_qwen3_asr.py: ``config.rope_scaling.get("mrope_section", [24, 20, 20])``），
#: 不是我们猜的。设错会让位置编码错，音符时序系统性偏移。
DEFAULT_MROPE: dict[str, Any] = {"type": "mrope", "mrope_section": [24, 20, 20]}


class QwenBackendError(RuntimeError):
    """``qwen_asr`` 后端不可用（未安装、版本不符、路径变化、缺子模块）。"""


def installed_version() -> str | None:
    """已安装的 ``qwen-asr`` 版本；未安装返回 ``None``。

    优先读包元数据（不 import 包，避免触发 ``nagisa`` 依赖）。
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("qwen-asr")
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - 极旧环境
        return None


def _package_root() -> Path | None:
    """``qwen_asr`` 的真实安装根目录。

    **不能只在 ``sys.modules`` / ``meta_path`` 里找**：第一次
    :func:`load_backend` 会把「空壳父包」放进 ``sys.modules``，之后
    ``importlib.util.find_spec("qwen_asr")`` 会读到那个壳的 ``__spec__``
    （``None``）并抛 ``ValueError``；``meta_path`` 里的
    ``PathFinder.find_spec`` 同样会复用 ``sys.modules`` 条目而返回 ``None``。

    实测后果：同进程**第二次** ``probe()`` / ``load_backend()`` 直接崩
    ——即「服务跑完第一个任务，第二个任务必失败」（P0）。
    由 ``tests/test_vocalparse_integration.py`` 的真实加载测试抓到。

    解法：**绕开导入系统，直接沿 ``sys.path`` 做文件系统查找**。
    """
    import sys as _sys

    for entry in list(_sys.path):
        if not entry:
            continue
        try:
            candidate = Path(entry) / "qwen_asr"
        except (OSError, ValueError):  # pragma: no cover - 非法路径项
            continue
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def check_import_paths() -> dict[str, Any]:
    """检查「三个子模块是否都能在预期位置找到」——不执行它们。

    返回 ``{"ok": bool, "root": str, "backend": str, "missing": [...], "reason": str}``。
    这个函数的价值在于：``qwen-asr`` 升级导致路径变化时，能在**探测阶段**就发现，
    而不是等到推理时才抛 ``ImportError``。
    """
    root = _package_root()
    if root is None:
        return {"ok": False, "root": "", "backend": "", "missing": list(REQUIRED_MODULES),
                "reason": "未安装 qwen-asr"}
    for subdir in BACKEND_SUBDIRS:
        backend = root / subdir
        if not backend.is_dir():
            continue
        missing = [name for name in REQUIRED_MODULES if not (backend / f"{name}.py").is_file()]
        if not missing:
            return {"ok": True, "root": str(root), "backend": str(backend), "missing": [],
                    "reason": ""}
        return {
            "ok": False, "root": str(root), "backend": str(backend), "missing": missing,
            "reason": f"{backend} 缺少子模块：{', '.join(missing)}",
        }
    tried = " / ".join(BACKEND_SUBDIRS)
    return {
        "ok": False, "root": str(root), "backend": "", "missing": list(REQUIRED_MODULES),
        "reason": f"在 {root} 下找不到后端子目录（试过 {tried}）",
    }


def probe() -> dict[str, Any]:
    """后端可用性总检（版本 + 路径），供 ``is_available`` 与 doctor 使用。"""
    version = installed_version()
    paths = check_import_paths()
    reason = ""
    if version is None:
        reason = "未安装 qwen-asr（Qwen3-ASR 建模代码不在标准 transformers 里）"
    elif version != PINNED_VERSION:
        # 不硬拒绝：路径检查通过就允许用，但要把版本差异讲清楚
        if not paths["ok"]:
            reason = (
                f"qwen-asr 版本 {version} ≠ 实测版本 {PINNED_VERSION}，"
                f"且后端路径不可用：{paths['reason']}"
            )
    if not reason and not paths["ok"]:
        reason = paths["reason"]
    return {
        "available": not reason,
        "version": version,
        "pinned_version": PINNED_VERSION,
        "version_matches_pin": version == PINNED_VERSION,
        "paths": paths,
        "reason": reason,
    }


def load_backend() -> dict[str, ModuleType]:
    """按文件路径加载三个建模子模块（不执行 ``qwen_asr/__init__.py``）。

    为什么按路径加载：包 ``__init__`` 会连带导入 ``nagisa``（分词，forced-aligner
    用）与可选 ``gradio``（demo 用），这些对建模推理没有作用，却让依赖变重、
    在没有它们的机器上直接失败。

    返回 ``{"configuration_qwen3_asr": module, "processing_qwen3_asr": module,
    "modeling_qwen3_asr": module}``。任何一步不成立都抛 :class:`QwenBackendError`。
    """
    info = check_import_paths()
    if not info["ok"]:
        raise QwenBackendError(info["reason"] or "qwen_asr 后端不可用")
    backend = Path(info["backend"])
    root = Path(info["root"])

    # 造空壳父包，让三个子模块的相对导入（`from .configuration_...`）成立。
    #
    # **关键**：``__path__`` 必须指向真实目录，不能留空列表。留空的后果是
    # ``sys.modules["qwen_asr"]`` 变成一个无 ``__path__`` 的空壳，后续
    # ``PathFinder.find_spec('qwen_asr')`` 会复用它而不再搜索 → 返回 None
    # → 第二次 probe/load 误判成「未安装」（实测 P0：同进程第二个任务必失败）。
    shell_paths = {
        "qwen_asr": [str(root)],
        "qwen_asr.core": [str(root / "core")],
        "qwen_asr.core.transformers_backend": [str(backend)],
    }
    for name, search_paths in shell_paths.items():
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__path__", None):
            continue
        module = ModuleType(name)
        module.__path__ = list(search_paths)
        module.__package__ = name
        sys.modules[name] = module

    loaded: dict[str, ModuleType] = {}
    for stem in REQUIRED_MODULES:
        full_name = f"qwen_asr.core.transformers_backend.{stem}"
        existing = sys.modules.get(full_name)
        if existing is not None:
            loaded[stem] = existing
            continue
        spec = importlib.util.spec_from_file_location(full_name, str(backend / f"{stem}.py"))
        if spec is None or spec.loader is None:
            raise QwenBackendError(f"无法为 {backend / f'{stem}.py'} 创建模块 spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(full_name, None)
            raise QwenBackendError(
                f"加载 {stem} 失败：{type(exc).__name__}: {exc}"
            ) from exc
        loaded[stem] = module
    return loaded


def mrope_scaling() -> dict[str, Any]:
    """返回一份 mrope 配置副本（调用方可以安全修改）。"""
    return {"type": DEFAULT_MROPE["type"], "mrope_section": list(DEFAULT_MROPE["mrope_section"])}


def validate_mrope(config: Any) -> list[str]:
    """校验 ``rope_scaling`` 是否可用；返回问题列表（空表示没问题）。

    mRoPE 少了 ``mrope_section`` 就是**位置编码静默失效**：不会报错，但音符时序
    会系统性偏移。所以这里显式检查，而不是信任默认值。
    """
    problems: list[str] = []
    scaling = getattr(config, "rope_scaling", None)
    if scaling is None:
        return ["rope_scaling 为空：mRoPE 位置编码不会生效"]
    if not hasattr(scaling, "get"):
        problems.append(f"rope_scaling 类型异常：{type(scaling).__name__}")
        return problems
    section = scaling.get("mrope_section")
    if not section:
        problems.append("rope_scaling 缺少 mrope_section")
    elif not isinstance(section, (list, tuple)) or len(section) != 3:
        problems.append(f"mrope_section 应为长度 3 的序列，实际 {section!r}")
    elif sum(int(item) for item in section) <= 0:
        problems.append(f"mrope_section 全为 0：{section!r}")
    return problems

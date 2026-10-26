"""``qwen_backend`` **重复调用**的回归测试。

## 这个 bug 是真实出现过的 P0

第一次 `load_backend()` 成功后，同进程内**第二次** `probe()` 会抛：

    ValueError: qwen_asr.__spec__ is None

后果：FastAPI 服务跑完第一个任务后，第二个任务在建符号契约时崩溃
（`is_available()` → `probe()`）。这是「跑一次能用、跑两次就挂」级别的缺陷，
由 `tests/test_vocalparse_integration.py` 的真实模型加载测试暴露。

根因：`load_backend()` 往 `sys.modules` 塞入**空壳父包**（`ModuleType`，
没有 `__spec__`）；之后 `importlib.util.find_spec("qwen_asr")` /
`PathFinder.find_spec(...)` 都会读到这个壳而失败。

修法：`_package_root()` 绕开导入系统，**直接沿 `sys.path` 做文件系统查找**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aitsm.transcribe import qwen_backend as Q

#: 本机是否真的装了 qwen-asr（决定能否真跑重复加载）
_REAL_INSTALLED = Q.installed_version() is not None


# ---------------------------------------------------------------- 纯逻辑（不依赖安装）


def test_package_root_does_not_depend_on_import_system(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """``_package_root()`` 必须走文件系统，即使 ``sys.modules`` 被污染也能找到。"""
    fake_pkg = tmp_path / "qwen_asr"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    # 塞一个没有 __spec__ 的空壳，模拟 load_backend 之后的真实状态
    import types

    monkeypatch.setitem(sys.modules, "qwen_asr", types.ModuleType("qwen_asr"))

    assert Q._package_root() == fake_pkg


def test_package_root_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "qwen_asr", raising=False)
    assert Q._package_root() is None


def test_package_root_survives_shell_module_in_sys_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """空壳父包的 ``__path__`` 为空列表时也必须能找到（这正是原 bug 的形态）。"""
    import types

    fake_pkg = tmp_path / "qwen_asr"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    shell = types.ModuleType("qwen_asr")
    shell.__path__ = []  # 原 bug：空列表
    monkeypatch.setitem(sys.modules, "qwen_asr", shell)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])

    assert Q._package_root() == fake_pkg


def test_probe_never_raises_with_shell_module(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """``probe()`` 对空壳状态也必须返回结论而不是抛异常。"""
    import types

    monkeypatch.setitem(sys.modules, "qwen_asr", types.ModuleType("qwen_asr"))
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    info = Q.probe()
    assert isinstance(info, dict)
    assert info["available"] is False
    assert info["reason"]


# ---------------------------------------------------------------- 真实重复加载


@pytest.mark.skipif(not _REAL_INSTALLED, reason="本机未安装 qwen-asr")
def test_repeated_load_backend_is_stable():
    """**核心回归**：连续两次 load_backend 都必须成功。

    这就是原来必崩的场景（第二次抛 ``qwen_asr.__spec__ is None``）。
    """
    first = Q.load_backend()
    assert set(first) == set(Q.REQUIRED_MODULES)

    # 此时 sys.modules 里已有空壳父包 —— 原 bug 从这里开始
    assert "qwen_asr" in sys.modules

    second = Q.load_backend()
    assert set(second) == set(Q.REQUIRED_MODULES)


@pytest.mark.skipif(not _REAL_INSTALLED, reason="本机未安装 qwen-asr")
def test_repeated_probe_is_stable():
    """连续多次 ``probe()`` 结果一致，不能第一次 True 之后就崩。"""
    results = [Q.probe()["available"] for _ in range(4)]
    assert all(item is True for item in results), results


@pytest.mark.skipif(not _REAL_INSTALLED, reason="本机未安装 qwen-asr")
def test_load_then_probe_then_load():
    """真实调用顺序：load → probe → load（服务跑第二个任务的路径）。"""
    Q.load_backend()
    assert Q.probe()["available"] is True
    Q.load_backend()
    assert Q.probe()["available"] is True


@pytest.mark.skipif(not _REAL_INSTALLED, reason="本机未安装 qwen-asr")
def test_vocal_is_available_after_backend_load(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """`vocal.is_available()` 在 backend 被加载过之后仍要正常工作。

    这是服务端的真实调用序：第一个任务加载模型 → 第二个任务先问
    `transcribe_vocal` 里的 `is_available()`。
    """
    from aitsm.transcribe import vocal as V

    weights = tmp_path / "VocalParse"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(V.VOCALPARSE_DIR_ENV, str(weights))

    Q.load_backend()
    assert V.is_available() is True
    Q.load_backend()
    assert V.is_available() is True

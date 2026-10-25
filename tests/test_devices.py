"""计算设备探测的测试（对应「页面上选 CPU 还是 GPU」这个需求）。

关键点：判断依据必须是 **torch 是否 CUDA 编译版**，而不是「机器有没有显卡」。
实测本机有 RTX 4060，但 torch 是 CPU 版 —— 这种情形必须如实告知用户，
否则他会以为选了 GPU 就在用 GPU。
"""

from __future__ import annotations

import pytest

from aitsm import devices as D


# ---------------------------------------------------------------- 解析与提示


def test_resolve_auto_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(D, "_torch_info", lambda: {"cuda_available": False})
    assert D.resolve_device("auto") == "cpu"


def test_resolve_auto_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(D, "_torch_info", lambda: {"cuda_available": True})
    assert D.resolve_device("auto") == "cuda"


def test_resolve_explicit_cuda_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    """用户明确要 GPU 但机器用不了 → 回落 CPU（不能让任务失败）。"""
    monkeypatch.setattr(D, "_torch_info", lambda: {"cuda_available": False})
    assert D.resolve_device("cuda") == "cpu"
    assert D.resolve_device("cuda:1") == "cpu"


def test_resolve_explicit_cpu_is_respected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(D, "_torch_info", lambda: {"cuda_available": True})
    assert D.resolve_device("cpu") == "cpu"


def test_resolve_handles_blank_and_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(D, "_torch_info", lambda: {"cuda_available": False})
    assert D.resolve_device(None) == "cpu"
    assert D.resolve_device("") == "cpu"
    assert D.resolve_device("  ") == "cpu"


def test_note_warns_when_cuda_requested_but_unavailable():
    note = D.device_note("cuda", "cpu")
    assert "GPU" in note and ("回落" in note or "不可用" in note)


def test_note_mentions_cpu_cost():
    assert "CPU" in D.device_note("auto", "cpu")


def test_note_reports_gpu_when_used():
    note = D.device_note("cuda", "cuda")
    assert "GPU" in note and "回落" not in note


# ---------------------------------------------------------------- 探测结构


def test_detect_devices_shape():
    info = D.detect_devices()
    for key in ("default", "gpu_usable", "cpu_available", "choices", "summary", "torch", "estimated_slow"):
        assert key in info
    assert info["choices"] == list(D.DEVICE_CHOICES)
    assert info["cpu_available"] is True
    assert info["default"] in ("cpu", "cuda")
    assert isinstance(info["summary"], str) and info["summary"]


def test_detect_devices_never_raises():
    """health 端点每次都会调它，绝不能抛异常。"""
    assert isinstance(D.detect_devices(), dict)


def test_detect_devices_estimated_slow_matches_gpu_state():
    info = D.detect_devices()
    assert info["estimated_slow"] == (not info["gpu_usable"])


def test_cpu_build_torch_is_explained(monkeypatch: pytest.MonkeyPatch):
    """torch 是 CPU 版时必须说明「装了 CUDA 版才能用 GPU」，而不是只说没卡。"""
    fake = {
        "installed": True,
        "version": "2.14.1+cpu",
        "cuda_build": False,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "reason": "当前 torch 是 CPU 版（未编译 CUDA）；要启用 GPU 需安装对应的 CUDA 版 torch",
    }
    monkeypatch.setattr(D, "_torch_info", lambda: fake)
    monkeypatch.setattr(D, "_nvidia_smi", lambda: {"present": True, "devices": [{"name": "RTX 4060", "memory": "8 GB"}]})
    info = D.detect_devices()
    assert info["gpu_usable"] is False
    assert "RTX 4060" in info["summary"], "有卡但 torch 不能用时，要同时说清两件事"
    assert "CPU 版" in info["summary"]


def test_torch_missing_is_handled(monkeypatch: pytest.MonkeyPatch):
    fake = {
        "installed": False,
        "version": "",
        "cuda_build": False,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "reason": "未安装 torch",
    }
    monkeypatch.setattr(D, "_torch_info", lambda: fake)
    monkeypatch.setattr(D, "_nvidia_smi", lambda: {"present": False, "devices": []})
    info = D.detect_devices()
    assert info["default"] == "cpu"
    assert "torch" in info["summary"] or "CPU" in info["summary"]


def test_gpu_usable_produces_structured_summary(monkeypatch: pytest.MonkeyPatch):
    fake = {
        "installed": True,
        "version": "2.6.0+cu124",
        "cuda_build": True,
        "cuda_available": True,
        "mps_available": False,
        "device_count": 1,
        "devices": [{"index": 0, "name": "RTX 4060", "total_memory_gb": 8.0, "capability": "8.9"}],
        "reason": "",
    }
    monkeypatch.setattr(D, "_torch_info", lambda: fake)
    monkeypatch.setattr(D, "_nvidia_smi", lambda: {"present": True, "devices": []})
    info = D.detect_devices()
    assert info["gpu_usable"] is True
    assert info["default"] == "cuda"
    assert "RTX 4060" in info["summary"]
    assert info["estimated_slow"] is False

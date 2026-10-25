"""分离层适配器测试：只验证「收敛成 SeparationResult + 如实上报降级」，不跑真 Demucs。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aitsm.contracts import SeparationResult
from aitsm.separation import engine
from aitsm.pipeline import separate as torch_separate


@pytest.fixture(autouse=True)
def _isolate_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """其它测试（--quality 预设）会写 AITSM_DEMUCS_MODEL，这里必须先清干净。"""
    monkeypatch.delenv("AITSM_DEMUCS_MODEL", raising=False)


@pytest.fixture()
def mix(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.wav"
    path.write_bytes(b"RIFF....WAVEfake")
    return path


def test_onnx_unavailable_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AITSM_DEMUCS_ONNX_DIR", raising=False)
    assert engine.onnx_available() is False


def test_onnx_requires_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AITSM_DEMUCS_ONNX_DIR", str(tmp_path / "missing"))
    assert engine.onnx_available() is False
    empty = tmp_path / "weights"
    empty.mkdir()
    monkeypatch.setenv("AITSM_DEMUCS_ONNX_DIR", str(empty))
    # 目录存在但没有 *.onnx 也没有 demucs_onnx 包
    assert engine.onnx_available() is False


def test_separate_maps_torch_result(
    monkeypatch: pytest.MonkeyPatch, mix: Path, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_separate(source: Path, work: Path, *, device: str = "auto") -> dict:
        captured["source"] = source
        captured["work"] = work
        captured["device"] = device
        captured["model_env"] = os.environ.get("AITSM_DEMUCS_MODEL")
        return {
            "engine": "demucs",
            "model": "htdemucs",
            "device": "cpu",
            "stems": {
                "vocals": str(tmp_path / "vocals.wav"),
                "drums": str(tmp_path / "drums.wav"),
                "bass": str(tmp_path / "bass.wav"),
                "other": str(tmp_path / "other.wav"),
            },
            "hq_stems": {},
            "warnings": ["复用已有分轨结果。"],
            "cached": True,
        }

    monkeypatch.setattr(torch_separate, "separate", fake_separate)
    result = engine.separate(mix, tmp_path / "work")

    assert isinstance(result, SeparationResult)
    assert result.engine == "demucs"
    assert result.model == "htdemucs"
    assert result.cached is True
    assert result.vocals and result.accompaniment
    assert set(result.stems) >= {"vocals", "drums", "bass", "other"}
    assert captured["model_env"] == "htdemucs"  # 默认值被写进环境
    # 调用结束后必须恢复环境，不能污染后续任务
    assert os.environ.get("AITSM_DEMUCS_MODEL") in (None, captured.get("previous"))


def test_separate_reports_architecture_gap(
    monkeypatch: pytest.MonkeyPatch, mix: Path, tmp_path: Path
) -> None:
    def fake_separate(source: Path, work: Path, *, device: str = "auto") -> dict:
        return {
            "engine": "demucs",
            "model": "htdemucs",
            "device": "cpu",
            "stems": {"vocals": "v.wav", "drums": "d.wav", "bass": "b.wav", "other": "o.wav"},
            "warnings": [],
            "cached": False,
        }

    monkeypatch.setattr(torch_separate, "separate", fake_separate)
    result = engine.separate(mix, tmp_path / "work")
    assert any("htdemucs_ft" in warning for warning in result.warnings)


def test_separate_restores_previous_model_env(
    monkeypatch: pytest.MonkeyPatch, mix: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "htdemucs_ft")
    seen: dict[str, str | None] = {}

    def fake_separate(source: Path, work: Path, *, device: str = "auto") -> dict:
        seen["model"] = os.environ.get("AITSM_DEMUCS_MODEL")
        return {"engine": "demucs", "model": "htdemucs_ft", "device": "cpu",
                "stems": {"vocals": "v.wav", "other": "o.wav"}, "warnings": [], "cached": False}

    monkeypatch.setattr(torch_separate, "separate", fake_separate)
    result = engine.separate(mix, tmp_path / "work", model="htdemucs")
    assert seen["model"] == "htdemucs"  # 显式参数优先
    assert os.environ["AITSM_DEMUCS_MODEL"] == "htdemucs_ft"  # 调用后恢复
    assert not any("首选" in warning for warning in result.warnings)


def test_separate_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        engine.separate(tmp_path / "nope.wav", tmp_path / "work")


def test_fallback_engine_name_is_explicit(monkeypatch: pytest.MonkeyPatch, mix: Path, tmp_path: Path) -> None:
    def fake_separate(source: Path, work: Path, *, device: str = "auto") -> dict:
        return {"engine": "fallback", "model": "center-hpss", "device": "cpu",
                "stems": {"vocals": "v.wav", "other": "o.wav"}, "warnings": ["降级"], "cached": False}

    monkeypatch.setattr(torch_separate, "separate", fake_separate)
    result = engine.separate(mix, tmp_path / "work")
    assert result.engine == "demucs-fallback"
    assert result.warnings == ["降级"]

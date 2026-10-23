"""``--quality`` 预设、分轨质量旋钮、伴奏轨（去人声）构造的单测。

这三件事共同决定“更慢但更准”这条路能不能走通：
- 预设必须真的把环境变量写进去，且不覆盖用户显式设置；
- ``separate.demucs_quality()`` 必须**调用时**读环境变量（否则同一进程内切档无效）；
- ``run._build_instrumental()`` 必须把鼓/贝斯/伴奏相加，缺轨时安全返回 None。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.cli import QUALITY_PRESETS, apply_quality_preset  # noqa: E402
from aitsm.pipeline import run as run_mod  # noqa: E402
from aitsm.pipeline import separate as separate_mod  # noqa: E402


# ------------------------------------------------------------------ 预设
def test_preset_names_and_escalation() -> None:
    assert set(QUALITY_PRESETS) == {"standard", "high", "max"}
    assert QUALITY_PRESETS["standard"] == {}
    # 档位越高越慢：shifts 递增、用微调分轨
    high = QUALITY_PRESETS["high"]
    maxed = QUALITY_PRESETS["max"]
    assert high["AITSM_DEMUCS_MODEL"] == maxed["AITSM_DEMUCS_MODEL"] == "htdemucs_ft"
    assert int(maxed["AITSM_DEMUCS_SHIFTS"]) > int(high["AITSM_DEMUCS_SHIFTS"])
    assert maxed["AITSM_WHISPER_MODEL"] == "large-v3"


def test_apply_preset_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AITSM_DEMUCS_MODEL", "AITSM_DEMUCS_SHIFTS", "AITSM_DEMUCS_OVERLAP", "AITSM_WHISPER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    applied = apply_quality_preset("max")
    assert applied["AITSM_DEMUCS_MODEL"] == "htdemucs_ft"
    assert applied["AITSM_WHISPER_MODEL"] == "large-v3"
    assert separate_mod.demucs_quality()[0] == "htdemucs_ft"


def test_apply_preset_does_not_override_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITSM_WHISPER_MODEL", "medium")
    monkeypatch.delenv("AITSM_DEMUCS_MODEL", raising=False)
    applied = apply_quality_preset("max")
    assert "AITSM_WHISPER_MODEL" not in applied
    assert applied["AITSM_DEMUCS_MODEL"] == "htdemucs_ft"


def test_apply_preset_standard_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AITSM_DEMUCS_MODEL", raising=False)
    assert apply_quality_preset("standard") == {}
    assert "AITSM_DEMUCS_MODEL" not in __import__("os").environ


# ------------------------------------------------------- 分轨质量旋钮
def test_demucs_quality_reads_env_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AITSM_DEMUCS_MODEL", raising=False)
    monkeypatch.delenv("AITSM_DEMUCS_SHIFTS", raising=False)
    monkeypatch.delenv("AITSM_DEMUCS_OVERLAP", raising=False)
    assert separate_mod.demucs_quality() == ("htdemucs", 1, 0.5)

    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("AITSM_DEMUCS_SHIFTS", "3")
    monkeypatch.setenv("AITSM_DEMUCS_OVERLAP", "0.8")
    assert separate_mod.demucs_quality() == ("htdemucs_ft", 3, 0.8)


def test_demucs_quality_clamps_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "   ")
    monkeypatch.setenv("AITSM_DEMUCS_SHIFTS", "not-a-number")
    monkeypatch.setenv("AITSM_DEMUCS_OVERLAP", "3")
    model, shifts, overlap = separate_mod.demucs_quality()
    assert model == "htdemucs"  # 空白回退默认
    assert shifts == 1  # 非法值回退默认
    assert overlap == 0.95  # 夹到上限


# ----------------------------------------------- 去人声伴奏轨（和弦输入）
def _write_stem(path: Path, value: float, seconds: float = 0.2, sr: int = 22050) -> Path:
    import numpy as np
    import soundfile as sf

    samples = np.full(int(sr * seconds), value, dtype="float32")
    sf.write(path, samples, sr)
    return path


def test_build_instrumental_sums_three_stems(tmp_path: Path) -> None:
    import soundfile as sf

    stems = {
        "vocals": str(_write_stem(tmp_path / "vocals.wav", 0.5)),
        "drums": str(_write_stem(tmp_path / "drums.wav", 0.1)),
        "bass": str(_write_stem(tmp_path / "bass.wav", 0.2)),
        "other": str(_write_stem(tmp_path / "other.wav", 0.05)),
    }
    out = run_mod._build_instrumental(stems, tmp_path)
    assert out is not None and Path(out).is_file()
    data, sr = sf.read(out, dtype="float32")
    assert sr == 22050
    # drums + bass + other，不含人声
    assert abs(float(data.mean()) - 0.35) < 1e-3


def test_build_instrumental_returns_none_without_stems(tmp_path: Path) -> None:
    assert run_mod._build_instrumental({}, tmp_path) is None
    assert run_mod._build_instrumental({"vocals": "x.wav"}, tmp_path) is None


def test_build_instrumental_handles_missing_file(tmp_path: Path) -> None:
    stems = {
        "drums": str(tmp_path / "nope-drums.wav"),
        "bass": str(tmp_path / "nope-bass.wav"),
        "other": str(tmp_path / "nope-other.wav"),
    }
    assert run_mod._build_instrumental(stems, tmp_path) is None


# ------------------------------------------------- separate() 端到端接线
def test_separate_records_quality_in_cache_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """回归：``separate()`` 曾在构造 cache_key 时引用未定义的 shifts/overlap，
    直接 NameError（被上层捕获成「分轨失败，改用混合音频」）。这里用假的
    ``_run_demucs`` 走完整接线，确保质量参数真的进了缓存键。"""
    import numpy as np
    import soundfile as sf

    mix = tmp_path / "mix.wav"
    sf.write(mix, np.zeros(22050, dtype="float32"), 22050)

    def fake_prepare(source, work):
        return mix, {"reused": False}

    def fake_run(mix_path, out_hq, out_analysis, *, device="auto", warnings=None):
        out_analysis.mkdir(parents=True, exist_ok=True)
        stems = {}
        for name in separate_mod.STEM_NAMES:
            path = out_analysis / f"{name}.wav"
            sf.write(path, np.zeros(2205, dtype="float32"), 22050)
            stems[name] = str(path)
        return {
            "engine": "demucs",
            "model": "htdemucs_ft",
            "device": "cpu",
            "stems": stems,
            "hq_stems": {},
            "warnings": [],
            "versions": {"demucs": "4.1.0"},
        }

    monkeypatch.setattr(separate_mod, "_prepare_mix", fake_prepare)
    monkeypatch.setattr(separate_mod, "_run_demucs", fake_run)
    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("AITSM_DEMUCS_SHIFTS", "2")
    monkeypatch.setenv("AITSM_DEMUCS_OVERLAP", "0.75")

    result = separate_mod.separate(mix, tmp_path / "work")
    assert result["cached"] is False
    assert set(result["stems"]) == set(separate_mod.STEM_NAMES)

    import json

    manifest = json.loads((tmp_path / "work" / "stems" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_key"]["model"] == "htdemucs_ft"
    assert manifest["cache_key"]["shifts"] == 2
    assert manifest["cache_key"]["overlap"] == 0.75

    # 参数不变 -> 第二次直接命中缓存，不再跑模型
    calls = {"n": 0}

    def counting_run(*args, **kwargs):  # pragma: no cover - 命中缓存时不该被调用
        calls["n"] += 1
        return fake_run(*args, **kwargs)

    monkeypatch.setattr(separate_mod, "_run_demucs", counting_run)
    again = separate_mod.separate(mix, tmp_path / "work")
    assert again["cached"] is True
    assert calls["n"] == 0

    # 换了质量参数 -> 缓存失效，必须重跑
    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "htdemucs")
    third = separate_mod.separate(mix, tmp_path / "work")
    assert third["cached"] is False
    assert calls["n"] == 1


def test_separate_reports_quality_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """manifest/pipeline 需要能解释「这份分轨是哪个档位产的」，所以结果里必须带
    model/shifts/overlap——包括复用缓存的路径。"""
    import numpy as np
    import soundfile as sf

    mix = tmp_path / "mix2.wav"
    sf.write(mix, np.zeros(22050, dtype="float32"), 22050)

    def fake_prepare(source, work):
        return mix, {"reused": False}

    def fake_run(mix_path, out_hq, out_analysis, *, device="auto", warnings=None):
        out_analysis.mkdir(parents=True, exist_ok=True)
        stems = {}
        for name in separate_mod.STEM_NAMES:
            path = out_analysis / f"{name}.wav"
            sf.write(path, np.zeros(2205, dtype="float32"), 22050)
            stems[name] = str(path)
        return {
            "engine": "demucs",
            "model": "htdemucs_ft",
            "device": "cpu",
            "stems": stems,
            "hq_stems": {},
            "warnings": [],
            "versions": {},
        }

    monkeypatch.setattr(separate_mod, "_prepare_mix", fake_prepare)
    monkeypatch.setattr(separate_mod, "_run_demucs", fake_run)
    monkeypatch.setenv("AITSM_DEMUCS_MODEL", "htdemucs_ft")
    monkeypatch.setenv("AITSM_DEMUCS_SHIFTS", "4")
    monkeypatch.setenv("AITSM_DEMUCS_OVERLAP", "0.9")

    fresh = separate_mod.separate(mix, tmp_path / "w")
    assert fresh["shifts"] == 4 and fresh["overlap"] == 0.9

    cached = separate_mod.separate(mix, tmp_path / "w")
    assert cached["cached"] is True
    assert cached["shifts"] == 4 and cached["overlap"] == 0.9

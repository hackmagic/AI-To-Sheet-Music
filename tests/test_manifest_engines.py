"""manifest 里的「哪个模型跑的」这条元信息必须落盘。

**为什么单独测**：`aitsm/pipeline/lyrics.py` 的公开接口 `transcribe_lyrics()` 返回的是
纯歌词行列表（渲染层契约，不能为了元信息去改它）。型号/设备因此走模块级
`last_run_info()`，再由 `aitsm/pipeline/run.py::_transcribe_lyrics()` 拼进 payload，
最后进 `analysis.json["tools"]["lyrics"]` 与 `manifest.json`。

这条链路以前是断的 —— 产物里只有 `engine: "whisper"`，没有型号，
导致 small（CER 0.64）与 medium（CER 0.50）跑出来的谱面差异在产物里无从归因。
"""

from __future__ import annotations

import pytest

from aitsm.pipeline import lyrics as lyrics_module
from aitsm.pipeline import run as run_module


class _FakeLyricsModule:
    """冒充 aitsm.pipeline.lyrics：只提供 run.py 用到的那几个名字。"""

    def __init__(self, *, lines=None, info=None):
        self.lines = lines or []
        self.info = info or {}
        self.calls = 0

    def transcribe_lyrics(self, path, *, duration=None, language="auto"):
        self.calls += 1
        return self.lines

    def last_run_info(self):
        return dict(self.info)


@pytest.fixture()
def patch_lyrics(monkeypatch):
    def apply(fake):
        monkeypatch.setattr(lyrics_module, "transcribe_lyrics", fake.transcribe_lyrics)
        monkeypatch.setattr(lyrics_module, "last_run_info", fake.last_run_info)
        return fake

    return apply


def _call(vocals: str = "fake_vocals.wav"):
    warnings: list[dict] = []
    return run_module._transcribe_lyrics(vocals, 200.0, warnings, language="en"), warnings


def test_payload_carries_model_and_device(patch_lyrics):
    fake = patch_lyrics(
        _FakeLyricsModule(
            lines=[{"start": 0.0, "end": 2.0, "text": "hello"}],
            info={"engine": "faster-whisper", "model": "medium", "device": "cpu"},
        )
    )
    payload, warnings = _call()
    assert fake.calls == 1
    assert payload["engine"] == "faster-whisper"
    assert payload["model"] == "medium"
    assert payload["device"] == "cpu"
    assert payload["available"] is True
    assert warnings == []


def test_payload_keeps_fallback_engine_when_info_missing(patch_lyrics):
    """老实现/第三方实现没有 last_run_info 时不能崩，engine 退回 'whisper'。"""
    patch_lyrics(_FakeLyricsModule(lines=[{"start": 0.0, "end": 1.0, "text": "hi"}], info={}))
    payload, _ = _call()
    assert payload["engine"] == "whisper"
    assert "model" not in payload
    assert payload["available"] is True


def test_empty_lines_still_records_model(patch_lyrics):
    """识别出 0 行也要记下型号：这才是「用 medium 也没识别到」的证据。"""
    patch_lyrics(
        _FakeLyricsModule(
            lines=[], info={"engine": "faster-whisper", "model": "medium", "device": "cuda"}
        )
    )
    payload, warnings = _call()
    assert payload["available"] is False
    assert payload["model"] == "medium"
    assert payload["device"] == "cuda"
    assert any(w["code"] == "lyrics.empty" for w in warnings)


def test_no_vocals_skips_engine_entirely():
    warnings: list[dict] = []
    payload = run_module._transcribe_lyrics(None, 200.0, warnings, language="en")
    assert payload is None
    assert any(w["code"] == "lyrics.no_vocals" for w in warnings)


def test_tools_block_uses_payload_fields():
    """`tools.lyrics` 必须把 model/device 透传出去，否则 manifest 又丢了。"""
    import inspect

    source = inspect.getsource(run_module)
    assert '"model": (lyrics or {}).get("model")' in source
    assert '"device": (lyrics or {}).get("device")' in source


def test_last_run_info_is_a_copy():
    """调用方拿到的是副本，改它不能污染模块状态。"""
    lyrics_module._LAST_RUN_INFO.clear()
    lyrics_module._LAST_RUN_INFO.update({"model": "medium"})
    info = lyrics_module.last_run_info()
    info["model"] = "small"
    assert lyrics_module.last_run_info()["model"] == "medium"
    lyrics_module._LAST_RUN_INFO.clear()


def test_faster_whisper_result_includes_model_and_device():
    """ASR 结果 dict 本身也要带型号：`_run_faster_whisper` 是唯一知道它的人。"""
    import inspect

    source = inspect.getsource(lyrics_module._run_faster_whisper)
    assert '"model": _whisper_model_name()' in source
    assert '"device": used_device' in source

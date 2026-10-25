"""伴奏联合转录适配器单测（合成 events + monkeypatch，不下载模型、不跑真音频）。

覆盖：

* events 两种形状（dict / list）的规范化与契约类型出口；
* ABC 调试输出（头必须存在、和弦注解、允许空旋律）；
* SheetSage2 可用性探测（无 transformers / 无本地权重均为不可用）；
* SheetSage2 成功、推理失败、空结果三条路径；
* 降级路径复用 pipeline 四个阶段、stem 缺失时回退整曲混音、进度回调两种签名。
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

from aitsm.contracts import AccompanimentResult, BeatGrid, KeySignature
from aitsm.transcribe import accompaniment


# --------------------------------------------------------------------------- #
# fixtures：模拟 pipeline 返回
# --------------------------------------------------------------------------- #


def _beats_dict() -> dict:
    times = [round(0.5 + 0.5 * index, 3) for index in range(16)]
    return {
        "bpm": 120.0,
        "bpm_display": 120.0,
        "confidence": 0.8,
        "beats": times,
        "downbeats": times[::4],
        "beats_per_bar": 4,
        "time_signature": "4/4",
        "first_downbeat_sec": times[0],
        "pickup_beats": 0,
        "duration_sec": 8.5,
        "warnings": ["节拍置信度偏低"],
    }


def _key_dict() -> dict:
    return {
        "tonic": "G",
        "mode": "major",
        "confidence": 0.7,
        "num_sharps_flats": 1,
        "tonic_pc": 7,
        "warnings": [],
    }


def _chords_list() -> list[dict]:
    return [
        {"start": 0.5, "end": 2.5, "symbol": "G", "confidence": 0.8, "source": "template"},
        {"start": 2.5, "end": 4.5, "symbol": "C", "confidence": 0.7, "source": "template"},
        {"start": 4.5, "end": 6.5, "symbol": "Am", "confidence": 0.6, "source": "template"},
    ]


def _sections_list() -> list[dict]:
    return [
        {"start": 0.0, "end": 4.0, "kind": "intro", "label": "前奏", "energy": 0.3, "confidence": 0.5},
        {"start": 4.0, "end": 8.0, "kind": "verse", "label": "主歌", "energy": 0.6, "confidence": 0.5},
    ]


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, calls: dict | None = None) -> dict:
    """把降级路径的四个 pipeline 调用替换成合成返回，并记录入参。"""
    calls = calls if calls is not None else {}

    def fake_beats(source):
        calls.setdefault("beats", []).append(source)
        return dict(_beats_dict())

    def fake_key(source, chords=None):
        key = "key_refine" if chords else "key"
        calls.setdefault(key, []).append(source)
        return dict(_key_dict())

    def fake_chords(source, beats, *, key=None, total_seconds=None):
        calls.setdefault("chords", []).append(
            {"source": source, "bpm": beats.get("bpm"), "key": key, "total_seconds": total_seconds}
        )
        return [dict(item) for item in _chords_list()]

    def fake_structure(source, beats, chords, total_seconds):
        calls.setdefault("structure", []).append(
            {"source": source, "n_chords": len(chords), "total_seconds": total_seconds}
        )
        return [dict(item) for item in _sections_list()]

    monkeypatch.setattr(accompaniment, "_pipeline_beats", fake_beats)
    monkeypatch.setattr(accompaniment, "_pipeline_key", fake_key)
    monkeypatch.setattr(accompaniment, "_pipeline_chords", fake_chords)
    monkeypatch.setattr(accompaniment, "_pipeline_structure", fake_structure)
    monkeypatch.setattr(accompaniment, "is_available", lambda: False)
    return calls


def _fake_transformers() -> tuple[types.SimpleNamespace, list[dict]]:
    """伪造一个只含 AutoModel 的 transformers 模块，并记录 transcribe 调用。"""
    log: list[dict] = []

    class _FakeModel:
        def to(self, device):
            log.append({"event": "to", "device": device})
            return self

        def eval(self):
            log.append({"event": "eval"})
            return self

        def transcribe(self, audio, output_dir=None):
            log.append({"event": "transcribe", "audio": audio, "output_dir": output_dir})
            return {
                "beats": [0.0, 0.5, 1.0, 1.5],
                "downbeats": [0.0],
                "tempo": 120.0,
                "key": {"tonic": "C", "mode": "major", "confidence": 0.9},
                "chords": [
                    {"start": 0.0, "end": 1.0, "symbol": "C", "confidence": 0.9},
                    {"start": 1.0, "end": 2.0, "symbol": "F", "confidence": 0.8},
                ],
                "sections": [
                    {"start": 0.0, "end": 2.0, "kind": "intro", "label": "前奏", "confidence": 0.7}
                ],
                "notes": [{"start": 0.0, "end": 0.5, "midi": 67, "confidence": 0.9}],
                "abc": "X:1\nT:sheetsage2\nK:C\n",
            }

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            log.append({"event": "from_pretrained", "path": path, "kwargs": kwargs})
            return _FakeModel()

    return types.SimpleNamespace(AutoModel=_AutoModel), log


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module, log = _fake_transformers()
    monkeypatch.setenv(accompaniment.SHEETSAGE2_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(accompaniment, "_import_transformers", lambda: module)
    return log


# --------------------------------------------------------------------------- #
# 可用性探测
# --------------------------------------------------------------------------- #


def test_is_available_requires_local_weights(monkeypatch, tmp_path):
    monkeypatch.delenv(accompaniment.SHEETSAGE2_DIR_ENV, raising=False)
    assert accompaniment.is_available() is False

    monkeypatch.setenv(accompaniment.SHEETSAGE2_DIR_ENV, str(tmp_path / "missing"))
    assert accompaniment.is_available() is False

    # 目录有效但 transformers 未安装时仍然不可用（本环境即如此）
    monkeypatch.setenv(accompaniment.SHEETSAGE2_DIR_ENV, str(tmp_path))
    expected = accompaniment._import_transformers() is not None
    assert accompaniment.is_available() is expected
    assert accompaniment.sheetsage2_dir() == tmp_path


def test_sheetsage2_dir_rejects_file(tmp_path):
    target = tmp_path / "weights.txt"
    target.write_text("not a dir", encoding="utf-8")
    assert accompaniment.sheetsage2_dir() is None


# --------------------------------------------------------------------------- #
# events 规范化
# --------------------------------------------------------------------------- #


def test_parse_events_list_shape():
    events = [
        {"type": "beat", "time": 1.0},
        {"type": "beat", "time": 1.5},
        {"type": "downbeat", "time": 1.0},
        {"type": "chord", "start": 1.0, "end": 2.5, "symbol": "C", "confidence": 0.8},
        {"type": "chord", "start": 2.5, "end": 4.0, "symbol": "G"},
        {"type": "section", "start": 0.0, "end": 4.0, "kind": "verse", "label": "主歌"},
        {"type": "note", "start": 1.0, "end": 1.5, "midi": 60},
        {"type": "note", "start": 1.5, "end": 2.0, "pitch": "E4"},
    ]
    result = accompaniment.parse_events(events)

    assert isinstance(result, AccompanimentResult)
    assert isinstance(result.beats, BeatGrid)
    assert result.beats.bpm == pytest.approx(120.0)
    assert result.beats.downbeats == [1.0]
    assert [chord.symbol for chord in result.chords] == ["C", "G"]
    assert result.chords[0].end == pytest.approx(2.5)
    assert result.sections[0].kind == "verse"
    assert [note.midi for note in result.melody_skeleton] == [60, 64]
    assert result.events
    assert result.engine == "events"
    assert "X:1" in result.abc


def test_parse_events_dict_shape_with_aliases():
    payload = {
        "tempo": 100,
        "time_signature": "3/4",
        "beat_times": [0.0, 0.6, 1.2, 1.8],
        "downbeats": [0.0, 1.8],
        "harmonies": [{"time": 0.0, "chord": "Am", "duration": 1.2}],
        "segments": [{"start": 0.0, "end": 1.8, "label": "主歌"}],
        "melody": [{"start": 0.0, "end": 0.6, "note": "A4"}],
        "key": {"tonic": "A", "mode": "minor", "confidence": 0.6},
        "abc": "X:1\nK:Am\n",
        "midi": "out.mid",
    }
    result = accompaniment.parse_events(payload)

    assert result.beats.bpm == pytest.approx(100.0)
    assert result.beats.beats_per_bar == 3
    assert result.beats.time_signature == "3/4"
    assert isinstance(result.key, KeySignature)
    assert (result.key.tonic, result.key.mode) == ("A", "minor")
    assert result.key.fifths == 0  # A 小调无升降
    assert [chord.symbol for chord in result.chords] == ["Am"]
    assert result.chords[0].end == pytest.approx(1.2)
    assert result.melody_skeleton[0].midi == 69
    assert result.abc == "X:1\nK:Am\n"  # 自带 abc 优先，不重算


def test_parse_events_nested_payload_and_bare_numbers():
    payload = {
        "result": {
            "beats": [0.0, 1.0, 2.0],
            "chords": [{"start": 0.0, "end": 2.0, "symbol": "C"}],
            "events": [3.0, {"type": "beat", "time": 4.0}],
        }
    }
    result = accompaniment.parse_events(payload)
    assert result.beats.beats == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [chord.symbol for chord in result.chords] == ["C"]


def test_normalize_events_closes_open_chords_and_parses_key_string():
    payload = accompaniment.normalize_events(
        {
            "key": "F# minor",
            "chords": [{"start": 0.0, "symbol": "F#m"}, {"start": 2.0, "symbol": "D"}],
            "sections": [{"start": 0.0, "end": 2.0}],
        }
    )
    assert payload["key"].mode == "minor"
    assert payload["key"].fifths == 3  # 升 F 小调 3 个升号
    assert payload["chords"][0].end == pytest.approx(2.0)  # 缺 end 补到下一个和弦起点
    assert payload["chords"][1].end > 2.0
    assert payload["sections"][0].kind == "other"


def test_parse_events_rejects_unsupported_input():
    with pytest.raises(TypeError):
        accompaniment.parse_events(42)  # type: ignore[arg-type]


def test_parse_events_empty_input_is_valid_but_empty():
    result = accompaniment.parse_events(None)
    assert result.chords == [] and result.melody_skeleton == []
    assert result.beats.bpm == pytest.approx(120.0)
    assert result.abc.startswith("X:1")


# --------------------------------------------------------------------------- #
# ABC 调试输出
# --------------------------------------------------------------------------- #


def _harmonic_events() -> dict:
    return {
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "downbeats": [0.0, 2.0],
        "tempo": 120.0,
        "key": {"tonic": "C", "mode": "major"},
        "chords": [
            {"start": 0.0, "end": 1.0, "symbol": "C"},
            {"start": 1.0, "end": 2.0, "symbol": "Am"},
            {"start": 2.0, "end": 3.0, "symbol": "F"},
            {"start": 3.0, "end": 4.0, "symbol": "G"},
        ],
        "notes": [
            {"start": 0.0, "end": 0.5, "midi": 60},
            {"start": 0.5, "end": 1.0, "midi": 62},
            {"start": 1.0, "end": 1.5, "midi": 64},
        ],
    }


def test_abc_from_result_has_headers_chords_and_notes():
    result = accompaniment.parse_events(_harmonic_events())
    abc = accompaniment.abc_from_result(result)

    assert abc.startswith("X:1\n")
    for header in ("T:", "M:4/4", "L:1/8", "Q:1/4=120", "K:C"):
        assert header in abc
    assert '"C"' in abc
    assert "|" in abc
    body = abc.split("K:C", 1)[1]
    assert "C" in body  # C4 的 ABC 写字母
    assert "D" in body


def test_abc_allows_empty_melody():
    result = accompaniment.parse_events(
        {
            "beats": [0.0, 0.5, 1.0, 1.5],
            "chords": [{"start": 0.0, "end": 2.0, "symbol": "Dm"}],
        }
    )
    abc = accompaniment.abc_from_result(result)
    body = abc.split("K:", 1)[1]
    assert '"Dm"' in abc
    assert "z" in body  # 无旋律 → 整小节休止 + 和弦标记


def test_abc_uses_flat_spelling_for_flat_keys():
    result = accompaniment.parse_events(
        {
            "beats": [0.0, 0.5, 1.0, 1.5],
            "key": {"tonic": "F", "mode": "major"},
            "notes": [{"start": 0.0, "end": 0.5, "midi": 70}],  # Bb4
        }
    )
    abc = accompaniment.abc_from_result(result)
    assert "_B" in abc


# --------------------------------------------------------------------------- #
# 降级路径
# --------------------------------------------------------------------------- #


def test_transcribe_degrades_to_pipeline(monkeypatch, tmp_path):
    source = tmp_path / "accomp.wav"
    source.write_bytes(b"RIFF")
    calls = _install_fake_pipeline(monkeypatch)

    result = accompaniment.transcribe_accompaniment(source, work_dir=tmp_path / "work")

    assert result.engine == "beats+chroma"
    assert result.requested == "SheetSage2"
    assert result.elapsed_seconds is not None and result.elapsed_seconds >= 0
    assert any("SheetSage2" in warning for warning in result.warnings)
    assert [chord.symbol for chord in result.chords] == ["G", "C", "Am"]
    assert (result.key.tonic, result.key.mode) == ("G", "major")
    assert result.beats.bpm == pytest.approx(120.0)
    assert [section.kind for section in result.sections] == ["intro", "verse"]
    assert calls["beats"] == [str(source)]
    assert calls["key"] == [str(source)] and calls["key_refine"] == [str(source)]
    assert calls["chords"][0]["total_seconds"] == pytest.approx(8.5)
    assert calls["structure"][0]["n_chords"] == 3
    assert (tmp_path / "work").is_dir()


def test_falls_back_to_full_mix_when_stem_missing(monkeypatch, tmp_path):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"RIFF")
    calls = _install_fake_pipeline(monkeypatch)

    result = accompaniment.transcribe_accompaniment(
        tmp_path / "missing_stem.wav", work_dir=tmp_path, full_mix=mix
    )

    assert calls["beats"] == [str(mix)]
    assert any("整曲混音" in warning for warning in result.warnings)


def test_requires_some_audio_path(tmp_path):
    with pytest.raises(ValueError):
        accompaniment.transcribe_accompaniment(None, work_dir=tmp_path)


def test_missing_source_raises_instead_of_pretending(monkeypatch, tmp_path):
    def boom(source):
        raise FileNotFoundError(f"no such file: {source}")

    monkeypatch.setattr(accompaniment, "_pipeline_beats", boom)
    monkeypatch.setattr(accompaniment, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="节拍分析不可用"):
        accompaniment.transcribe_accompaniment(tmp_path / "nope.wav", work_dir=tmp_path)


def test_progress_supports_both_signatures(monkeypatch, tmp_path):
    _install_fake_pipeline(monkeypatch)
    triples: list[tuple[str, float]] = []
    accompaniment.transcribe_accompaniment(
        tmp_path / "a.wav",
        work_dir=tmp_path,
        progress=lambda stage, percent, message: triples.append((stage, percent)),
    )
    assert triples and triples[-1][1] == 100.0

    dicts: list[dict] = []
    accompaniment.transcribe_accompaniment(
        tmp_path / "a.wav", work_dir=tmp_path, progress=dicts.append
    )
    assert dicts and dicts[-1]["percent"] == 100.0


# --------------------------------------------------------------------------- #
# SheetSage2 路径
# --------------------------------------------------------------------------- #


def test_sheetsage2_success_path_skips_pipeline(monkeypatch, tmp_path):
    log = _install_fake_transformers(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("SheetSage2 可用时不应走音频管线")

    for name in ("_pipeline_beats", "_pipeline_key", "_pipeline_chords", "_pipeline_structure"):
        monkeypatch.setattr(accompaniment, name, forbidden)

    result = accompaniment.transcribe_accompaniment(tmp_path / "accomp.wav", work_dir=tmp_path)

    assert result.engine == "sheetsage2"
    assert result.requested == "SheetSage2"
    assert [chord.symbol for chord in result.chords] == ["C", "F"]
    assert result.key.tonic == "C"
    assert result.melody_skeleton[0].midi == 67
    assert result.abc == "X:1\nT:sheetsage2\nK:C\n"
    assert any(entry["event"] == "transcribe" for entry in log)
    assert any(entry.get("path") == str(tmp_path) for entry in log)


def test_sheetsage2_inference_failure_degrades(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, tmp_path)

    class _BoomModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def transcribe(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _BoomModel()

    monkeypatch.setattr(
        accompaniment, "_import_transformers", lambda: types.SimpleNamespace(AutoModel=_AutoModel)
    )
    calls = _install_fake_pipeline(monkeypatch)
    monkeypatch.setattr(accompaniment, "is_available", lambda: True)

    result = accompaniment.transcribe_accompaniment(tmp_path / "a.wav", work_dir=tmp_path)

    assert result.engine == "beats+chroma"
    assert any("降级" in warning for warning in result.warnings)
    assert calls["beats"] == [str(tmp_path / "a.wav")]


def test_sheetsage2_empty_output_degrades(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, tmp_path)

    class _EmptyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def transcribe(self, *args, **kwargs):
            return {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _EmptyModel()

    monkeypatch.setattr(
        accompaniment, "_import_transformers", lambda: types.SimpleNamespace(AutoModel=_AutoModel)
    )
    _install_fake_pipeline(monkeypatch)
    monkeypatch.setattr(accompaniment, "is_available", lambda: True)

    result = accompaniment.transcribe_accompaniment(tmp_path / "a.wav", work_dir=tmp_path)

    assert result.engine == "beats+chroma"
    assert any("未返回可用事件" in warning for warning in result.warnings)


def test_sheetsage2_transcribe_without_output_dir_kwarg(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, tmp_path)

    class _NoKwargModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def transcribe(self, audio):
            return {"beats": [0.0, 0.5], "chords": [{"start": 0.0, "end": 1.0, "symbol": "C"}]}

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _NoKwargModel()

    monkeypatch.setattr(
        accompaniment, "_import_transformers", lambda: types.SimpleNamespace(AutoModel=_AutoModel)
    )

    result = accompaniment.transcribe_accompaniment(tmp_path / "a.wav", work_dir=tmp_path)

    assert result.engine == "sheetsage2"
    assert [chord.symbol for chord in result.chords] == ["C"]


def test_sheetsage2_midi_artifact_is_recorded(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, tmp_path)

    class _MidiModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def transcribe(self, audio, output_dir=None):
            Path(output_dir, "result.mid").write_bytes(b"MThd")
            return {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _MidiModel()

    monkeypatch.setattr(
        accompaniment, "_import_transformers", lambda: types.SimpleNamespace(AutoModel=_AutoModel)
    )

    result = accompaniment.transcribe_accompaniment(tmp_path / "a.wav", work_dir=tmp_path)

    assert result.engine == "sheetsage2"
    artifacts = [event for event in result.events if event.get("type") == "artifact"]
    assert artifacts and artifacts[0]["kind"] == "midi"
    assert artifacts[0]["path"].endswith("result.mid")


def test_transcribe_with_sheetsage_returns_none_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv(accompaniment.SHEETSAGE2_DIR_ENV, raising=False)
    monkeypatch.setattr(accompaniment, "_import_transformers", lambda: None)
    sink: list[str] = []

    result = accompaniment.transcribe_with_sheetsage(
        tmp_path / "a.wav", work_dir=tmp_path, warnings=sink
    )

    assert result is None
    assert sink and "SheetSage2" in sink[0]


def test_transcribe_with_sheetsage_returns_result_when_available(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, tmp_path)

    result = accompaniment.transcribe_with_sheetsage(tmp_path / "a.wav", work_dir=tmp_path)

    assert isinstance(result, AccompanimentResult)
    assert result.engine == "sheetsage2"
    assert [chord.symbol for chord in result.chords] == ["C", "F"]


def test_midi_without_pretty_midi_reports_reason(tmp_path):
    if importlib.util.find_spec("pretty_midi") is not None:
        pytest.skip("环境已安装 pretty_midi，跳过缺失分支")
    notes, reason = accompaniment._read_midi_notes(tmp_path / "x.mid")
    assert notes == []
    assert "pretty_midi" in reason


def test_emit_swallows_callback_errors(tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("callback bug")

    accompaniment._emit(broken, "stage", 10.0, "message")  # 不抛

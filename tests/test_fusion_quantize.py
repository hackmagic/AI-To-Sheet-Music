"""量化契约：DuET 缺失时显式降级、时值合法化、provenance 记录 requested vs engine。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aitsm.contracts import (
    BeatGrid,
    ChordEvent,
    KeySignature,
    MelodyNote,
    Section,
    SymbolicScore,
)
from aitsm.fusion import (
    DUET_ENV,
    ENGINE_HMM,
    quantize_symbolic,
    write_musicxml,
)
from aitsm.fusion import quantize as quantize_module
from aitsm.pipeline.quantize import legal_durations

LEGAL = legal_durations()


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


def _beats(bpm: float = 120.0, bars: int = 2) -> BeatGrid:
    period = 60.0 / bpm
    count = bars * 4
    return BeatGrid(
        bpm=bpm,
        beats=[index * period for index in range(count + 1)],
        downbeats=[bar * 4 * period for bar in range(bars + 1)],
        beats_per_bar=4,
        time_signature="4/4",
        confidence=0.8,
    )


def _score() -> SymbolicScore:
    """故意让人声稍微不齐（±20–90 ms），模拟 pyin 的抖动。"""
    beats = _beats()
    return SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C", mode="major", confidence=0.7, fifths=0),
        chords=[
            ChordEvent(start=0.0, end=2.0, symbol="C", confidence=0.6),
            ChordEvent(start=2.0, end=4.0, symbol="G", confidence=0.6),
        ],
        sections=[Section(start=0.0, end=4.0, kind="verse", label="Verse 1")],
        melody=[
            MelodyNote(start=0.02, end=0.55, midi=60, confidence=0.8, lyric="Hel"),
            MelodyNote(start=0.53, end=1.08, midi=62, confidence=0.7, lyric="lo"),
            MelodyNote(start=1.1, end=1.6, midi=64, confidence=0.6),
            MelodyNote(start=2.03, end=2.49, midi=65, confidence=0.5),
            MelodyNote(start=2.52, end=3.4, midi=67, confidence=0.5),
        ],
    )


def _dur_quarters(notes: list[MelodyNote], beat_period: float) -> list[float]:
    return [(note.end - note.start) / beat_period for note in notes]


def _assert_legal(notes: list[MelodyNote], beat_period: float) -> None:
    for quarters in _dur_quarters(notes, beat_period):
        assert min(abs(quarters - value) for value in LEGAL) < 1e-4, f"时值 {quarters} 不在 legal_durations()"


# --------------------------------------------------------------------------- #
# probe / is_available：能力探测明确降级
# --------------------------------------------------------------------------- #


def test_probe_degrades_when_transformers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quantize_module, "_find_spec", lambda name: False)
    monkeypatch.delenv(DUET_ENV, raising=False)

    result = quantize_module.probe()
    assert result["available"] is False
    assert "transformers" in result["reason"]
    assert result["transformers"] is False
    assert quantize_module.is_available() is False


def test_probe_reports_missing_local_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # transformers 在、权重目录在，但适配器模块不存在
    monkeypatch.setattr(quantize_module, "_find_spec", lambda name: name == "transformers")
    monkeypatch.setenv(DUET_ENV, str(tmp_path))

    result = quantize_module.probe()
    assert result["available"] is False
    assert "适配器" in result["reason"]
    assert result["duet_dir_exists"] is True
    assert result["adapter"] is False


def test_probe_accepts_adapter_when_everything_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(quantize_module, "_find_spec", lambda name: True)
    monkeypatch.setenv(DUET_ENV, str(tmp_path))

    result = quantize_module.probe()
    assert result["available"] is True
    assert result["reason"] == ""
    assert quantize_module.is_available() is True


# --------------------------------------------------------------------------- #
# quantize_symbolic：降级路径 + 合法时值
# --------------------------------------------------------------------------- #


def test_quantize_symbolic_falls_back_to_hmm_and_legalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quantize_module,
        "probe",
        lambda device="auto": {"available": False, "reason": "未安装 transformers（离线）", "checks": {}, "device": "cpu"},
    )
    score = _score()
    original = score.to_dict()

    quantized, provenance = quantize_symbolic(score)

    assert provenance.layer == "quantization"
    assert provenance.engine == ENGINE_HMM
    assert provenance.requested == "duet"
    assert provenance.degraded is True, "requested≠engine 必须被标记为降级"
    assert "transformers" in provenance.detail["fallback_reason"]
    assert provenance.detail["stats"]["engine"] == "metrical-hmm"
    assert score.to_dict() == original, "不得就地修改输入总谱"

    assert len(quantized.melody) == len(score.melody), "量化不增删音符"
    assert [note.midi for note in quantized.melody] == [60, 62, 64, 65, 67]
    beat_period = quantized.beats.beat_period
    _assert_legal(quantized.melody, beat_period)

    starts = [note.start for note in quantized.melody]
    assert starts == sorted(starts), "量化后起点必须单调不减"
    for note in quantized.melody:
        assert note.end >= note.start

    assert quantized.meta["quantization_engine"] == ENGINE_HMM
    assert quantized.meta["quantization_degraded"] is True
    assert quantized.source["quantizer"] == ENGINE_HMM
    assert quantized.get_provenance("quantization") is not None
    codes = {(item.get("stage"), item.get("code")) for item in quantized.warnings}
    assert ("quantization", "duet_unavailable") in codes


def test_quantize_symbolic_uses_adapter_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAdapter:
        @staticmethod
        def quantize_melody(**kwargs: Any) -> list[dict]:
            notes = []
            for note in kwargs["notes"]:
                start = round(note["start"] / 0.5) * 0.5
                notes.append({**note, "start": start, "end": start + 0.5})
            return notes

    monkeypatch.setattr(
        quantize_module,
        "probe",
        lambda device="auto": {"available": True, "reason": "", "checks": {}, "device": "cpu"},
    )
    monkeypatch.setitem(__import__("sys").modules, "aitsm.fusion.duet", FakeAdapter)

    quantized, provenance = quantize_symbolic(_score())
    assert provenance.engine == "duet"
    assert provenance.degraded is False
    _assert_legal(quantized.melody, quantized.beats.beat_period)
    assert quantized.meta["quantization_degraded"] is False


def test_quantize_symbolic_survives_broken_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: Any) -> list[dict]:
        raise RuntimeError("权重文件损坏")

    class BrokenAdapter:
        quantize_melody = staticmethod(boom)

    monkeypatch.setattr(
        quantize_module,
        "probe",
        lambda device="auto": {"available": True, "reason": "", "checks": {}, "device": "cpu"},
    )
    monkeypatch.setitem(__import__("sys").modules, "aitsm.fusion.duet", BrokenAdapter)

    quantized, provenance = quantize_symbolic(_score())
    assert provenance.engine == ENGINE_HMM
    assert provenance.degraded is True
    codes = {(item.get("stage"), item.get("code")) for item in quantized.warnings}
    assert ("quantization", "duet_failed") in codes
    _assert_legal(quantized.melody, quantized.beats.beat_period)


def test_quantized_score_is_valid_musicxml() -> None:
    quantized, _ = quantize_symbolic(_score())
    xml = write_musicxml(quantized)
    restored = SymbolicScore.from_musicxml(xml, sidecar=quantized.to_dict())
    assert len(restored.melody) == len(quantized.melody)
    _assert_legal(restored.melody, restored.beats.beat_period)


def test_quantize_preserves_chords_key_and_lyrics() -> None:
    score = _score()
    quantized, _ = quantize_symbolic(score)
    assert [chord.symbol for chord in quantized.chords] == ["C", "G"]
    assert [chord.start for chord in quantized.chords] == [0.0, 2.0]
    assert quantized.key.tonic == "C" and quantized.key.fifths == 0
    assert [note.lyric for note in quantized.melody] == ["Hel", "lo", "", "", ""]
    assert [section.label for section in quantized.sections] == ["Verse 1"]


def test_quantize_empty_melody_is_noop() -> None:
    score = _score()
    score.melody = []
    quantized, provenance = quantize_symbolic(score)
    assert quantized.melody == []
    assert provenance.engine == ENGINE_HMM
    assert [chord.symbol for chord in quantized.chords] == ["C", "G"]


def test_quantize_rejects_non_score() -> None:
    with pytest.raises(TypeError):
        quantize_symbolic({"beats": {}})  # type: ignore[arg-type]

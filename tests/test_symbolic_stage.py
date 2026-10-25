"""v2 符号契约阶段测试：analysis → VocalResult/AccompanimentResult → SymbolicScore → 落盘。

不跑音频、不下载模型；quantize 走 HMM 降级（本机没有 DuET 权重）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.pipeline import symbolic_stage

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "analysis.json"

BPM = 90.0
BEAT = 60.0 / BPM


def _synthetic_analysis() -> dict:
    beats = [round(index * BEAT, 4) for index in range(9)]
    return {
        "schema": "aitsm.analysis/v1",
        "created_at": "2024-01-01T00:00:00+00:00",
        "source": {"original_name": "synth.wav", "sha256": "0" * 64, "path": "synth.wav",
                   "size_bytes": 1, "source_duration": 10.0},
        "audio": {"analysis_wav": "synth.wav", "playback_audio": "synth.wav", "duration": 10.0,
                  "sample_rate": 44100, "channels": 2},
        "stems": {"engine": "demucs", "model": "htdemucs", "vocals": "v.wav", "other": "o.wav"},
        "beats": {"bpm": BPM, "beats": beats, "downbeats": [beats[0], beats[4], beats[8]],
                  "beats_per_bar": 4, "time_signature": "4/4", "confidence": 0.8,
                  "pickup_beats": 0, "first_downbeat_sec": beats[0]},
        "key": {"tonic": "C", "mode": "major", "name": "C major", "num_sharps_flats": 0,
                "confidence": 0.7},
        "chords": [
            {"start": beats[0], "end": beats[4], "symbol": "C", "confidence": 0.9},
            {"start": beats[4], "end": beats[8], "symbol": "G", "confidence": 0.8},
        ],
        "sections": [{"start": beats[0], "end": beats[8], "kind": "verse", "label": "A",
                      "energy": 0.5, "repeat_of": None, "confidence": 0.6}],
        "lyrics": [{"start": beats[0], "end": beats[4], "text": "hello world",
                    "words": [{"start": beats[0], "end": beats[2], "text": "hello"},
                              {"start": beats[2], "end": beats[4], "text": "world"}],
                    "confidence": 0.7, "interjection": False}],
        "melody": {"available": True, "engine": "pyin", "notes": [
            {"start": beats[0], "end": beats[1], "midi": 60, "confidence": 0.9, "lyric": "hello"},
            {"start": beats[1], "end": beats[2], "midi": 62, "confidence": 0.8, "lyric": ""},
            {"start": beats[2], "end": beats[4], "midi": 64, "confidence": 0.7, "lyric": "world"},
        ]},
        "tools": {"lyrics": {"engine": "whisper", "available": True},
                  "melody": {"engine": "pyin", "note_count": 3},
                  "separate": {"engine": "demucs", "model": "htdemucs"}},
        "warnings": [],
        "meta": {"engines": {"separation": "demucs", "chords": "chroma-template+viterbi"}},
    }


def test_vocal_result_from_analysis_reports_degradation() -> None:
    vocal = symbolic_stage.vocal_from_analysis(_synthetic_analysis())
    assert vocal.requested == "VocalParse"
    assert vocal.engine == "pyin"
    assert len(vocal.notes) == 3
    assert len(vocal.lyrics) == 1
    assert vocal.lyrics[0].words[0].text == "hello"
    assert any("VocalParse" in message for message in vocal.warnings)


def test_accompaniment_result_from_analysis_keeps_time_skeleton() -> None:
    accompaniment = symbolic_stage.accompaniment_from_analysis(_synthetic_analysis())
    assert accompaniment.requested == "SheetSage2"
    assert accompaniment.beats.bpm == pytest.approx(BPM, abs=1e-6)
    assert len(accompaniment.beats.beats) == 9
    assert [chord.symbol for chord in accompaniment.chords] == ["C", "G"]
    assert accompaniment.sections and accompaniment.sections[0].label == "A"
    assert any("SheetSage2" in message for message in accompaniment.warnings)


def test_build_symbolic_saves_contract(tmp_path: Path) -> None:
    result = symbolic_stage.build_symbolic(_synthetic_analysis(), tmp_path)
    assert result["available"] is True
    assert result["stats"]["melody_notes"] == 3
    assert result["stats"]["chords"] == 2
    assert Path(result["paths"]["json"]).is_file()
    layers = [item["layer"] for item in result["provenance"]]
    assert "fusion" in layers
    # 本机没有 DuET 权重：量化必须如实降级
    assert "quantization" in result["degraded_layers"]
    assert "vocal_transcription" in result["degraded_layers"]


def test_build_symbolic_falls_back_when_fusion_breaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aitsm.fusion import skeleton

    def boom(*args, **kwargs):
        raise RuntimeError("fusion offline")

    monkeypatch.setattr(skeleton, "fuse", boom)
    result = symbolic_stage.build_symbolic(_synthetic_analysis(), tmp_path)
    assert result["available"] is True
    assert any("fusion" in message for message in result["warnings"])
    assert Path(result["paths"]["json"]).is_file()


@pytest.mark.skipif(not ARTIFACTS.is_file(), reason="缺少 artifacts/analysis.json")
def test_real_artifact_produces_symbolic_score(tmp_path: Path) -> None:
    analysis = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    result = symbolic_stage.build_symbolic(analysis, tmp_path, quantize=False)
    assert result["available"] is True
    assert result["stats"]["melody_notes"] > 200
    assert result["stats"]["chords"] > 20
    assert result["stats"]["lyric_notes"] > 100

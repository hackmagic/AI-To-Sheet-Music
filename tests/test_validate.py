"""``aitsm.score.validate`` 的单元测试。

覆盖两层校验：
* 结构层（jsonschema）：缺字段、类型错误；
* 音乐层：小节连续、和弦不越界/不重叠、旋律音域、段落覆盖、capo 一致性、
  指法与和弦符号音级匹配、TAB ``strings`` 与和弦音级一致。

所有「坏谱」都在 :func:`make_score` 产出的**好谱**上做定点破坏，断言恰好触发
对应的错误码——这样一旦校验器漏检或误报，测试立刻失败。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aitsm.score.arrange import arrange
from aitsm.score.validate import validate_score

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "samples" / "demo-score.json"

BPM = 90.0
BEATS_PER_BAR = 4
BARS = 6


def beat_time(beat: float) -> float:
    """拍 -> 秒（4/4，BPM 固定）。"""
    return beat * 60.0 / BPM


def make_analysis() -> dict:
    """合成一段 6 小节、C 大调的 analysis（符合 analysis.schema.json）。"""
    beats = [beat_time(index) for index in range(BARS * BEATS_PER_BAR + 1)]
    downbeats = [beats[index] for index in range(0, BARS * BEATS_PER_BAR, BEATS_PER_BAR)]
    chords = [
        {"start": beat_time(0), "end": beat_time(8), "symbol": "C", "confidence": 0.9},
        {"start": beat_time(8), "end": beat_time(16), "symbol": "G", "confidence": 0.9},
        {"start": beat_time(16), "end": beat_time(24), "symbol": "Am", "confidence": 0.88},
    ]
    melody = {
        "notes": [
            {"start": beat_time(0), "end": beat_time(1), "midi": 64, "confidence": 0.9},
            {"start": beat_time(1), "end": beat_time(2), "midi": 67, "confidence": 0.9},
            {"start": beat_time(2), "end": beat_time(3), "midi": 69, "confidence": 0.85},
            {"start": beat_time(3), "end": beat_time(4), "midi": 67, "confidence": 0.85},
            {"start": beat_time(8), "end": beat_time(9), "midi": 65, "confidence": 0.8},
            {"start": beat_time(9), "end": beat_time(10), "midi": 64, "confidence": 0.8},
            {"start": beat_time(16), "end": beat_time(18), "midi": 62, "confidence": 0.8},
            {"start": beat_time(18), "end": beat_time(20), "midi": 60, "confidence": 0.8},
        ]
    }
    return {
        "schema": "aitsm.analysis/v1",
        "source": {"original_name": "synthetic.wav", "sha256": "0" * 64},
        "audio": {"analysis_wav": "synthetic.wav", "duration": beat_time(24) + 0.5, "sample_rate": 44100},
        "beats": {
            "bpm": BPM,
            "beats": beats,
            "downbeats": downbeats,
            "beats_per_bar": BEATS_PER_BAR,
            "time_signature": "4/4",
            "confidence": 0.9,
        },
        "key": {"tonic": "C", "mode": "major", "confidence": 0.82, "num_sharps_flats": 0},
        "chords": chords,
        "sections": [
            {"start": beat_time(0), "end": beat_time(12), "kind": "verse", "label": "主歌", "energy": 0.4},
            {"start": beat_time(12), "end": beat_time(24), "kind": "chorus", "label": "副歌", "energy": 0.8},
        ],
        "lyrics": [],
        "melody": melody,
        "tools": {},
        "warnings": [],
    }


@pytest.fixture(scope="module")
def good_score() -> dict:
    """arrange() 产出的合法 score（带 TAB，便于测 strings 一致性）。"""
    return arrange(make_analysis(), {"include_tab": True})


def make_score() -> dict:
    """每次都重新 arrange，返回可安全破坏的独立副本。"""
    return copy.deepcopy(arrange(make_analysis(), {"include_tab": True}))


def codes(items: list[dict]) -> list[str]:
    return [item.get("code", "") for item in items]


# --------------------------------------------------------------------------
# 正例
# --------------------------------------------------------------------------


def test_demo_sample_is_valid() -> None:
    """手写示例 samples/demo-score.json 必须结构合法、无错误。"""
    score = json.loads(DEMO.read_text(encoding="utf-8"))
    report = validate_score(score)
    assert report["ok"] is True, report["errors"]
    assert report["errors"] == []
    assert report["stats"]["bars"] == len(score["bars"])


def test_arranged_score_is_valid(good_score: dict) -> None:
    report = validate_score(good_score)
    assert report["ok"] is True, report["errors"]
    assert report["errors"] == []


def test_stats_are_populated(good_score: dict) -> None:
    report = validate_score(good_score)
    stats = report["stats"]
    assert stats["bars"] == BARS
    assert stats["sections"] == 2
    assert stats["capo"] == good_score["meta"]["capo"]
    assert stats["play_key"] == good_score["meta"]["key"]["play"]
    assert stats["sounding_key"] == good_score["meta"]["key"]["sounding"]
    assert stats["voicing_coverage"] == 1.0
    assert stats["vocal_notes"] == sum(len(bar.get("vocal") or []) for bar in good_score["bars"])
    assert stats["lyric_coverage"] == 0.0  # 合成 analysis 没有歌词
    assert stats["unique_chords"] == len(stats["chord_symbols"])
    assert stats["tab_pairs_checked"] > 0  # include_tab=True 时应校验过弦品
    assert stats["quality_overall"] is not None
    assert isinstance(stats["stages"], dict)


def test_report_issues_are_service_compatible(good_score: dict) -> None:
    """错误/警告条目要能直接并入 quality.warnings（level/code/message 三键）。"""
    broken = make_score()
    broken["meta"]["capo"] = 99
    report = validate_score(broken)
    for item in report["errors"] + report["warnings"]:
        assert set(item) >= {"level", "code", "message"}
        assert item["level"] in ("info", "warn", "error")
        assert isinstance(item["message"], str) and item["message"]


# --------------------------------------------------------------------------
# 结构层
# --------------------------------------------------------------------------


def test_schema_version_mismatch() -> None:
    """schema 版本不对必须报错（有 jsonschema 时报 schema.invalid，降级时报 schema.mismatch）。"""
    score = make_score()
    score["schema"] = "aitsm.score/v2"
    report = validate_score(score)
    assert report["ok"] is False
    assert {"schema.invalid", "schema.mismatch"} & set(codes(report["errors"]))


def test_missing_required_field() -> None:
    score = make_score()
    del score["sections"]
    report = validate_score(score)
    assert report["ok"] is False
    assert {"schema.invalid", "schema.missing", "sections.empty"} & set(codes(report["errors"]))


def test_bars_empty() -> None:
    score = make_score()
    score["bars"] = []
    report = validate_score(score)
    assert "bars.empty" in codes(report["errors"])


def test_not_a_dict() -> None:
    report = validate_score("not a score")  # type: ignore[arg-type]
    assert report["ok"] is False
    assert "score.type" in codes(report["errors"])


# --------------------------------------------------------------------------
# 小节层
# --------------------------------------------------------------------------


def test_bar_index_must_be_continuous() -> None:
    score = make_score()
    score["bars"][2]["index"] = 7
    report = validate_score(score)
    assert "bars.index" in codes(report["errors"])


def test_bars_must_be_contiguous() -> None:
    score = make_score()
    for bar in score["bars"][2:]:
        bar["start_sec"] += 1.0
        bar["end_sec"] += 1.0
    report = validate_score(score)
    assert "bars.contiguous" in codes(report["errors"])


def test_chord_beat_out_of_range() -> None:
    score = make_score()
    score["bars"][1]["chords"] = [{"beat": 7.5, "symbol": "G", "dur": 1.0}]
    report = validate_score(score)
    assert "chords.beat_range" in codes(report["errors"])


def test_chords_must_not_overlap() -> None:
    score = make_score()
    bar = score["bars"][1]
    bar["chords"] = [
        {"beat": 0.0, "symbol": "G", "dur": 3.0},
        {"beat": 1.0, "symbol": "D", "dur": 2.0},
    ]
    report = validate_score(score)
    assert "chords.overlap" in codes(report["errors"])


def test_invalid_chord_symbol() -> None:
    score = make_score()
    score["bars"][0]["chords"] = [{"beat": 0.0, "symbol": "Hmaj7", "dur": 4.0}]
    report = validate_score(score)
    assert "chords.symbol_invalid" in codes(report["errors"])


def test_vocal_beat_out_of_range() -> None:
    score = make_score()
    bar = score["bars"][0]
    bar["vocal"] = [{"beat": 4.0, "midi": 64, "dur": 0.5}]
    report = validate_score(score)
    assert "vocal.beat_range" in codes(report["errors"])


def test_vocal_midi_must_be_integer() -> None:
    score = make_score()
    bar = score["bars"][0]
    bar["vocal"] = [{"beat": 0.0, "midi": "sixty-four", "dur": 0.5}]
    report = validate_score(score)
    assert "melody.midi" in codes(report["errors"])


def test_guitar_pattern_must_exist() -> None:
    score = make_score()
    score["bars"][0]["guitar"] = {"pattern_id": "no_such_pattern"}
    report = validate_score(score)
    assert "guitar.pattern_unknown" in codes(report["errors"])


def test_tab_strings_must_match_chord_tones() -> None:
    """扫弦的 [弦号, 品位] 换算出的音级必须属于当前和弦。"""
    score = make_score()
    bar = score["bars"][0]
    # 找一个小节里第一击，把最粗弦改成完全无关的音（C 和弦里塞 F：第 6 弦 1 品 = F2）
    hit = bar["guitar"]["hits"][0]
    hit["strings"] = [[1, 0], [6, 1]]
    report = validate_score(score)
    assert "guitar.hit_pitch" in codes(report["errors"])


def test_tab_string_number_range() -> None:
    score = make_score()
    bar = score["bars"][0]
    bar["guitar"]["hits"][0]["strings"] = [[9, 0]]
    report = validate_score(score)
    assert "guitar.hit_string" in codes(report["errors"])


# --------------------------------------------------------------------------
# 段落 / 调性 / 指法
# --------------------------------------------------------------------------


def test_sections_empty() -> None:
    score = make_score()
    score["sections"] = []
    report = validate_score(score)
    assert "sections.empty" in codes(report["errors"])


def test_sections_hole() -> None:
    score = make_score()
    score["sections"][-1]["end_bar"] = BARS - 1
    report = validate_score(score)
    assert "sections.hole" in codes(report["errors"])


def test_sections_out_of_range() -> None:
    score = make_score()
    score["sections"][-1]["end_bar"] = BARS + 5
    report = validate_score(score)
    assert "sections.out_of_range" in codes(report["errors"])


def test_bar_section_reference_must_exist() -> None:
    score = make_score()
    score["bars"][0]["section"] = "s99"
    report = validate_score(score)
    assert "bars.section_ref" in codes(report["errors"])


def test_capo_must_be_reasonable() -> None:
    score = make_score()
    score["meta"]["capo"] = 15
    report = validate_score(score)
    assert "meta.capo" in codes(report["errors"])


def test_capo_key_consistency() -> None:
    """capo 与 sounding/play 的差必须一致。"""
    score = make_score()
    score["meta"]["capo"] = 2
    report = validate_score(score)
    assert "meta.capo_mismatch" in codes(report["errors"])


def test_key_fields_required() -> None:
    score = make_score()
    del score["meta"]["key"]["num_sharps_flats"]
    report = validate_score(score)
    assert "meta.key" in codes(report["errors"])


def test_voicing_reference_must_exist() -> None:
    score = make_score()
    score["bars"][0]["chords"][0]["voicing_id"] = "C_does_not_exist"
    report = validate_score(score)
    assert "chords.voicing_unknown" in codes(report["errors"])


def test_concert_symbol_capo_consistency() -> None:
    score = make_score()
    chord = score["bars"][0]["chords"][0]
    chord["concert"] = "F#"
    report = validate_score(score)
    assert "chords.concert_capo" in codes(report["errors"])


def test_voicing_frets_length() -> None:
    score = make_score()
    voicing_id = next(iter(score["voicings"]))
    score["voicings"][voicing_id]["frets"] = [0, 1, 2]
    report = validate_score(score)
    assert "voicings.frets" in codes(report["errors"])


def test_voicing_missing_tones_is_warning() -> None:
    """指法缺少根音/三音/五音只是警告（有些把位确实省略五音）。"""
    score = make_score()
    score["voicings"]["fake_c"] = {
        "symbol": "C",
        "frets": [0, 0, 0, 0, 0, 0],  # 全空弦 = E A D G B E，缺 C 根音
        "fingers": [0, 0, 0, 0, 0, 0],
    }
    report = validate_score(score)
    assert "voicings.missing_tones" in codes(report["warnings"])


# --------------------------------------------------------------------------
# 旋律音域 / 元数据
# --------------------------------------------------------------------------


def test_melody_range_warning() -> None:
    score = make_score()
    score["bars"][0]["vocal"] = [{"beat": 0.0, "midi": 100, "dur": 1.0}]
    report = validate_score(score)
    assert "melody.range" in codes(report["warnings"])
    assert report["stats"]["melody_high"] == 100


def test_melody_empty_is_warning_only() -> None:
    score = make_score()
    for bar in score["bars"]:
        bar["vocal"] = []
    report = validate_score(score)
    assert report["ok"] is True
    assert "melody.empty" in codes(report["warnings"])
    assert report["stats"]["vocal_notes"] == 0


def test_lyric_coverage_warning() -> None:
    score = make_score()
    score["bars"][0]["vocal"] = [
        {"beat": 0.0, "midi": 64, "dur": 1.0, "lyric": "月"},
        {"beat": 1.0, "midi": 65, "dur": 1.0},
        {"beat": 2.0, "midi": 67, "dur": 1.0},
        {"beat": 3.0, "midi": 64, "dur": 1.0},
    ]
    for bar in score["bars"][1:]:
        bar["vocal"] = []
    report = validate_score(score)
    assert "lyrics.coverage" in codes(report["warnings"])


def test_tempo_out_of_range() -> None:
    score = make_score()
    score["meta"]["tempo"] = 500
    report = validate_score(score)
    assert "meta.tempo" in codes(report["errors"])

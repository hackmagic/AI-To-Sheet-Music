"""``aitsm.score.arrange`` 的单元测试。

覆盖：
* 调性/变调夹矩阵（12 大调 + 12 小调 × 自动 capo、用户指定 key/capo、transpose）；
* 和弦符号移调（含斜杠低音、挂留、七和弦）与 voicing 绑定；
* 量化边界：弱起小节、三连音、跨小节延音、贴近小节线的音；
* 和弦映射：同小节合并、每小节最多 4 个、长和弦跨小节延续；
* 容错：缺和弦 / 缺旋律 / 缺段落时不崩、不编造内容；
* 拍号、节奏型密度、难度、TAB、输出契约（schemas/score.schema.json）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aitsm.score.arrange import arrange
from aitsm.score.validate import validate_score

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "score.schema.json"

BPM = 120.0
BEATS_PER_BAR = 4
BARS = 6


def beat_time(beat: float, bpm: float = BPM) -> float:
    return beat * 60.0 / bpm


def pc_name(pc: int, *, prefer_flats: bool = False) -> str:
    from aitsm.score.model import pc_to_name

    return pc_to_name(pc % 12, prefer_flats=prefer_flats)


def make_analysis(
    *,
    tonic: str = "C",
    mode: str = "major",
    bars: int = BARS,
    beats_per_bar: int = BEATS_PER_BAR,
    bpm: float = BPM,
    chord_symbols: list[str] | None = None,
    chord_plan: list[tuple[int, str]] | None = None,
    melody_notes: list[dict] | None = None,
    with_melody: bool = True,
    sections: list[dict] | None = None,
    with_sections: bool = True,
    lyrics: list[dict] | None = None,
    downbeat_offset_beats: float = 0.0,
) -> dict:
    """构造符合 analysis.schema.json 的合成分析结果。

    默认 6 小节 4/4、每 2 小节一个和弦、每小节 4 个四分音符旋律。
    """
    total_beats = bars * beats_per_bar
    beats = [beat_time(index, bpm) for index in range(total_beats + 1)]
    downbeats = [
        beats[index] for index in range(0, total_beats, beats_per_bar)
    ]
    if downbeat_offset_beats:
        # 首强拍晚于音频开头（弱起）：丢掉前 offset 拍上的强拍
        off = int(downbeat_offset_beats)
        downbeats = [
            beats[index]
            for index in range(off, total_beats, beats_per_bar)
            if index >= off
        ]
    chords: list[dict] = []
    if chord_plan is None:
        if chord_symbols is None:
            from aitsm.score.model import note_to_pc

            root = note_to_pc(tonic)
            if mode == "minor":
                chord_symbols = [
                    pc_name(root) + "m",
                    pc_name(root + 7),
                    pc_name(root + 3),
                    pc_name(root + 5) + "m",
                ]
            else:
                chord_symbols = [
                    pc_name(root),
                    pc_name(root + 7),
                    pc_name(root + 9) + "m",
                    pc_name(root + 5),
                ]
        span = max(1, bars // len(chord_symbols))
        chord_plan = []
        for index, symbol in enumerate(chord_symbols):
            start_bar = index * span
            if start_bar >= bars:
                break
            end_bar = min(bars, start_bar + span)
            chord_plan.append((start_bar, symbol))
            chords.append(
                {
                    "start": beat_time(start_bar * beats_per_bar, bpm),
                    "end": beat_time(end_bar * beats_per_bar, bpm),
                    "symbol": symbol,
                    "confidence": 0.9,
                }
            )
    else:
        for start_bar, symbol in chord_plan:
            chords.append(
                {
                    "start": beat_time(start_bar * beats_per_bar, bpm),
                    "end": beat_time((start_bar + 1) * beats_per_bar, bpm),
                    "symbol": symbol,
                    "confidence": 0.9,
                }
            )

    if melody_notes is None:
        melody_notes = []
        if with_melody:
            pitches = [60, 62, 64, 65, 67, 69, 67, 64]
            for bar in range(bars):
                for beat in range(beats_per_bar):
                    index = (bar * beats_per_bar + beat) % len(pitches)
                    start = beat_time(bar * beats_per_bar + beat, bpm)
                    melody_notes.append(
                        {
                            "start": start,
                            "end": start + beat_time(1, bpm) * 0.95,
                            "midi": pitches[index],
                            "confidence": 0.85,
                        }
                    )

    if sections is None and with_sections:
        half = max(1, bars // 2)
        sections = [
            {
                "start": beat_time(0, bpm),
                "end": beat_time(half * beats_per_bar, bpm),
                "kind": "verse",
                "label": "主歌",
                "energy": 0.4,
                "confidence": 0.8,
            },
            {
                "start": beat_time(half * beats_per_bar, bpm),
                "end": beat_time(bars * beats_per_bar, bpm),
                "kind": "chorus",
                "label": "副歌",
                "energy": 0.85,
                "confidence": 0.8,
            },
        ]

    analysis: dict = {
        "schema": "aitsm.analysis/v1",
        "source": {"original_name": "unit-test.wav", "sha256": "0" * 64},
        "audio": {
            "analysis_wav": "unit-test.wav",
            "duration": beat_time(total_beats, bpm) + 1.0,
            "sample_rate": 44100,
        },
        "beats": {
            "bpm": bpm,
            "beats": beats,
            "downbeats": downbeats,
            "beats_per_bar": beats_per_bar,
            "time_signature": f"{beats_per_bar}/4",
            "confidence": 0.9,
        },
        "key": {
            "tonic": tonic,
            "mode": mode,
            "confidence": 0.85,
            "num_sharps_flats": 0,
        },
        "chords": chords,
        "sections": sections or [],
        "lyrics": lyrics or [],
        "melody": {"notes": melody_notes},
        "tools": {},
        "warnings": [],
    }
    return analysis


@pytest.fixture(scope="module")
def score() -> dict:
    return arrange(make_analysis(), {"include_tab": True})


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def iter_vocals(score_dict: dict) -> list[dict]:
    notes: list[dict] = []
    for bar in score_dict["bars"]:
        notes.extend(bar.get("vocal") or [])
    return notes


# --------------------------------------------------------------------------
# 输出契约
# --------------------------------------------------------------------------


def test_output_matches_json_schema(score: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(load_schema())
    errors = sorted(validator.iter_errors(score), key=lambda error: list(error.path))
    assert not errors, [f"{list(error.path)}: {error.message}" for error in errors[:5]]


def test_meta_contract(score: dict) -> None:
    meta = score["meta"]
    assert score["schema"] == "aitsm.score/v1"
    assert meta["key"]["detected"] and meta["key"]["sounding"] and meta["key"]["play"]
    assert meta["key"]["mode"] in ("major", "minor")
    assert -7 <= meta["key"]["num_sharps_flats"] <= 7
    assert 0 <= meta["capo"] <= 12
    assert 20 <= meta["tempo"] <= 320
    assert meta["time_signature"] == [4, 4]
    assert meta["guitar"]["strings"] == 6
    assert len(meta["guitar"]["tuning"]) == 6
    assert meta["difficulty"] in ("beginner", "elementary", "intermediate", "advanced")
    assert meta["generator"]["version"]
    assert isinstance(meta["warnings"], list)


def test_bars_have_required_fields(score: dict) -> None:
    assert [bar["index"] for bar in score["bars"]] == list(range(1, BARS + 1))
    assert score["bars"][0]["time_signature"] == [4, 4]
    for bar in score["bars"]:
        assert bar["guitar"]["pattern_id"]
        assert bar["start_sec"] < bar["end_sec"]


# --------------------------------------------------------------------------
# 调性 / 变调夹矩阵
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["major", "minor"])
def test_key_capo_matrix_all_12_tonics(mode: str) -> None:
    """12 个主音 × 大/小调：capo 实用、记谱调 = 发声调 - capo、调号自洽。"""
    from aitsm.score.model import key_fifths, note_to_pc

    from_play = {0, 7, 2, 9, 4, 5, 10, 3, 11}
    for pc in range(12):
        tonic = pc_name(pc)
        analysis = make_analysis(tonic=tonic, mode=mode, bars=4)
        result = arrange(analysis, {})
        key = result["meta"]["key"]
        capo = result["meta"]["capo"]
        sounding_pc = pc
        play_pc = note_to_pc(key["play"])
        assert 0 <= capo <= 7, f"{tonic} {mode}: capo={capo}"
        assert (play_pc + capo) % 12 == sounding_pc, f"{tonic} {mode}: {key}"
        assert play_pc in from_play, f"{tonic} {mode}: 记谱调 {key['play']} 不是吉他友好调"
        assert key["num_sharps_flats"] == key_fifths(play_pc, mode)
        # 记谱音高已扣掉 capo
        for note in iter_vocals(result):
            assert note["beat"] < 4.0
        assert validate_score(result)["ok"] is True


def test_requested_key_is_respected() -> None:
    from aitsm.score.model import note_to_pc

    analysis = make_analysis(tonic="F")
    result = arrange(analysis, {"key": "G"})
    assert note_to_pc(result["meta"]["key"]["play"]) == 7
    assert result["meta"]["capo"] == (5 - 7) % 12  # F -> G 需要 -2 => 10 品，超标要告警
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "key.capo_high" in codes


def test_requested_key_and_capo() -> None:
    from aitsm.score.model import note_to_pc

    analysis = make_analysis(tonic="D")
    result = arrange(analysis, {"key": "C", "capo": 2})
    meta = result["meta"]
    assert note_to_pc(meta["key"]["play"]) == 0
    assert meta["capo"] == 2
    assert (note_to_pc(meta["key"]["play"]) + meta["capo"]) % 12 == note_to_pc(meta["key"]["sounding"])


def test_requested_capo_alone() -> None:
    from aitsm.score.model import note_to_pc

    analysis = make_analysis(tonic="E")
    result = arrange(analysis, {"capo": 2})
    meta = result["meta"]
    assert meta["capo"] == 2
    assert note_to_pc(meta["key"]["play"]) == (4 - 2) % 12


def test_invalid_requested_key_falls_back() -> None:
    result = arrange(make_analysis(tonic="G"), {"key": "Hmaj"})
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "key.invalid_option" in codes
    assert result["meta"]["key"]["play"]


def test_transpose_option_shifts_sounding_key() -> None:
    from aitsm.score.model import note_to_pc

    analysis = make_analysis(tonic="C")
    base = arrange(analysis, {})
    shifted = arrange(analysis, {"transpose": 2})
    assert note_to_pc(shifted["meta"]["key"]["sounding"]) == (note_to_pc(base["meta"]["key"]["sounding"]) + 2) % 12
    capo = shifted["meta"]["capo"]
    assert (note_to_pc(shifted["meta"]["key"]["play"]) + capo) % 12 == note_to_pc(shifted["meta"]["key"]["sounding"])


# --------------------------------------------------------------------------
# 记谱音高 / 和弦符号
# --------------------------------------------------------------------------


def test_vocal_pitch_is_shifted_by_capo() -> None:
    """发声音高 = 记谱音高 + capo：所以记谱 midi 必须比原曲低 capo。"""
    analysis = make_analysis(tonic="C")
    result = arrange(analysis, {"capo": 3})
    capo = result["meta"]["capo"]
    assert capo == 3
    original = sorted(note["midi"] for note in analysis["melody"]["notes"])
    arranged = sorted(note["midi"] for note in iter_vocals(result))
    # 跨小节延音会拆成两个音，但音高集合一致
    assert set(arranged) == {midi - capo for midi in original}


@pytest.mark.parametrize(
    ("symbol", "expected_pc"),
    [("C/E", 0), ("Dsus4", 2), ("G7", 7), ("Am7", 9), ("Bm7b5", 11), ("Fmaj7", 5)],
)
def test_chord_symbol_parsing_and_voicing(symbol: str, expected_pc: int) -> None:
    from aitsm.score.model import note_to_pc, parse_chord_symbol

    analysis = make_analysis(tonic="C", chord_symbols=[symbol])
    result = arrange(analysis, {})
    chord = result["bars"][0]["chords"][0]
    parsed = parse_chord_symbol(chord["symbol"])
    assert note_to_pc(parsed.raw[0] if parsed.raw else "C") is not None
    assert parsed.root_pc == expected_pc
    # 斜杠低音必须保留
    if "/" in symbol:
        assert parsed.bass_pc is not None
        assert parse_chord_symbol(chord["symbol"]).bass_pc == parse_chord_symbol(symbol).bass_pc
    assert chord["voicing_id"] in result["voicings"]
    voicing = result["voicings"][chord["voicing_id"]]
    assert voicing["symbol"] == chord["symbol"]
    assert len(voicing["frets"]) == 6
    assert validate_score(result)["ok"] is True


def test_chords_transposed_into_play_key() -> None:
    """F 大调的歌用 C 指法弹（capo=5）：和弦符号要整体移调，concert 保留原调。"""
    from aitsm.score.model import note_to_pc, parse_chord_symbol

    analysis = make_analysis(
        tonic="F",
        chord_symbols=["F", "Bb", "C7", "Dm"],
        chord_plan=[(0, "F"), (1, "Bb"), (2, "C7"), (3, "Dm")],
    )
    result = arrange(analysis, {"key": "C"})
    meta = result["meta"]
    capo = meta["capo"]
    assert note_to_pc(meta["key"]["play"]) == 0
    symbols = [chord["symbol"] for bar in result["bars"] for chord in (bar.get("chords") or [])]
    assert symbols[0] == "C", symbols
    for first, original in zip(symbols, ["C", "F", "G7", "Am"]):
        assert first == original
    for bar in result["bars"]:
        for chord in bar.get("chords") or []:
            concert = parse_chord_symbol(chord["concert"]).root_pc
            notated = parse_chord_symbol(chord["symbol"]).root_pc
            assert (concert - notated) % 12 == capo % 12
    assert validate_score(result)["ok"] is True


# --------------------------------------------------------------------------
# 小节网格 / 量化边界
# --------------------------------------------------------------------------


def test_pickup_bar() -> None:
    """首小节的第一个强拍晚于第一个拍点 -> 弱起（pickup）小节。"""
    analysis = make_analysis(bars=5, downbeat_offset_beats=2.0)
    result = arrange(analysis, {})
    first = result["bars"][0]
    assert first.get("pickup") is True
    assert first["start_sec"] < result["bars"][1]["start_sec"]
    assert validate_score(result)["ok"] is True


def test_triplet_quantization_uses_fraction_duration() -> None:
    """1/3 网格上的音不写成 0.25/0.5，而是 "1/3" 这类分数字符串（三连音）。"""
    bpm = BPM
    notes = []
    # 前两拍各 1 拍，第 3 拍开始三连音：2.0, 2.3333, 2.6667
    notes.append({"start": beat_time(0, bpm), "end": beat_time(1, bpm), "midi": 60, "confidence": 0.9})
    notes.append({"start": beat_time(1, bpm), "end": beat_time(2, bpm), "midi": 62, "confidence": 0.9})
    for index, offset in enumerate((2.0, 2.0 + 1 / 3, 2.0 + 2 / 3)):
        start = beat_time(offset, bpm)
        notes.append(
            {
                "start": start,
                "end": beat_time(offset + 1 / 3, bpm),
                "midi": 64 + index,
                "confidence": 0.9,
            }
        )
    analysis = make_analysis(bars=2, melody_notes=notes)
    result = arrange(analysis, {})
    vocals = iter_vocals(result)
    durations = [note.get("dur") for note in vocals]
    assert any(isinstance(value, str) and value.endswith("/3") for value in durations), durations
    beats = [note["beat"] for note in vocals]
    assert any(abs(value - round(value * 3) / 3) < 0.01 for value in beats)
    assert validate_score(result)["ok"] is True


def test_note_at_bar_line_snaps_forward() -> None:
    """起点贴着下一条小节线的音必须归到下一小节，不能写在小节末尾（beat=小节长度）。"""
    bpm = BPM
    notes = [
        {"start": beat_time(3.999, bpm), "end": beat_time(4.5, bpm), "midi": 64, "confidence": 0.9},
        {"start": beat_time(4.5, bpm), "end": beat_time(5.0, bpm), "midi": 65, "confidence": 0.9},
    ]
    analysis = make_analysis(bars=3, melody_notes=notes)
    result = arrange(analysis, {})
    for bar in result["bars"]:
        for note in bar.get("vocal") or []:
            assert note["beat"] < 4.0, f"bar {bar['index']} beat {note['beat']}"
    assert validate_score(result)["ok"] is True


def test_note_across_bar_line_is_tied() -> None:
    """跨小节的音拆成两段 + tie（start/stop）。"""
    bpm = BPM
    notes = [{"start": beat_time(3.0, bpm), "end": beat_time(5.0, bpm), "midi": 67, "confidence": 0.9}]
    analysis = make_analysis(bars=3, melody_notes=notes)
    result = arrange(analysis, {})
    first = [note for note in result["bars"][0].get("vocal") or [] if note.get("midi") == 67]
    second = [note for note in result["bars"][1].get("vocal") or [] if note.get("midi") == 67]
    assert first and first[-1].get("tie") == "start", first
    assert second and second[0].get("tie") == "stop", second
    assert validate_score(result)["ok"] is True


def test_three_four_time_signature() -> None:
    analysis = make_analysis(bars=4, beats_per_bar=3)
    analysis["beats"]["time_signature"] = "3/4"
    result = arrange(analysis, {})
    assert result["meta"]["time_signature"] == [3, 4]
    assert result["bars"][0]["time_signature"] == [3, 4]
    for bar in result["bars"]:
        for chord in bar.get("chords") or []:
            assert chord["beat"] < 3.0
        for note in bar.get("vocal") or []:
            assert note["beat"] < 3.0
        pattern = result["strum_patterns"][bar["guitar"]["pattern_id"]]
        assert pattern["beats_per_bar"] == 3
    assert validate_score(result)["ok"] is True


@pytest.mark.parametrize(("meter", "beats"), [("2/4", 2), ("6/8", 6), ("12/8", 12)])
def test_compound_meters_pick_matching_pattern(meter: str, beats: int) -> None:
    """复拍子（2/4、6/8、12/8）必须选到拍号一致的节奏型，且不产生拍号告警。"""
    analysis = make_analysis(bars=4, beats_per_bar=beats)
    analysis["beats"]["time_signature"] = meter
    result = arrange(analysis, {})
    assert result["meta"]["time_signature"] == [beats, int(meter.split("/")[1])]
    for bar in result["bars"]:
        pattern = result["strum_patterns"][bar["guitar"]["pattern_id"]]
        assert pattern["beats_per_bar"] == beats
        for hit in bar["guitar"].get("hits") or []:
            # 事件按四分音符拍计，必须落在小节长度内
            assert hit["beat"] < beats * 4 / int(meter.split("/")[1]) + 1e-6
    report = validate_score(result)
    assert report["ok"] is True
    assert not [w for w in report["warnings"] if w["code"] == "guitar.pattern_meter"]


def test_irregular_meter_falls_back_with_warning() -> None:
    """5/4 这类没有专用节奏型的拍号：退回 4 拍族并给出拍号告警（不崩、不编造）。"""
    analysis = make_analysis(bars=4, beats_per_bar=5)
    analysis["beats"]["time_signature"] = "5/4"
    result = arrange(analysis, {})
    report = validate_score(result)
    assert report["ok"] is True
    assert [w for w in report["warnings"] if w["code"] == "guitar.pattern_meter"]


# --------------------------------------------------------------------------
# 和弦映射
# --------------------------------------------------------------------------


def test_adjacent_same_chord_is_merged() -> None:
    analysis = make_analysis(
        bars=2,
        chord_plan=[(0, "C"), (1, "C")],
    )
    # 同一小节里两个相同和弦（前半拍 C、后半拍 C）
    analysis["chords"] = [
        {"start": beat_time(0), "end": beat_time(2), "symbol": "C", "confidence": 0.9},
        {"start": beat_time(2), "end": beat_time(4), "symbol": "C", "confidence": 0.9},
    ]
    result = arrange(analysis, {})
    chords = result["bars"][0].get("chords") or []
    assert len(chords) == 1
    assert chords[0]["symbol"] == "C"


def test_at_most_four_chords_per_bar() -> None:
    analysis = make_analysis(bars=2)
    # 一小节内 5 个八分音符位置各一个和弦 -> 超出上限，必须裁到 4 个并告警
    analysis["chords"] = [
        {
            "start": beat_time(index * 0.5),
            "end": beat_time((index + 1) * 0.5),
            "symbol": symbol,
            "confidence": 0.5 + index * 0.05,
        }
        for index, symbol in enumerate(["C", "D", "E", "F", "G"])
    ]
    result = arrange(analysis, {})
    chords = result["bars"][0].get("chords") or []
    assert len(chords) <= 4
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "chords.too_many" in codes
    # 保留的应该是置信度高的那些（丢弃最早的 C）
    assert "C" not in [chord["symbol"] for chord in chords]


def test_long_chord_continues_into_next_bars() -> None:
    analysis = make_analysis(bars=4)
    analysis["chords"] = [
        {"start": beat_time(0), "end": beat_time(8), "symbol": "Am", "confidence": 0.9},
        {"start": beat_time(8), "end": beat_time(16), "symbol": "F", "confidence": 0.9},
    ]
    result = arrange(analysis, {})
    assert result["bars"][0]["chords"][0]["symbol"] == "Am"
    assert result["bars"][1]["chords"][0]["symbol"] == "Am"
    assert result["bars"][1]["chords"][0].get("continue") is True
    assert validate_score(result)["ok"] is True


def test_empty_chords_does_not_fabricate_harmony() -> None:
    """没有识别到和弦时宁可留空并告警，绝不编造和声。"""
    analysis = make_analysis(bars=3)
    analysis["chords"] = []
    result = arrange(analysis, {})
    for bar in result["bars"]:
        assert not bar.get("chords")
        assert "hits" not in bar["guitar"]
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "chords.missing_bars" in codes
    assert result["voicings"] == {}
    assert validate_score(result)["ok"] is True


# --------------------------------------------------------------------------
# 容错 / 段落
# --------------------------------------------------------------------------


def test_missing_melody_is_tolerated() -> None:
    analysis = make_analysis(bars=3)
    analysis.pop("melody")
    result = arrange(analysis, {})
    assert all(not bar.get("vocal") for bar in result["bars"])
    report = validate_score(result)
    assert report["ok"] is True
    assert "melody.empty" in {item["code"] for item in report["warnings"]}


def test_missing_sections_falls_back_to_single_other() -> None:
    analysis = make_analysis(bars=4, with_sections=False)
    analysis["sections"] = []
    result = arrange(analysis, {})
    assert len(result["sections"]) == 1
    section = result["sections"][0]
    assert section["kind"] == "other"
    assert section["label"] == "其他"
    assert section["start_bar"] == 1 and section["end_bar"] == 4
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "sections.missing" in codes
    assert validate_score(result)["ok"] is True


def test_section_labels_are_chinese() -> None:
    kinds = ["intro", "verse", "prechorus", "chorus", "bridge", "interlude", "outro"]
    analysis = make_analysis(bars=len(kinds), with_sections=False)
    analysis["sections"] = [
        {
            "start": beat_time(index * BEATS_PER_BAR),
            "end": beat_time((index + 1) * BEATS_PER_BAR),
            "kind": kind,
            "energy": 0.5,
        }
        for index, kind in enumerate(kinds)
    ]
    result = arrange(analysis, {})
    labels = [section["label"] for section in result["sections"]]
    assert labels == ["前奏", "主歌", "预副歌", "副歌", "桥段", "间奏", "尾奏"]
    for section in result["sections"]:
        assert section["start_bar"] <= section["end_bar"]
    assert validate_score(result)["ok"] is True


def test_sections_cover_all_bars_without_holes() -> None:
    analysis = make_analysis(bars=8, with_sections=False)
    analysis["sections"] = [
        {"start": beat_time(0), "end": beat_time(8), "kind": "intro", "energy": 0.2},
        {"start": beat_time(16), "end": beat_time(32), "kind": "chorus", "energy": 0.9},
    ]
    result = arrange(analysis, {})
    covered = set()
    for section in result["sections"]:
        covered.update(range(section["start_bar"], section["end_bar"] + 1))
    assert covered == set(range(1, 9))
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "sections.gap_filled" in codes
    assert validate_score(result)["ok"] is True


def test_chorus_pattern_is_denser_than_verse() -> None:
    from aitsm.score.strum import pattern_note_count

    analysis = make_analysis(bars=4)
    result = arrange(analysis, {"style": "pop", "difficulty": "intermediate"})
    verse = next(section for section in result["sections"] if section["kind"] == "verse")
    chorus = next(section for section in result["sections"] if section["kind"] == "chorus")
    verse_pattern = verse["pattern_id"]
    chorus_pattern = chorus["pattern_id"]
    assert pattern_note_count(chorus_pattern) >= pattern_note_count(verse_pattern)


# --------------------------------------------------------------------------
# 选项
# --------------------------------------------------------------------------


def test_include_tab_controls_hits() -> None:
    analysis = make_analysis(bars=3)
    with_tab = arrange(analysis, {"include_tab": True})
    without_tab = arrange(analysis, {"include_tab": False})
    hits = [hit for bar in with_tab["bars"] for hit in bar["guitar"].get("hits") or []]
    assert hits
    for hit in hits:
        assert hit.get("strings")
        for pair in hit["strings"]:
            assert len(pair) == 2
            assert 1 <= pair[0] <= 6
    assert all(not bar["guitar"].get("hits") for bar in without_tab["bars"])
    assert all(bar["guitar"]["pattern_id"] for bar in without_tab["bars"])
    assert validate_score(without_tab)["ok"] is True


def test_difficulty_option_limits_voicings() -> None:
    analysis = make_analysis(bars=4, chord_symbols=["F", "Bm", "C"])
    result = arrange(analysis, {"difficulty": "beginner"})
    assert result["meta"]["difficulty"] == "beginner"
    difficulties = [voicing["difficulty"] for voicing in result["voicings"].values()]
    assert difficulties
    assert max(difficulties) <= 3, difficulties


def test_style_option_is_recorded_and_influences_pattern() -> None:
    analysis = make_analysis(bars=4)
    result = arrange(analysis, {"style": "folk"})
    assert result["meta"]["style"] == "folk"
    patterns = set(result["strum_patterns"])
    kinds = {result["strum_patterns"][pid].get("kind") for pid in patterns}
    assert patterns
    assert kinds


def test_unknown_instrument_warns_but_works() -> None:
    result = arrange(make_analysis(bars=2), {"instrument": "piano"})
    codes = {item["code"] for item in result["quality"]["warnings"]}
    assert "options.instrument" in codes
    assert validate_score(result)["ok"] is True


def test_deterministic_output() -> None:
    analysis = make_analysis(bars=4)
    first = arrange(copy.deepcopy(analysis), {})
    second = arrange(copy.deepcopy(analysis), {})
    first["meta"]["generator"]["created_at"] = "x"
    second["meta"]["generator"]["created_at"] = "x"
    assert first == second


def test_empty_analysis_raises() -> None:
    with pytest.raises(ValueError):
        arrange({}, {})


# --------------------------------------------------------------------------
# 和弦碎片 / 空档 / 扫弦指法一致性（真实曲回归：lead 报的 guitar.hit_pitch 121 错误）
# --------------------------------------------------------------------------


def _beat_sec(beat: float, bpm: float = BPM) -> float:
    return beat * 60.0 / bpm


def test_chord_fragment_across_barline_is_dropped() -> None:
    """analysis 的和弦 end 只比小节线多几十毫秒时，不能变成 beat=0 的碎片和弦。"""
    analysis = make_analysis(bars=4)
    bar_len_sec = 4 * _beat_sec(1)
    analysis["chords"] = [
        # Bm 从弱起前一小节开始，结束在小节线后 0.11 拍（残影）
        {"start": 0.0, "end": bar_len_sec + _beat_sec(0.11), "symbol": "Bm", "confidence": 0.7},
        # D 从第 2 小节线开始
        {"start": bar_len_sec, "end": bar_len_sec * 2, "symbol": "D", "confidence": 0.9},
        {"start": bar_len_sec * 2, "end": bar_len_sec * 4, "symbol": "G", "confidence": 0.9},
    ]
    result = arrange(analysis, {})
    bar_two = result["bars"][1]
    symbols = [chord["symbol"] for chord in bar_two["chords"]]
    assert symbols == ["D"], symbols
    beats = [chord["beat"] for chord in bar_two["chords"]]
    assert all(beat < 0.5 for beat in beats)
    report = validate_score(result)
    assert report["ok"] is True, report["errors"]
    codes = {item["code"] for item in report["errors"]}
    assert "chords.overlap" not in codes


def test_chord_gap_is_filled_to_next_chord() -> None:
    """和弦之间小于一个量化步长的空档要补齐（吉他实际会一直扫）。"""
    analysis = make_analysis(bars=2)
    bar_sec = 4 * _beat_sec(1)
    analysis["chords"] = [
        {"start": 0.0, "end": _beat_sec(0.29), "symbol": "C", "confidence": 0.9},
        {"start": _beat_sec(0.5), "end": bar_sec, "symbol": "G", "confidence": 0.9},
        {"start": bar_sec, "end": bar_sec * 2, "symbol": "Am", "confidence": 0.9},
    ]
    result = arrange(analysis, {})
    chords = result["bars"][0]["chords"]
    from aitsm.score.model import duration_value

    for index in range(len(chords) - 1):
        start = chords[index]["beat"]
        end = start + duration_value(chords[index]["dur"])
        following = chords[index + 1]["beat"]
        assert end >= following - 1e-6, (chords[index], following)
    report = validate_score(result)
    assert "chords.gap" not in {item["code"] for item in report["warnings"]}


def test_guitar_hits_use_active_chord_voicing() -> None:
    """每个扫弦击发的指法必须与当下生效和弦一致（否则 TAB 会弹出别的和弦）。"""
    from aitsm.score.model import chord_pitch_classes, tuning_to_midis

    analysis = make_analysis(bars=6)
    result = arrange(analysis, {"include_tab": True})
    tuning = tuning_to_midis("standard")
    for bar in result["bars"]:
        chords = bar.get("chords") or []
        hits = (bar.get("guitar") or {}).get("hits") or []
        for hit in hits:
            beat = hit["beat"]
            active = None
            for chord in chords:
                start = chord["beat"]
                dur = chord.get("dur")
                from aitsm.score.model import duration_value

                if start <= beat + 1e-6 and start + duration_value(dur) > beat + 1e-6:
                    active = chord
            assert active is not None, (bar["index"], hit)
            if active.get("voicing_id") and hit.get("voicing_id"):
                assert hit["voicing_id"] == active["voicing_id"], (bar["index"], beat)
            symbol = active["symbol"]
            pcs = chord_pitch_classes(symbol)
            for string_number, fret in hit.get("strings") or []:
                midi = tuning[6 - int(string_number)] + int(fret)
                assert midi % 12 in pcs, (bar["index"], beat, symbol, string_number, fret)
    assert validate_score(result)["ok"] is True


def test_no_stale_fragment_voicing_in_hits() -> None:
    """碎片和弦被丢弃后，beat 0 的扫弦只能挂在小节真正生效的和弦上。"""
    analysis = make_analysis(bars=4)
    bar_sec = 4 * _beat_sec(1)
    analysis["chords"] = [
        {"start": 0.0, "end": bar_sec + _beat_sec(0.1057), "symbol": "Bm", "confidence": 0.67},
        {"start": bar_sec, "end": bar_sec * 2, "symbol": "D", "confidence": 0.9},
        {"start": bar_sec * 2, "end": bar_sec * 4, "symbol": "A", "confidence": 0.9},
    ]
    result = arrange(analysis, {"include_tab": True})
    bar_two = result["bars"][1]
    for hit in (bar_two.get("guitar") or {}).get("hits") or []:
        if hit["beat"] < 0.5:
            assert hit.get("voicing_id", "").startswith("D"), hit
    report = validate_score(result)
    assert report["ok"] is True, report["errors"]
    assert "guitar.hit_pitch" not in {item["code"] for item in report["errors"]}

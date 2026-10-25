"""MusicXML 往返契约：写出的文件合法、每小节时值合规、和弦/歌词/弱起可还原。"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from aitsm.contracts import (
    BeatGrid,
    ChordEvent,
    KeySignature,
    LyricLine,
    LyricWord,
    MelodyNote,
    Section,
    SymbolicScore,
)
from aitsm.fusion import (
    DIVISIONS,
    MUSICXML_VERSION,
    format_chord_symbol,
    parse_chord_symbol,
    read_musicxml,
    write_musicxml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = REPO_ROOT / "artifacts" / "analysis.json"


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


def _demo_score() -> SymbolicScore:
    """120 BPM / 4/4 的两小节小谱：跨小节长音 + 三个和弦 + 歌词 + 段落。"""
    beats = BeatGrid(
        bpm=120.0,
        beats=[index * 0.5 for index in range(13)],
        downbeats=[0.0, 2.0, 4.0, 6.0],
        beats_per_bar=4,
        time_signature="4/4",
        confidence=0.9,
    )
    melody = [
        MelodyNote(start=0.0, end=0.5, midi=60, confidence=0.8, lyric="Hel"),
        MelodyNote(start=0.5, end=1.0, midi=62, confidence=0.8, lyric="lo"),
        MelodyNote(start=1.0, end=2.5, midi=64, confidence=0.7, lyric="world"),
    ]
    lyrics = [
        LyricLine(
            start=0.0,
            end=1.5,
            text="Hel lo world",
            words=[LyricWord(start=0.0, end=0.5, text="Hel"), LyricWord(start=0.5, end=1.0, text="lo")],
            confidence=0.8,
        )
    ]
    return SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C", mode="major", confidence=0.9, fifths=0),
        chords=[
            ChordEvent(start=0.0, end=2.0, symbol="C", confidence=0.7),
            ChordEvent(start=2.0, end=4.0, symbol="Am7", confidence=0.6),
            ChordEvent(start=4.0, end=6.0, symbol="G/B", confidence=0.6),
        ],
        sections=[Section(start=0.0, end=4.0, kind="verse", label="Verse 1", confidence=0.7)],
        melody=melody,
        lyrics=lyrics,
        meta={"title": "Demo Song"},
    )


def _note_durations(measure: ET.Element) -> list[int]:
    return [
        int(float(note.findtext("duration")))
        for note in measure.findall("note")
        if note.find("duration") is not None
    ]


def _measures(xml: str) -> list[ET.Element]:
    root = ET.fromstring(xml.encode("utf-8"))
    return list(root.iter("measure"))


# --------------------------------------------------------------------------- #
# 写出：格式合法性
# --------------------------------------------------------------------------- #


def test_write_musicxml_is_wellformed_partwise() -> None:
    xml = write_musicxml(_demo_score())
    assert xml.startswith("<?xml")
    assert "<!DOCTYPE score-partwise" in xml
    root = ET.fromstring(xml.encode("utf-8"))
    assert root.tag == "score-partwise"
    assert root.get("version") == MUSICXML_VERSION

    divisions = [int(item.text) for item in root.iter("divisions")]
    assert divisions and set(divisions) == {DIVISIONS}

    # 首小节带 tempo direction（标准元素，不靠自定义命名空间）
    first = next(root.iter("measure"))
    sounds = [item for item in first.iter("sound") if item.get("tempo")]
    assert sounds, "第一小节必须写出 <sound tempo=...>"
    assert abs(float(sounds[0].get("tempo")) - 120.0) < 1e-6

    # 段落用 <rehearsal> 写出
    labels = [item.text for item in root.iter("rehearsal")]
    assert "Verse 1" in labels


# --------------------------------------------------------------------------- #
# 往返一致
# --------------------------------------------------------------------------- #


def test_roundtrip_score_musicxml_score() -> None:
    score = _demo_score()
    xml = write_musicxml(score)
    back = SymbolicScore.from_musicxml(xml, sidecar=score.to_dict())

    assert back.beats.bpm == pytest.approx(120.0, abs=1e-3)
    assert back.beats.time_signature == "4/4"
    assert back.beats.beats_per_bar == 4
    assert back.key.tonic == "C"
    assert back.key.mode == "major"
    assert back.key.fifths == 0

    assert [chord.symbol for chord in back.chords] == ["C", "Am7", "G/B"]
    for original, restored in zip(score.chords, back.chords):
        assert restored.start == pytest.approx(original.start, abs=1e-3)
        assert restored.end == pytest.approx(original.end, abs=1e-3)

    assert [note.midi for note in back.melody] == [60, 62, 64]
    for original, restored in zip(score.melody, back.melody):
        assert restored.start == pytest.approx(original.start, abs=1e-3)
        assert restored.end == pytest.approx(original.end, abs=1e-3)
        assert restored.lyric == original.lyric

    assert [line.text for line in back.lyrics] == ["Hel lo world"]
    assert [section.label for section in back.sections] == ["Verse 1"]
    assert back.sections[0].start == pytest.approx(0.0, abs=1e-3)
    assert back.sections[0].kind == "verse"
    assert back.meta.get("title") == "Demo Song"


def test_read_without_sidecar_uses_tempo_map() -> None:
    xml = write_musicxml(_demo_score())
    back = read_musicxml(xml)
    assert back.beats.bpm == pytest.approx(120.0, abs=1e-3)
    assert [note.midi for note in back.melody] == [60, 62, 64]
    # 没有 sidecar 时按 tempo map 推算：120 BPM、每四分音符 24 divisions，
    # 0.5 s = 1 拍，读回来的绝对时间应精确落在 0.5 s 网格上。
    assert back.melody[1].start == pytest.approx(0.5, abs=1e-3)
    assert back.melody[2].start == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 每小节时值总和
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("time_signature", "beats_per_bar", "expected_quarters"),
    [("4/4", 4, 4.0), ("3/4", 3, 3.0)],
)
def test_each_measure_durations_sum_to_time_signature(
    time_signature: str, beats_per_bar: int, expected_quarters: float
) -> None:
    beat_period = 0.5
    beats = BeatGrid(
        bpm=120.0,
        beats=[index * beat_period for index in range(beats_per_bar * 2 + 1)],
        beats_per_bar=beats_per_bar,
        time_signature=time_signature,
    )
    score = SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="G", fifths=1),
        chords=[ChordEvent(start=0.0, end=beat_period * 2, symbol="G")],
        melody=[
            MelodyNote(start=0.0, end=0.75, midi=67),
            MelodyNote(start=0.75, end=beat_period * 3, midi=69),
        ],
    )
    xml = write_musicxml(score)
    expected = int(round(expected_quarters * DIVISIONS))
    measures = _measures(xml)
    assert len(measures) >= 2
    for measure in measures:
        assert sum(_note_durations(measure)) == expected


def test_pickup_measure_is_implicit_and_roundtrips() -> None:
    beats = BeatGrid(
        bpm=120.0,
        beats=[0.5 + index * 0.5 for index in range(9)],
        downbeats=[0.5, 2.5, 4.5],
        beats_per_bar=4,
        first_downbeat_sec=0.5,
        pickup_beats=1,
    )
    score = SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C"),
        chords=[ChordEvent(start=0.0, end=2.0, symbol="C")],
        melody=[MelodyNote(start=0.25, end=0.5, midi=60), MelodyNote(start=0.5, end=1.5, midi=62)],
    )
    xml = write_musicxml(score)
    measures = _measures(xml)
    assert measures[0].get("implicit") == "yes"
    assert sum(_note_durations(measures[0])) == DIVISIONS  # 1 拍 = 24 divisions 的弱起

    back = read_musicxml(xml)
    assert back.beats.pickup_beats == 1
    assert back.beats.first_downbeat_sec == pytest.approx(0.5, abs=1e-3)
    assert [note.start for note in back.melody] == pytest.approx([0.25, 0.5], abs=1e-3)


def test_long_intro_splits_into_full_bars_and_partial_pickup() -> None:
    """前奏长于一小节时不能让单个 implicit 小节超过拍号（artifact 里踩到过 97/96）。"""
    beats = BeatGrid(
        bpm=120.0,
        beats=[3.25 + index * 0.5 for index in range(9)],
        downbeats=[3.25, 5.25, 7.25],
        beats_per_bar=4,
        first_downbeat_sec=3.25,
    )
    score = SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C"),
        chords=[ChordEvent(start=3.25, end=5.25, symbol="C")],
        melody=[MelodyNote(start=3.25, end=3.75, midi=60), MelodyNote(start=3.75, end=4.25, midi=62)],
    )
    xml = write_musicxml(score)
    measures = _measures(xml)
    lengths = [sum(_note_durations(measure)) for measure in measures]
    bar_div = 4 * DIVISIONS
    assert all(0 < length <= bar_div for length in lengths), lengths
    assert measures[0].get("implicit") == "yes"
    assert lengths[0] == bar_div, "整小节前奏写成完整 implicit 小节"
    assert measures[1].get("implicit") == "yes"
    assert lengths[1] == 156 - bar_div, "剩余 1.625 小节 - 1 小节 = 60 divisions"

    back = read_musicxml(xml)
    assert back.beats.first_downbeat_sec == pytest.approx(3.25, abs=1e-3)
    assert [note.start for note in back.melody] == pytest.approx([3.25, 3.75], abs=1e-3)


def test_note_crossing_barline_is_split_with_ties() -> None:
    beats = BeatGrid(bpm=120.0, beats=[index * 0.5 for index in range(13)], beats_per_bar=4)
    score = SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C"),
        melody=[MelodyNote(start=0.0, end=2.5, midi=64), MelodyNote(start=2.5, end=3.0, midi=65)],
    )
    xml = write_musicxml(score)
    root = ET.fromstring(xml.encode("utf-8"))
    ties = [(tie.get("type"), tie.tag) for tie in root.iter() if tie.tag in ("tie", "tied")]
    assert ("start", "tie") in ties
    assert ("stop", "tie") in ties

    back = read_musicxml(xml)
    assert len(back.melody) == 2, "跨小节长音必须被 tie 合并回一个音"
    assert back.melody[0].start == pytest.approx(0.0, abs=1e-3)
    assert back.melody[0].end == pytest.approx(2.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# 和弦符号
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("symbol", "root_step", "suffix", "bass_step"),
    [
        ("C", "C", "", None),
        ("Am7", "A", "m7", None),
        ("G/B", "G", "", "B"),
        ("Bbmaj7", "B", "maj7", None),
        ("F#m7b5", "F", "m7b5", None),
    ],
)
def test_parse_chord_symbol(symbol: str, root_step: str, suffix: str, bass_step: str | None) -> None:
    parsed = parse_chord_symbol(symbol)
    assert parsed is not None
    assert parsed["root_step"] == root_step
    assert parsed["suffix"] == suffix
    assert parsed["bass_step"] == bass_step
    assert format_chord_symbol(
        parsed["root_step"], parsed["root_alter"], parsed["suffix"], parsed["bass_step"], parsed["bass_alter"]
    ) == symbol


def test_parse_chord_symbol_rejects_garbage() -> None:
    assert parse_chord_symbol("") is None
    assert parse_chord_symbol("H7") is None
    assert parse_chord_symbol(123) is None  # type: ignore[arg-type]


def test_two_chords_in_one_measure_survive_roundtrip() -> None:
    """Lead 报的回归：同一小节 C（第 1 拍）+ G（第 3 拍）必须写出两个 <harmony>。"""
    beats = BeatGrid(bpm=120.0, beats=[index * 0.5 for index in range(5)], beats_per_bar=4)
    score = SymbolicScore(
        beats=beats,
        key=KeySignature(tonic="C"),
        chords=[ChordEvent(start=0.0, end=1.0, symbol="C"), ChordEvent(start=1.0, end=2.0, symbol="G")],
        melody=[MelodyNote(start=0.0, end=0.5, midi=60), MelodyNote(start=1.0, end=1.5, midi=67)],
    )
    xml = write_musicxml(score)
    root = ET.fromstring(xml.encode("utf-8"))
    measures = list(root.iter("measure"))
    assert len(measures) == 1
    harmonies = list(measures[0].iter("harmony"))
    assert len(harmonies) == 2
    assert harmonies[0].findtext("root/root-step") == "C"
    assert harmonies[1].findtext("root/root-step") == "G"
    # G 挂在第 3 拍那一下：前两个 <note>（C 音 + 休止）之后、G 音之前
    children = list(measures[0])
    position = children.index(harmonies[1])
    notes_before = [child for child in children[:position] if child.tag == "note"]
    assert len(notes_before) == 2
    assert harmonies[1].findtext("offset") in (None, "0")

    back = read_musicxml(xml)
    assert [(chord.start, chord.end, chord.symbol) for chord in back.chords] == pytest.approx(
        [(0.0, 1.0, "C"), (1.0, 2.0, "G")]
    )


# --------------------------------------------------------------------------- #
# 与 analysis.json 同形状的总谱能落盘
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not ANALYSIS_PATH.exists(), reason="缺少 artifacts/analysis.json")
def test_save_analysis_shaped_score_writes_musicxml(tmp_path: Path) -> None:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    score = SymbolicScore.from_analysis(analysis)
    assert score.melody, "analysis.json 转出来的总谱应该带旋律"

    paths = score.save(tmp_path)
    assert set(paths) == {"json", "musicxml"}
    musicxml = tmp_path / "symbolic.musicxml"
    assert musicxml.exists()
    assert musicxml.stat().st_size > 1000
    ET.fromstring(musicxml.read_text(encoding="utf-8").encode("utf-8"))

    loaded = SymbolicScore.load(tmp_path)
    assert len(loaded.melody) == len(score.melody)
    assert len(loaded.chords) == len(score.chords)
    assert loaded.beats.bpm == pytest.approx(score.beats.bpm, abs=1e-3)


# --------------------------------------------------------------------------- #
# 边界
# --------------------------------------------------------------------------- #


def test_include_lyrics_false_omits_lyric_elements() -> None:
    xml = write_musicxml(_demo_score(), include_lyrics=False)
    root = ET.fromstring(xml.encode("utf-8"))
    assert not list(root.iter("lyric"))


def test_read_musicxml_rejects_non_musicxml() -> None:
    with pytest.raises(ValueError):
        read_musicxml("<not-a-score/>")


def test_write_musicxml_rejects_non_score() -> None:
    with pytest.raises(TypeError):
        write_musicxml({"beats": {}})  # type: ignore[arg-type]

"""融合骨架契约：旋律优先级、骨架补洞、beat 对齐、performance SMF 合法。"""

from __future__ import annotations

from pathlib import Path

import pytest

from aitsm.contracts import (
    AccompanimentResult,
    BeatGrid,
    ChordEvent,
    KeySignature,
    LyricLine,
    LyricWord,
    MelodyNote,
    Section,
    SymbolicScore,
    VocalResult,
)
from aitsm.fusion import (
    FUSION_ENGINE,
    fuse,
    snap_notes_to_beats,
    write_musicxml,
    write_performance_midi,
)
from aitsm.score.export.midi import PPQ, parse_midi

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


def _beats(bpm: float = 120.0, bars: int = 3, beats_per_bar: int = 4) -> BeatGrid:
    period = 60.0 / bpm
    count = bars * beats_per_bar
    return BeatGrid(
        bpm=bpm,
        beats=[index * period for index in range(count + 1)],
        downbeats=[bar * beats_per_bar * period for bar in range(bars + 1)],
        beats_per_bar=beats_per_bar,
        time_signature=f"{beats_per_bar}/4",
        confidence=0.8,
    )


def _vocal() -> VocalResult:
    return VocalResult(
        notes=[
            MelodyNote(start=0.0, end=0.5, midi=60, confidence=0.9, lyric="Hel"),
            MelodyNote(start=0.5, end=1.0, midi=62, confidence=0.85, lyric="lo"),
            MelodyNote(start=1.0, end=1.5, midi=64, confidence=0.6, lyric="world"),
            # 与 1.0 同起点同 midi 的重复检测项（置信度更低，必须被丢弃）
            MelodyNote(start=1.0, end=1.2, midi=64, confidence=0.3),
        ],
        lyrics=[
            LyricLine(
                start=0.0,
                end=1.5,
                text="Hel lo world",
                words=[
                    LyricWord(start=0.0, end=0.5, text="Hel"),
                    LyricWord(start=0.5, end=1.0, text="lo"),
                    LyricWord(start=1.0, end=1.5, text="world"),
                ],
                confidence=0.8,
            )
        ],
        bpm=120.0,
        engine="pyin",
        requested="VocalParse",
        warnings=["pyin 未做八度纠错"],
    )


def _accompaniment() -> AccompanimentResult:
    return AccompanimentResult(
        beats=_beats(),
        key=KeySignature(tonic="C", mode="major", confidence=0.7, fifths=0),
        chords=[
            ChordEvent(start=0.0, end=2.0, symbol="C", confidence=0.6),
            ChordEvent(start=2.0, end=4.0, symbol="Am7", confidence=0.6),
            ChordEvent(start=4.0, end=6.0, symbol="G/B", confidence=0.5),
        ],
        sections=[Section(start=0.0, end=4.0, kind="verse", label="Verse 1", confidence=0.6)],
        melody_skeleton=[
            # 落在 1.5–3.0 的人声空档里，整小节（1 bar = 2.0 s）可容纳
            MelodyNote(start=2.0, end=2.5, midi=72, confidence=0.55),
            MelodyNote(start=2.5, end=3.0, midi=71, confidence=0.5),
            # 与人声重叠的骨架音不许补进来
            MelodyNote(start=0.75, end=1.25, midi=67, confidence=0.9),
        ],
        engine="chroma-template",
        requested="SheetSage2",
        warnings=["beats 来自 librosa"],
    )


# --------------------------------------------------------------------------- #
# fuse
# --------------------------------------------------------------------------- #


def test_fuse_prefers_vocal_and_fills_skeleton_gap() -> None:
    score = fuse(_vocal(), _accompaniment())

    melody = [(note.start, note.end, note.midi) for note in score.melody]
    assert melody == [
        (0.0, 0.5, 60),
        (0.5, 1.0, 62),
        (1.0, 1.5, 64),
        (2.0, 2.5, 72),
        (2.5, 3.0, 71),
    ], "骨架只补人声空档，且重复检测项必须被去掉"
    assert {note.source for note in score.melody if note.midi in (72, 71)} == {"skeleton"}
    assert not any(note.midi == 67 for note in score.melody)

    assert score.beats.bpm == pytest.approx(120.0)
    assert score.key.tonic == "C"
    assert [chord.symbol for chord in score.chords] == ["C", "Am7", "G/B"]
    assert [section.label for section in score.sections] == ["Verse 1"]
    assert len(score.lyrics) == 1 and score.lyrics[0].text == "Hel lo world"

    # provenance：三条 layer，requested vs engine 都记清楚
    layers = {item.layer: item for item in score.provenance}
    assert set(layers) == {"vocal_transcription", "accompaniment_transcription", "fusion"}
    assert layers["vocal_transcription"].requested == "VocalParse"
    assert layers["vocal_transcription"].engine == "pyin"
    assert layers["accompaniment_transcription"].requested == "SheetSage2"
    assert layers["accompaniment_transcription"].engine == "chroma-template"
    assert layers["fusion"].engine == FUSION_ENGINE
    assert layers["fusion"].detail["skeleton_filled"] == 2
    assert layers["fusion"].detail["skeleton_filled_seconds"] == pytest.approx(1.0)

    # 原始警告被合并成 dict（stage 标明来源），带 code
    warning_entries = [(item.get("stage"), item.get("code"), item.get("level")) for item in score.warnings]
    assert ("fusion", "skeleton_fill", "info") in warning_entries
    assert ("vocal_transcription", "vocal_warning", "warn") in warning_entries
    assert ("accompaniment_transcription", "accompaniment_warning", "warn") in warning_entries

    assert score.meta["melody_source"] == "fused"
    assert score.source["vocal_engine"] == "pyin"


def test_fuse_without_vocal_melody_uses_skeleton() -> None:
    vocal = VocalResult(notes=[], lyrics=[], engine="none", requested="VocalParse", warnings=[])
    score = fuse(vocal, _accompaniment())
    codes = {(item.get("stage"), item.get("code")) for item in score.warnings}
    assert ("fusion", "melody_from_skeleton") in codes
    assert [note.midi for note in score.melody] == [67, 72, 71]
    assert all(note.source == "skeleton" for note in score.melody)


def test_fuse_without_any_melody_keeps_chords_and_warns() -> None:
    vocal = VocalResult(notes=[], lyrics=[], engine="none", requested="VocalParse", warnings=[])
    accompaniment = _accompaniment()
    accompaniment.melody_skeleton = []
    score = fuse(vocal, accompaniment)
    assert score.melody == []
    assert [chord.symbol for chord in score.chords] == ["C", "Am7", "G/B"]
    codes = {(item.get("stage"), item.get("code")) for item in score.warnings}
    assert ("fusion", "no_melody") in codes


def test_fuse_derives_lyrics_from_note_text_when_missing() -> None:
    vocal = _vocal()
    vocal.lyrics = []
    score = fuse(vocal, _accompaniment())
    assert len(score.lyrics) == 1
    assert score.lyrics[0].text == "world" or "Hel" in score.lyrics[0].text


def test_fuse_rejects_wrong_types() -> None:
    with pytest.raises(TypeError):
        fuse({"notes": []}, _accompaniment())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# snap_notes_to_beats
# --------------------------------------------------------------------------- #


def test_snap_notes_moves_only_onsets_within_tolerance() -> None:
    notes = [
        MelodyNote(start=0.02, end=0.52, midi=60),
        MelodyNote(start=0.5, end=1.0, midi=62),
        MelodyNote(start=1.30, end=1.60, midi=64),  # 偏离 0.30s，超容差不许动
    ]
    snapped = snap_notes_to_beats(notes, _beats(), tolerance=0.06)
    assert [note.start for note in snapped] == pytest.approx([0.0, 0.5, 1.30])
    assert [note.end - note.start for note in snapped] == pytest.approx([0.5, 0.5, 0.30])
    assert snapped[0].source == "snapped"
    assert snapped[1].source == ""
    assert snapped[2].source == ""


def test_snap_notes_keeps_monotonic_and_clamps_end() -> None:
    notes = [
        MelodyNote(start=0.98, end=1.7, midi=60),
        MelodyNote(start=1.01, end=1.9, midi=64),
    ]
    snapped = snap_notes_to_beats(notes, _beats(), tolerance=0.1)
    starts = [note.start for note in snapped]
    assert starts == sorted(starts), "起点必须单调不减"
    for note in snapped:
        assert note.end >= note.start
    assert snapped[0].start == pytest.approx(1.0)


def test_snap_notes_empty_and_no_grid() -> None:
    assert snap_notes_to_beats([], _beats()) == []
    note = MelodyNote(start=0.123, end=0.4, midi=60)
    no_grid = BeatGrid(bpm=120.0, beats=[], downbeats=[])
    assert snap_notes_to_beats([note], no_grid) == [note]


# --------------------------------------------------------------------------- #
# performance SMF
# --------------------------------------------------------------------------- #


def test_write_performance_midi_is_valid_smf(tmp_path: Path) -> None:
    score = fuse(_vocal(), _accompaniment())
    path = tmp_path / "performance.mid"
    report = write_performance_midi(score, path)

    data = path.read_bytes()
    assert data.startswith(b"MThd"), "SMF 必须以 MThd 开头"
    assert len(data) == report["size_bytes"]

    parsed = parse_midi(data)
    assert parsed["format"] == 1
    assert parsed["division"] == PPQ
    assert [track["name"] for track in parsed["tracks"]] == ["tempo", "melody", "chords"]
    assert report["notes"]["melody"] == len(score.melody)
    assert parsed["tracks"][1]["notes"].__len__() == len(score.melody)
    assert report["duration_seconds"] >= 3.0

    # tempo 事件写进了第 0 轨
    tempo_meta = [meta for meta in parsed["tracks"][0]["meta"] if meta["type"] == 0x51]
    assert tempo_meta
    microseconds = int.from_bytes(tempo_meta[0]["data"], "big")
    assert 60_000_000 / microseconds == pytest.approx(120.0, abs=1e-3)


def test_write_performance_midi_still_legal_after_musicxml_roundtrip(tmp_path: Path) -> None:
    score = fuse(_vocal(), _accompaniment())
    restored = SymbolicScore.from_musicxml(write_musicxml(score), sidecar=score.to_dict())
    path = tmp_path / "restored.mid"
    report = write_performance_midi(restored, path)
    assert path.read_bytes().startswith(b"MThd")
    assert report["notes"]["melody"] == len(score.melody)


def test_write_performance_midi_rejects_non_score(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_performance_midi("not a score", tmp_path / "bad.mid")  # type: ignore[arg-type]

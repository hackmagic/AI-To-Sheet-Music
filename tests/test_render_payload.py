"""渲染层测试：绝对时间轴、ABC 兜底谱与网页渲染载荷。

只依赖 ``artifacts/score.json``（真实样例），不启动服务、不联网；
整个文件应在数秒内跑完。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.render import build_payload, build_timing, write_abc

ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "artifacts" / "score.json"

PAYLOAD_KEYS = {
    "schema",
    "title",
    "meta",
    "sections",
    "stats",
    "musicxml",
    "abc",
    "timing",
    "chord_diagrams",
    "audio_url",
}


@pytest.fixture(scope="module")
def score() -> dict:
    with SCORE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def payload(score: dict) -> dict:
    return build_payload(score)


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def test_timing_structure_and_monotonicity(score: dict) -> None:
    timing = build_timing(score)

    assert set(timing) == {"bars", "notes", "chords", "lyrics"}
    assert len(timing["bars"]) == len(score["bars"])
    assert timing["bars"], "样例谱面至少应有一小节"

    indices = [bar["index"] for bar in score["bars"]]
    assert [bar["index"] for bar in timing["bars"]] == indices

    previous_start = -1.0
    previous_end = -1.0
    for bar in timing["bars"]:
        assert bar["end"] >= bar["start"] >= 0.0
        assert bar["start"] >= previous_start - 1e-6
        assert bar["end"] >= previous_end - 1e-6
        previous_start, previous_end = bar["start"], bar["end"]

    bounds = {bar["index"]: (bar["start"], bar["end"]) for bar in timing["bars"]}

    assert timing["notes"] == sorted(
        timing["notes"], key=lambda item: (item["start"], item["end"], item["bar"])
    )
    assert timing["chords"] == sorted(
        timing["chords"], key=lambda item: (item["start"], item["end"], item["bar"])
    )
    assert timing["lyrics"] == sorted(
        timing["lyrics"], key=lambda item: (item["start"], item["end"], item["bar"])
    )

    for item in timing["notes"]:
        assert set(item) == {"start", "end", "midi", "lyric", "bar"}
        assert 0.0 <= item["start"] <= item["end"]
        assert item["midi"] is None or isinstance(item["midi"], int)
        bar_start, bar_end = bounds[item["bar"]]
        assert bar_start - 1e-6 <= item["start"] <= bar_end + 1e-6
        assert item["end"] <= bar_end + 1e-6

    for item in timing["chords"]:
        assert set(item) == {"start", "end", "symbol", "bar"}
        assert item["symbol"]
        assert 0.0 <= item["start"] <= item["end"]
        bar_start, bar_end = bounds[item["bar"]]
        assert bar_start - 1e-6 <= item["start"] <= bar_end + 1e-6
        assert item["end"] <= bar_end + 1e-6

    for item in timing["lyrics"]:
        assert set(item) == {"start", "end", "text", "bar"}
        assert item["text"].strip()
        assert 0.0 <= item["start"] <= item["end"]

    # 时间轴覆盖整首歌，且和弦/人声数量与谱面一致
    assert timing["bars"][-1]["end"] > timing["bars"][0]["start"]
    assert len(timing["chords"]) == sum(len(bar.get("chords") or []) for bar in score["bars"])
    assert len(timing["notes"]) == sum(len(bar.get("vocal") or []) for bar in score["bars"])
    lyric_notes = sum(
        1
        for bar in score["bars"]
        for note in bar.get("vocal") or []
        if str(note.get("lyric") or "").strip()
    )
    assert len(timing["lyrics"]) == lyric_notes


def test_timing_pickup_bar_is_shorter(score: dict) -> None:
    """弱起小节按真实时间跨度收窄，而不是硬撑满 4 拍。"""
    timing = build_timing(score)
    first = timing["bars"][0]
    span = first["end"] - first["start"]
    seconds_per_beat = 60.0 / float(score["meta"]["tempo"])
    assert 2.5 * seconds_per_beat < span < 3.5 * seconds_per_beat


# --------------------------------------------------------------------------- #
# abc
# --------------------------------------------------------------------------- #


def test_abc_header_and_chords(score: dict, payload: dict) -> None:
    abc = payload["abc"]
    lines = abc.splitlines()

    assert lines[0] == "X:1"
    assert any(line.startswith("T:") for line in lines)
    assert any(line.startswith("M:") for line in lines)
    assert "L:1/4" in lines
    assert any(line.startswith("Q:1/4=") for line in lines)
    assert any(line.startswith("K:") for line in lines)
    music_lines = [
        line
        for line in lines
        if line and not line.startswith(("X:", "T:", "M:", "L:", "Q:", "K:", "w:"))
    ]
    assert music_lines and music_lines[-1].endswith("|]")

    # 和弦行非空：谱面里出现过的和弦符号都要带引号标注
    symbols = {
        str(chord.get("symbol")).strip()
        for bar in score["bars"]
        for chord in bar.get("chords") or []
        if str(chord.get("symbol") or "").strip()
    }
    assert symbols
    for symbol in symbols:
        assert f'"{symbol}"' in abc, f"ABC 缺少和弦标注：{symbol}"
    assert abc.count('"') >= 2 * len(symbols)

    # 有歌词时必须有 w: 行
    assert any(line.startswith("w:") for line in lines)


def test_abc_accepts_symbolic_score() -> None:
    symbolic = {
        "schema": "aitsm.symbolic/v1",
        "beats": {"bpm": 120.0, "time_signature": "4/4", "downbeats": [0.0, 2.0]},
        "key": {"tonic": "G", "mode": "major", "fifths": 1},
        "chords": [{"start": 0.0, "end": 2.0, "symbol": "G"}],
        "melody": [{"start": 0.0, "end": 0.5, "midi": 67, "lyric": "hi"}],
        "meta": {"title": "Demo"},
    }
    abc = write_abc(symbolic)

    assert abc.startswith("X:1")
    assert "T:Demo" in abc
    assert "K:G" in abc
    assert '"G"' in abc
    assert "w: hi" in abc
    assert "|]" in abc


def test_abc_empty_score() -> None:
    abc = write_abc({"schema": "aitsm.score/v1", "meta": {"title": "空"}, "bars": []})
    assert abc.startswith("X:1")
    assert "K:C" in abc
    assert "z4 |]" in abc


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #


def test_payload_keys_and_json(score: dict, payload: dict) -> None:
    assert set(payload) == PAYLOAD_KEYS
    assert payload["schema"] == "aitsm.payload/v1"
    assert payload["title"] == score["meta"]["title"]
    assert len(payload["sections"]) == len(score["sections"])
    assert payload["audio_url"] == ""
    assert payload["musicxml"] == ""

    stats = payload["stats"]
    assert stats["bars"] == len(score["bars"])
    assert stats["sections"] == len(score["sections"])
    assert stats["chords"] == len(payload["timing"]["chords"])
    assert stats["notes"] == len(payload["timing"]["notes"])
    assert stats["lyrics"] == len(payload["timing"]["lyrics"])
    assert stats["voicings"] == len(score["voicings"])
    assert stats["duration_sec"] == pytest.approx(payload["timing"]["bars"][-1]["end"])

    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert json.loads(dumped) == payload


def test_payload_is_pure(score: dict) -> None:
    before = json.dumps(score, ensure_ascii=False, sort_keys=True)
    build_payload(score)
    assert json.dumps(score, ensure_ascii=False, sort_keys=True) == before


def test_chord_diagrams_match_every_bar(score: dict, payload: dict) -> None:
    diagrams = payload["chord_diagrams"]
    assert set(diagrams) == {"tuning", "capo", "voicings", "bars", "count"}

    assert diagrams["tuning"] == score["meta"]["guitar"]["tuning"]
    assert diagrams["capo"] == int(score["meta"].get("capo") or 0)
    assert set(diagrams["voicings"]) == set(score["voicings"])

    expected_total = sum(len(bar.get("chords") or []) for bar in score["bars"])
    assert diagrams["count"] == expected_total
    assert len(diagrams["bars"]) == len(score["bars"])

    seen = 0
    for source_bar, entry in zip(score["bars"], diagrams["bars"]):
        assert entry["index"] == source_bar["index"]
        source_chords = source_bar.get("chords") or []
        assert len(entry["chords"]) == len(source_chords)
        for source_chord, chord in zip(source_chords, entry["chords"]):
            assert chord["symbol"] == source_chord["symbol"]
            assert chord["beat"] == pytest.approx(float(source_chord.get("beat") or 0.0))
            assert chord["voicing_id"] == source_chord.get("voicing_id", "")
            diagram = chord["diagram"]
            assert diagram is not None, f"小节 {source_bar['index']} 的 {chord['symbol']} 缺少指法图"
            assert set(diagram) == {
                "id",
                "symbol",
                "frets",
                "fingers",
                "base_fret",
                "barres",
                "difficulty",
                "label",
            }
            assert len(diagram["frets"]) == len(diagrams["tuning"])
            if diagram["fingers"]:
                assert len(diagram["fingers"]) == len(diagram["frets"])
            assert isinstance(diagram["barres"], list)
            seen += 1
    assert seen == expected_total


def test_payload_overrides_and_missing_voicings(score: dict) -> None:
    trimmed = {
        "schema": score["schema"],
        "meta": score["meta"],
        "sections": score["sections"],
        "bars": score["bars"][:2],
    }
    payload = build_payload(
        trimmed,
        symbolic={"musicxml": "<score-partwise/>"},
        audio_url="  /jobs/demo/playback.mp3  ",
        abc="X:1\nT:手工\nL:1/4\nK:C\nC4 |]\n",
    )

    assert payload["musicxml"] == "<score-partwise/>"
    assert payload["audio_url"] == "/jobs/demo/playback.mp3"
    assert payload["abc"].startswith("X:1\nT:手工")
    assert payload["chord_diagrams"]["voicings"] == {}
    for source_bar, entry in zip(trimmed["bars"], payload["chord_diagrams"]["bars"]):
        for source_chord, chord in zip(source_bar.get("chords") or [], entry["chords"]):
            assert chord["voicing_id"] == source_chord.get("voicing_id", "")
            assert chord["diagram"] is None
    assert payload["stats"]["duration_sec"] >= 0.0

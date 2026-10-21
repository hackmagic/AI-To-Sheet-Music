"""歌词 ↔ 旋律对齐层的离线单测（纯算法，无模型依赖）。

覆盖：1:1 对齐、一字多音（melisma + tie 链）、一音多字（lyric_extend）、
音符多于歌词、歌词多于音符、带标点/英文歌词、越界音符容差、
无歌词/空旋律降级、重复调用幂等、入参不被修改、schema 校验。

运行方式（pytest 或直接 python 均可）：
    .venv/Scripts/python.exe tests/test_align.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.pipeline.align import (  # noqa: E402
    align_lyrics_to_melody,
    assign_notes_to_lines,
    syllables_from_lyrics,
)
from aitsm.pipeline import melody as M  # noqa: E402

BEATS = {"beats": [index * 0.5 for index in range(20)], "bpm": 120, "beats_per_bar": 4}


def _approx(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _melody(spans: list[tuple[float, float]], midis: list[int] | None = None) -> dict:
    notes = [
        {
            "start": start,
            "end": end,
            "midi": midis[index] if midis else 64,
            "confidence": 0.8,
            "onset": start,
            "lyric": None,
        }
        for index, (start, end) in enumerate(spans)
    ]
    return {"notes": notes, "range": {"low_midi": 60, "high_midi": 72}, "warnings": []}


# ---------------------------------------------------------------------------
# 音节展开 / 音符归属
# ---------------------------------------------------------------------------


def test_syllables_from_lyrics_marks_line_index() -> None:
    syllables, lines = syllables_from_lyrics(
        [
            {"start": 0.0, "end": 1.0, "text": "一个人"},
            {"start": 2.0, "end": 3.0, "text": "[Music]"},
        ]
    )
    assert len(lines) == 1
    assert [item["text"] for item in syllables] == ["一", "个", "人"]
    assert {item["line"] for item in syllables} == {0}


def test_syllables_from_lyrics_skips_empty_text() -> None:
    syllables, lines = syllables_from_lyrics([{"start": 0.0, "end": 1.0, "text": "  "}])
    assert syllables == []
    assert lines == []


def test_assign_notes_to_lines_by_overlap() -> None:
    notes = _melody([(0.2, 0.8), (3.2, 3.8)])["notes"]
    lines = [
        {"start": 0.0, "end": 1.0, "text": "一"},
        {"start": 3.0, "end": 4.0, "text": "二"},
    ]
    groups = assign_notes_to_lines(notes, lines, tolerance=0.25)
    assert groups == [[0], [1]]


def test_assign_notes_without_overlap_needs_tolerance() -> None:
    notes = _melody([(0.2, 0.4)])["notes"]
    lines = [{"start": 0.5, "end": 1.0, "text": "一"}]
    # 距离 0.1s：容差 0.25s 内可归属，紧容差下不归属
    assert assign_notes_to_lines(notes, lines, tolerance=0.25) == [[0]]
    assert assign_notes_to_lines(notes, lines, tolerance=0.01) == [[]]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def test_align_one_syllable_per_note() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)], [60, 62, 64, 65])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 2.0, "text": "一个人走"}], BEATS)
    assert [note["lyric"] for note in result["notes"]] == ["一", "个", "人", "走"]
    assert all("tie" not in note for note in result["notes"])
    assert result["alignment"]["coverage"] == 1.0
    assert result["alignment"]["matched_syllables"] == 4
    assert result["alignment"]["melisma"] == 0
    assert result["alignment"]["shared"] == 0


def test_align_melisma_uses_tie_chain() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0), (1.0, 1.6)], [64, 64, 64])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.6, "text": "啊"}], BEATS)
    notes = result["notes"]
    assert notes[0]["lyric"] == "啊"
    assert notes[0]["tie"] == "start"
    assert notes[1]["lyric"] is None
    assert notes[1]["tie"] == "continue"
    assert notes[2]["lyric"] is None
    assert notes[2]["tie"] == "stop"
    assert result["alignment"]["melisma"] == 1


def test_align_melisma_two_notes_uses_start_stop() -> None:
    melody = _melody([(0.0, 0.6), (0.6, 1.2)], [67, 69])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.2, "text": "梦"}], BEATS)
    notes = result["notes"]
    assert notes[0]["lyric"] == "梦"
    assert notes[0]["tie"] == "start"
    assert notes[1]["lyric"] is None
    assert notes[1]["tie"] == "stop"


def test_align_lyric_extend_when_more_syllables_than_notes() -> None:
    melody = _melody([(0.0, 1.0), (1.0, 2.0)], [60, 62])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 2.0, "text": "一个人走"}], BEATS)
    notes = result["notes"]
    assert all(note.get("lyric_extend") is True for note in notes)
    sung: list[str] = []
    for note in notes:
        sung.extend(str(note["lyric"]).split("-"))
    assert sung == ["一", "个", "人", "走"]
    assert result["alignment"]["shared"] >= 1


def test_align_more_notes_than_syllables_leaves_rest_without_lyric() -> None:
    melody = _melody([(0.0, 0.6), (0.6, 1.2), (2.0, 2.2), (3.0, 3.2), (4.0, 4.2)])
    lyrics = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "你好",
            "words": [
                {"text": "你", "start": 0.0, "end": 0.6},
                {"text": "好", "start": 0.6, "end": 1.2},
            ],
        }
    ]
    result = align_lyrics_to_melody(melody, lyrics, BEATS)
    notes = result["notes"]
    assert notes[0]["lyric"] == "你"
    assert notes[1]["lyric"] == "好"
    assert all(note["lyric"] is None for note in notes[2:])
    assert result["alignment"]["matched_syllables"] == 2
    assert result["alignment"]["notes_with_lyric"] == 2


def test_align_skips_stray_notes_inside_line() -> None:
    # 两个音节之间夹着一段无歌词的音符（伴奏/气声），不得硬塞歌词
    melody = _melody([(0.0, 0.4), (1.5, 1.9), (2.0, 2.4)], [60, 62, 64])
    lyrics = [
        {
            "start": 0.0,
            "end": 2.4,
            "text": "一二",
            "words": [
                {"text": "一", "start": 0.0, "end": 0.4},
                {"text": "二", "start": 2.0, "end": 2.4},
            ],
        }
    ]
    result = align_lyrics_to_melody(melody, lyrics, BEATS)
    lyrics_after = [note["lyric"] for note in result["notes"]]
    assert lyrics_after[0] == "一"
    assert lyrics_after[1] is None
    assert lyrics_after[2] == "二"
    assert result["alignment"]["matched_syllables"] == 2


def test_align_handles_punctuation_and_english() -> None:
    melody = _melody([(0.0, 0.4), (0.4, 0.8), (0.8, 1.2)])
    result = align_lyrics_to_melody(
        melody, [{"start": 0.0, "end": 1.2, "text": "I love you"}], BEATS
    )
    assert [note["lyric"] for note in result["notes"]] == ["I", "love", "you"]

    melody_zh = _melody([(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)])
    result_zh = align_lyrics_to_melody(
        melody_zh, [{"start": 0.0, "end": 2.0, "text": "一个人，走。"}], BEATS
    )
    assert [note["lyric"] for note in result_zh["notes"]] == ["一", "个", "人", "走"]


def test_align_accepts_note_slightly_before_line() -> None:
    melody = _melody([(0.2, 0.45)])
    result = align_lyrics_to_melody(melody, [{"start": 0.5, "end": 1.0, "text": "一"}], BEATS)
    assert result["notes"][0]["lyric"] == "一"
    assert result["alignment"]["lines_without_notes"] == 0


def test_align_long_note_single_syllable() -> None:
    melody = _melody([(0.0, 2.0)], [72])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 2.0, "text": "天"}], BEATS)
    assert result["notes"][0]["lyric"] == "天"
    assert "tie" not in result["notes"][0]


def test_align_two_lines_keep_order() -> None:
    melody = _melody(
        [(0.0, 0.5), (0.5, 1.0), (3.0, 3.5), (3.5, 4.0)], [60, 62, 64, 65]
    )
    result = align_lyrics_to_melody(
        melody,
        [
            {"start": 0.0, "end": 1.0, "text": "你好"},
            {"start": 3.0, "end": 4.0, "text": "再见"},
        ],
        BEATS,
    )
    assert [note["lyric"] for note in result["notes"]] == ["你", "好", "再", "见"]
    assert result["alignment"]["lines"] == 2


# ---------------------------------------------------------------------------
# 降级 / 边界
# ---------------------------------------------------------------------------


def test_align_empty_melody_returns_empty_with_warning() -> None:
    result = align_lyrics_to_melody({}, [{"start": 0.0, "end": 1.0, "text": "一"}], BEATS)
    assert result["notes"] == []
    assert result["alignment"]["notes"] == 0
    assert any("旋律" in warning for warning in result["warnings"])


def test_align_without_lyrics_leaves_notes_clean() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0)])
    melody["notes"][0]["lyric"] = "旧"
    melody["notes"][0]["tie"] = "start"
    result = align_lyrics_to_melody(melody, [], BEATS)
    assert [note["lyric"] for note in result["notes"]] == [None, None]
    assert all("tie" not in note for note in result["notes"])
    assert any("无歌词" in warning for warning in result["warnings"])


def test_align_far_line_reports_unmatched() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0)])
    result = align_lyrics_to_melody(
        melody, [{"start": 10.0, "end": 12.0, "text": "远方"}], BEATS
    )
    assert result["alignment"]["lines_without_notes"] == 1
    assert len(result["alignment"]["unmatched"]) == 2
    assert any("未找到对应旋律" in warning for warning in result["warnings"])


def test_align_is_pure_and_idempotent() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)], [60, 62, 64])
    snapshot = copy.deepcopy(melody)
    lyrics = [{"start": 0.0, "end": 1.5, "text": "一二三"}]
    first = align_lyrics_to_melody(melody, lyrics, BEATS)
    assert melody == snapshot, "入参不得被修改"
    second = align_lyrics_to_melody(melody, lyrics, BEATS)
    assert [note["lyric"] for note in first["notes"]] == [
        note["lyric"] for note in second["notes"]
    ]
    assert first["notes"] is not melody["notes"]


def test_aligning_twice_replaces_old_lyrics() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0)])
    first = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.0, "text": "旧词"}], BEATS)
    second = align_lyrics_to_melody(first, [{"start": 0.0, "end": 1.0, "text": "新词"}], BEATS)
    assert [note["lyric"] for note in second["notes"]] == ["新", "词"]
    assert all("lyric_extend" not in note for note in second["notes"])


def test_align_without_beats_still_works() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0)])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.0, "text": "你好"}], None)
    assert [note["lyric"] for note in result["notes"]] == ["你", "好"]


def test_melody_module_exposes_same_aligner() -> None:
    melody = _melody([(0.0, 0.5), (0.5, 1.0)])
    direct = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.0, "text": "你好"}], BEATS)
    wrapped = M.align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.0, "text": "你好"}], BEATS)
    assert [note["lyric"] for note in wrapped["notes"]] == [
        note["lyric"] for note in direct["notes"]
    ]


def test_align_output_passes_schema() -> None:
    try:
        import jsonschema
    except Exception:
        print("  (skip) jsonschema 未安装，跳过 schema 校验")
        return
    schema = json.loads((ROOT / "schemas" / "analysis.schema.json").read_text(encoding="utf-8"))
    melody_schema = schema["properties"]["melody"]
    melody = _melody([(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)], [60, 62, 64])
    result = align_lyrics_to_melody(melody, [{"start": 0.0, "end": 1.5, "text": "一二三"}], BEATS)
    jsonschema.validate(result, melody_schema)


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except Exception:  # noqa: BLE001 - 简易测试运行器
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
            else:
                print(f"PASS {name}")
    print(f"\n{'OK' if failures == 0 else f'{failures} FAILED'}")
    raise SystemExit(1 if failures else 0)

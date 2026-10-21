"""端到端自检：合成 analysis.json -> arrange() -> validate_score() -> score.json。

这份合成 analysis **严格按 `schemas/analysis.schema.json` 构造**（8 小节、84 BPM、
4/4、C 大调、4 个和弦、逐字歌词、10 个旋律音），因此可以当作「没有真实音频时的
回归基线」：只要 arrange/validate 改动破坏了主链路，这里就会先炸。

产物写到 ``work/score-selftest/score.json``，方便前端与导出器联调时直接取用。

运行方式：
    .venv/Scripts/python.exe tests/test_arrange_e2e.py
    .venv/Scripts/python.exe -m pytest tests/test_arrange_e2e.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.score.arrange import arrange  # noqa: E402
from aitsm.score.validate import validate_score  # noqa: E402

OUT_DIR = ROOT / "work" / "score-selftest"

BPM = 84.0
BEAT_SEC = 60.0 / BPM
BARS = 8
BEATS_PER_BAR = 4


def beat_time(beat: float) -> float:
    """拍序号 -> 秒。"""
    return round(beat * BEAT_SEC, 4)


def make_analysis() -> dict:
    """构造一份符合 analysis.schema.json 的合成分析结果（8 小节）。"""
    total_beats = BARS * BEATS_PER_BAR
    beats = [beat_time(index) for index in range(total_beats + 1)]
    downbeats = [beat_time(index * BEATS_PER_BAR) for index in range(BARS)]

    # 和弦进行 C - G - Am - F | C - G - F - C，每小节一个和弦
    progression = ["C", "G", "Am", "F", "C", "G", "F", "C"]
    chords = [
        {
            "start": beat_time(index * 4),
            "end": beat_time((index + 1) * 4),
            "symbol": symbol,
            "confidence": 0.92,
            "root_pc": {"C": 0, "G": 7, "Am": 9, "F": 5}[symbol],
            "bass_pc": {"C": 0, "G": 7, "Am": 9, "F": 5}[symbol],
        }
        for index, symbol in enumerate(progression)
    ]

    # 主歌 1-4 小节，副歌 5-8 小节
    sections = [
        {
            "start": beat_time(0),
            "end": beat_time(16),
            "kind": "verse",
            "label": "主歌",
            "energy": 0.42,
            "confidence": 0.85,
        },
        {
            "start": beat_time(16),
            "end": beat_time(32),
            "kind": "chorus",
            "label": "副歌",
            "energy": 0.78,
            "confidence": 0.85,
        },
    ]

    lyric_text = "月亮代表我的心呀真呀"  # 10 个字，与 melody_notes 一一对应
    chars = list(lyric_text)
    # 前 10 拍每拍一个字；歌词 words 与 melody 一一对应
    words = [
        {"start": beat_time(index), "end": beat_time(index + 1), "text": char}
        for index, char in enumerate(chars)
    ]
    melody_notes = [
        {
            "start": words[index]["start"],
            "end": words[index]["end"],
            "midi": midi,
            "confidence": 0.8,
            "lyric": words[index]["text"],
        }
        for index, midi in enumerate([64, 67, 69, 67, 65, 64, 62, 60, 62, 64])
    ]

    return {
        "schema": "aitsm.analysis/v1",
        "source": {
            "original_name": "selftest.mp3",
            "sha256": "0" * 64,
            "path": "work/score-selftest/selftest.mp3",
        },
        "audio": {
            "analysis_wav": "work/score-selftest/selftest.wav",
            "duration": beat_time(total_beats),
            "sample_rate": 22050,
        },
        "stems": {"vocals": None, "drums": None, "bass": None, "other": None},
        "beats": {
            "bpm": BPM,
            "bpm_display": BPM,
            "beats": beats,
            "downbeats": downbeats,
            "beats_per_bar": BEATS_PER_BAR,
            "time_signature": [4, 4],
            "confidence": 0.9,
        },
        "key": {
            "tonic": "C",
            "mode": "major",
            "confidence": 0.82,
            "num_sharps_flats": 0,
        },
        "chords": chords,
        "sections": sections,
        "lyrics": [
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": lyric_text,
                "words": words,
            }
        ],
        "melody": {"notes": melody_notes},
        "tools": {"beats": "synthetic", "chords": "synthetic", "melody": "synthetic"},
        "warnings": [],
    }


def test_arrange_e2e_produces_valid_score() -> None:
    """合成 analysis -> arrange -> validate，全链路必须通过。"""
    analysis = make_analysis()
    score = arrange(analysis, {"difficulty": "beginner", "style": "pop", "include_tab": True})
    report = validate_score(score)

    assert report["ok"], f"校验失败: {json.dumps(report['errors'], ensure_ascii=False, indent=2)}"
    assert score["schema"] == "aitsm.score/v1"
    assert len(score["bars"]) == BARS
    # 每小节都要有和弦（跨小节长和弦必须延续）
    for bar in score["bars"]:
        assert bar.get("chords"), f"第 {bar['index']} 小节没有和弦"
        assert bar["guitar"].get("pattern_id")
    # 段落覆盖 1..8，首段主歌、末段副歌
    assert score["sections"][0]["start_bar"] == 1
    assert score["sections"][-1]["end_bar"] == BARS
    assert score["sections"][0]["kind"] == "verse"
    assert score["sections"][-1]["kind"] == "chorus"
    # 旋律 + 歌词都要落到小节里
    vocal = [note for bar in score["bars"] for note in bar.get("vocal") or []]
    assert len(vocal) == 10
    assert sum(1 for note in vocal if note.get("lyric")) == 10
    # 调性与 capo 契约完整
    key = score["meta"]["key"]
    for field in ("detected", "sounding", "play", "mode", "num_sharps_flats", "confidence"):
        assert field in key
    assert score["meta"]["capo"] == 0
    # 每个和弦都能查到指法
    for bar in score["bars"]:
        for chord in bar["chords"]:
            assert chord.get("voicing_id"), f"{chord['symbol']} 没选到指法"
            assert chord["voicing_id"] in score["voicings"]

    # 落盘，供前端 / 导出联调
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "score.json"
    path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    assert path.is_file() and path.stat().st_size > 1000


def test_arrange_e2e_options_tab_and_difficulty() -> None:
    """include_tab / difficulty 选项要真实生效。"""
    analysis = make_analysis()
    with_tab = arrange(analysis, {"include_tab": True})
    without_tab = arrange(analysis, {"include_tab": False})
    assert any(bar["guitar"].get("hits") for bar in with_tab["bars"])
    assert not any(bar["guitar"].get("hits") for bar in without_tab["bars"])
    assert validate_score(without_tab)["ok"]

    beginner = arrange(analysis, {"difficulty": "beginner"})
    advanced = arrange(analysis, {"difficulty": "advanced"})
    beginner_max = max(v["difficulty"] for v in beginner["voicings"].values())
    advanced_max = max(v["difficulty"] for v in advanced["voicings"].values())
    assert beginner_max <= 3, f"beginner 用了难度 {beginner_max} 的指法"
    assert advanced_max >= beginner_max


def test_arrange_e2e_capo_user_override() -> None:
    """用户指定 capo 时，记谱音高必须整体减 capo，发声音高不变。"""
    analysis = make_analysis()
    score = arrange(analysis, {"capo": 2})
    assert score["meta"]["capo"] == 2
    assert score["meta"]["key"]["play"] == "Bb" or score["meta"]["key"]["play"] == "A#"
    # C 大调 + capo2 -> 记谱 Bb 大调
    assert score["meta"]["key"]["num_sharps_flats"] == -2
    report = validate_score(score)
    assert report["ok"], json.dumps(report["errors"], ensure_ascii=False, indent=2)
    # 旋律音高比原曲低 2 个半音
    original = arrange(make_analysis(), {"capo": 0})
    first_original = [n for bar in original["bars"] for n in bar.get("vocal") or []][0]
    first_capo = [n for bar in score["bars"] for n in bar.get("vocal") or []][0]
    assert first_capo["midi"] == first_original["midi"] - 2


def test_arrange_e2e_deterministic() -> None:
    """同样的输入必须产出同样的谱（除时间戳外逐字段一致）。"""
    analysis = make_analysis()
    first = arrange(analysis, {})
    second = arrange(analysis, {})
    first["meta"]["generator"].pop("created_at", None)
    second["meta"]["generator"].pop("created_at", None)
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(
        second, sort_keys=True, ensure_ascii=False
    )


def main() -> int:
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
            else:
                print(f"[ok] {name}")
    print(f"\n{'全部通过' if not failures else f'{failures} 个失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

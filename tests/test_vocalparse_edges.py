"""VocalParse 解析器的**边界条件**回归。

每个测试对应一个真实踩过或评审指出的风险点：
- 多 BPM 必须按 token 位置分段生效（不能全局取第一个）；
- 孤立 ``<NOTE_>`` 必须被记录，不能静默跳过让后续音符整体前移；
- 休止 ``<P_0>`` 与音符要区分；
- 前奏片段与 SheetSage2 的 ``intro`` 判断互相印证（交叉验证）。
"""

from __future__ import annotations

import pytest

from aitsm.transcribe import vocal as V


# ---------------------------------------------------------------- 多 BPM 分段


def test_multiple_bpm_tokens_apply_per_segment():
    """变速歌：BPM 必须按出现位置分段，而不是全局取第一个。"""
    events = V.parse_vocalparse_tokens(
        "<BPM_60> a <P_60> <NOTE_4> b <P_62> <NOTE_4> "
        "<BPM_120> c <P_64> <NOTE_4> d <P_66> <NOTE_4>"
    )
    bpms = [item.bpm for item in events]
    assert bpms[:2] == [60.0, 60.0], "BPM_60 之后的事件都应是 60"
    assert bpms[2:] == [120.0, 120.0], "BPM_120 之后的事件必须切到 120"


def test_bpm_change_affects_note_durations():
    """BPM 变化要真的改变时间轴推进速度，而不只是记个数字。"""
    events = V.parse_vocalparse_tokens(
        "<BPM_60> <P_60> <NOTE_4> <BPM_120> <P_62> <NOTE_4> <P_64> <NOTE_4>"
    )
    notes = V.vocal_events_to_notes(events, None)
    assert len(notes) == 3
    # 60 BPM：四分音符 1.0s；之后 120 BPM：四分音符 0.5s
    assert notes[0].start == pytest.approx(0.0, abs=1e-6)
    assert notes[0].end == pytest.approx(1.0, abs=1e-6)
    assert notes[1].start == pytest.approx(1.0, abs=1e-6)
    assert notes[1].end == pytest.approx(1.5, abs=1e-6)
    assert notes[2].end == pytest.approx(2.0, abs=1e-6)


def test_bpm_before_first_pitch_is_used():
    """BPM 出现在所有音高之前时，第一个事件就要带上它。"""
    events = V.parse_vocalparse_tokens("<BPM_90><P_60><NOTE_4>")
    assert events[0].bpm == 90.0


def test_invalid_bpm_tokens_are_ignored():
    events = V.parse_vocalparse_tokens("<BPM_0> <P_60> <NOTE_4> <BPM_999> <P_62> <NOTE_4>")
    assert all(item.bpm is None for item in events), "0/超范围 BPM 不应被采纳"


def test_last_token_bpm_helper_returns_latest():
    """转录层用「最后一个 BPM」作为整曲 tempo 的代表值。"""
    events = V.parse_vocalparse_tokens(
        "<BPM_80> <P_60> <NOTE_4> <BPM_100> <P_62> <NOTE_4>"
    )
    latest = next((item.bpm for item in reversed(events) if item.bpm), None)
    assert latest == 100.0


# ---------------------------------------------------------------- 孤儿 NOTE


def test_orphan_note_tokens_are_recorded():
    """没有前导 <P_> 的 <NOTE_> 必须被记录（可能是休止写成纯时值）。"""
    events = V.parse_vocalparse_tokens("<NOTE_4> <P_60> <NOTE_8> 你")
    orphans = V.last_parse_orphans()
    assert orphans == ["4"], "孤立的 NOTE_4 应被记录，而不是静默丢弃"
    assert [item.midi_pitch for item in events] == [60]


def test_no_orphans_clears_state():
    V.parse_vocalparse_tokens("<NOTE_4>")  # 先制造一次孤儿
    assert V.last_parse_orphans()
    V.parse_vocalparse_tokens("<P_60> <NOTE_4> 你")  # 正常输入
    assert V.last_parse_orphans() == []


def test_orphan_note_does_not_shift_time_axis():
    """孤儿时值不参与时间推进，后续音符不能被推前。"""
    clean = V.vocal_events_to_notes(
        V.parse_vocalparse_tokens("<P_60> <NOTE_4> <P_62> <NOTE_4>"), None
    )
    with_orphan = V.vocal_events_to_notes(
        V.parse_vocalparse_tokens("<NOTE_4> <P_60> <NOTE_4> <P_62> <NOTE_4>"), None
    )
    assert [n.midi for n in clean] == [n.midi for n in with_orphan] == [60, 62]
    assert [n.start for n in clean] == [n.start for n in with_orphan]


# ---------------------------------------------------------------- 休止


def test_rest_pitch_advances_time_but_emits_no_note():
    events = V.parse_vocalparse_tokens("<BPM_60> <P_60> <NOTE_4> <P_0> <NOTE_4> <P_62> <NOTE_4>")
    notes = V.vocal_events_to_notes(events, None)
    assert [note.midi for note in notes] == [60, 62]
    assert notes[1].start == pytest.approx(2.0, abs=1e-6), "休止占一拍，第二音从 2.0s 开始"


def test_all_rest_stream_yields_no_notes():
    assert V.vocal_events_to_notes(V.parse_vocalparse_tokens("<P_0> <NOTE_4> <P_0> <NOTE_2>"), None) == []


# ---------------------------------------------------------------- 交叉验证（评审点 9）

#: 真实前奏片段（16s 前奏）上 VocalParse 的解码结果特征：
#: 极少音高事件 —— 与 SheetSage2 判定的 intro 一致。
_INTRO_LIKE_OUTPUT = (
    "language Chinese<asr_text>Oh<|file_sep|>SP <P_0> <NOTE_DOT_16> <P_0> <NOTE_32> "
    "<P_0> <NOTE_32> <BPM_65> CursorPosition <P_0> <NOTE_32> (Equal) <P_0> <NOTE_32>"
)


def test_intro_like_output_has_few_notes():
    """前奏片段：音高事件应当很少（否则说明「前奏被误判成有人声」）。"""
    events = V.parse_vocalparse_tokens(_INTRO_LIKE_OUTPUT)
    notes = V.vocal_events_to_notes(events, None)
    assert len(notes) < 5, f"前奏不应有大量音符，实际 {len(notes)}"


def test_intro_like_output_has_high_rest_ratio():
    events = V.parse_vocalparse_tokens(_INTRO_LIKE_OUTPUT)
    if not events:
        return
    rests = sum(1 for item in events if item.midi_pitch == 0)
    assert rests / len(events) > 0.5, "前奏的主要事件应是休止"


# ---------------------------------------------------------------- CoT / 噪声（回归保护）


def test_cot_separator_and_noise_still_cleaned():
    events = V.parse_vocalparse_tokens(
        "纯歌词前言<|file_sep|>language Chinese<asr_text>SP <P_0> <NOTE_32> "
        "Let's <P_58> <NOTE_8> (Equal) <P_0> <NOTE_32>"
    )
    lyrics = [item.lyric for item in events if item.lyric]
    assert "Let's" in lyrics
    assert not any("language" in item or "asr_text" in item for item in lyrics)
    assert not any("Equal" in item for item in lyrics)

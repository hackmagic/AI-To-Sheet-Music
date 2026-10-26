"""VocalParse **真实输出**格式的回归测试。

样本来自 `pymaster/VocalParse` 在真实音频上的实际生成结果（2026-10 实测），
不是文档示例。这些结构固定下来，避免以后改动解析器时又把这些噪声当歌词。
"""

from __future__ import annotations

import pytest

from aitsm.transcribe import vocal as V

#: 真实生成的一段（已裁剪，保留全部结构性噪声）
REAL_OUTPUT = (
    "language Chinese<asr_text>Let's break it down哦哦哦Let's break it down"
    "<|file_sep|>SP <P_0> <NOTE_DOT_16> Let's <P_58> <NOTE_DOT_8> <P_60> <NOTE_DOT_16> "
    "<P_62> <NOTE_2> <P_66> <NOTE_2> <P_68> <NOTE_DOT_2> "
    "SP <P_0> <NOTE_DOT_16> break <P_0> <NOTE_32> it <P_64> <NOTE_32> down <P_0> <NOTE_32> "
    "<BPM_80> CursorPosition <P_0> <NOTE_32> (Equal) <P_0> <NOTE_32> <BPM_65>"
)


# ---------------------------------------------------------------- CoT 分隔


def test_cot_prefix_is_dropped():
    """`<|file_sep|>` 之前是纯歌词，不能混进结构化序列。"""
    events = V.parse_vocalparse_tokens(REAL_OUTPUT)
    lyrics = " ".join(item.lyric for item in events if item.lyric)
    assert "language Chinese" not in lyrics
    assert "<asr_text>" not in lyrics
    assert "break it down哦" not in lyrics


def test_sequence_without_separator_is_parsed_whole():
    """标准 interleaved 输出（无分隔符）仍然完整解析。"""
    events = V.parse_vocalparse_tokens("<BPM_89><P_68><NOTE_4>你<P_69><NOTE_8>好")
    assert [(item.midi_pitch, item.lyric) for item in events] == [(68, "你"), (69, "好")]


# ---------------------------------------------------------------- 占位符噪声


@pytest.mark.parametrize("noise", ["SP", "CursorPosition", "Equal", "(Equal)", "[SP]", "   "])
def test_placeholders_are_not_lyrics(noise: str) -> None:
    events = V.parse_vocalparse_tokens(f"{noise} <P_60> <NOTE_4> {noise}")
    assert all(item.lyric == "" for item in events)


def test_bracketed_noise_does_not_leak_into_lyric():
    events = V.parse_vocalparse_tokens("(Equal) <P_64> <NOTE_4> 你 (Equal) <P_65> <NOTE_4> 好")
    assert [item.lyric for item in events] == ["你", "好"]


# ---------------------------------------------------------------- 休止 <P_0>


def test_pitch_zero_is_a_rest_not_midi_zero():
    """`<P_0>` 是休止：不产出音符，但时间轴照常推进。"""
    events = V.parse_vocalparse_tokens("SP <P_0> <NOTE_4> 哦 <P_60> <NOTE_4> 好")
    notes = V.vocal_events_to_notes(events, None)
    assert [note.midi for note in notes] == [60]
    # 休止占一个四分音符：120 BPM 下 0.5s，所以真音符从 0.5s 开始
    assert notes[0].start == pytest.approx(0.5, abs=1e-6)


def test_rest_keeps_time_axis_advancing():
    events = V.parse_vocalparse_tokens("<P_60> <NOTE_4> a <P_0> <NOTE_4> <P_62> <NOTE_4> b")
    notes = V.vocal_events_to_notes(events, None)
    assert [note.midi for note in notes] == [60, 62]
    assert notes[1].start == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- 歌词挂接方向


def test_lyric_before_pitch_attaches_to_that_note():
    """真实格式：`Let's <P_58> <NOTE_x>`。"""
    events = V.parse_vocalparse_tokens("Let's <P_58> <NOTE_8> break <P_60> <NOTE_8>")
    assert [(item.midi_pitch, item.lyric) for item in events] == [(58, "Let's"), (60, "break")]


def test_lyric_after_pitch_attaches_to_that_note():
    """文档格式：`<P_68> <NOTE_4> 你`。"""
    events = V.parse_vocalparse_tokens("<P_68> <NOTE_4> 你 <P_69> <NOTE_4> 好")
    assert [(item.midi_pitch, item.lyric) for item in events] == [(68, "你"), (69, "好")]


def test_real_output_carries_recognisable_lyrics():
    events = V.parse_vocalparse_tokens(REAL_OUTPUT)
    lyrics = [item.lyric for item in events if item.lyric]
    assert "Let's" in lyrics
    # 真实音频里 whisper 级别的英文识别即可，不强求全对
    assert any(item in lyrics for item in ("it", "break"))


def test_real_output_bpm_is_captured():
    events = V.parse_vocalparse_tokens(REAL_OUTPUT)
    bpms = {item.bpm for item in events if item.bpm}
    assert 80.0 in bpms
    # 后面的 BPM_65 是模型自我修正，不应覆盖前面的有效值以外的语义
    assert all(0 < bpm <= 255 for bpm in bpms)


def test_real_output_event_count_is_reasonable():
    """真实串解析出的音高事件数量不应爆炸（噪声被清理后应可控）。"""
    events = V.parse_vocalparse_tokens(REAL_OUTPUT)
    assert 5 <= len(events) <= 30


# ---------------------------------------------------------------- 健壮性


def test_empty_and_garbage_inputs():
    for value in ("", None, "   ", "noise without tokens"):
        assert V.parse_vocalparse_tokens(value) == []


def test_rest_only_stream_produces_no_notes():
    events = V.parse_vocalparse_tokens("<P_0> <NOTE_4> <P_0> <NOTE_4>")
    assert V.vocal_events_to_notes(events, None) == []

"""VocalParse 分段推理与 BPM 统计的测试（不需要模型）。

VocalParse 模型 card 明确「>30s 应先分段，否则后半段质量下降」。
本文件锁住：窗口切分、段内偏移拼接、片段 BPM 的稳健取值。
"""

from __future__ import annotations

import pytest

from aitsm.transcribe import vocal as V


# ---------------------------------------------------------------- 窗口切分


class _FakeAudio:
    """最小可切片对象：``_split_windows`` 只要求 ``len()`` 与切片。"""

    def __init__(self, length: int):
        self.length = length
        self.slices: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item):
        start, stop = item.start or 0, item.stop if item.stop is not None else self.length
        self.slices.append((start, stop))
        return ("slice", start, stop)


def test_short_audio_is_single_window():
    audio = _FakeAudio(16000 * 10)  # 10s
    windows = V._split_windows(audio, 16000, 30.0)
    assert len(windows) == 1
    assert windows[0][0] == 0.0


def test_exactly_max_duration_is_single_window():
    audio = _FakeAudio(16000 * 30)
    assert len(V._split_windows(audio, 16000, 30.0)) == 1


def test_long_audio_is_split_into_30s_windows():
    """75s 音频应切成 3 段（30 + 30 + 15），不丢尾巴。"""
    audio = _FakeAudio(16000 * 75)
    windows = V._split_windows(audio, 16000, 30.0)
    assert len(windows) == 3
    offsets = [offset for offset, _chunk in windows]
    assert offsets == [pytest.approx(0.0), pytest.approx(30.0), pytest.approx(60.0)]


def test_window_offsets_are_in_seconds():
    """偏移必须以秒为单位（下游用它拼回绝对时间轴）。"""
    audio = _FakeAudio(16000 * 95)
    windows = V._split_windows(audio, 16000, 30.0)
    assert [round(offset) for offset, _ in windows] == [0, 30, 60, 90]


def test_windows_cover_entire_audio():
    """切片的 [start, stop) 必须无缝覆盖整段，不能丢样本也不能重叠。"""
    audio = _FakeAudio(16000 * 100)
    V._split_windows(audio, 16000, 30.0)
    assert audio.slices, "应当发生了切片"
    assert audio.slices[0][0] == 0
    assert audio.slices[-1][1] == 16000 * 100
    for previous, current in zip(audio.slices, audio.slices[1:]):
        assert previous[1] == current[0], "相邻切片必须首尾相接"


def test_windows_do_not_exceed_max():
    audio = _FakeAudio(16000 * 100)
    for _offset, chunk in V._split_windows(audio, 16000, 30.0):
        assert len(chunk) <= 16000 * 30


def test_zero_length_audio_is_single_window():
    assert len(V._split_windows(_FakeAudio(0), 16000, 30.0)) == 1


def test_max_window_constant_matches_model_card():
    """30s 不是随便取的：模型 card 的 Limitations 明确写了 >30s 要分段。"""
    assert V.VOCALPARSE_MAX_WINDOW == 30.0
    assert V.SAMPLE_RATE_16K == 16000


# ---------------------------------------------------------------- BPM 稳健取值


def _events(*bpms):
    return [V.VocalEvent(lyric="", midi_pitch=60, note_value="4", bpm=bpm) for bpm in bpms]


def test_dominant_bpm_picks_most_frequent():
    """多数派的 BPM 代表全曲，而不是「最后一个」。"""
    assert V._dominant_bpm(_events(120, 120, 120, 60)) == 120


def test_dominant_bpm_ignores_trailing_outlier():
    """最后一段是慢速尾奏时，不能被它带偏。"""
    assert V._dominant_bpm(_events(96, 96, 96, 96, 55)) == 96


def test_dominant_bpm_tie_takes_first_seen():
    assert V._dominant_bpm(_events(100, 80)) == 100


def test_dominant_bpm_none_when_no_bpm():
    assert V._dominant_bpm(_events(None, None)) is None


def test_dominant_bpm_empty():
    assert V._dominant_bpm([]) is None


# ---------------------------------------------------------------- 多段拼接语义


def test_multi_window_notes_are_offset_and_sorted():
    """段内音符拼回绝对时间时：加偏移 + 保持有序（模拟两段拼接）。"""
    chunk_a = V.parse_vocalparse_tokens("<BPM_120> <P_60> <NOTE_4> a")
    chunk_b = V.parse_vocalparse_tokens("<BPM_120> <P_64> <NOTE_4> b")

    shifted = []
    for offset, chunk in ((0.0, chunk_a), (30.0, chunk_b)):
        for note in V.vocal_events_to_notes(chunk, None):
            shifted.append((round(note.start + offset, 6), note.midi, note.lyric))
    shifted.sort()

    assert shifted == [(0.0, 60, "a"), (30.0, 64, "b")]


def test_multi_window_resets_syllable_index():
    """跨段后音节序号要重编，避免第二段从 0 重来造成重复。"""
    chunk_a = V.parse_vocalparse_tokens("<P_60> <NOTE_4> a <P_62> <NOTE_4> b")
    chunk_b = V.parse_vocalparse_tokens("<P_64> <NOTE_4> c")
    merged = []
    for offset, events in ((0.0, chunk_a), (30.0, chunk_b)):
        for note in V.vocal_events_to_notes(events, None):
            merged.append(note)
    cursor = 0
    for note in merged:
        if note.lyric:
            note.syllable_index = cursor
            cursor += 1
    assert [note.syllable_index for note in merged] == [0, 1, 2]

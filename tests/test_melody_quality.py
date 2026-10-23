"""旋律（f0）与歌词对齐的「音乐约束」单测。

覆盖 4 项改进中的两项：
- pyin 前置带通 + F0 智能中值平滑（melody）
- Whisper 时间戳降级为软锚点 + 节拍/休止约束（align）
"""

from __future__ import annotations

import numpy as np
import pytest

from aitsm.pipeline import align as align_mod
from aitsm.pipeline import melody as melody_mod


# --- F0 智能中值平滑 --------------------------------------------------------


def test_smooth_f0_track_removes_single_frame_octave_spike():
    """孤立一帧的倍频误判（+12 半音）应被修掉，周围的真实音高不动。"""
    values = [60.0] * 5 + [72.0] + [60.0] * 5
    smoothed, fixed = melody_mod.smooth_f0_track(values, kernel=5, max_deviation=1.5, min_run=2)
    assert fixed == 1
    assert smoothed[5] == pytest.approx(60.0)


def test_smooth_f0_track_keeps_real_two_frame_run():
    """真实的连续两帧高八度（快速跳进）不该被抹掉——min_run 保护在这里。"""
    values = [60.0] * 4 + [72.0, 72.0] + [60.0] * 4
    smoothed, fixed = melody_mod.smooth_f0_track(values, kernel=5, max_deviation=1.5, min_run=2)
    assert fixed == 0
    assert smoothed[4] == pytest.approx(72.0)


def test_smooth_f0_track_does_not_fill_unvoiced_frames():
    """None（未发声）必须保持 None，绝不因为平滑而变成有声帧。"""
    values = [60.0, None, 60.0, None, 61.0]
    smoothed, _fixed = melody_mod.smooth_f0_track(values)
    assert smoothed[1] is None
    assert smoothed[3] is None


# --- 前置带通 ----------------------------------------------------------------


def test_bandpass_removes_subsonic_energy():
    sr = 22050
    times = np.arange(sr, dtype=float) / sr
    # 20 Hz 次声 + 440 Hz 基频
    tone = 0.5 * np.sin(2 * np.pi * 20.0 * times) + 0.5 * np.sin(2 * np.pi * 440.0 * times)
    filtered, info = melody_mod._bandpass_audio(tone, sr, np=np, low=65.0, high=1200.0)
    assert info.startswith("bandpass")
    spectrum_before = np.abs(np.fft.rfft(tone))
    spectrum_after = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(tone), 1.0 / sr)
    low_band = (freqs >= 15) & (freqs <= 30)
    mid_band = (freqs >= 430) & (freqs <= 450)
    assert spectrum_after[low_band].max() < 0.05 * spectrum_before[low_band].max()
    assert spectrum_after[mid_band].max() > 0.5 * spectrum_before[mid_band].max()


def test_bandpass_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AITSM_F0_BANDPASS", "0")
    tone = np.zeros(22050)
    filtered, info = melody_mod._bandpass_audio(tone, 22050, np=np)
    assert info == "disabled"
    assert filtered is tone


def test_bandpass_tolerates_bad_limits():
    tone = np.zeros(22050)
    _filtered, info = melody_mod._bandpass_audio(tone, 22050, np=np, low=2000.0, high=100.0)
    assert info == "skipped"


# --- 歌词对齐：软锚点 --------------------------------------------------------


def test_match_cost_saturates_beyond_anchor_cap():
    """时间差超过锚点上限后代价不再增长——Whisper 整段错位不该压倒音乐结构。

    两个音符都与音节完全不相交（overlap 项相同），只差距离：一个超出上限 2 s，
    另一个超出上限 9 s，代价必须相等。
    """
    breach_2s = align_mod._match_cost(0.0, 0.5, 2.0, 2.5)
    breach_9s = align_mod._match_cost(0.0, 0.5, 9.0, 9.5)
    assert breach_2s == pytest.approx(breach_9s)


def test_match_cost_beat_bonus_lowers_cost():
    base = align_mod._match_cost(1.0, 1.3, 1.0, 1.3)
    bonus = align_mod._match_cost(1.0, 1.3, 1.0, 1.3, beat_bonus=-align_mod._BEAT_BONUS)
    assert bonus == pytest.approx(base - align_mod._BEAT_BONUS)


def test_match_cost_rest_penalty_counts_capped():
    base = align_mod._match_cost(1.0, 1.3, 1.0, 1.3)
    one = align_mod._match_cost(1.0, 1.3, 1.0, 1.3, rest_count=1)
    many = align_mod._match_cost(1.0, 1.3, 1.0, 1.3, rest_count=5)
    assert one == pytest.approx(base + align_mod._REST_PENALTY)
    assert many == pytest.approx(base + 2 * align_mod._REST_PENALTY)  # 上限 2


def test_align_prefers_on_beat_note_when_timestamps_are_ambiguous():
    """两个音符时间证据相同（都在锚点上限之外），节拍约束决定挂给谁。"""
    beats = {"bpm": 120.0, "beats": [0.0, 0.5, 1.0, 1.5], "beats_per_bar": 4}
    notes = [
        {"start": 0.0, "end": 0.2, "midi": 60, "confidence": 0.8},  # 在拍点
        {"start": 0.3, "end": 0.55, "midi": 62, "confidence": 0.8},  # 不在拍点
    ]
    syllables = [{"text": "la", "start": 5.0, "end": 5.3, "line": 0}]
    ops = align_mod.align_syllables_to_notes(
        syllables, notes, beat_times=beats["beats"], beat_period=0.5
    )
    matched = [op for op in ops if op[0] == "syllable"]
    assert matched, ops
    assert matched[0][2] == 0  # 挂到第 0 个音符（在拍点上）


def test_align_melisma_across_rest_is_penalised():
    """跨长休止的一字多音代价更高：同样两个音，中间有休止时不该被判成 melisma。"""
    beats = {"bpm": 120.0, "beats": [0.0, 0.5, 1.0], "beats_per_bar": 4}
    notes = [
        {"start": 0.0, "end": 0.2, "midi": 60, "confidence": 0.8},
        {"start": 2.0, "end": 2.2, "midi": 62, "confidence": 0.8},  # 长休止后
    ]
    syllables = [{"text": "la", "start": 0.0, "end": 0.25, "line": 0}]
    ops = align_mod.align_syllables_to_notes(
        syllables, notes, beat_times=beats["beats"], beat_period=0.5
    )
    # 音节覆盖两个音是 melisma；因为跨休止被惩罚，DP 更愿意只挂第一个音
    assert ("syllable", 0, 0, 1) in ops
    assert ("syllable", 0, 0, 2) not in ops


def test_align_reports_beat_anchors():
    beats = {"bpm": 120.0, "beats": [0.0, 0.5, 1.0], "beats_per_bar": 4}
    melody = {"notes": [{"start": 0.0, "end": 0.5, "midi": 60, "confidence": 0.9}]}
    lyrics = [{"text": "la la", "start": 0.0, "end": 0.6}]
    aligned = align_mod.align_lyrics_to_melody(melody, lyrics, beats)
    assert aligned["alignment"]["beat_anchors"] == 3

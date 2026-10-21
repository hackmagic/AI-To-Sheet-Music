"""节拍/速度/拍号单测：纯函数用合成数组，端到端用合成 click 轨。"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from aitsm.pipeline import beats as beats_module

np = pytest.importorskip("numpy")


def write_click_track(
    path: Path,
    *,
    bpm: float = 120.0,
    seconds: float = 16.0,
    sample_rate: int = 22050,
    beats_per_bar: int = 4,
) -> Path:
    """写 4/4 强弱拍 click：每小节第一拍是低频底鼓，其余是高频率。"""
    frames = int(seconds * sample_rate)
    data = np.zeros(frames, dtype=np.float64)
    interval = 60.0 / bpm
    time_cursor = 0.0
    beat_index = 0
    while time_cursor < seconds - 0.05:
        start = int(time_cursor * sample_rate)
        length = int(0.05 * sample_rate)
        phase = beat_index % beats_per_bar
        frequency = 70.0 if phase == 0 else 1400.0
        amplitude = 1.0 if phase == 0 else 0.45
        local = np.arange(length) / sample_rate
        envelope = np.exp(-local * 70.0)
        segment = amplitude * envelope * np.sin(2.0 * np.pi * frequency * local)
        end = min(frames, start + length)
        data[start:end] += segment[: end - start]
        time_cursor += interval
        beat_index += 1
    pcm = np.clip(data * 16000.0, -32767, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def write_silence(path: Path, *, seconds: float = 4.0, sample_rate: int = 22050) -> Path:
    frames = int(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def make_beats(bpm: float = 120.0, seconds: float = 8.0, beats_per_bar: int = 4) -> dict:
    interval = 60.0 / bpm
    times = [round(index * interval, 4) for index in range(int(seconds / interval))]
    return {
        "bpm": bpm,
        "bpm_display": bpm,
        "beats": times,
        "downbeats": times[::beats_per_bar],
        "beats_per_bar": beats_per_bar,
        "time_signature": f"{beats_per_bar}/4",
        "confidence": 0.8,
    }


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #

def test_autocorrelate_is_normalized_and_peaks_at_lag_zero():
    signal = np.zeros(200)
    signal[::20] = 1.0  # 每 20 帧一个脉冲
    ac = beats_module.autocorrelate(signal, 100)
    assert ac[0] == pytest.approx(1.0, abs=1e-6)
    # 周期性 → 第 20 帧是除 0 外的最大峰（脉冲串只占 1/20 帧，故小于 1）
    assert ac[20] > 0.8
    assert int(np.argmax(ac[1:])) + 1 == 20
    assert abs(float(ac[10])) < 0.3


def test_tempo_prior_prefers_musical_range():
    assert beats_module.tempo_prior(100.0) > beats_module.tempo_prior(30.0)
    assert beats_module.tempo_prior(100.0) > beats_module.tempo_prior(240.0)
    assert 0.0 <= beats_module.tempo_prior(60.0) <= 1.0


def test_choose_display_tempo_avoids_double_time():
    sr, hop = 22050, 512
    # 造一个 160BPM 的 onset 包络自相关（帧间隔 = 60/160*22050/512 ≈ 16.1 帧）
    envelope = np.zeros(400)
    envelope[::16] = 1.0
    ac = beats_module.autocorrelate(envelope, 400)
    result = beats_module.choose_display_tempo(160.0, ac, sr, hop)
    assert 45.0 <= result["bpm"] <= 210.0
    assert result["bpm"] <= 160.0 + 1e-6
    if result["bpm"] < 160.0:
        assert result["normalization"] == "half_time"


def test_beat_regularity_uniform_is_high():
    times = [index * 0.5 for index in range(16)]
    assert beats_module.beat_regularity(times) > 0.95
    jittered = [index * 0.5 + (0.12 if index % 2 else -0.12) for index in range(16)]
    assert beats_module.beat_regularity(jittered) < beats_module.beat_regularity(times)


def test_detect_meter_finds_four_four_phase():
    # 4 拍一小节，第 0 拍低频重音强、和声也在第 0 拍变化
    low = [1.0 if index % 4 == 0 else 0.15 for index in range(32)]
    change = [1.0 if index % 4 == 0 else 0.1 for index in range(32)]
    meter = beats_module.detect_meter(low, change)
    assert meter["beats_per_bar"] == 4
    assert meter["phase"] == 0
    assert meter["time_signature"] == "4/4"
    assert meter["confidence"] > 0.3


def test_detect_meter_finds_three_four():
    low = [1.0 if index % 3 == 0 else 0.1 for index in range(30)]
    change = [0.9 if index % 3 == 0 else 0.1 for index in range(30)]
    meter = beats_module.detect_meter(low, change)
    assert meter["beats_per_bar"] == 3
    assert meter["time_signature"] == "3/4"


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #

def test_analyze_beats_click_track(tmp_path: Path):
    pytest.importorskip("librosa")
    source = write_click_track(tmp_path / "click120.wav", bpm=120.0, seconds=16.0)
    result = beats_module.analyze_beats(source)

    assert 20.0 <= result["bpm"] <= 320.0
    assert result["bpm"] == pytest.approx(120.0, abs=5.0)
    assert result["bpm_display"] == pytest.approx(120.0, abs=5.0)
    assert result["beats_per_bar"] == 4
    assert result["time_signature"] == "4/4"
    assert result["confidence"] > 0.3
    beat_times = result["beats"]
    assert len(beat_times) >= 20
    assert all(0.0 <= value <= 16.5 for value in beat_times)
    assert all(a < b for a, b in zip(beat_times, beat_times[1:]))
    assert result["downbeats"], "应给出 downbeats"
    assert result["first_downbeat_sec"] is not None
    assert len(result["beat_strength"]) == len(beat_times)


def test_analyze_beats_is_selectable_by_bpm(tmp_path: Path):
    pytest.importorskip("librosa")
    source = write_click_track(tmp_path / "click90.wav", bpm=90.0, seconds=18.0)
    result = beats_module.analyze_beats(source)
    assert result["bpm"] == pytest.approx(90.0, abs=6.0)
    assert result["bpm_display"] == pytest.approx(90.0, abs=6.0)


def test_analyze_beats_silence_degrades_with_warning(tmp_path: Path):
    pytest.importorskip("librosa")
    source = write_silence(tmp_path / "silence.wav")
    result = beats_module.analyze_beats(source)
    assert 20.0 <= result["bpm"] <= 320.0
    assert result["confidence"] <= 0.35
    assert result["beats"], "静音也要返回可用网格，不能返回空"
    assert result["warnings"], "降级必须写 warnings"


# --------------------------------------------------------------------------- #
# 拍号：3/4 与 6/8 的区分（合成曲回归）
# --------------------------------------------------------------------------- #

def test_fit_beat_grid_recovers_regular_tempo():
    """帧量化的拍点用整串拟合后应恢复真实速度，并给出很小的残差。"""
    np = pytest.importorskip("numpy")
    true_period = 60.0 / 138.0
    rng = np.random.default_rng(7)
    # 模拟 hop 量化：拍点被吸到 23ms 的网格上
    quantized = np.round((np.arange(64) * true_period) / 0.02322) * 0.02322
    jittered = [float(value) for value in quantized]

    grid = beats_module.fit_beat_grid(jittered)
    assert grid["bpm"] == pytest.approx(138.0, abs=0.6)
    assert grid["max_residual"] <= 0.02
    assert len(grid["times"]) == len(jittered)


def test_detect_meter_demotes_slow_six_eight_to_three_four():
    """3/4 曲目里 6 拍候选会因「抽样增强」占便宜，拍太慢时必须降级为 3/4。"""
    low = [1.0 if index % 6 == 1 else (0.5 if index % 6 == 4 else 0.2) for index in range(60)]
    change = [0.9 if index % 6 == 2 else (0.5 if index % 3 == 2 else 0.05) for index in range(60)]

    with_fast_beats = beats_module.detect_meter(low, change, beat_interval=0.63)
    assert with_fast_beats["time_signature"] == "3/4"
    assert with_fast_beats["beats_per_bar"] == 3
    assert with_fast_beats["demoted_from"] == 6
    assert "note" in with_fast_beats


def test_detect_meter_keeps_six_eight_for_fast_beats():
    """真 6/8：八分音符拍够快 + 存在半小节次强拍，应保留 6/8。"""
    low = [1.0 if index % 6 == 0 else (0.6 if index % 3 == 0 else 0.1) for index in range(72)]
    change = [0.95 if index % 6 == 0 else 0.05 for index in range(72)]
    meter = beats_module.detect_meter(low, change, beat_interval=0.22)
    assert meter["time_signature"] == "6/8"
    assert meter["beats_per_bar"] == 6
    assert "demoted_from" not in meter

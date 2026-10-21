"""人声旋律提取的离线单测（纯算法层不依赖 numpy/librosa）。

覆盖：Hz/MIDI 换算、中值滤波、八度跳变修正、音域约束、帧序列 -> 音符切分
（长音/断音/能量起音/碎片过滤）、节拍网格与量化、音域统计、以及 librosa
缺失时的安全降级。末尾附一个合成正弦波集成测试（有 librosa 时才真正跑推理）。

运行方式（pytest 或直接 python 均可）：
    .venv/Scripts/python.exe tests/test_melody.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aitsm.pipeline import melody as M  # noqa: E402


def _approx(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _has_librosa() -> bool:
    try:
        import librosa  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def _write_sine_wav(path: Path, frequency: float, seconds: float, sample_rate: int = 22050) -> None:
    """用标准库生成单音正弦 WAV，避免引入额外测试依赖。"""
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        samples = bytearray()
        for index in range(frames):
            value = int(0.3 * 32767 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            samples += struct.pack("<h", value)
        handle.writeframes(bytes(samples))


# ---------------------------------------------------------------------------
# 基础换算
# ---------------------------------------------------------------------------


def test_hz_to_midi_and_back() -> None:
    assert _approx(M.hz_to_midi(440.0), 69.0, 1e-6)
    assert _approx(M.midi_to_hz(69.0), 440.0, 1e-6)
    for midi in (48, 60, 64, 72, 88):
        assert _approx(M.hz_to_midi(M.midi_to_hz(midi)), float(midi), 1e-6)
    assert math.isnan(M.hz_to_midi(0.0))


# ---------------------------------------------------------------------------
# 平滑 / 修正 / 约束
# ---------------------------------------------------------------------------


def test_median_filter_removes_spike_and_keeps_holes() -> None:
    filtered = M.median_filter_values([60, 60, 60, 84, 60, 60, 60], kernel=5)
    assert filtered == [60, 60, 60, 60, 60, 60, 60]

    with_holes = M.median_filter_values([60.0, None, 62.0], kernel=3)
    assert with_holes[0] is not None
    assert with_holes[1] is None  # 检测空洞不得被填成有声
    assert with_holes[2] is not None


def test_fix_octave_jumps_corrects_false_octave() -> None:
    values = [60.0, 60.0, 60.0, 84.0, 84.0, 84.0, 62.0, 62.0]
    fixed_values, fixed_count = M.fix_octave_jumps(values)
    assert fixed_count >= 1
    assert _approx(fixed_values[3], 60.0)
    assert _approx(fixed_values[5], 60.0)


def test_fix_octave_jumps_keeps_real_melody() -> None:
    values = [60.0, 62.0, 64.0, 65.0, 67.0, 69.0, 71.0, 72.0]
    fixed_values, fixed_count = M.fix_octave_jumps(values)
    assert fixed_count == 0
    assert fixed_values == values


def test_constrain_pitch_range_shifts_or_drops() -> None:
    values = [40.0, 52.0, 100.0, 10.0, 60.0, None]
    constrained, shifted, dropped = M.constrain_pitch_range(values)
    assert constrained == [52.0, 52.0, 88.0, None, 60.0, None]
    assert shifted == 2
    assert dropped == 1


# ---------------------------------------------------------------------------
# 帧序列 -> 音符
# ---------------------------------------------------------------------------


def test_segment_notes_splits_on_pitch_change() -> None:
    times = [index * 0.01 for index in range(60)]
    midis: list[float | None] = [60.0] * 20 + [62.0] * 20 + [64.0] * 20
    notes = M.segment_notes(times, midis)
    assert [note["midi"] for note in notes] == [60, 62, 64]
    assert _approx(notes[0]["start"], 0.0, 1e-6)
    assert _approx(notes[1]["start"], 0.2, 1e-6)
    assert _approx(notes[2]["start"], 0.4, 1e-6)
    for note in notes:
        assert note["end"] - note["start"] >= M._MIN_NOTE_SECONDS - 1e-6


def test_segment_notes_keeps_sustain_as_single_note() -> None:
    times = [index * 0.01 for index in range(50)]
    notes = M.segment_notes(times, [60.0] * 50)
    assert len(notes) == 1
    assert notes[0]["midi"] == 60
    assert _approx(notes[0]["end"] - notes[0]["start"], 0.5, 1e-6)


def test_segment_notes_splits_only_with_energy_evidence() -> None:
    times = [index * 0.01 for index in range(40)]
    midis: list[float | None] = [60.0] * 40
    energies: list[float | None] = [0.1] * 20 + [0.5] * 20
    onsets = [0.2]

    with_energy = M.segment_notes(times, midis, energies=energies, onsets=onsets)
    assert len(with_energy) == 2
    assert _approx(with_energy[0]["end"], 0.2, 1e-6)

    # 没有能量证据（检测空洞/颤音）时保持长音，不得乱切
    without_energy = M.segment_notes(times, midis, onsets=onsets)
    assert len(without_energy) == 1


def test_segment_notes_drops_isolated_blip() -> None:
    times = [index * 0.01 for index in range(21)]
    midis: list[float | None] = [None] * 10 + [72.0] + [None] * 10
    assert M.segment_notes(times, midis) == []


def test_segment_notes_handles_mismatched_input() -> None:
    assert M.segment_notes([], []) == []
    assert M.segment_notes([0.0, 0.1], [60.0]) == []


def test_note_range_reports_empty_without_fabricating() -> None:
    assert M.note_range([]) == {"low_midi": 0, "high_midi": 0}
    notes = [{"midi": 60}, {"midi": 72}, {"midi": 64}]
    assert M.note_range(notes) == {"low_midi": 60, "high_midi": 72}


# ---------------------------------------------------------------------------
# 节拍网格 / 量化
# ---------------------------------------------------------------------------


def test_build_beat_grid_from_beats() -> None:
    grid, period = M.build_beat_grid({"beats": [0.0, 0.5, 1.0, 1.5], "bpm": 120}, subdivision=2)
    assert _approx(period, 0.5, 1e-6)
    assert _approx(grid[0], 0.0, 1e-6)
    assert any(_approx(point, 0.25, 1e-6) for point in grid)
    assert any(_approx(point, 0.75, 1e-6) for point in grid)
    assert grid == sorted(grid)


def test_build_beat_grid_from_bpm_only() -> None:
    grid, period = M.build_beat_grid({"bpm": 100}, subdivision=2)
    assert _approx(period, 0.6, 1e-6)
    assert any(_approx(point, 0.3, 1e-6) for point in grid)


def test_build_beat_grid_without_beats() -> None:
    grid, period = M.build_beat_grid(None)
    assert grid == []
    assert _approx(period, 0.5, 1e-6)


def test_quantize_notes_snaps_and_keeps_raw() -> None:
    beats = {"beats": [0.0, 0.5, 1.0, 1.5], "bpm": 120}
    notes = [{"start": 0.26, "end": 0.52, "midi": 60, "confidence": 0.8, "onset": 0.26}]
    quantized, stats = M.quantize_notes(notes, beats)
    assert _approx(quantized[0]["start"], 0.25, 1e-6)
    assert _approx(quantized[0]["end"], 0.5, 1e-6)
    assert _approx(quantized[0]["start_raw"], 0.26, 1e-6)
    assert stats["snapped"] == 1
    assert stats["grid_points"] > 0


def test_quantize_notes_without_grid_keeps_original_times() -> None:
    notes = [{"start": 0.26, "end": 0.52, "midi": 60, "confidence": 0.8}]
    quantized, stats = M.quantize_notes(notes, None)
    assert _approx(quantized[0]["start"], 0.26, 1e-6)
    assert stats["kept_raw"] == 1


def test_quantize_notes_repairs_overlap() -> None:
    beats = {"beats": [0.0, 0.5, 1.0, 1.5], "bpm": 120}
    # 20ms 内的两个碎片：并成一个音（同格冲突不再一路后移）
    notes = [
        {"start": 0.24, "end": 0.51, "midi": 60, "confidence": 0.8},
        {"start": 0.26, "end": 0.49, "midi": 62, "confidence": 0.8},
    ]
    quantized, stats = M.quantize_notes(notes, beats)
    assert len(quantized) == 1
    assert stats["adjusted_overlap"] == 1
    assert quantized[0]["end"] > quantized[0]["start"]
    # 正常相邻的两个音不合并，且不重叠
    separate = [
        {"start": 0.24, "end": 0.51, "midi": 60, "confidence": 0.8},
        {"start": 0.62, "end": 0.9, "midi": 62, "confidence": 0.8},
    ]
    quantized2, _ = M.quantize_notes(separate, beats)
    assert len(quantized2) == 2
    assert quantized2[1]["start"] >= quantized2[0]["end"] - 1e-6


# ---------------------------------------------------------------------------
# 降级与集成
# ---------------------------------------------------------------------------


def test_detect_onsets_from_envelope_picks_peaks() -> None:
    # 10ms 一帧、共 300 帧（3 秒）；在 0.5s/1.5s 处放明显尖峰
    times = [index * 0.01 for index in range(300)]
    envelope = [0.02] * 300
    envelope[50] = 0.9
    envelope[150] = 1.0
    onsets = M.detect_onsets_from_envelope(envelope, times)
    assert len(onsets) == 2, onsets
    assert _approx(onsets[0], 0.5, 0.02)
    assert _approx(onsets[1], 1.5, 0.02)


def test_detect_onsets_from_envelope_respects_wait_and_delta() -> None:
    times = [index * 0.01 for index in range(300)]
    envelope = [0.0] * 300
    # 两个峰只隔 20ms，wait=0.1s 时只保留第一个
    envelope[100] = 0.8
    envelope[102] = 0.7
    onsets = M.detect_onsets_from_envelope(envelope, times, wait=0.1)
    assert len(onsets) == 1, onsets
    assert _approx(onsets[0], 1.0, 0.02)
    # 平坦信号（无峰）不产生起音
    assert M.detect_onsets_from_envelope([0.0] * 300, times) == []
    # 过短输入安全返回空
    assert M.detect_onsets_from_envelope([1.0], [0.0]) == []


def test_resolve_probability_gate_adapts_to_engine_confidence() -> None:
    # 混音里 pyin 的 prob 普遍很低 -> 不按概率过滤（否则丢 97% 有声帧）
    low = [0.1, 0.2, 0.25, 0.05, None]
    assert M.resolve_probability_gate(low) is None
    # 干净人声 stem 概率高 -> 启用 0.15 门限
    assert M.resolve_probability_gate([0.8, 0.9, 0.7, 0.6]) == 0.15
    assert M.resolve_probability_gate([]) is None


def test_round_pitch_values_snaps_jitter() -> None:
    assert M.round_pitch_values([60.4, 59.7, None, 62.5]) == [60.0, 60.0, None, 62.0]
    assert M.round_pitch_values([]) == []


def test_absorb_ornament_notes_swallows_edge_jitter() -> None:
    # 主音 60(0-0.5) + 边缘毛刺 61(0.5-0.56) + 主音 60(0.56-1.0)
    notes = [
        {"start": 0.0, "end": 0.5, "midi": 60, "confidence": 0.9, "onset": 0.0},
        {"start": 0.5, "end": 0.56, "midi": 61, "confidence": 0.8, "onset": 0.5},
        {"start": 0.56, "end": 1.0, "midi": 60, "confidence": 0.9, "onset": 0.56},
    ]
    absorbed = M.absorb_ornament_notes(notes)
    assert len(absorbed) == 2, absorbed
    merged = M._merge_adjacent_same_pitch(absorbed, max_gap=0.02)
    assert len(merged) == 1, merged
    assert merged[0]["start"] == 0.0 and merged[0]["end"] == 1.0


def test_absorb_ornament_notes_keeps_real_short_notes() -> None:
    # 与邻居相差 >1 个半音的短音是真装饰音，不能吞
    notes = [
        {"start": 0.0, "end": 0.5, "midi": 60, "confidence": 0.9, "onset": 0.0},
        {"start": 0.5, "end": 0.56, "midi": 67, "confidence": 0.9, "onset": 0.5},
        {"start": 0.56, "end": 1.0, "midi": 60, "confidence": 0.9, "onset": 0.56},
    ]
    absorbed = M.absorb_ornament_notes(notes)
    assert len(absorbed) == 3, absorbed
    assert [note["midi"] for note in absorbed] == [60, 67, 60]


def test_transcribe_melody_missing_file_returns_empty_with_warning() -> None:
    result = M.transcribe_melody(str(ROOT / "no-such-file.wav"), {"beats": []})
    assert result["notes"] == []
    assert result["range"] == {"low_midi": 0, "high_midi": 0}
    assert result["warnings"]


def test_transcribe_melody_warns_when_input_is_not_vocal_stem() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mix.wav"
        _write_sine_wav(path, 440.0, 0.6)
        result = M.transcribe_melody(str(path), {"beats": [0.0, 0.5, 1.0], "bpm": 120}, duration=0.6)
        assert any("人声" in warning for warning in result["warnings"])


def test_transcribe_melody_on_synthetic_vocal_sequence() -> None:
    """合成已知音高序列（颤音 + 谐波 + 15dB 噪声）验证 pyin→音符链路准确率。"""
    if not _has_librosa():
        print("  (skip) librosa/numpy 未安装，跳过合成人声集成测试")
        return
    import wave

    import numpy as np

    expected = [60, 62, 64, 67, 69, 67, 64, 62, 60, 65, 69, 72, 71, 69, 67, 65]
    sr, note_dur, gap = 22050, 0.45, 0.05
    total = len(expected) * (note_dur + gap)
    times = np.arange(int(sr * total)) / sr
    signal = np.zeros_like(times)
    for index, midi in enumerate(expected):
        freq = M.midi_to_hz(midi)
        left = int(index * (note_dur + gap) * sr)
        right = int((index * (note_dur + gap) + note_dur) * sr)
        local = times[left:right] - times[left]
        envelope = np.minimum(1.0, local / 0.03) * np.minimum(1.0, (note_dur - local) / 0.06)
        # 真正的颤音必须是相位积分，否则相位调制会产生音高漂移
        depth, rate = 0.008, 5.5
        phase = 2 * np.pi * freq * (
            local + depth * (1 - np.cos(2 * np.pi * rate * local)) / (2 * np.pi * rate)
        )
        harmonics = sum(amp * np.sin(order * phase) for order, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
        signal[left:right] += 0.25 * envelope * harmonics
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(signal.shape)
    noise *= np.sqrt(np.mean(signal**2) / (10 ** (15 / 10)) / np.mean(noise**2))
    signal = signal + noise

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocals_sequence.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
        result = M.transcribe_melody(str(path), {"bpm": 120.0, "beats_per_bar": 4}, duration=total)

    notes = result["notes"]
    assert len(notes) == len(expected), [note["midi"] for note in notes]
    hits = 0
    for note, (index, midi) in zip(notes, enumerate(expected)):
        start = index * (note_dur + gap)
        if abs(note["start"] - start) <= 0.15 and note["midi"] == midi:
            hits += 1
    assert hits >= len(expected) - 1, f"合成旋律命中 {hits}/{len(expected)}：{[(round(n['start'], 2), n['midi']) for n in notes]}"
    assert all(note["onset"] is not None for note in notes)


def test_transcribe_melody_on_synthetic_sine() -> None:
    if not _has_librosa():
        print("  (skip) librosa/numpy 未安装，跳过合成音频集成测试")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocals_sine.wav"
        _write_sine_wav(path, 440.0, 1.0)
        result = M.transcribe_melody(
            str(path), {"beats": [index * 0.5 for index in range(4)], "bpm": 120}, duration=1.0
        )
        assert isinstance(result["notes"], list)
        assert result["notes"], f"应识别到旋律：{result['warnings']}"
        pitches = [note["midi"] for note in result["notes"]]
        assert all(66 <= pitch <= 72 for pitch in pitches), pitches  # 440Hz ≈ A4


def test_fold_notes_to_range_moves_octave_outliers() -> None:
    notes = [
        {"start": 0.0, "end": 1.0, "midi": 50, "confidence": 0.5, "onset": 0.0},
        {"start": 1.0, "end": 2.0, "midi": 60, "confidence": 0.5, "onset": 1.0},
        {"start": 2.0, "end": 3.0, "midi": 88, "confidence": 0.5, "onset": 2.0},
    ]
    folded, shifted = M.fold_notes_to_range(notes, low=55, high=84)
    assert shifted == 2
    # 50 与邻居 60 连续 → +12 得 62；88 与邻居 62/60 连续 → -24 得 64（不是只看最近八度）
    assert [note["midi"] for note in folded] == [62, 60, 64]
    assert folded[0]["raw_midi"] == 50 and folded[2]["raw_midi"] == 88
    assert "raw_midi" not in folded[1]
    # 都不丢音符
    assert len(folded) == len(notes)


def test_estimate_note_confidence_prefers_stable_and_loud() -> None:
    stable = M.estimate_note_confidence(probability=0.2, stability=0.95, energy_ratio=1.2)
    shaky = M.estimate_note_confidence(probability=0.1, stability=0.2, energy_ratio=0.2)
    empty = M.estimate_note_confidence()
    assert 0.0 <= empty <= 1.0
    assert stable > 0.75, stable
    assert shaky < stable - 0.3, (shaky, stable)
    assert empty < 0.3


def test_refine_note_confidences_uses_frame_evidence() -> None:
    times = [index * 0.01 for index in range(20)]
    midis = [60.0] * 10 + [64.0] * 10
    energies = [0.2] * 20
    probabilities = [0.3] * 20
    notes = [
        {"start": 0.0, "end": 0.1, "midi": 60, "confidence": 0.0, "onset": 0.0},
        {"start": 0.1, "end": 0.2, "midi": 64, "confidence": 0.0, "onset": 0.1},
    ]
    refined = M.refine_note_confidences(
        notes, times, midis, probabilities=probabilities, energies=energies
    )
    assert all(0.5 <= note["confidence"] <= 1.0 for note in refined), [
        note["confidence"] for note in refined
    ]
    # 音高与帧证据不符的音符置信度必须明显更低
    wrong = M.refine_note_confidences(
        [{"start": 0.0, "end": 0.1, "midi": 72, "confidence": 0.0, "onset": 0.0}],
        times,
        midis,
        probabilities=probabilities,
        energies=energies,
    )
    assert wrong[0]["confidence"] < refined[0]["confidence"] - 0.2


def test_melody_output_passes_schema() -> None:
    try:
        import jsonschema
    except Exception:
        print("  (skip) jsonschema 未安装，跳过 schema 校验")
        return
    schema = json.loads((ROOT / "schemas" / "analysis.schema.json").read_text(encoding="utf-8"))
    melody_schema = schema["properties"]["melody"]
    times = [index * 0.01 for index in range(30)]
    notes = M.segment_notes(times, [60.0] * 10 + [64.0] * 20)
    payload = {"notes": notes, "range": M.note_range(notes), "warnings": ["test"]}
    jsonschema.validate(payload, melody_schema)


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

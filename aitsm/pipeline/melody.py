"""人声旋律转谱（F10）：f0 提取 + 纯算法音符切分/量化/音域约束。

分层设计：
- 纯算法层（可离线单测，不依赖 numpy/librosa）：midi/Hz 换算、中值滤波、八度跳变修正、
  音域约束、帧序列→音符切分、onsets 校正、节拍网格构建与量化。
- 模型层：librosa.pyin（缺失/失败时降级 piptrack；librosa 也缺失时返回空 notes + warning）。

公共契约::

    transcribe_melody(vocals_wav, beats, *, duration=None)
        -> {"notes": [{"start","end","midi","confidence","onset","lyric"}],
            "range": {"low_midi": int, "high_midi": int},
            "warnings": [str]}

保真纪律（参考 To-Sheet-Music-Skill/references/transcription-fidelity.md）：
- 检测空洞（f0 未定/低能量）不等于休止：只在有能量证据时才按 onset 切分，宁可保留长音；
- 不为了“好看”删除超音域或密集同音：超音域先按八度归位，归不进去的帧计数并告警，不静默编造；
- 所有结果带 confidence，低置信度交给上层质量面板标注。
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from bisect import bisect_left
from typing import Any, Callable, Iterable, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = [
    "transcribe_melody",
    "align_lyrics_to_melody",
    "hz_to_midi",
    "midi_to_hz",
    "median_filter_values",
    "fix_octave_jumps",
    "constrain_pitch_range",
    "round_pitch_values",
    "segment_notes",
    "absorb_ornament_notes",
    "build_beat_grid",
    "snap_to_grid",
    "quantize_notes",
    "note_range",
    "trim_note_overlaps",
    "fold_notes_to_range",
    "estimate_note_confidence",
    "refine_note_confidences",
    "DEFAULT_LOW_MIDI",
    "DEFAULT_HIGH_MIDI",
    "DEFAULT_VOCAL_LOW_MIDI",
    "DEFAULT_VOCAL_HIGH_MIDI",
]

# 人声音域约束：C3(48) ~ E6(88)。超出范围先按八度归位，归不进去才丢弃。
DEFAULT_LOW_MIDI = 48
DEFAULT_HIGH_MIDI = 88
# 主旋律的“合理”音域：G3(55) ~ C6(84)。女声主旋律不应低到 D3；
# 超出这个范围的音符按八度归位（男声低音歌可用 AITSM_MELODY_LOW_MIDI=48 放开）。
DEFAULT_VOCAL_LOW_MIDI = 55
DEFAULT_VOCAL_HIGH_MIDI = 84
DEFAULT_FMIN = 130.81  # C3
DEFAULT_FMAX = 1046.5  # C6
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_LENGTH = 256
DEFAULT_FRAME_LENGTH = 2048
_MIN_NOTE_SECONDS = 0.09
_MAX_GAP_SECONDS = 0.06
_PITCH_TOLERANCE = 0.8  # 半音；同音符内允许的基频抖动


# ---------------------------------------------------------------------------
# 基础换算与统计
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def hz_to_midi(frequency: float) -> float:
    """频率 -> 连续 MIDI 音高（440Hz = A4 = 69）。"""
    if frequency is None or frequency <= 0:
        return float("nan")
    return 69.0 + 12.0 * math.log2(float(frequency) / 440.0)


def midi_to_hz(midi: float) -> float:
    """连续 MIDI 音高 -> 频率。"""
    return 440.0 * 2.0 ** ((float(midi) - 69.0) / 12.0)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def median_filter_values(values: Sequence[float | None], kernel: int = 5) -> list[float | None]:
    """对 f0/MIDI 序列做中值滤波。

    只平滑“有声帧”；None（未定/静音）保持 None，避免把检测空洞填成有声。
    """
    if kernel < 3:
        return list(values)
    radius = kernel // 2
    result: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        window = [
            other
            for other in values[max(0, index - radius) : index + radius + 1]
            if other is not None
        ]
        result.append(_median(window) if window else value)
    return result


def fix_octave_jumps(
    midi_values: Sequence[float | None], *, window: int = 7, jump: float = 10.0
) -> tuple[list[float | None], int]:
    """修正 pyin 常见的八度误判：相对最近有声帧中值跳 >10 半音时尝试移八度。

    只在“移八度后明显更接近邻域中值”时生效；真实的大跳旋律保留原样。
    返回 (新序列, 修正帧数)。
    """
    result: list[float | None] = list(midi_values)
    history: list[float] = []
    fixed = 0
    for index, value in enumerate(midi_values):
        if value is None:
            continue
        current = value
        if len(history) >= 3:
            reference = _median(history[-window:])
            if abs(current - reference) > jump:
                best = current
                best_distance = abs(current - reference)
                for shift in (-12.0, 12.0, -24.0, 24.0):
                    candidate = current + shift
                    distance = abs(candidate - reference)
                    if distance < best_distance - 1.0:
                        best = candidate
                        best_distance = distance
                if best != current and abs(best - reference) <= jump:
                    current = best
                    fixed += 1
        result[index] = current
        history.append(current)
    return result, fixed


def constrain_pitch_range(
    midi_values: Sequence[float | None],
    *,
    low: int = DEFAULT_LOW_MIDI,
    high: int = DEFAULT_HIGH_MIDI,
) -> tuple[list[float | None], int, int]:
    """把音高约束到人声音域：先按八度归位，归不进去的帧记为丢弃。

    返回 (新序列, 八度归位帧数, 丢弃帧数)。
    """
    result: list[float | None] = []
    shifted = 0
    dropped = 0
    for value in midi_values:
        if value is None:
            result.append(None)
            continue
        if low <= value <= high:
            result.append(value)
            continue
        target: float | None = None
        for shift in (0.0, -12.0, 12.0, -24.0, 24.0, -36.0, 36.0):
            candidate = value + shift
            if low <= candidate <= high:
                target = candidate
                break
        if target is None:
            result.append(None)
            dropped += 1
        else:
            result.append(target)
            shifted += 1
    return result, shifted, dropped


def round_pitch_values(midi_values: Sequence[float | None]) -> list[float | None]:
    """把连续帧音高吸附到半音刻度（None 保持 None）。

    人声演唱有 ±0.5 半音的自然抖动和滑音；不先吸附，切分时会把一个长音
    切成「60 / 59 / 61」这样的一串碎片（实测合成人声 16 个音被切成 48 个）。
    """
    return [None if value is None else float(round(float(value))) for value in midi_values]


# ---------------------------------------------------------------------------
# 帧序列 -> 音符
# ---------------------------------------------------------------------------


def _frame_mean(values: Sequence[float | None], times: Sequence[float], start: float, end: float) -> float | None:
    """取 [start, end] 时间窗内有效帧的均值。"""
    if not values:
        return None
    left = bisect_left(times, start)
    right = bisect_left(times, end)
    window = [value for value in values[left:right] if value is not None]
    if not window:
        return None
    return sum(window) / len(window)


def _nearest_onset(time_value: float, onsets: Sequence[float], tolerance: float = 0.09) -> float:
    """在 ±tolerance 内找最近 onset；找不到返回原时间。"""
    if not onsets:
        return time_value
    position = bisect_left(onsets, time_value)
    candidates: list[float] = []
    if position < len(onsets):
        candidates.append(onsets[position])
    if position > 0:
        candidates.append(onsets[position - 1])
    if not candidates:
        return time_value
    best = min(candidates, key=lambda onset: abs(onset - time_value))
    return best if abs(best - time_value) <= tolerance else time_value


def detect_onsets_from_envelope(
    envelope: Sequence[float],
    times: Sequence[float],
    *,
    delta: float = 0.05,
    wait: float = 0.05,
    pre_max: float = 0.03,
    post_max: float = 0.03,
    pre_avg: float = 0.10,
    post_avg: float = 0.10,
) -> list[float]:
    """在起音强度包络上挑峰，返回起音时间（秒，绝对时间轴）。

    为什么不用 ``librosa.onset.onset_detect``：librosa 1.0.0 的 onset_detect 依赖
    numba 的 ``util.peak_pick``，在本项目的 numpy 2.5 环境下会直接段错误（整个进程
    崩溃，Python 层 ``except`` 捕获不到）。这里用纯 Python 复刻同样的 peak_pick 语义：
    局部极大 + 高出局部均值 ``delta`` + 与上一个起音至少间隔 ``wait`` 秒。
    """
    if not envelope or not times:
        return []
    count = int(min(len(envelope), len(times)))
    if count < 3:
        return []
    values = [float(value) for value in envelope[:count]]
    time_values = [float(value) for value in times[:count]]
    span = time_values[-1] - time_values[0]
    frame_dt = span / (count - 1) if span > 0 else 1.0
    if frame_dt <= 0:
        frame_dt = 1.0

    def frames(seconds: float, minimum: int) -> int:
        return max(minimum, int(round(seconds / frame_dt)))

    pre_max_frames = frames(pre_max, 1)
    post_max_frames = frames(post_max, 0)
    pre_avg_frames = frames(pre_avg, 1)
    post_avg_frames = frames(post_avg, 0)
    wait_frames = frames(wait, 1)
    threshold = max(float(delta), 1e-6)

    onsets: list[float] = []
    last_peak_index = -(10**9)
    for index, value in enumerate(values):
        if value <= 0:
            continue
        max_left = max(0, index - pre_max_frames)
        max_right = min(count, index + post_max_frames + 1)
        if value < max(values[max_left:max_right]):
            continue
        avg_left = max(0, index - pre_avg_frames)
        avg_right = min(count, index + post_avg_frames + 1)
        window = values[avg_left:avg_right]
        local_mean = sum(window) / len(window) if window else 0.0
        if value < local_mean + threshold:
            continue
        if index - last_peak_index < wait_frames:
            continue
        onsets.append(time_values[index])
        last_peak_index = index
    return onsets



def _segments_to_notes(
    times: Sequence[float],
    midis: Sequence[float | None],
    confidences: Sequence[float | None] | None,
    energies: Sequence[float | None] | None,
    onsets: Sequence[float],
    *,
    frame_dt: float,
    pitch_tolerance: float,
) -> list[dict]:
    """把连续有声帧切成音符：先找有声连续段，再按音高变化细分。"""
    frame_count = len(times)
    notes: list[dict] = []
    index = 0
    while index < frame_count:
        if midis[index] is None:
            index += 1
            continue
        segment: list[int] = [index]
        cursor = index + 1
        last_voiced = index
        while cursor < frame_count:
            if midis[cursor] is None:
                probe = cursor
                while (
                    probe < frame_count
                    and midis[probe] is None
                    and (probe - last_voiced) * frame_dt <= _MAX_GAP_SECONDS
                ):
                    probe += 1
                if probe >= frame_count or midis[probe] is None:
                    break
                reference = _median([midis[item] for item in segment[-5:]])  # type: ignore[list-item]
                if abs(midis[probe] - reference) > pitch_tolerance:  # type: ignore[operator]
                    break
                cursor = probe
                continue
            reference = _median([midis[item] for item in segment[-5:]])  # type: ignore[list-item]
            if abs(midis[cursor] - reference) > pitch_tolerance:  # type: ignore[operator]
                break
            segment.append(cursor)
            last_voiced = cursor
            cursor += 1
        notes.extend(
            _split_pitch_segment(
                times,
                midis,
                confidences,
                onsets,
                segment,
                frame_dt=frame_dt,
                pitch_tolerance=pitch_tolerance,
            )
        )
        index = last_voiced + 1
    return notes


def _split_pitch_segment(
    times: Sequence[float],
    midis: Sequence[float | None],
    confidences: Sequence[float | None] | None,
    onsets: Sequence[float],
    segment: Sequence[int],
    *,
    frame_dt: float,
    pitch_tolerance: float,
) -> list[dict]:
    """在一个有声段内按音高跳变继续切分，并产出音符 dict。"""
    if not segment:
        return []
    chunks: list[list[int]] = [[segment[0]]]
    for frame in segment[1:]:
        reference = _median([midis[item] for item in chunks[-1][-5:]])  # type: ignore[list-item]
        if abs(midis[frame] - reference) > pitch_tolerance:  # type: ignore[operator]
            chunks.append([frame])
        else:
            chunks[-1].append(frame)

    notes: list[dict] = []
    for chunk in chunks:
        pitch_values = [midis[item] for item in chunk if midis[item] is not None]
        if not pitch_values:
            continue
        pitch = int(round(_median(pitch_values)))  # type: ignore[arg-type]
        raw_start = float(times[chunk[0]])
        raw_end = float(times[chunk[-1]]) + frame_dt
        onset = _nearest_onset(raw_start, onsets, tolerance=0.09)
        if confidences is not None:
            conf_values = [confidences[item] for item in chunk if confidences[item] is not None]
            confidence = float(sum(conf_values) / len(conf_values)) if conf_values else 0.5
        else:
            confidence = 0.5
        notes.append(
            {
                "start": round(onset, 3),
                "end": round(max(raw_end, onset + frame_dt), 3),
                "midi": int(_clamp(pitch, 12, 108)),
                "confidence": round(_clamp(confidence, 0.0, 1.0), 3),
                "onset": round(onset, 3),
            }
        )
    return notes


def _env_midi(name: str, default: int) -> int:
    """读取 MIDI 音域配置（环境变量优先，非法值回落到默认）。"""
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if 12 <= value <= 108 else default


def vocal_range_limits() -> tuple[int, int]:
    """主旋律合理音域（默认 G3-C6 = 55-84，可用环境变量覆盖）。"""
    return (
        _env_midi("AITSM_MELODY_LOW_MIDI", DEFAULT_VOCAL_LOW_MIDI),
        _env_midi("AITSM_MELODY_HIGH_MIDI", DEFAULT_VOCAL_HIGH_MIDI),
    )


def fold_notes_to_range(
    notes: Sequence[dict],
    *,
    low: int | None = None,
    high: int | None = None,
    max_interval: int = 9,
) -> tuple[list[dict], int]:
    """把超出合理音域的音符按八度归位（不丢音符），用局部旋律连续性选八度。

    pyin 常把整句唱低的旋律识别成低八度（女声主旋律落到 D3），这类音符整体
    偏 12 的整数倍，按八度搬回音域内是安全操作。选哪个八度看上下文：
    优先选“音符跨度不超过大六度（``max_interval``）”且与前后音最接近的候选，
    没有候选满足时才退回音域内最近八度。原始音高保留在 ``raw_midi``。
    """
    limit_low, limit_high = vocal_range_limits()
    low = limit_low if low is None else int(low)
    high = limit_high if high is None else int(high)

    items = [dict(note) for note in notes]
    original = [int(note.get("midi") or 0) for note in items]
    shifted = 0
    for index, note in enumerate(items):
        midi = original[index]
        if low <= midi <= high:
            continue
        neighbor_values: list[int] = []
        for offset in (-2, -1, 1, 2):
            item = index + offset
            if not 0 <= item < len(items):
                continue
            if item < index:
                neighbor_values.append(int(items[item].get("midi") or 0))  # 已归位
            elif low <= original[item] <= high:
                neighbor_values.append(original[item])  # 未归位且本来就合理
        reference = _median(neighbor_values) if neighbor_values else None
        candidates = [midi + 12 * step for step in (-3, -2, -1, 1, 2, 3)]
        inside = [candidate for candidate in candidates if low <= candidate <= high]
        if not inside:
            continue
        if reference is not None:
            continuous = [
                candidate for candidate in inside if abs(candidate - reference) <= max_interval
            ]
            if continuous:
                inside = continuous
                best = min(inside, key=lambda candidate: abs(candidate - reference))
            else:
                best = min(inside, key=lambda candidate: abs(candidate - midi))
        else:
            best = min(inside, key=lambda candidate: abs(candidate - midi))
        note["raw_midi"] = midi
        note["midi"] = best
        shifted += 1
    return items, shifted


def estimate_note_confidence(
    *,
    probability: float | None = None,
    stability: float = 0.0,
    energy_ratio: float = 0.0,
) -> float:
    """综合音符置信度（0..1）：引擎概率 + 音高稳定度 + 相对能量。

    pyin 的 ``voiced_prob`` 在真实人声上普遍只有 0.1-0.3，单用它会把所有
    音符都标成“不可信”。这里把三项加权成更可用的质量分：
    - stability：音符时段内落在该音高 ±0.6 半音内的有声帧占比；
    - energy_ratio：该音符能量相对人声段 80 分位能量（>1 视为饱满）；
    - probability：引擎给出的有声概率（权重最小）。
    """
    prob = 0.0 if probability is None else _clamp(float(probability), 0.0, 1.0)
    stable = _clamp(float(stability), 0.0, 1.0)
    energy = _clamp(float(energy_ratio), 0.0, 1.0)
    return round(_clamp(0.15 + 0.45 * stable + 0.30 * energy + 0.10 * min(1.0, prob * 2.0), 0.0, 1.0), 3)


def refine_note_confidences(
    notes: Sequence[dict],
    times: Sequence[float],
    midis: Sequence[float | None],
    *,
    probabilities: Sequence[float | None] | None = None,
    energies: Sequence[float | None] | None = None,
    tolerance: float = 0.6,
) -> list[dict]:
    """用逐帧证据重算每个音符的 confidence（见 ``estimate_note_confidence``）。"""
    voiced = [index for index, value in enumerate(midis) if value is not None]
    reference_energy = 0.0
    if energies is not None and voiced:
        voiced_energies = sorted(
            float(energies[index])
            for index in voiced
            if index < len(energies) and energies[index] is not None
        )
        if voiced_energies:
            reference_energy = voiced_energies[int(0.8 * (len(voiced_energies) - 1))]

    refined: list[dict] = []
    for note in notes:
        note = dict(note)
        start = float(note.get("start") or 0.0)
        end = float(note.get("end") or start)
        pitch = float(note.get("midi") or 0.0)
        frames = [
            index
            for index in voiced
            if index < len(times) and start - 1e-6 <= float(times[index]) <= end + 1e-6
        ]
        if frames:
            near = sum(1 for index in frames if abs(float(midis[index]) - pitch) <= tolerance)
            stability = near / len(frames)
        else:
            stability = 0.0
        energy_ratio = 1.0
        if reference_energy > 0.0 and energies is not None:
            values = [
                float(energies[index])
                for index in frames
                if index < len(energies) and energies[index] is not None
            ]
            if values:
                energy_ratio = (sum(values) / len(values)) / reference_energy
        probability = None
        if probabilities is not None:
            values = [
                float(probabilities[index])
                for index in frames
                if index < len(probabilities) and probabilities[index] is not None
            ]
            if values:
                probability = sum(values) / len(values)
        note["confidence"] = estimate_note_confidence(
            probability=probability, stability=stability, energy_ratio=energy_ratio
        )
        refined.append(note)
    return refined


def trim_note_overlaps(notes: Sequence[dict], *, min_duration: float = 0.04) -> list[dict]:
    """用下一个音符的起音裁掉前一个音符的尾巴，保证音符不重叠。

    onset 校正会把起音往前挪（最多 0.09s），因此相邻音符的 ``start`` 常常
    早于前一个音符的 ``end``。若放任不管，量化时会一路往后推、节奏整体漂移；
    这里先按“后一个起音”收尾，量化只需要处理网格吸附本身。
    """
    result: list[dict] = []
    for note in notes:
        current = dict(note)
        if result:
            previous = result[-1]
            if current["start"] < previous["end"]:
                previous["end"] = round(
                    max(previous["start"] + min_duration, current["start"]), 3
                )
        result.append(current)
    return result


def _merge_adjacent_same_pitch(notes: list[dict], *, max_gap: float = 0.08) -> list[dict]:
    """合并被基频抖动切开的同音高碎片（有声间隙很小才算同一个音）。"""
    merged: list[dict] = []
    for note in notes:
        if merged:
            previous = merged[-1]
            gap = note["start"] - previous["end"]
            if note["midi"] == previous["midi"] and gap <= max_gap:
                previous["end"] = max(previous["end"], note["end"])
                previous["confidence"] = round(
                    min(previous["confidence"], note["confidence"]), 3
                )
                continue
        merged.append(dict(note))
    return merged


def _split_on_onsets(
    notes: list[dict],
    *,
    times: Sequence[float],
    energies: Sequence[float | None] | None,
    onsets: Sequence[float],
) -> list[dict]:
    """只在有明显能量再起（能量先降后升）时才按 onset 切开同音高长音。

    没有能量证据时不动——避免把检测空洞/颤音当成重新起音。
    """
    if not onsets:
        return notes
    result: list[dict] = []
    for note in notes:
        inner = [onset for onset in onsets if note["start"] + 0.10 < onset < note["end"] - 0.04]
        if not inner or energies is None:
            result.append(note)
            continue
        cursor = dict(note)
        for onset in inner:
            before = _frame_mean(energies, times, onset - 0.12, onset - 0.02)
            after = _frame_mean(energies, times, onset, onset + 0.08)
            if before is None or after is None or before <= 0:
                continue
            # energy 先降后升（>1.5 倍）才认定为重新起音；否则保持长音
            if after < before * 1.5:
                continue
            piece = dict(cursor)
            piece["end"] = round(onset, 3)
            if piece["end"] - piece["start"] <= 0.05:
                continue
            result.append(piece)
            cursor = {
                "start": round(onset, 3),
                "end": round(note["end"], 3),
                "midi": note["midi"],
                "confidence": round(note["confidence"] * 0.9, 3),
                "onset": round(onset, 3),
            }
        if cursor["end"] - cursor["start"] > 0.05:
            result.append(cursor)
    return result


def segment_notes(
    times: Sequence[float],
    midi_values: Sequence[float | None],
    confidences: Sequence[float | None] | None = None,
    energies: Sequence[float | None] | None = None,
    onsets: Sequence[float] | None = None,
    *,
    min_duration: float = _MIN_NOTE_SECONDS,
    pitch_tolerance: float = _PITCH_TOLERANCE,
) -> list[dict]:
    """纯算法：帧级 f0（MIDI 序列）-> 音符列表。

    参数:
        times: 每帧时间（秒，递增）
        midi_values: 每帧 MIDI（float，None 表示未定/静音）
        confidences: 每帧置信度（如 pyin voiced_prob），可选
        energies: 每帧能量（RMS），用于判断重新起音，可选
        onsets: 起音时刻列表（秒），可选
    返回:
        [{"start","end","midi","confidence","onset"}]，按时间排序。
    """
    if not times or not midi_values or len(times) != len(midi_values):
        return []
    times = [float(value) for value in times]
    if len(times) < 2:
        frame_dt = 0.02
    else:
        deltas = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
        frame_dt = _median(deltas) if deltas else 0.02
    onset_list = sorted(float(item) for item in (onsets or []))

    notes = _segments_to_notes(
        times,
        midi_values,
        confidences,
        energies,
        onset_list,
        frame_dt=frame_dt,
        pitch_tolerance=pitch_tolerance,
    )
    notes = _merge_adjacent_same_pitch(notes, max_gap=max(frame_dt * 2, 0.05))
    notes = _split_on_onsets(notes, times=times, energies=energies, onsets=onset_list)

    filtered: list[dict] = []
    for note in notes:
        duration = note["end"] - note["start"]
        if duration < min_duration:
            # 很短的碎片只有在高置信且接近最短时长时才保留，避免颤音/滑音碎片污染谱面
            if duration >= max(0.04, min_duration * 0.6) and note["confidence"] >= 0.85:
                note = dict(note)
                note["end"] = round(note["start"] + min_duration, 3)
            else:
                continue
        filtered.append(note)
    filtered.sort(key=lambda item: item["start"])
    return filtered


def absorb_ornament_notes(
    notes: Sequence[dict],
    *,
    max_duration: float = 0.12,
    max_pitch_diff: int = 1,
    max_gap: float = 0.08,
) -> list[dict]:
    """把起音/收音抖动产生的短碎片并回相邻主音（纯算法，可离线测试）。

    规则：碎片时长 < max_duration、与相邻音相差 <= max_pitch_diff 个半音、
    间隔 <= max_gap 时并入相邻音（优先并入更长的邻居）；否则原样保留。
    只吞边缘毛刺，不删除真正的短促装饰音（真正的快音有更长的邻居约束不住）。
    """
    items = [dict(note) for note in notes]
    result: list[dict] = []
    index = 0
    while index < len(items):
        note = items[index]
        duration = float(note["end"]) - float(note["start"])
        if duration >= max_duration or len(items) == 1:
            result.append(note)
            index += 1
            continue
        candidates: list[tuple[str, dict, float]] = []
        previous = result[-1] if result else None
        if (
            previous is not None
            and abs(int(previous["midi"]) - int(note["midi"])) <= max_pitch_diff
            and float(note["start"]) - float(previous["end"]) <= max_gap
        ):
            candidates.append(("previous", previous, float(previous["end"]) - float(previous["start"])))
        following = items[index + 1] if index + 1 < len(items) else None
        if (
            following is not None
            and abs(int(following["midi"]) - int(note["midi"])) <= max_pitch_diff
            and float(following["start"]) - float(note["end"]) <= max_gap
        ):
            candidates.append(("following", following, float(following["end"]) - float(following["start"])))
        if not candidates:
            result.append(note)
            index += 1
            continue
        kind, target, _ = max(candidates, key=lambda item: item[2])
        if kind == "previous":
            merged = dict(target)
            merged["end"] = round(max(float(merged["end"]), float(note["end"])), 3)
            result[-1] = merged
        else:
            merged = dict(target)
            merged["start"] = round(min(float(merged["start"]), float(note["start"])), 3)
            merged["onset"] = merged.get("onset", merged["start"])
            items[index + 1] = merged
        index += 1
    return result


def note_range(notes: Sequence[dict]) -> dict:
    """音符音域；没有音符时返回 0（禁止编造音域）。"""
    pitches = [int(note["midi"]) for note in notes if isinstance(note.get("midi"), (int, float))]
    if not pitches:
        return {"low_midi": 0, "high_midi": 0}
    return {"low_midi": int(min(pitches)), "high_midi": int(max(pitches))}


# ---------------------------------------------------------------------------
# 节拍网格与量化
# ---------------------------------------------------------------------------


def build_beat_grid(
    beats: dict | None,
    *,
    subdivision: int = 2,
    until: float | None = None,
) -> tuple[list[float], float]:
    """由 beats dict 构造细分节拍网格（默认八分音符网格）。

    返回 (grid_times, beat_period)。beats 不可用时返回 ([], 推算的 beat 周期)。
    """
    subdivision = max(1, int(subdivision))
    if not isinstance(beats, dict):
        return [], 0.5
    raw = sorted(
        float(value)
        for value in (beats.get("beats") or [])
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0
    )
    bpm = _as_float(beats.get("bpm"))
    if len(raw) >= 2:
        gaps = [later - earlier for earlier, later in zip(raw, raw[1:]) if later - earlier > 1e-4]
        period = _median(gaps) if gaps else (60.0 / bpm if bpm and bpm > 0 else 0.5)
    elif bpm and bpm > 0:
        period = 60.0 / bpm
        if not raw:
            raw = [0.0]
    else:
        return [], 0.5
    period = max(period, 0.05)
    step = period / subdivision

    grid: list[float] = []
    if len(raw) >= 2:
        for earlier, later in zip(raw, raw[1:]):
            span = later - earlier
            divisions = max(1, int(round(span / step)))
            for offset in range(divisions):
                grid.append(earlier + offset * span / divisions)
        grid.append(raw[-1])
    else:
        grid.append(raw[0])

    # 向后补齐（量化常需要覆盖最后一个 beat 之后）
    horizon = until if until is not None and until > 0 else raw[-1] + period * 2
    cursor = (raw[-1] if raw else 0.0) + step
    while cursor <= horizon + 1e-9:
        grid.append(cursor)
        cursor += step
    # 向前补齐到 0（弱起音符也要有网格）
    cursor = (raw[0] if raw else 0.0) - step
    while cursor >= -1e-9:
        grid.append(cursor)
        cursor -= step

    unique = sorted({round(item, 6) for item in grid if item >= -1e-9})
    return unique, period


def snap_to_grid(time_value: float, grid: Sequence[float], tolerance: float) -> float:
    """时间吸附到最近网格点；距离超过 tolerance 时保持原值。"""
    if not grid:
        return time_value
    position = bisect_left(grid, time_value)
    candidates: list[float] = []
    if position < len(grid):
        candidates.append(grid[position])
    if position > 0:
        candidates.append(grid[position - 1])
    if not candidates:
        return time_value
    best = min(candidates, key=lambda point: abs(point - time_value))
    return best if abs(best - time_value) <= tolerance else time_value


def quantize_notes(
    notes: Sequence[dict],
    beats: dict | None,
    *,
    subdivision: int = 2,
    max_shift_ratio: float = 0.3,
    min_duration: float = _MIN_NOTE_SECONDS,
    until: float | None = None,
) -> tuple[list[dict], dict]:
    """把音符起止吸附到节拍网格；吸附幅度受 max_shift_ratio * beat 限制。

    原始时间保留在 start_raw/end_raw，onset 保留真实起音时刻，便于保真复核。
    返回 (新音符列表, 统计 dict)。
    """
    grid, period = build_beat_grid(beats, subdivision=subdivision, until=until)
    tolerance = max_shift_ratio * period
    stats = {
        "grid_points": len(grid),
        "beat_period": round(period, 4),
        "snapped": 0,
        "kept_raw": 0,
        "adjusted_overlap": 0,
    }
    result: list[dict] = []
    if not grid:
        for note in notes:
            start_raw = float(note["start"])
            end_raw = float(note.get("end", start_raw + min_duration))
            new_note = dict(note)
            new_note["start_raw"] = round(start_raw, 3)
            new_note["end_raw"] = round(end_raw, 3)
            new_note.setdefault("onset", round(start_raw, 3))
            result.append(new_note)
            stats["kept_raw"] += 1
        return result, stats

    # 第一步：只量化起音。起音必须严格递增；撞到同一个网格点时保留原始起音
    # （而不是一路后移——连锁后移会让整段旋律与真实演唱脱节 0.3s 以上）。
    entries: list[dict] = []
    for note in notes:
        start_raw = float(note["start"])
        snapped_start = snap_to_grid(start_raw, grid, tolerance)
        if entries:
            previous = entries[-1]
            if snapped_start <= previous["start"] + 1e-6:
                stats["adjusted_overlap"] += 1
                if start_raw - previous["start"] < min_duration:
                    # 两个碎片几乎同时起音：并成一个音（保留先出现的音高）
                    end_raw = float(note.get("end", start_raw + min_duration))
                    previous["end_raw"] = max(previous["end_raw"], end_raw)
                    previous["confidence"] = max(
                        previous.get("confidence", 0.0), float(note.get("confidence") or 0.0)
                    )
                    continue
                snapped_start = start_raw
        entries.append(
            {
                "note": dict(note),
                "start": snapped_start,
                "start_raw": start_raw,
                "end_raw": float(note.get("end", start_raw + min_duration)),
                "confidence": float(note.get("confidence") or 0.0),
            }
        )

    # 第二步：量化结束时间，但不越过下一个音符的起音；短于最短时值就补足。
    for index, entry in enumerate(entries):
        start = entry["start"]
        end = snap_to_grid(entry["end_raw"], grid, tolerance)
        if index + 1 < len(entries):
            end = min(end, entries[index + 1]["start"])
        if end - start < min_duration:
            end = start + min_duration
        new_note = dict(entry["note"])
        new_note["start"] = round(start, 3)
        new_note["end"] = round(end, 3)
        new_note["start_raw"] = round(entry["start_raw"], 3)
        new_note["end_raw"] = round(entry["end_raw"], 3)
        new_note.setdefault("onset", round(entry["start_raw"], 3))
        result.append(new_note)
        moved = abs(start - entry["start_raw"]) > 1e-6 or abs(end - entry["end_raw"]) > 1e-6
        stats["snapped" if moved else "kept_raw"] += 1
    return result, stats


# ---------------------------------------------------------------------------
# librosa 后端
# ---------------------------------------------------------------------------


def _load_audio(path: str, duration: float | None) -> tuple[Any, int, Any, Any]:
    """返回 (y, sr, numpy, librosa)；不可用时抛 RuntimeError（由调用方转成 warning）。"""
    try:
        import numpy as np  # type: ignore
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - 依赖环境
        raise RuntimeError(f"librosa/numpy 不可用（{exc}）") from exc
    try:
        y, sr = librosa.load(
            path,
            sr=DEFAULT_SAMPLE_RATE,
            mono=True,
            duration=float(duration) if duration and duration > 0 else None,
        )
    except Exception as exc:
        raise RuntimeError(f"音频读取失败（{exc}）") from exc
    return y, int(sr), np, librosa


def _estimate_f0(
    y: Any, sr: int, *, np: Any, librosa: Any
) -> tuple[Any, Any, str, list[str]]:
    """pyin 提 f0；失败时降级 piptrack。返回 (f0, voiced_prob, engine, warnings)。"""
    warnings: list[str] = []
    try:
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            y=y,
            fmin=DEFAULT_FMIN,
            fmax=DEFAULT_FMAX,
            sr=sr,
            frame_length=DEFAULT_FRAME_LENGTH,
            hop_length=DEFAULT_HOP_LENGTH,
            center=True,
            fill_na=np.nan,
        )
        return f0, voiced_prob, "pyin", warnings
    except Exception as exc:
        warnings.append(f"pyin 失败（{exc}），降级为 piptrack 音高估计")
        LOGGER.warning("pyin 失败，降级 piptrack：%s", exc)

    try:
        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            fmin=DEFAULT_FMIN,
            fmax=DEFAULT_FMAX,
            hop_length=DEFAULT_HOP_LENGTH,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"piptrack 也失败（{exc}）") from exc
    frame_count = pitches.shape[1]
    f0 = np.full(frame_count, np.nan, dtype=float)
    voiced_prob = np.zeros(frame_count, dtype=float)
    for frame in range(frame_count):
        column = magnitudes[:, frame]
        if column.size == 0:
            continue
        peak = int(np.argmax(column))
        if column[peak] <= 0:
            continue
        f0[frame] = pitches[peak, frame]
        voiced_prob[frame] = float(min(1.0, column[peak] / (np.max(magnitudes) or 1.0)))
    return f0, voiced_prob, "piptrack", warnings


def _silence_gate(energies: Sequence[float | None]) -> float:
    """按能量分布取静音门限：90 分位能量的 8%，避免把弱唱当静音。"""
    positive = sorted(float(value) for value in energies if value is not None and float(value) > 0)
    if not positive:
        return 0.0
    reference = positive[int(0.9 * (len(positive) - 1))]
    return max(1e-6, reference * 0.08)


def resolve_probability_gate(probabilities: Sequence[float | None]) -> float | None:
    """决定 pyin ``voiced_prob`` 的硬门限；返回 None 表示不按概率过滤。

    实测：整曲混音（不是人声 stem）上 pyin 的 voiced_prob 中位数 < 0.3，
    固定按 0.15 过滤会丢掉约 97% 的有声帧（60s 混音只剩 4 个音符）。这时
    只信 pyin 自己的清浊判决（未发声处 f0 为 NaN）叠加能量与音域门限；
    人声 stem 上概率分布高，才启用 0.15 的概率门限过滤噪声帧。
    """
    values = sorted(
        float(value)
        for value in probabilities
        if value is not None and not math.isnan(float(value))
    )
    if not values:
        return None
    median = values[len(values) // 2]
    if median < 0.3:
        return None
    return 0.15


def _detect_onsets(y: Any, sr: int, *, np: Any, librosa: Any, warnings: list[str]) -> list[float]:
    """用谱通量（spectral flux）找起音候选，再交给纯算法挑峰。

    只用到 stft / frames_to_time（纯 numpy+scipy）；绝不调用
    ``librosa.onset.onset_detect``——它的 numba peak_pick 在本环境会段错误。
    """
    try:
        magnitude = np.abs(
            librosa.stft(
                y,
                n_fft=DEFAULT_FRAME_LENGTH,
                hop_length=DEFAULT_HOP_LENGTH,
                center=True,
            )
        )
    except Exception as exc:  # pragma: no cover - 依赖环境
        warnings.append(f"起音检测失败（STFT 不可用：{exc}），起音只依赖 pyin 帧")
        return []
    if magnitude.shape[1] < 3:
        return []
    flux = np.maximum(0.0, np.diff(magnitude, axis=1)).sum(axis=0)
    peak = float(flux.max()) if flux.size else 0.0
    if peak <= 0:
        return []
    envelope = (flux / peak).tolist()
    try:
        times = librosa.frames_to_time(
            np.arange(1, magnitude.shape[1]), sr=sr, hop_length=DEFAULT_HOP_LENGTH
        ).tolist()
    except Exception as exc:  # pragma: no cover - 依赖环境
        warnings.append(f"起音时间换算失败（{exc}），起音只依赖 pyin 帧")
        return []
    return detect_onsets_from_envelope(envelope, times)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def transcribe_melody(
    vocals_wav: str, beats: dict, *, duration: float | None = None
) -> dict:
    """从（人声）音频提取单声部主旋律，并量化到节拍网格。

    参数:
        vocals_wav: 人声 stem 或整曲 WAV 路径。
        beats: analyze_beats 的输出（使用 beats/bpm/beats_per_bar）。
        duration: 音频时长（秒），None 表示用文件全部。

    返回:
        {"notes": [...], "range": {...}, "warnings": [...]}；失败时 notes=[] 并给出原因。
    """
    warnings: list[str] = []
    result: dict = {
        "notes": [],
        "range": {"low_midi": 0, "high_midi": 0},
        "warnings": warnings,
        "engine": "none",
    }

    path = str(vocals_wav or "").strip()
    if not path or not os.path.isfile(path):
        warnings.append(f"旋律提取输入音频不存在：{path}")
        return result

    name_lower = os.path.basename(path).lower()
    if "vocal" not in name_lower:
        warnings.append(
            "输入文件疑似非人声分轨，旋律提取可能受伴奏干扰（八度/噪声误差），建议使用 vocals stem"
        )

    try:
        y, sr, np, librosa = _load_audio(path, duration)
    except RuntimeError as exc:
        warnings.append(str(exc))
        LOGGER.warning("旋律提取降级：%s", exc)
        return result

    if y is None or len(y) == 0:
        warnings.append("音频为空，无法提取旋律")
        return result

    total_seconds = float(len(y)) / sr
    hop = DEFAULT_HOP_LENGTH
    frame_length = DEFAULT_FRAME_LENGTH

    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True)[0]
    except Exception as exc:
        warnings.append(f"能量计算失败（{exc}），跳过静音门限")
        rms = None

    try:
        f0, voiced_prob, engine, engine_warnings = _estimate_f0(y, sr, np=np, librosa=librosa)
    except RuntimeError as exc:
        warnings.append(str(exc))
        return result
    warnings.extend(engine_warnings)
    result["engine"] = engine

    frame_count = int(len(f0))
    if frame_count == 0:
        warnings.append("f0 提取得到 0 帧，无法生成旋律")
        return result

    times = librosa.times_like(f0, sr=sr, hop_length=hop).tolist()
    energies: list[float | None] = []
    if rms is not None:
        rms_values = rms.tolist()
        if len(rms_values) < frame_count:
            rms_values = rms_values + [rms_values[-1] if rms_values else 0.0] * (
                frame_count - len(rms_values)
            )
        energies = [float(value) for value in rms_values[:frame_count]]

    confidence_values: list[float | None] = [
        float(value) if value is not None and not math.isnan(float(value)) else None
        for value in voiced_prob[:frame_count]
    ]

    gate = _silence_gate(energies) if energies else 0.0
    probability_gate = resolve_probability_gate(confidence_values)
    midis: list[float | None] = []
    gated_frames = 0
    for index, frequency in enumerate(f0[:frame_count]):
        value = float(frequency)
        if math.isnan(value) or value <= 0:
            midis.append(None)
            continue
        if energies and energies[index] is not None and energies[index] < gate:
            midis.append(None)
            gated_frames += 1
            continue
        probability = confidence_values[index]
        if probability_gate is not None and probability is not None and probability < probability_gate:
            midis.append(None)
            continue
        midis.append(hz_to_midi(value))

    midis = median_filter_values(midis, kernel=5)
    midis, octave_fixed = fix_octave_jumps(midis)
    midis, range_shifted, range_dropped = constrain_pitch_range(midis)
    midis = round_pitch_values(midis)
    notes = segment_notes(
        times,
        midis,
        confidences=confidence_values,
        energies=energies or None,
        onsets=_detect_onsets(y, sr, np=np, librosa=librosa, warnings=warnings),
    )
    # 先用逐帧证据重算置信度（必须在八度归位前，否则音高已与帧证据不一致）
    notes = refine_note_confidences(
        notes, times, midis, probabilities=confidence_values, energies=energies or None
    )
    # 再把超出合理音域（G3-C6）的音符按八度归位，避免把女声主旋律标成 D3
    notes, range_folded = fold_notes_to_range(notes)
    notes = _merge_adjacent_same_pitch(absorb_ornament_notes(notes), max_gap=0.02)
    # 量化前先消掉 onset 校正造成的音符重叠，避免量化时整体后推
    notes = trim_note_overlaps(notes)
    notes, quantize_stats = quantize_notes(
        notes,
        beats,
        until=duration if duration and duration > 0 else total_seconds,
    )
    result["notes"] = notes
    result["range"] = note_range(notes)
    result["quantize"] = quantize_stats

    if octave_fixed:
        warnings.append(f"八度修正 {octave_fixed} 帧（pyin 常见八度误判）")
    if range_shifted:
        warnings.append(f"超音域帧按八度归位 {range_shifted} 帧")
    if range_folded:
        low_limit, high_limit = vocal_range_limits()
        warnings.append(
            f"{range_folded} 个音符超出主旋律合理音域（{low_limit}-{high_limit}），已按八度归位"
        )
    if range_dropped:
        warnings.append(f"超出人声音域且无法归位的帧 {range_dropped} 帧已丢弃（低置信度噪声）")
    if gated_frames:
        LOGGER.info("旋律提取：能量门限过滤 %d 帧", gated_frames)
    if not notes:
        warnings.append("未提取到有效旋律音符，可能是纯伴奏/静音或人声太弱")
    elif len(notes) < 8:
        warnings.append(f"仅提取到 {len(notes)} 个音符，旋律可能不完整")
    else:
        LOGGER.info(
            "旋律提取完成：engine=%s notes=%d range=%s-%s",
            engine,
            len(notes),
            result["range"]["low_midi"],
            result["range"]["high_midi"],
        )
    return result


# 兼容文档中的写法：melody.py 也暴露对齐入口（实现与 align.py 完全一致）。
def align_lyrics_to_melody(melody: dict, lyrics: list[dict], beats: dict) -> dict:  # noqa: D401
    """给每个音符挂歌词音节，返回新的 melody dict（实现见 aitsm.pipeline.align）。"""
    from .align import align_lyrics_to_melody as _align

    return _align(melody, lyrics, beats)

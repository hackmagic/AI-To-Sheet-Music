"""编配引擎（F11~F16）：把 ``analysis.json``（秒 / 绝对时间轴）编译成弹唱谱 ``score.json``。

设计原则
--------
1. **只搬运证据，不编造音乐**：旋律音只做量化与移调，和弦只做映射与选指法；
   任何无法确定的内容一律留空并写入 ``quality.warnings``。
2. **确定性**：同一份 analysis + options 永远得到同一份 score（便于回归测试与 diff）。
3. **可弹唱优先**：调性选吉他友好调 + 变调夹，和弦优先开放把位，节奏型跟随段落能量。

时间轴约定
----------
analysis 全部是**秒**；score 全部是**四分音符拍**，小节内相对（``beat`` 从 0 起）。
``analysis.beats.beats`` 的「分析拍」通过 :func:`aitsm.score.model.quarter_beats_per_beat`
换算成四分音符倍率（4/4 → 1.0，6/8 按八分打拍 → 0.5）。
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Any, Iterable, Sequence

from . import strum as strum_lib
from .model import (
    bar_length_quarters,
    duration_value,
    format_duration,
    key_fifths,
    note_to_pc,
    parse_chord_symbol,
    parse_time_signature,
    pc_to_name,
    pc_to_name_for_fifths,
    quarter_beats_per_beat,
    quantize,
    spell_midi,
    tuning_to_midis,
    _json_default,
    STANDARD_TUNING,
)
from .voicings import voicing_for

__all__ = ["arrange", "SCORE_VERSION", "DEFAULT_OPTIONS"]

SCORE_VERSION = "0.1.0"

DEFAULT_OPTIONS: dict[str, Any] = {
    "key": None,
    "capo": None,
    "difficulty": "intermediate",
    "style": "pop",
    "transpose": 0,
    "include_tab": True,
    "instrument": "guitar",
}

#: 吉他友好的大调主音（按常用程度排序）：C G D A E F Bb
_MAJOR_PLAY_ORDER = [0, 7, 2, 9, 4, 5, 10, 3, 8, 1, 6, 11]
#: 吉他友好的小调主音：Am Em Bm Dm Gm Cm Fm ...
_MINOR_PLAY_ORDER = [9, 4, 11, 2, 7, 0, 5, 3, 8, 1, 6, 10]
#: 真的能用开放把位弹的大调（C G D A E F Bb）；其余调（C#/Eb/F#/Ab/B）几乎全是大横按
_MAJOR_OPEN_KEYS = frozenset({0, 7, 2, 9, 4, 5, 10})
#: 有常用开放/半开放指法的小调（Am Em Dm Bm）
_MINOR_OPEN_KEYS = frozenset({9, 4, 2, 11})

#: 段落 kind -> 中文 label
_SECTION_LABELS = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "预副歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "interlude": "间奏",
    "solo": "吉他 solo",
    "outro": "尾奏",
    "other": "其他",
}

#: 难度 -> 指法偏好
_DIFFICULTY_PREFER: dict[str, dict[str, Any]] = {
    "beginner": {"max_difficulty": 2},
    "elementary": {"max_difficulty": 2},
    "intermediate": {"min_sounding": 5},
    "advanced": {"min_sounding": 6},
}

_VOICING_DIFFICULTY = {"beginner": 1, "elementary": 2, "intermediate": 3, "advanced": 4}

_EPS = 1e-6


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    """宽容地取一个有限浮点数。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _sec_sort(items: Iterable[dict], key: str = "start") -> list[dict]:
    return sorted(items, key=lambda item: _number(item.get(key)))


def _median(values: Sequence[float]) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


# --------------------------------------------------------------------------
# 1) 拍网格 / 小节切分
# --------------------------------------------------------------------------


class _BeatGrid:
    """秒 ↔ 拍 的双向映射（分段线性，拍外按平均拍长外推）。"""

    def __init__(self, beat_times: Sequence[float], duration: float, bpm: float) -> None:
        times = sorted({round(float(value), 6) for value in beat_times if _number(value) >= 0})
        if len(times) < 2:
            # 没有可用拍点时按 BPM 合成（40~208 之间，防止极端值）
            bpm = min(240.0, max(30.0, _number(bpm, 90.0) or 90.0))
            interval = 60.0 / bpm
            count = max(2, int(math.ceil(max(duration, interval) / interval)) + 1)
            times = [round(index * interval, 6) for index in range(count)]
        cleaned: list[float] = [times[0]]
        for value in times[1:]:
            if value - cleaned[-1] > 1e-3:
                cleaned.append(value)
        self.times = cleaned
        self.interval = _median([b - a for a, b in zip(cleaned, cleaned[1:])]) or 0.5
        self.duration = max(_number(duration, 0.0), cleaned[-1])

    def sec_to_beat(self, sec: float) -> float:
        """秒 -> 拍序号（浮点，允许小节内小数）。"""
        sec = _number(sec)
        times = self.times
        if sec <= times[0]:
            return (sec - times[0]) / self.interval
        if sec >= times[-1]:
            return (len(times) - 1) + (sec - times[-1]) / self.interval
        # 二分找区间
        low, high = 0, len(times) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if times[mid] <= sec:
                low = mid
            else:
                high = mid
        span = times[high] - times[low]
        if span <= _EPS:
            return float(low)
        return low + (sec - times[low]) / span

    def beat_to_sec(self, beat: float) -> float:
        """拍序号 -> 秒。"""
        beat = _number(beat)
        times = self.times
        if beat <= 0:
            return times[0] + beat * self.interval
        if beat >= len(times) - 1:
            return times[-1] + (beat - (len(times) - 1)) * self.interval
        index = int(math.floor(beat))
        frac = beat - index
        return times[index] + frac * (times[index + 1] - times[index])


def _build_bars(
    analysis: dict,
    grid: _BeatGrid,
    options: dict,
    warnings: list[dict],
) -> tuple[list[dict], tuple[int, int], int, float]:
    """切小节网格，返回 ``(bars, time_signature, beats_per_bar, bar_len_quarters)``。"""
    beats_info = analysis.get("beats") or {}
    ts = parse_time_signature(beats_info.get("time_signature"), (4, 4))
    bpb = int(_number(beats_info.get("beats_per_bar"), ts[0]) or ts[0])
    bpb = max(2, min(12, bpb))
    bar_len = bar_length_quarters(ts)
    qpb = quarter_beats_per_beat(bpb, ts)

    downbeats = [float(value) for value in (beats_info.get("downbeats") or []) if _number(value) >= 0]
    bar_start_secs: list[float] = []
    if len(downbeats) >= 2:
        bar_start_secs = sorted(set(round(value, 6) for value in downbeats))
        # 校验 downbeats 与小节长度一致；不一致时退回「每 bpb 拍一个强拍」
        expected = bar_len / qpb  # 一小节应有的拍数
        gaps = [b - a for a, b in zip(bar_start_secs, bar_start_secs[1:])]
        if gaps and any(abs(gap / grid.interval - expected) > 1.5 for gap in gaps):
            warnings.append(
                {
                    "level": "warn",
                    "code": "bars.downbeat_irregular",
                    "message": "强拍间隔与拍号不一致，已按每小节拍数重新切分",
                    "stage": "bars",
                }
            )
            bar_start_secs = []
    if not bar_start_secs:
        step = max(1, int(round(bar_len / qpb)))
        bar_start_secs = [grid.times[index] for index in range(0, len(grid.times), step)]

    # 弱起：第一小节之前还有拍点（或音频从 0 开始但首强拍不在 0）
    pickup_beats = 0
    expected_bar_beats = bar_len / qpb
    if bar_start_secs and bar_start_secs[0] > grid.times[0] + grid.interval * 0.4:
        candidate = max(1, int(round((bar_start_secs[0] - grid.times[0]) / grid.interval)))
        if candidate < expected_bar_beats - 0.5:
            pickup_beats = candidate
            bar_start_secs = [grid.times[0]] + bar_start_secs
        else:
            # 首强拍前已有整小节以上的素材（前奏/静音），不是弱起，按常规网格切分
            step = max(1, int(round(expected_bar_beats)))
            bar_start_secs = [grid.times[index] for index in range(0, len(grid.times), step)]

    bars: list[dict] = []
    duration = min(grid.duration, max(grid.duration, grid.times[-1]))
    starts = bar_start_secs
    for index, start_sec in enumerate(starts):
        end_sec = starts[index + 1] if index + 1 < len(starts) else grid.duration
        if end_sec - start_sec < grid.interval * 0.35:
            continue
        start_beat = grid.sec_to_beat(start_sec)
        end_beat = grid.sec_to_beat(end_sec)
        bars.append(
            {
                "start_beat": start_beat,
                "end_beat": end_beat,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "pickup": False,
            }
        )
    if not bars:  # pragma: no cover - 极端防御
        bars = [
            {
                "start_beat": 0.0,
                "end_beat": max(1.0, bar_len / qpb),
                "start_sec": 0.0,
                "end_sec": grid.duration,
                "pickup": False,
            }
        ]

    # 尾部不完整小节：不足半小节则并入上一小节，否则保留（标记 pickup=False + 警告）
    if len(bars) >= 2:
        last = bars[-1]
        ideal = bar_len / qpb
        if (last["end_beat"] - last["start_beat"]) < ideal * 0.5:
            warnings.append(
                {
                    "level": "info",
                    "code": "bars.tail_merged",
                    "message": f"尾奏只剩 {last['end_beat'] - last['start_beat']:.2f} 拍，已并入第 {len(bars) - 1} 小节",
                    "stage": "bars",
                }
            )
            bars[-2]["end_beat"] = last["end_beat"]
            bars[-2]["end_sec"] = last["end_sec"]
            bars.pop()

    if pickup_beats:
        bars[0]["pickup"] = True
        warnings.append(
            {
                "level": "info",
                "code": "bars.pickup",
                "message": f"检测到弱起小节（约 {pickup_beats} 拍）",
                "stage": "bars",
            }
        )

    # 小节内速度（支持可选 tempo_map：[{time,bpm}] 或按拍点实测）
    tempo_map = beats_info.get("tempo_map")
    base_bpm = _number(beats_info.get("bpm_display") or beats_info.get("bpm"), 90.0)
    for bar in bars:
        bar["tempo"] = base_bpm
    if isinstance(tempo_map, list) and tempo_map:
        points = [
            (_number(item.get("time")), _number(item.get("bpm"), base_bpm))
            for item in tempo_map
            if isinstance(item, dict) and _number(item.get("bpm")) > 20
        ]
        points.sort()
        for bar in bars:
            current = base_bpm
            for time_sec, bpm in points:
                if time_sec <= bar["start_sec"] + _EPS:
                    current = bpm
            bar["tempo"] = current
    else:
        # 没有显式速度表：用每小节的拍数与实际秒数算局部 BPM（能捕捉渐慢/渐快）
        for bar in bars:
            span_beats = (bar["end_beat"] - bar["start_beat"]) / qpb
            span_sec = bar["end_sec"] - bar["start_sec"]
            if span_beats > 0 and span_sec > 0.05:
                local_bpm = span_beats * 60.0 / span_sec
                if abs(local_bpm - base_bpm) / base_bpm > 0.02:
                    bar["tempo"] = round(local_bpm, 3)
    if len({round(bar["tempo"], 3) for bar in bars}) > 1:
        warnings.append(
            {
                "level": "info",
                "code": "bars.tempo_changes",
                "message": f"检测到速度波动（{min(bar['tempo'] for bar in bars):.0f}~{max(bar['tempo'] for bar in bars):.0f} BPM），已写入各小节 tempo",
                "stage": "bars",
            }
        )

    bars[0]["time_signature"] = list(ts)
    for index, bar in enumerate(bars):
        bar["index"] = index + 1
        bar["_qpb"] = qpb
        bar["_bar_len"] = bar_len
    return bars, ts, bpb, bar_len


def _bar_at(bars: Sequence[dict], sec: float, grid: _BeatGrid) -> int:
    """秒 -> 小节下标（超出末尾夹到最后一小节）。

    先换算成拍序号再比较，避免 ``start_sec`` 四舍五入造成的边界抖动；
    落点恰好落在小节线上时归入**后**一小节。
    """
    beat = grid.sec_to_beat(_number(sec))
    index = 0
    for position, bar in enumerate(bars):
        if beat >= bar["start_beat"] - _EPS:
            index = position
    return index


def _position_in_bar(bar: dict, beat_float: float) -> float:
    """全局拍序号 -> 小节内相对拍（四分音符拍）。"""
    return max(0.0, (beat_float - bar["start_beat"]) * bar["_qpb"])


def _snap_to_next_bar(
    bars: Sequence[dict],
    bar_index: int,
    start_sec: float,
    grid: "_BeatGrid",
) -> tuple[int, float]:
    """贴近小节线的旋律音归入下一小节。

    分析结果里音符起点常带 ±几毫秒误差，量化到 1/4 网格后可能正好落到
    ``beat == 小节长度``——那是下一小节的强拍，留在本小节就是非法位置
    （校验器会报 ``vocal.beat_range``）。
    """
    if not bars:
        return bar_index, start_sec
    bar_index = max(0, min(bar_index, len(bars) - 1))
    bar = bars[bar_index]
    local = _position_in_bar(bar, grid.sec_to_beat(start_sec))
    bar_len = float(bar["_bar_len"])
    if _quantize_melody(local, bar_len=bar_len) < bar_len - _EPS:
        return bar_index, start_sec
    if bar_index + 1 < len(bars):
        return bar_index + 1, bars[bar_index + 1]["start_sec"]
    # 已经是最后一小节：只能退到小节内最后一个网格点
    return bar_index, start_sec


# --------------------------------------------------------------------------
# 2) 调性 / 变调夹
# --------------------------------------------------------------------------


def _choose_key(
    sounding_pc: int,
    mode: str,
    options: dict,
    warnings: list[dict],
) -> tuple[int, int]:
    """选记谱调与变调夹，返回 ``(play_pc, capo)``。

    - 用户指定 ``options["key"]`` 时直接采用该调；
    - 否则在吉他友好调里挑 capo 最小（≤7）且最常用的那个；
    - 用户指定 ``options["capo"]`` 时以用户为准，记谱调顺推。
    """
    requested_key = options.get("key")
    requested_capo = options.get("capo")
    order = _MINOR_PLAY_ORDER if mode == "minor" else _MAJOR_PLAY_ORDER

    if requested_key:
        try:
            play_pc = note_to_pc(str(requested_key))
        except ValueError:
            warnings.append(
                {
                    "level": "warn",
                    "code": "key.invalid_option",
                    "message": f"无法识别指定的调 {requested_key!r}，改为自动选择",
                    "stage": "key",
                }
            )
            play_pc = None  # type: ignore[assignment]
        if play_pc is not None:
            if requested_capo is not None:
                capo = max(0, min(12, int(round(_number(requested_capo, 0)))))
                return play_pc, capo
            capo = (sounding_pc - play_pc) % 12
            if capo > 7:
                warnings.append(
                    {
                        "level": "warn",
                        "code": "key.capo_high",
                        "message": f"指定调 {requested_key} 需要 {capo} 品变调夹，超过 7 品",
                        "stage": "key",
                    }
                )
            return play_pc, capo

    if requested_capo is not None:
        capo = max(0, min(12, int(round(_number(requested_capo, 0)))))
        play_pc = (sounding_pc - capo) % 12
        if play_pc not in order:
            warnings.append(
                {
                    "level": "info",
                    "code": "key.non_guitar_friendly",
                    "message": f"按指定 {capo} 品变调夹后记谱调为 {pc_to_name(play_pc)}，不是最常用的吉他调",
                    "stage": "key",
                }
            )
        return play_pc, capo

    best: tuple[int, int, int] | None = None
    open_friendly = _MINOR_OPEN_KEYS if mode == "minor" else _MAJOR_OPEN_KEYS
    for preference, candidate in enumerate(order):
        capo = (sounding_pc - candidate) % 12
        # 记谱调必须能用开放把位弹（C/G/D/A/E/F/Bb…）；非友好调（C#/F#/B…）全是大横按，
        # 即使 capo 更小也不划算 —— 重罚后只在万不得已时才会选中。
        penalty = 0 if candidate in open_friendly else 8
        # 超过 7 品的变调夹不实用 -> 重罚；capo 越小越好；偏好顺序作最后 tie-break
        score = (penalty + (capo if capo <= 7 else 100 + capo), preference, candidate)
        if best is None or score < best:
            best = score
    assert best is not None
    capo = (sounding_pc - best[2]) % 12
    return best[2] % 12, capo


# --------------------------------------------------------------------------
# 3) 和弦映射
# --------------------------------------------------------------------------


def _transpose_symbol(symbol: str, delta: int, *, prefer_flats: bool) -> str:
    """把和弦符号整体移调（斜杠低音一起移）。"""
    try:
        chord = parse_chord_symbol(symbol)
    except ValueError:
        return symbol
    root = (chord.root_pc + delta) % 12
    bass = None if chord.bass_pc is None else (chord.bass_pc + delta) % 12
    from .model import format_chord_symbol

    return format_chord_symbol(root, chord.quality, bass, prefer_flats=prefer_flats)


def _choose_voicing(
    symbol: str,
    difficulty: str,
    previous: dict | None,
) -> dict | None:
    """按难度与换把平滑度选指法。"""
    prefer: dict[str, Any] = dict(_DIFFICULTY_PREFER.get(str(difficulty).lower(), {}))
    if previous:
        prefer["previous"] = previous
    voicing = voicing_for(symbol, prefer=prefer)
    if voicing is None and prefer:  # 极少数和弦在难度约束下无解 -> 放开约束
        voicing = voicing_for(symbol)
    return voicing


def _map_chords(
    analysis: dict,
    bars: list[dict],
    grid: _BeatGrid,
    *,
    delta: int,
    capo: int,
    prefer_flats: bool,
    difficulty: str,
    warnings: list[dict],
) -> tuple[dict[int, list[dict]], dict[str, dict], float]:
    """和弦事件 -> 每小节 ``chords[]``（键为 1 起的小节号）。

    跨小节的长和弦会在经过的每个小节各生成一条记录（``beat=0`` 表示延续），
    这样前端按小节渲染时不会出现"和弦消失"的空档。
    """
    events = [dict(item) for item in (analysis.get("chords") or []) if item.get("symbol")]
    events = _sec_sort(events)
    bar_chords: dict[int, list[dict]] = {}
    voicings: dict[str, dict] = {}
    confidences: list[float] = []
    previous_voicing: dict | None = None

    for event in events:
        start_sec = _number(event.get("start"))
        end_sec = _number(event.get("end"), start_sec + 1.0)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.25
        symbol_raw = str(event["symbol"])
        symbol = _transpose_symbol(symbol_raw, delta, prefer_flats=prefer_flats)
        concert = _transpose_symbol(symbol_raw, delta + capo, prefer_flats=prefer_flats)
        confidence = _number(event.get("confidence"), 0.5)
        confidences.append(confidence)
        start_beat = grid.sec_to_beat(start_sec)
        end_beat = grid.sec_to_beat(end_sec)
        start_bar_index = _bar_at(bars, start_sec, grid)

        voicing = _choose_voicing(symbol, difficulty, previous_voicing)
        if voicing is not None:
            previous_voicing = voicing
            voicings[voicing["id"]] = _public_voicing(voicing)

        for position in range(start_bar_index, len(bars)):
            bar = bars[position]
            if bar["start_beat"] >= end_beat - _EPS and position > start_bar_index:
                break
            grid_step = 0.5 if abs(bar["_bar_len"] % 0.5) < _EPS else 0.25
            bar_len = bar["_bar_len"]
            if position == start_bar_index:
                beat_in_bar = _quantize_grid(_position_in_bar(bar, start_beat), grid_step, bar_len)
            else:
                beat_in_bar = 0.0
            if beat_in_bar >= bar_len - _EPS:
                continue
            # 结束位置夹到当前小节内
            span_beats = _position_in_bar(bar, end_beat)
            bar_end_beat = min(_position_in_bar(bar, bar["end_beat"]), bar_len)
            dur_beats = min(span_beats, bar_end_beat) - beat_in_bar
            if dur_beats <= 0:
                dur_beats = max(grid_step, bar_end_beat - beat_in_bar)
            entry: dict[str, Any] = {
                "beat": round(beat_in_bar, 4),
                "dur": format_duration(round(dur_beats, 4)),
                "symbol": symbol,
                "concert": concert,
                "confidence": round(confidence, 3),
                "start_sec": round(grid.beat_to_sec(bar["start_beat"] + beat_in_bar / bar["_qpb"]), 4)
                if position != start_bar_index
                else round(start_sec, 4),
            }
            if voicing is not None:
                entry["voicing_id"] = voicing["id"]
            if position != start_bar_index:
                entry["continue"] = True
            bar_chords.setdefault(bar["index"], []).append(entry)

    # 每小节最多 4 个和弦位置：超出时保留置信度高的（并给出警告）
    for bar_number, entries in list(bar_chords.items()):
        entries.sort(key=lambda item: (item["beat"], -_number(item.get("confidence"), 0.5)))
        if len(entries) > 4:
            kept = sorted(entries, key=lambda item: -_number(item.get("confidence"), 0.5))[:4]
            kept.sort(key=lambda item: item["beat"])
            warnings.append(
                {
                    "level": "warn",
                    "code": "chords.too_many",
                    "message": f"第 {bar_number} 小节和弦超过 4 个，已按置信度保留 4 个",
                    "stage": "chords",
                    "target": f"bar:{bar_number}",
                }
            )
            entries = kept
        # 同小节相同和弦合并（相邻且同符号）
        step = 0.5 if abs(bars[bar_number - 1]["_bar_len"] % 0.5) < _EPS else 0.25
        entries = _resolve_chord_overlaps(entries, step)
        merged: list[dict] = []
        for entry in entries:
            if merged and merged[-1]["symbol"] == entry["symbol"]:
                merged[-1]["dur"] = _merge_duration(merged[-1]["dur"], entry["dur"])
                merged[-1]["confidence"] = max(merged[-1].get("confidence", 0), entry.get("confidence", 0))
                continue
            merged.append(entry)
        merged = _fill_chord_gaps(merged, bars[bar_number - 1]["_bar_len"], step)
        bar_chords[bar_number] = merged

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return bar_chords, voicings, mean_conf


def _fill_chord_gaps(entries: list[dict], bar_len: float, step: float) -> list[dict]:
    """把和弦之间 / 小节末尾的小空档补齐到下一个和弦起点或小节线。

    analysis 给出的和弦 ``end`` 常比下一个和弦 ``start`` 早零点几拍（量化误差），
    真实演奏时吉他会一直扫到换和弦。只补**小于一个量化步长**的空档，
    更大的空档视为真正的休止不动。
    """
    for index in range(len(entries) - 1):
        current = entries[index]
        following = entries[index + 1]
        start = _number(current.get("beat"))
        try:
            dur = duration_value(current.get("dur"))
        except ValueError:  # pragma: no cover - 防御
            continue
        gap = _number(following.get("beat")) - (start + dur)
        if 0 <= gap < step - _EPS:
            current["dur"] = format_duration(round(max(step, _number(following.get("beat")) - start), 4))
    if entries:
        last = entries[-1]
        start = _number(last.get("beat"))
        try:
            dur = duration_value(last.get("dur"))
        except ValueError:  # pragma: no cover - 防御
            dur = 0.0
        gap = bar_len - (start + dur)
        if 0 <= gap < step - _EPS:
            last["dur"] = format_duration(round(bar_len - start, 4))
    return entries


def _resolve_chord_overlaps(entries: list[dict], step: float) -> list[dict]:
    """消除同小节和弦重叠。

    跨小节长和弦的残段（分析给的 end 只比小节线多几十毫秒）会产生
    ``beat=0, dur=0.1`` 的碎片，和新和弦在 0 拍撞车。这里把碎片直接丢掉，
    其余重叠则把前一个和弦截到后一个起点。
    """
    resolved: list[dict] = []
    for entry in entries:
        try:
            beat = _number(entry.get("beat"))
            dur = duration_value(entry.get("dur"))
        except ValueError:  # pragma: no cover - 防御
            continue
        drop_current = False
        while resolved:
            previous = resolved[-1]
            previous_beat = _number(previous.get("beat"))
            try:
                previous_dur = duration_value(previous.get("dur"))
            except ValueError:  # pragma: no cover - 防御
                previous_dur = 0.0
            if previous_beat + previous_dur <= beat + _EPS:
                break
            remaining = beat - previous_beat
            if remaining < step - _EPS:
                # 同一位置的两个和弦事件（残影 / 重复识别）：保留时值更长的
                if dur > previous_dur:
                    resolved[-1] = entry
                drop_current = True
                break
            previous["dur"] = format_duration(round(remaining, 4))
            break
        if drop_current or dur <= 0:
            continue
        resolved.append(entry)
    return resolved


def _quantize_grid(value: float, step: float, bar_len: float) -> float:
    """量化到 ``step`` 网格，并夹在小节范围内。"""
    if step <= 0:
        return max(0.0, min(value, bar_len))
    snapped = round(value / step) * step
    if snapped >= bar_len - _EPS:
        snapped = max(0.0, bar_len - step)
    return max(0.0, snapped)


def _merge_duration(first: Any, second: Any) -> Any:
    """合并两个时值（支持分数字符串）。"""
    try:
        total = duration_value(first) + duration_value(second)
    except ValueError:  # pragma: no cover - 防御
        return first
    return format_duration(round(total, 4))


def _public_voicing(voicing: dict) -> dict:
    """精简为 score.voicings 需要的字段（前端与导出器共用）。"""
    keys = ("symbol", "frets", "fingers", "base_fret", "barres", "difficulty", "label")
    result: dict[str, Any] = {}
    for key in keys:
        if key not in voicing:
            continue
        value = voicing[key]
        if key == "frets" and isinstance(value, list):
            # schema：-1 表示不弹；统一保留 -1（不转 null，前端好判断）
            value = [int(item) if item is not None else -1 for item in value]
        result[key] = value
    return result


# --------------------------------------------------------------------------
# 4) 旋律量化 / 歌词
# --------------------------------------------------------------------------


def _triplet_proximity(value: float) -> float:
    """value 距离最近的 1/3 网格有多远（用于判断是否三连音）。"""
    scaled = value * 3.0
    return abs(scaled - round(scaled)) / 3.0


def _quantize_melody(value: float, *, bar_len: float) -> float:
    """把旋律位置量化到 1/4 拍网格（默认），三连音需显式开启。

    为什么默认不做三连音：导出层（MusicXML）与网页渲染器都只可靠支持二分网格，
    1/3 网格会被拆成连音或被吸附到十六分音符——`renderer.js` 会打出
    「时值无法精确记谱」的告警，导出的 MusicXML 也要靠连音凑。
    真需要三连音时设 ``AITSM_MELODY_TRIPLETS=1``，此时保持精确的 1/3 值。
    """
    if os.environ.get("AITSM_MELODY_TRIPLETS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        triplet = round(value * 3.0) / 3.0
        quarter = round(value * 4.0) / 4.0
        if _triplet_proximity(value) < 0.02 and _triplet_proximity(value) <= abs(value - quarter):
            # 不四舍五入：1/3 网格必须保持精确值，否则 format_duration 认不出三连音
            return max(0.0, min(round(triplet, 9), bar_len))
    result = round(value * 4.0) / 4.0
    return max(0.0, min(round(result, 9), bar_len))


# 英语常见独立词（用于判断「残词」：不在表里的短词大概率是被拆开的音节）
# Whisper 会把 baby 切成 "bab" + "y"，这里靠「前一个词不是独立词 + 紧跟」来合并。
_COMMON_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been being
    am do does did done have has had having will would can could shall should may might must
    i you he she it we they me him her us them my your his its our their mine yours
    this that these those there here where when what who whom which why how
    not no nor so than then too very just only also even still yet ever never always
    let lets let's im i'm ive i've ill i'll id i'd you're you've we're they're don't didn't
    oh now now's yeah yea ah uh um la na da hey ho hmm mmm whoa ohh ahh
    baby love night light right time mind heart soul road home gone done come go goin going
    gonna wanna gotta know think feel want need take make give get got say said see look
    up down out off over under again away back round around through into onto upon
    one two three four five six seven eight nine ten
    """.split()
)


def _clean_lyric(value: Any) -> str | None:
    """把 ``lyric`` 字段归一化成字符串或 None（过滤空白/纯符号）。"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _is_word_fragment(word: str) -> bool:
    """判断一个 token 是否像被切开的音节（用于合并 bab + y -> baby）。"""
    if not word.isalpha() or not word.isascii():
        return False
    return word.lower() not in _COMMON_WORDS


def _merge_fragment_words(words: list[dict]) -> int:
    """合并同一英文单词被切开的相邻 token（原地修改），返回合并次数。"""
    merged = 0
    index = 0
    while index + 1 < len(words):
        current = words[index]
        nxt = words[index + 1]
        first = str(current.get("text") or "")
        second = str(nxt.get("text") or "")
        if not (_is_word_fragment(first) and second.isalpha() and second.isascii()):
            index += 1
            continue
        if len(second) > 3:
            index += 1
            continue
        if second.lower() in _COMMON_WORDS:
            index += 1
            continue
        gap = _number(nxt.get("start")) - _number(current.get("end"))
        if not (-0.3 <= gap <= 0.35):
            index += 1
            continue
        current["text"] = first + second
        current["end"] = _number(nxt.get("end"), _number(current.get("end")))
        words.pop(index + 1)
        merged += 1
    return merged


def _lyric_index(analysis: dict) -> list[dict]:
    """把 ``analysis.lyrics`` 摊平成按时间排序的「词 -> 时间」列表。"""
    words: list[dict] = []
    for line in analysis.get("lyrics") or []:
        line_words = line.get("words")
        if isinstance(line_words, list) and line_words:
            for word in line_words:
                text = str(word.get("text") or "").strip()
                if not text:
                    continue
                words.append(
                    {
                        "start": _number(word.get("start"), _number(line.get("start"))),
                        "end": _number(word.get("end"), _number(line.get("end"))),
                        "text": text,
                    }
                )
        else:
            text = str(line.get("text") or "").strip()
            if not text:
                continue
            start = _number(line.get("start"))
            end = _number(line.get("end"), start)
            # 行内按字平均分配时间（中文歌词常见做法）
            chars = [char for char in text if not char.isspace()]
            if not chars:
                continue
            span = max(end - start, 0.05)
            for index, char in enumerate(chars):
                words.append(
                    {
                        "start": start + span * index / len(chars),
                        "end": start + span * (index + 1) / len(chars),
                        "text": char,
                    }
                )
    words.sort(key=lambda item: item["start"])
    _merge_fragment_words(words)
    return words


def _nearest_lyric(words: Sequence[dict], start_sec: float, end_sec: float, used: set[int]) -> str | None:
    """给一个旋律音找最贴近的歌词字。"""
    best_index: int | None = None
    best_distance = 1e9
    for index, word in enumerate(words):
        if index in used:
            continue
        if word["start"] <= start_sec + 0.08 and word["end"] >= end_sec - 0.08:
            distance = abs(word["start"] - start_sec)
        else:
            distance = abs(word["start"] - start_sec) + 0.2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index is None or best_distance > 1.2:
        return None
    used.add(best_index)
    return str(words[best_index]["text"])


def _merge_note_fragments(notes: list[dict]) -> int:
    """把被切开的英文残词合并回整词（bab + y -> baby），原地修改并返回次数。"""
    merged = 0
    for index in range(len(notes) - 1):
        current = notes[index]
        nxt = notes[index + 1]
        first = _clean_lyric(current.get("lyric"))
        second = _clean_lyric(nxt.get("lyric"))
        if not first or not second:
            continue
        if not (_is_word_fragment(first) and second.isalpha() and second.isascii() and len(second) <= 3):
            continue
        # 后一个 token 若本身是常用独立词（oh/ah/ya...），说明是两个词，不要合并
        if second.lower() in _COMMON_WORDS:
            continue
        gap = _number(nxt.get("start")) - _number(current.get("end"))
        if not (-0.35 <= gap <= 0.35):
            continue
        current["lyric"] = first + second
        # 后一个 token 归入同一个词：不再挂词，标记为延音（一字多音）
        nxt["lyric"] = None
        nxt.setdefault("_fragment", True)
        merged += 1
    return merged


def _map_melody(
    analysis: dict,
    bars: list[dict],
    grid: _BeatGrid,
    *,
    midi_delta: int,
    warnings: list[dict],
) -> tuple[dict[int, list[dict]], dict[str, float]]:
    """旋律音 -> 每小节 ``vocal[]``（跨小节拆分 + tie）。"""
    notes = [dict(item) for item in ((analysis.get("melody") or {}).get("notes") or []) if item.get("midi") is not None]
    notes = _sec_sort(notes)
    merged_fragments = _merge_note_fragments(notes)
    lyrics_words = _lyric_index(analysis)
    used_lyrics: set[int] = set()
    bar_vocals: dict[int, list[dict]] = {}
    confidences: list[float] = []
    unmatched = 0

    for note in notes:
        start_sec = _number(note.get("start"))
        end_sec = _number(note.get("end"), start_sec + 0.25)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.12
        midi = int(round(_number(note.get("midi"), 0))) + midi_delta
        confidence = _number(note.get("confidence"), 0.5)
        confidences.append(confidence)
        tie = note.get("tie")
        # 规则（lead 明确）：显式 lyric:null / tie=stop|continue 的音符绝不回填歌词，
        # 它是一字多音的延续，只靠 tie / lyric_extend 表示连唱。
        if "lyric" in note:
            lyric_text = _clean_lyric(note.get("lyric"))
        elif tie in ("stop", "continue"):
            lyric_text = None
        else:
            lyric_text = _nearest_lyric(lyrics_words, start_sec, end_sec, used_lyrics)
        if lyric_text is None and tie not in ("stop", "continue") and not note.get("_fragment"):
            unmatched += 1
        extends = lyric_text is None and (tie in ("stop", "continue") or bool(note.get("_fragment")))

        start_bar_index = _bar_at(bars, start_sec, grid)
        start_bar_index, start_sec = _snap_to_next_bar(bars, start_bar_index, start_sec, grid)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.12
        end_bar_index = _bar_at(bars, end_sec, grid)
        if end_bar_index < start_bar_index:
            end_bar_index = start_bar_index
        segments: list[tuple[int, float, float]] = []
        for bar_position in range(start_bar_index, end_bar_index + 1):
            bar = bars[bar_position]
            seg_start_sec = max(start_sec, bar["start_sec"])
            seg_end_sec = min(end_sec, bar["end_sec"])
            if seg_end_sec - seg_start_sec <= 1e-4:
                continue
            beat_start = _quantize_melody(
                _position_in_bar(bar, grid.sec_to_beat(seg_start_sec)), bar_len=bar["_bar_len"]
            )
            if beat_start >= bar["_bar_len"] - _EPS:
                # 兜底（最后一小节无法后移）：退到小节内最后一个 16 分位置
                beat_start = max(0.0, float(bar["_bar_len"]) - 0.25)
            beat_end = _quantize_melody(
                _position_in_bar(bar, grid.sec_to_beat(seg_end_sec)), bar_len=bar["_bar_len"]
            )
            if beat_end <= beat_start:
                beat_end = min(bar["_bar_len"], beat_start + 0.25)
            segments.append((bar_position, beat_start, beat_end))
        if not segments:
            continue
        for position, (bar_position, beat_start, beat_end) in enumerate(segments):
            bar = bars[bar_position]
            entry: dict[str, Any] = {
                "beat": round(beat_start, 6),
                "dur": format_duration(beat_end - beat_start),
                "midi": midi,
                "confidence": round(confidence, 3),
                "start_sec": round(grid.beat_to_sec(bar["start_beat"] + beat_start / bar["_qpb"]), 4),
            }
            if position == 0:
                entry["lyric"] = lyric_text
                if extends:
                    entry["lyric_extend"] = True
            else:
                entry["lyric"] = None
            if len(segments) > 1:
                if position == 0:
                    entry["tie"] = "start"
                elif position == len(segments) - 1:
                    entry["tie"] = "stop"
                else:
                    entry["tie"] = "continue"
            bar_vocals.setdefault(bar["index"], []).append(entry)

    for bar_index, entries in bar_vocals.items():
        entries.sort(key=lambda item: item["beat"])
        bar_len = float(bars[bar_index - 1]["_bar_len"])
        bar_vocals[bar_index] = _fix_vocal_overlaps(entries, warnings, bar_len=bar_len)

    if unmatched:
        warnings.append(
            {
                "level": "warn",
                "code": "lyrics.missing",
                "message": f"{unmatched} 个旋律音没有匹配到歌词，暂留空",
                "stage": "lyrics",
            }
        )
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return bar_vocals, {"melody": mean_conf, "lyric_match": 1.0 - unmatched / max(1, len(notes))}


def _fix_vocal_overlaps(entries: list[dict], warnings: list[dict], *, bar_len: float) -> list[dict]:
    """同小节内旋律音不允许重叠：去重同网格音、截断重叠、钳到小节长度。"""
    # 1) 同一网格位置（如两个音都量化到 1.75 拍）：只保留信息量大的那个
    cleaned: list[dict] = []
    for entry in entries:
        if cleaned and _number(entry.get("beat")) - _number(cleaned[-1].get("beat")) < 0.05:
            previous = cleaned[-1]
            if _vocal_weight(entry) > _vocal_weight(previous):
                cleaned[-1] = entry
            continue
        cleaned.append(entry)
    # 2) 前一个音截到后一个音起点
    for index in range(len(cleaned) - 1):
        current = cleaned[index]
        nxt = cleaned[index + 1]
        try:
            dur = duration_value(current.get("dur"))
        except ValueError:  # pragma: no cover
            dur = 0.25
        limit = _number(nxt.get("beat")) - _number(current.get("beat"))
        if dur > limit + _EPS:
            if limit <= 0.05:
                warnings.append(
                    {
                        "level": "info",
                        "code": "melody.overlap",
                        "message": f"第 {index + 1} 个音与下一个音起点重合，已略去",
                        "stage": "melody",
                    }
                )
                current["dur"] = format_duration(0.05)
            else:
                current["dur"] = format_duration(limit)
    # 3) 最后一个音不能伸出小节
    if cleaned:
        last = cleaned[-1]
        limit = bar_len - _number(last.get("beat"))
        try:
            dur = duration_value(last.get("dur"))
        except ValueError:  # pragma: no cover
            dur = 0.0
        if limit <= 0.0:
            cleaned.pop()
        elif dur > limit + _EPS:
            last["dur"] = format_duration(max(0.05, limit))
    return cleaned


def _vocal_weight(entry: dict) -> tuple[int, float, float]:
    """同网格重复音保留哪一个：有词 > 时值长 > 置信度高。"""
    try:
        dur = duration_value(entry.get("dur"))
    except ValueError:  # pragma: no cover
        dur = 0.0
    return (
        1 if entry.get("lyric") else 0,
        round(dur, 6),
        _number(entry.get("confidence"), 0.0),
    )



# --------------------------------------------------------------------------
# 5) 段落
# --------------------------------------------------------------------------


def _looks_mojibake(text: str) -> bool:
    """粗略判断字符串是否为乱码（替换符 / 控制字符 / 非法代理）。"""
    if not text:
        return True
    for char in text:
        code = ord(char)
        if char == "\ufffd" or (code < 0x20 and char not in "\t "):
            return True
        if 0xD800 <= code <= 0xDFFF:
            return True
    return False


def _section_bars(section: dict) -> int:
    return int(section["end_bar"]) - int(section["start_bar"]) + 1


def _merge_section_into(target: dict, follower: dict) -> None:
    """把后一个段落并入前一个（保持 target 的 kind / label / pattern）。"""
    target_bars = _section_bars(target)
    follower_bars = _section_bars(follower)
    energy = (
        _number(target.get("energy"), 0.5) * target_bars
        + _number(follower.get("energy"), 0.5) * follower_bars
    ) / max(1, target_bars + follower_bars)
    target["end_bar"] = max(int(target["end_bar"]), int(follower["end_bar"]))
    target["end_sec"] = max(_number(target.get("end_sec")), _number(follower.get("end_sec")))
    target["energy"] = round(energy, 3)
    if follower.get("repeat_of"):
        target["repeat_of"] = follower["repeat_of"]



def _normalize_sections(
    sections: list[dict],
    meta_for_pattern: dict,
    warnings: list[dict],
) -> list[dict]:
    """段落结构兜底：合并同 kind、修正顺序、控制数量、重新编号。

    上游段落切分常常把一段副歌切成 2~3 段，或让副歌出现在主歌之前。人类看的
    弹唱谱必须结构清晰，所以这里做一遍「无损」重整（只合并/改名，不编造段落）。
    """
    if not sections:
        return sections
    sections = [item for item in sections if _section_bars(item) >= 1]
    sections.sort(key=lambda item: (item["start_bar"], item["end_bar"]))
    merged_count = 0

    # 1) 修正重叠：后一段不得压在前一段上
    kept: list[dict] = []
    for item in sections:
        if kept and int(item["start_bar"]) <= int(kept[-1]["end_bar"]):
            item["start_bar"] = int(kept[-1]["end_bar"]) + 1
        if kept and int(item["start_bar"]) > int(item["end_bar"]):
            continue
        kept.append(item)
    sections = kept

    # 2) 合并相邻同 kind 段落（截断点相同才合并，避免丢段落）
    index = 1
    while index < len(sections):
        previous = sections[index - 1]
        current = sections[index]
        if previous["kind"] == current["kind"]:
            _merge_section_into(previous, current)
            sections.pop(index)
            merged_count += 1
            continue
        index += 1

    # 3) 顺序兜底：第一个非 intro 段落不能是副歌（流行歌惯例）
    changed = 0
    for item in sections:
        if item["kind"] == "intro":
            continue
        if item["kind"] == "chorus":
            item["kind"] = "verse"
            changed += 1
        break
    # 重新合并（改 kind 后可能与前一段同 kind）
    index = 1
    while index < len(sections):
        previous = sections[index - 1]
        current = sections[index]
        if previous["kind"] == current["kind"]:
            _merge_section_into(previous, current)
            sections.pop(index)
            merged_count += 1
            continue
        index += 1

    # 4) 段落数 >10 时合并最短的相邻对（优先同 kind）
    while len(sections) > 10:
        best_index = None
        best_score = None
        for position in range(len(sections) - 1):
            left = sections[position]
            right = sections[position + 1]
            if left["kind"] in ("intro", "outro") or right["kind"] in ("intro", "outro"):
                continue
            length = _section_bars(left) + _section_bars(right)
            score = (0 if left["kind"] == right["kind"] else 1, length)
            if best_score is None or score < best_score:
                best_score = score
                best_index = position
        if best_index is None:
            break
        left = sections[best_index]
        right = sections[best_index + 1]
        if left["kind"] != right["kind"]:
            # 异 kind 合并：保留较长那段（听众感知以主段为准）
            primary, follower = (left, right) if _section_bars(left) >= _section_bars(right) else (right, left)
            merged = primary
            merged["start_bar"] = min(int(left["start_bar"]), int(right["start_bar"]))
            merged["end_bar"] = max(int(left["end_bar"]), int(right["end_bar"]))
            merged["end_sec"] = max(_number(left.get("end_sec")), _number(right.get("end_sec")))
            merged["energy"] = round((_number(left.get("energy"), 0.5) + _number(right.get("energy"), 0.5)) / 2, 3)
            sections[best_index] = merged
        else:
            _merge_section_into(left, right)
        sections.pop(best_index + 1)
        merged_count += 1

    # 5) 重新编号 + 中文 label（同名段落编号，如「主歌 1/2」）
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for item in sections:
        totals[item["kind"]] = totals.get(item["kind"], 0) + 1
    for order, item in enumerate(sections):
        kind = item["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        base = _SECTION_LABELS.get(kind, _SECTION_LABELS["other"])
        item["label"] = f"{base} {counts[kind]}" if totals.get(kind, 1) > 1 else base
        item["id"] = f"s{order + 1}"
        item["pattern_id"] = strum_lib.pick_pattern(
            {"kind": kind, "label": item["label"], "energy": _number(item.get("energy"), 0.5)},
            meta_for_pattern,
        )

    if merged_count:
        warnings.append(
            {
                "level": "info",
                "code": "sections.merged",
                "message": f"已合并 {merged_count} 个相邻的重复段落，最终 {len(sections)} 段",
                "stage": "sections",
            }
        )
    if changed:
        warnings.append(
            {
                "level": "info",
                "code": "sections.kind_fixed",
                "message": "首个非前奏段落原为副歌，已按流行歌惯例改为主歌",
                "stage": "sections",
            }
        )
    return sections


def _map_sections(
    analysis: dict,
    bars: list[dict],
    grid: _BeatGrid,
    meta_for_pattern: dict,
    warnings: list[dict],
) -> list[dict]:
    """把 analysis.sections（秒）映射到小节范围，并挑节奏型。"""
    raw = [dict(item) for item in (analysis.get("sections") or []) if item.get("kind")]
    raw = _sec_sort(raw)
    if not raw:
        warnings.append(
            {
                "level": "warn",
                "code": "sections.missing",
                "message": "analysis 未提供段落信息，整曲按「其他」处理",
                "stage": "sections",
            }
        )
        raw = [{"start": bars[0]["start_sec"], "end": bars[-1]["end_sec"], "kind": "other"}]

    sections: list[dict] = []
    for order, item in enumerate(raw):
        start_sec = _number(item.get("start"))
        end_sec = _number(item.get("end"), start_sec)
        start_index = _bar_at(bars, start_sec, grid)
        end_index = _bar_at(bars, max(start_sec, end_sec - 1e-4), grid)
        if end_index < start_index:
            end_index = start_index
        kind = str(item.get("kind") or "other").lower()
        if kind not in _SECTION_LABELS:
            kind = "other"
        label = str(item.get("label") or _SECTION_LABELS[kind])
        if _looks_mojibake(label):
            label = _SECTION_LABELS[kind]
        section_meta = dict(meta_for_pattern)
        section_meta["energy"] = item.get("energy")
        pattern_id = strum_lib.pick_pattern(
            {
                "kind": kind,
                "label": label,
                "energy": _number(item.get("energy"), 0.5),
            },
            section_meta,
        )
        sections.append(
            {
                "id": f"s{order + 1}",
                "label": label,
                "kind": kind,
                "start_bar": bars[start_index]["index"],
                "end_bar": bars[end_index]["index"],
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "repeat_of": item.get("repeat_of") if isinstance(item.get("repeat_of"), str) else None,
                "pattern_id": pattern_id,
                "energy": round(_number(item.get("energy"), 0.5), 3),
            }
        )

    # 补齐空洞：某小节没有段落时并入上一段（首段之前并入首段）
    _fill_section_gaps(sections, bars, warnings)
    # 段落结构兜底：合并同 kind、修正顺序、控制数量、重新编号
    sections = _normalize_sections(sections, meta_for_pattern, warnings)
    _fill_section_gaps(sections, bars, warnings)
    return sections


def _fill_section_gaps(sections: list[dict], bars: list[dict], warnings: list[dict]) -> None:
    """把没有被段落覆盖的小节并入相邻段落（原地修改）。"""
    if not sections:
        return
    filled = 0
    for bar in bars:
        bar_number = bar["index"]
        covered = any(section["start_bar"] <= bar_number <= section["end_bar"] for section in sections)
        if covered:
            continue
        filled += 1
        target = None
        for section in sections:
            if section["start_bar"] <= bar_number:
                target = section
        target = target or sections[0]
        target["start_bar"] = min(target["start_bar"], bar_number)
        target["end_bar"] = max(target["end_bar"], bar_number)
    sections.sort(key=lambda item: item["start_bar"])
    if filled:
        warnings.append(
            {
                "level": "info",
                "code": "sections.gap_filled",
                "message": f"{filled} 个小节没有段落信息，已并入相邻段落",
                "stage": "sections",
            }
        )


def _active_chord_at(chords: Sequence[dict], beat: float) -> dict | None:
    """返回在 ``beat`` 处生效的和弦事件（起点最晚且未结束者）。"""
    active: dict | None = None
    for chord in chords:
        start = _number(chord.get("beat"))
        if start > beat + 1e-6:
            continue
        try:
            dur = duration_value(chord.get("dur"))
        except ValueError:  # pragma: no cover - 防御
            dur = 0.0
        if dur > 0 and start + dur <= beat + 1e-6:
            continue
        active = chord
    return active


def _section_for_bar(sections: Sequence[dict], bar_number: int) -> dict | None:
    for section in sections:
        if section["start_bar"] <= bar_number <= section["end_bar"]:
            return section
    return None


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def arrange(analysis: dict, options: dict | None = None) -> dict:
    """把分析证据编译成弹唱谱（``aitsm.score/v1``）。

    参数
    ----
    analysis: ``aitsm.analysis/v1`` 字典（秒 / 绝对时间轴）。
    options: 见 :data:`DEFAULT_OPTIONS`；缺失项用默认值，未知项忽略。

    返回
    ----
    符合 ``schemas/score.schema.json`` 的 score 字典。
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    warnings: list[dict] = []

    if not isinstance(analysis, dict) or not analysis:
        raise ValueError("analysis 不能为空")
    schema = str(analysis.get("schema") or "")
    if schema and schema != "aitsm.analysis/v1":
        warnings.append(
            {
                "level": "warn",
                "code": "analysis.schema",
                "message": f"analysis.schema = {schema!r}，期望 aitsm.analysis/v1",
                "stage": "input",
            }
        )
    instrument = str(options.get("instrument") or "guitar").lower()
    if instrument != "guitar":
        warnings.append(
            {
                "level": "warn",
                "code": "options.instrument",
                "message": f"暂只支持吉他弹唱，已按 guitar 处理（请求 {instrument}）",
                "stage": "input",
            }
        )

    audio = analysis.get("audio") or {}
    source = analysis.get("source") or {}
    duration = _number(audio.get("duration"), 0.0)
    beats_info = analysis.get("beats") or {}
    bpm = _number(beats_info.get("bpm_display") or beats_info.get("bpm"), 90.0)
    bpm = min(320.0, max(20.0, bpm or 90.0))
    grid = _BeatGrid([_number(value) for value in (beats_info.get("beats") or [])], duration, bpm)

    bars, ts, bpb, bar_len = _build_bars(analysis, grid, options, warnings)

    # ---- 调性 / 变调夹 ----
    key_info = analysis.get("key") or {}
    detected_mode = str(key_info.get("mode") or "major").lower()
    detected_mode = "minor" if detected_mode.startswith("min") else "major"
    try:
        detected_pc = note_to_pc(str(key_info.get("tonic") or "C"))
    except ValueError:
        detected_pc = 0
        warnings.append(
            {
                "level": "warn",
                "code": "key.invalid",
                "message": f"无法识别调性主音 {key_info.get('tonic')!r}，按 C 处理",
                "stage": "key",
            }
        )
    detected_name = f"{pc_to_name(detected_pc)} {detected_mode}"
    transpose = int(round(_number(options.get("transpose"), 0)))
    transpose = max(-12, min(12, transpose))
    sounding_pc = (detected_pc + transpose) % 12
    mode = detected_mode
    play_pc, capo = _choose_key(sounding_pc, mode, options, warnings)

    # 记谱移调量：发声音高 - capo（变调夹夹上去后，手按的音比实际发声低 capo 个半音）
    midi_delta = transpose - capo
    notation_pc = (detected_pc + midi_delta) % 12
    if notation_pc != play_pc:  # 理论上不会发生：以实际移调结果为准
        play_pc = notation_pc
    delta = midi_delta % 12
    if delta >= 6:
        delta -= 12
    fifths = key_fifths(play_pc, mode)
    prefer_flats = fifths < 0

    # ---- 和弦 / 旋律 ----
    difficulty = str(options.get("difficulty") or "intermediate").lower()
    if difficulty not in _DIFFICULTY_PREFER:
        difficulty = "intermediate"
    bar_chords, voicings, chord_conf = _map_chords(
        analysis,
        bars,
        grid,
        delta=delta,
        capo=capo,
        prefer_flats=prefer_flats,
        difficulty=difficulty,
        warnings=warnings,
    )
    bar_vocals, melody_stats = _map_melody(
        analysis, bars, grid, midi_delta=midi_delta, warnings=warnings
    )

    # ---- 段落 / 节奏型 ----
    meta_for_pattern = {
        "tempo": bpm,
        "time_signature": list(ts),
        "difficulty": difficulty,
        "style": str(options.get("style") or "pop"),
        "beats_per_bar": bpb,
    }
    sections = _map_sections(analysis, bars, grid, meta_for_pattern, warnings)

    # ---- 组装小节 ----
    include_tab = bool(options.get("include_tab", True))
    used_patterns: dict[str, dict] = {}
    score_bars: list[dict] = []
    for bar in bars:
        section = _section_for_bar(sections, bar["index"])
        pattern_id = section.get("pattern_id") if section else None
        if not pattern_id:
            pattern_id = strum_lib.pick_pattern({"kind": "verse", "energy": 0.45}, meta_for_pattern)
        pattern = strum_lib.get_pattern(pattern_id)
        if pattern is not None:
            used_patterns[pattern_id] = pattern
        chords = bar_chords.get(bar["index"], [])
        guitar: dict[str, Any] = {"pattern_id": pattern_id}
        if include_tab and chords and pattern is not None:
            hits: list[dict] = []
            for chord in chords:
                frets = None
                voicing_id = chord.get("voicing_id")
                if voicing_id and voicing_id in voicings:
                    frets = voicings[voicing_id].get("frets")
                expanded = strum_lib.expand_pattern(pattern_id, frets, include_strings=True)
                try:
                    chord_dur = duration_value(chord.get("dur"))
                except ValueError:
                    chord_dur = bar["_bar_len"]
                for hit in expanded:
                    if _number(hit.get("beat")) >= bar["_bar_len"] - _EPS:
                        continue
                    relative = {**hit, "beat": round(_number(hit.get("beat")) + _number(chord.get("beat")), 4)}
                    if voicing_id and "voicing_id" not in relative:
                        relative["voicing_id"] = voicing_id
                    # 超出该和弦时长的击发丢弃（和弦提前切换）
                    offset = relative["beat"] - _number(chord.get("beat"))
                    if offset >= chord_dur - _EPS and offset > 0:
                        continue
                    hits.append(relative)
            hits.sort(key=lambda item: _number(item.get("beat")))
            # 断言：每个击发用的指法必须与该 beat 生效和弦一致（防止串指法）
            checked: list[dict] = []
            for hit in hits:
                active = _active_chord_at(chords, _number(hit.get("beat")))
                active_id = active.get("voicing_id") if active else None
                hit_id = hit.get("voicing_id")
                if active_id and hit_id and hit_id != active_id:
                    warnings.append(
                        {
                            "level": "info",
                            "code": "guitar.voicing_mismatch",
                            "message": f"第 {bar['index']} 小节 beat={hit['beat']:g} 的扫弦指法与当前和弦不一致，已丢弃",
                            "stage": "guitar",
                            "target": f"bar:{bar['index']}",
                        }
                    )
                    continue
                checked.append(hit)
            if checked:
                guitar["hits"] = checked

        score_bar: dict[str, Any] = {
            "index": bar["index"],
            "start_sec": round(bar["start_sec"], 4),
            "end_sec": round(bar["end_sec"], 4),
        }
        if section is not None:
            score_bar["section"] = section["id"]
        if bar.get("pickup"):
            score_bar["pickup"] = True
        if bar["index"] == 1 or bar.get("time_signature"):
            score_bar["time_signature"] = list(ts)
        if abs(_number(bar.get("tempo"), bpm) - bpm) > 0.5:
            score_bar["tempo"] = round(_number(bar.get("tempo"), bpm), 3)
        if chords:
            score_bar["chords"] = chords
        if bar_vocals.get(bar["index"]):
            score_bar["vocal"] = bar_vocals[bar["index"]]
        score_bar["guitar"] = guitar
        score_bars.append(score_bar)

    # ---- 质量 ----
    beats_conf = _number(beats_info.get("confidence"), 0.0)
    key_conf = _number(key_info.get("confidence"), 0.0)
    lyric_lines = analysis.get("lyrics") or []
    lyrics_conf = (
        sum(_number(item.get("confidence"), 0.5) for item in lyric_lines) / len(lyric_lines)
        if lyric_lines
        else 0.0
    )
    stages = {
        "beats": round(beats_conf, 3),
        "key": round(key_conf, 3),
        "chords": round(chord_conf, 3),
        "melody": round(melody_stats.get("melody", 0.0), 3),
        "lyrics": round(lyrics_conf, 3),
        "lyric_match": round(melody_stats.get("lyric_match", 0.0), 3),
    }
    _quality_warnings(analysis, bars, score_bars, warnings, stages)
    assignable = [value for key_name, value in stages.items() if key_name != "lyric_match"]
    overall = round(sum(assignable) / len(assignable), 3) if assignable else 0.0

    title = str(source.get("original_name") or "未命名").rsplit(".", 1)[0]
    if not title:
        title = "未命名"
    score: dict[str, Any] = {
        "schema": "aitsm.score/v1",
        "meta": {
            "title": f"{title} · 弹唱谱",
            "artist": str(analysis.get("artist") or ""),
            "source": {
                "file": str(source.get("path") or ""),
                "original_name": str(source.get("original_name") or ""),
                "duration_sec": round(duration, 3),
                "sha256": str(source.get("sha256") or ""),
                "audio_url": str((analysis.get("audio") or {}).get("playback_audio") or ""),
            },
            "tempo": round(bpm, 3),
            "tempo_source": "beat-track" if beats_conf > 0 else "default",
            "time_signature": list(ts),
            "key": {
                "detected": detected_name,
                "sounding": pc_to_name_for_fifths(sounding_pc, fifths),
                "mode": mode,
                "play": pc_to_name_for_fifths(play_pc, fifths),
                "num_sharps_flats": fifths,
                "confidence": round(key_conf, 3),
            },
            "capo": capo,
            "guitar": {
                "strings": 6,
                "tuning": list(STANDARD_TUNING),
                "tuning_name": "standard",
            },
            "difficulty": difficulty,
            "style": str(options.get("style") or "pop"),
            "generator": {
                "version": SCORE_VERSION,
                "created_at": _now_iso(),
                "engines": {
                    "arranger": f"aitsm.score.arrange {SCORE_VERSION}",
                    "chords": str((analysis.get("tools") or {}).get("chords") or "analysis"),
                    "melody": str((analysis.get("tools") or {}).get("melody") or "analysis"),
                    "voicings": "aitsm.score.voicings",
                },
            },
            "warnings": [],
        },
        "sections": sections,
        "bars": score_bars,
        "voicings": voicings,
        "strum_patterns": used_patterns,
        "quality": {
            "overall": overall,
            "stages": stages,
            "warnings": warnings,
        },
    }
    # meta.warnings 保留 error 级（schema 允许两处都写）
    score["meta"]["warnings"] = [item for item in warnings if item.get("level") == "error"]
    return _json_safe(score)


def _quality_warnings(
    analysis: dict,
    bars: Sequence[dict],
    score_bars: Sequence[dict],
    warnings: list[dict],
    stages: dict,
) -> None:
    """把低置信度 / 降级说明汇总成 warnings。"""
    if stages.get("chords", 0) and stages["chords"] < 0.6:
        warnings.append(
            {
                "level": "warn",
                "code": "chords.low_confidence",
                "message": f"和弦平均置信度 {stages['chords']:.2f}，建议人工确认",
                "stage": "chords",
            }
        )
    if stages.get("melody", 0) and stages["melody"] < 0.5:
        warnings.append(
            {
                "level": "warn",
                "code": "melody.low_confidence",
                "message": f"旋律平均置信度 {stages['melody']:.2f}",
                "stage": "melody",
            }
        )
    if stages.get("lyrics", 0) and stages["lyrics"] < 0.5:
        warnings.append(
            {
                "level": "warn",
                "code": "lyrics.low_confidence",
                "message": f"歌词平均置信度 {stages['lyrics']:.2f}",
                "stage": "lyrics",
            }
        )
    # 旋律音域（记谱音高）检查
    out_of_range = 0
    for score_bar in score_bars:
        for note in score_bar.get("vocal") or []:
            midi = note.get("midi")
            if isinstance(midi, int) and not (48 <= midi <= 88):
                out_of_range += 1
    if out_of_range:
        warnings.append(
            {
                "level": "warn",
                "code": "melody.range",
                "message": f"{out_of_range} 个记谱音高超出建议音域 48-88（可能需整体八度调整）",
                "stage": "melody",
            }
        )
    # 没有旋律的小节
    empty_vocal = [bar["index"] for bar in score_bars if not bar.get("vocal")]
    if empty_vocal and len(empty_vocal) < len(score_bars):
        warnings.append(
            {
                "level": "info",
                "code": "melody.silent_bars",
                "message": f"{len(empty_vocal)} 个小节没有旋律（纯伴奏）",
                "stage": "melody",
            }
        )
    # 无和弦的小节
    empty_chord = [bar["index"] for bar in score_bars if not bar.get("chords")]
    if empty_chord:
        warnings.append(
            {
                "level": "warn",
                "code": "chords.missing_bars",
                "message": f"{len(empty_chord)} 个小节没有和弦：{empty_chord[:8]}",
                "stage": "chords",
            }
        )


def _json_safe(value: Any) -> Any:
    """递归把非 JSON 类型（如 numpy 标量）转成内建类型。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 6) if math.isfinite(float(value)) else 0.0
    if hasattr(value, "item"):  # numpy 标量
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):  # pragma: no cover
            return str(value)
    return _json_default(value) if callable(_json_default) else str(value)

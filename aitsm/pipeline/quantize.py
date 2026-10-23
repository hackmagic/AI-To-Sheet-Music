"""节奏量化：把演奏/演唱的真实时刻吸到「有音乐含义」的时值上（metrical HMM）。

为什么不能就地 ``round(t / 0.125)``
-----------------------------------
就近吸附有两个结构性缺陷，都会直接变成谱面噪声：

1. **相邻音会互相挤**。人唱的 onset 是「相对节拍有系统性滞后/超前」的，两个相邻音各自
   吸附会把 0.23 拍的间隔吸成 0.25，但把 0.30 吸成 0.25、把 0.27 吸成 0.5 —— 差异被随机
   放大，于是出现 1/6 拍、5/16 拍这类**没有对应音符形状**的时值（导出时只能被强行吸到
   最近的合法值，等于在谱面上制造错位）。
2. **丢失节拍重音先验**。落在正拍、半拍、十六分位置的证据强度完全不同；音乐里绝大多数
   起音发生在强位置，量化器应该主动偏向它们，而不是等距对待所有网格点。

做法
----
把量化建模成**在节拍网格上的最短路径问题**（Viterbi）：

- 状态：音符 i 的候选起音位置（原起音 ± ``max_shift_ratio`` × 拍长 内的网格点）；
- 发射代价：``偏移代价`` + ``节拍重音代价``（正拍最便宜，十六分位置最贵）；
- 转移代价：``时值不合法代价``（离最近的可记谱时值有多远）+ ``过短惩罚``；
- 多声部一致：旋律起音**落在和弦变化点上**时给少量奖励——伴奏换和弦的地方几乎总是
  旋律的乐句起点，这是两个声部之间唯一可靠的免费证据。

输出保证每个音符的时值都是可记谱值（1/16、1/8、附点八分、1/4、附点四分、二分、附点
二分、全音符），因此导出层不需要再做「强行吸附」。

公共契约::

    quantize_notes_hmm(notes, beats, *, harmony_onsets=None, until=None)
        -> (新音符列表, 统计 dict)

音符 dict 与 :func:`aitsm.pipeline.melody.quantize_notes` 完全一致：写 ``start`` /
``end``，并把原始时刻保留在 ``start_raw`` / ``end_raw``，便于保真复核。
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Sequence

from .melody import DEFAULT_HOP_LENGTH  # noqa: F401  （复用常量，保持采样域一致）
from .melody import build_beat_grid, snap_to_grid

LOGGER = logging.getLogger(__name__)

__all__ = [
    "quantize_notes_hmm",
    "metrical_strength",
    "legal_durations",
    "duration_cost",
    "LEGAL_DURATIONS",
    "DEFAULT_SUBDIVISION",
]

# 网格：每个四分音符 4 格（十六分音符）。三连音不在此列——它需要 tuplet 记谱，
# 而 MuseScore 对「混合时值的三连音组」算不准小节总长（实测 exit 1320），
# 所以旋律一律吸到二进制网格上（与导出层的节奏吸附一致）。
DEFAULT_SUBDIVISION = 4

# 可记谱时值（单位：四分音符）。附点值 = 1.5 × 基准值。
LEGAL_DURATIONS: tuple[float, ...] = (
    0.25,  # 十六分
    0.375,  # 附点十六分（罕见但可记谱）
    0.5,  # 八分
    0.75,  # 附点八分
    1.0,  # 四分
    1.5,  # 附点四分
    2.0,  # 二分
    3.0,  # 附点二分
    4.0,  # 全音符
)

# 代价权重（经验值，均由 A/B 得出：见 docs/10-rhythm-quantize.md）
_SHIFT_WEIGHT = 1.0  # 每偏移一拍的代价
_METRE_COST = (0.0, 0.05, 0.12, 0.22)  # 按重音等级：正拍/拍/半拍/十六分
_DURATION_WEIGHT = 0.55  # 时值不合法代价
_TOO_SHORT_PENALTY = 0.35  # 时值短于十六分
_HARMONY_BONUS = 0.12  # 起音与和弦变化点重合
_HARMONY_TOLERANCE = 0.045  # 秒
_REST_GAP_RATIO = 0.5  # 原始间隔超过半个拍就视为休止（不强制连奏）
_MIN_DURATION = 0.25  # 最短时值（四分音符计）


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _env_switch(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def legal_durations() -> tuple[float, ...]:
    """可记谱时值（四分音符计），供测试与其他模块复用。"""
    return LEGAL_DURATIONS


# 合法「网格步数」（1 步 = 1/4 拍 = 十六分）：1/2/3/4/6/8/12/16 步对应
# 0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 / 4 拍，覆盖十六分到全音符含附点。
# 休止后的音符按步数吸时值，保证经过小节映射后仍是可记谱值。
_LEGAL_STEPS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)


def metrical_strength(position_quarters: float, bar_len: float = 4.0) -> int:
    """位置（相对小节线，单位四分音符）的节拍重音等级：3=正拍，2=拍，1=半拍，0=十六分。"""
    if bar_len <= 0:
        bar_len = 4.0
    position = float(position_quarters) % float(bar_len)
    if abs(position) < 1e-6 or abs(position - bar_len) < 1e-6:
        return 3
    if abs(position - round(position)) < 1e-6:
        return 2
    if abs(position * 2.0 - round(position * 2.0)) < 1e-6:
        return 1
    return 0


def duration_cost(duration_quarters: float) -> float:
    """时值离「最近的可记谱值」有多远（相对误差），短于十六分额外惩罚。"""
    value = float(duration_quarters)
    if value <= 0:
        return float("inf")
    best = min(abs(value - legal) / legal for legal in LEGAL_DURATIONS)
    cost = _DURATION_WEIGHT * best
    if value < _MIN_DURATION - 1e-9:
        cost += _TOO_SHORT_PENALTY
    return cost


def _nearest_legal_duration(value: float) -> float:
    return min(LEGAL_DURATIONS, key=lambda legal: abs(value - legal))


def _beat_period(beats: dict[str, Any] | None) -> float:
    bpm = 0.0
    if isinstance(beats, dict):
        try:
            bpm = float(beats.get("bpm") or 0.0)
        except (TypeError, ValueError):
            bpm = 0.0
    if bpm <= 0:
        return 0.5
    return 60.0 / bpm


def _bar_len(beats: dict[str, Any] | None) -> float:
    if isinstance(beats, dict):
        try:
            value = float(beats.get("beats_per_bar") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 4.0


def _candidates(
    raw_start: float,
    grid: Sequence[float],
    tolerance: float,
    *,
    limit: int,
) -> list[float]:
    """原起音附近（±tolerance）的网格点，按距离排序取前 limit 个。"""
    if not grid:
        return [raw_start]
    lo, hi = raw_start - tolerance, raw_start + tolerance
    nearby = [point for point in grid if lo - 1e-9 <= point <= hi + 1e-9]
    if not nearby:
        return [raw_start]
    nearby.sort(key=lambda point: abs(point - raw_start))
    return sorted(nearby[: max(1, int(limit))])


def _harmony_hits(times: Sequence[float], harmony_onsets: Sequence[float], tolerance: float) -> set[int]:
    """哪些候选时刻同时是和弦变化点（用集合返回，避免每步都线性扫）。"""
    hits: set[int] = set()
    if not harmony_onsets:
        return hits
    ordered = sorted(float(item) for item in harmony_onsets)
    for index, time_value in enumerate(times):
        nearest = min(ordered, key=lambda anchor: abs(anchor - time_value))
        if abs(nearest - time_value) <= tolerance:
            hits.add(index)
    return hits


def quantize_notes_hmm(
    notes: Sequence[dict],
    beats: dict[str, Any] | None,
    *,
    harmony_onsets: Sequence[float] | None = None,
    until: float | None = None,
    subdivision: int = DEFAULT_SUBDIVISION,
    max_shift_ratio: float | None = None,
    candidate_limit: int = 5,
    min_duration: float = 0.09,
) -> tuple[list[dict], dict]:
    """用 metrical HMM 量化音符起止，返回 (新音符列表, 统计)。

    参数:
        notes: 音符列表（读 ``start`` / ``end``，其余字段原样透传）。
        beats: analyze_beats 的输出（``bpm`` / ``beats`` / ``beats_per_bar`` / ``downbeats``）。
        harmony_onsets: 和弦变化时刻（秒）。给了就作为「乐句起点」的免费证据。
        until: 音频时长，用于网格右边界。
        subdivision: 每个四分音符的网格数（4 = 十六分）。
        max_shift_ratio: 允许的最大偏移（拍长倍数），默认读环境变量
            ``AITSM_QUANTIZE_MAX_SHIFT``（0.4）。
        candidate_limit: 每个音符最多保留几个候选网格点（控制 DP 规模）。
        min_duration: 最短音符（秒），短于此的音符视为切分碎片。

    返回的每个音符都有 ``start`` / ``end``（量化后）与 ``start_raw`` / ``end_raw``（原始）。
    """
    stats: dict[str, Any] = {
        "engine": "metrical-hmm",
        "grid_points": 0,
        "notes": len(notes),
        "snapped": 0,
        "kept_raw": 0,
        "mean_shift_ms": 0.0,
        "max_shift_ms": 0.0,
        "legal_duration_ratio": 1.0,
        "harmony_anchored": 0,
        "rests": 0,
    }
    if not notes:
        stats["legal_duration_ratio"] = 0.0
        return [], stats

    period = _beat_period(beats)
    bar_len = _bar_len(beats)
    shift_ratio = (
        _env_float("AITSM_QUANTIZE_MAX_SHIFT", 0.40) if max_shift_ratio is None else float(max_shift_ratio)
    )
    tolerance = shift_ratio * period

    grid, grid_period = build_beat_grid(beats, subdivision=max(1, int(subdivision)), until=until)
    if not grid:
        # 没有节拍信息：原样返回，只补齐 raw 字段
        result: list[dict] = []
        for note in notes:
            raw_start = float(note["start"])
            raw_end = float(note.get("end", raw_start + min_duration))
            item = dict(note)
            item["start_raw"] = round(raw_start, 3)
            item["end_raw"] = round(raw_end, 3)
            item.setdefault("onset", round(raw_start, 3))
            result.append(item)
            stats["kept_raw"] += 1
        stats["grid_points"] = 0
        return result, stats
    if grid_period > 0:
        period = float(grid_period)
    stats["grid_points"] = len(grid)

    ordered = sorted(notes, key=lambda note: (float(note["start"]), float(note.get("end", 0.0))))
    raw_starts = [float(note["start"]) for note in ordered]
    raw_ends = [float(note.get("end", raw_starts[index] + min_duration)) for index, note in enumerate(ordered)]

    # 候选时刻 + 每候选的发射代价（偏移 + 节拍重音 + 和弦锚点奖励）
    downbeats = []
    if isinstance(beats, dict):
        downbeats = sorted(
            float(item) for item in (beats.get("downbeats") or []) if isinstance(item, (int, float))
        )
    candidate_lists: list[list[float]] = []
    emission_lists: list[list[float]] = []
    for raw_start in raw_starts:
        candidates = _candidates(raw_start, grid, tolerance, limit=candidate_limit)
        hits = _harmony_hits(candidates, list(harmony_onsets or []), _HARMONY_TOLERANCE)
        emissions: list[float] = []
        for index, candidate in enumerate(candidates):
            offset = abs(candidate - raw_start) / period
            if downbeats:
                nearest_bar = min(downbeats, key=lambda anchor: abs(anchor - candidate))
                position = (candidate - nearest_bar) / period
            else:
                position = candidate / period
            strength = metrical_strength(position, bar_len)
            cost = _SHIFT_WEIGHT * offset + _METRE_COST[3 - strength]
            if index in hits:
                cost -= _HARMONY_BONUS
                stats["harmony_anchored"] += 1
            emissions.append(cost)
        candidate_lists.append(candidates)
        emission_lists.append(emissions)

    # Viterbi：dp[i][k] = 处理到第 i 个音符、它落在第 k 个候选时的最小代价
    count = len(ordered)
    dp: list[list[float]] = []
    back: list[list[int]] = []
    dp.append(list(emission_lists[0]))
    back.append([-1] * len(candidate_lists[0]))
    for index in range(1, count):
        previous_times = candidate_lists[index - 1]
        previous_costs = dp[index - 1]
        current_lists = candidate_lists[index]
        row: list[float] = []
        row_back: list[int] = []
        for position, candidate in enumerate(current_lists):
            best_cost = float("inf")
            best_previous = -1
            for previous_index, previous_time in enumerate(previous_times):
                duration = candidate - previous_time
                if duration <= 1e-6:
                    continue
                gap = max(0.0, raw_starts[index] - raw_ends[index - 1])
                transition = duration_cost(duration / period)
                if gap <= _REST_GAP_RATIO * period:
                    # 连奏：时值由后面那个起音决定，不希望它太短
                    if duration < min_duration:
                        transition += 0.2
                total = previous_costs[previous_index] + transition
                if total < best_cost - 1e-12:
                    best_cost = total
                    best_previous = previous_index
            if best_previous < 0:
                # 所有前驱都非法（时间倒退）：退回最近候选，保证不丢音
                best_cost = min(previous_costs) + 0.5
                best_previous = min(
                    range(len(previous_costs)), key=lambda item: previous_costs[item]
                )
            row.append(best_cost + emission_lists[index][position])
            row_back.append(best_previous)
        dp.append(row)
        back.append(row_back)

    # 回溯
    chosen: list[int] = [0] * count
    chosen[count - 1] = min(range(len(dp[count - 1])), key=lambda item: dp[count - 1][item])
    for index in range(count - 1, 0, -1):
        chosen[index - 1] = back[index][chosen[index]]

    starts = [candidate_lists[index][chosen[index]] for index in range(count)]
    grid_index = {round(point, 6): index for index, point in enumerate(grid)}

    # 结束时间：有休止的按原始结束吸附且取可记谱值；连奏的交给下一个起音
    result = []
    shifts: list[float] = []
    legal_count = 0
    for index, note in enumerate(ordered):
        start = starts[index]
        raw_start = raw_starts[index]
        raw_end = raw_ends[index]
        next_start = starts[index + 1] if index + 1 < count else None
        next_raw_start = raw_starts[index + 1] if index + 1 < count else None
        gap = None
        if next_raw_start is not None:
            gap = max(0.0, next_raw_start - raw_end)
        if next_start is not None and (gap is None or gap <= _REST_GAP_RATIO * period):
            end = next_start
        else:
            end = snap_to_grid(raw_end, grid, tolerance)
            if next_start is not None:
                end = min(end, next_start)
            # 休止后的音符：把时值吸到「合法步数」上，起音不动。
            # 注意必须按网格步数吸（1 步 = 1/4 拍），不能用全局平均拍长换算秒：
            # 实际拍间隔逐拍有 ±8% 波动（0.60~0.65 s），用全局拍长换算出来的
            # 「合法时值」再经小节映射会变成 0.138 拍这种不可记谱的值。
            start_index = grid_index.get(round(start, 6))
            stepped = None
            if start_index is not None:
                shortest = max(1, int(round(min_duration / max(period / 4.0, 1e-6))))
                for steps in _LEGAL_STEPS:
                    if steps < shortest:
                        continue
                    target = start_index + steps
                    if target >= len(grid):
                        continue
                    candidate_end = grid[target]
                    if next_start is not None and candidate_end > next_start + 1e-6:
                        continue
                    distance = abs(candidate_end - raw_end)
                    if stepped is None or distance < stepped[0]:
                        stepped = (distance, candidate_end)
            if stepped is not None:
                end = stepped[1]
                stats["rests"] += 1
        item = dict(note)
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        item["start_raw"] = round(raw_start, 3)
        item["end_raw"] = round(raw_end, 3)
        item.setdefault("onset", round(raw_start, 3))
        result.append(item)
        shift = abs(start - raw_start)
        shifts.append(shift)
        if abs(start - raw_start) > 1e-6 or abs(end - raw_end) > 1e-6:
            stats["snapped"] += 1
        else:
            stats["kept_raw"] += 1
        duration_quarters = (end - start) / period
        if duration_cost(duration_quarters) <= _DURATION_WEIGHT * 1e-6 + 1e-9:
            legal_count += 1

    if shifts:
        stats["mean_shift_ms"] = round(1000.0 * sum(shifts) / len(shifts), 1)
        stats["max_shift_ms"] = round(1000.0 * max(shifts), 1)
    stats["legal_duration_ratio"] = round(legal_count / max(1, len(result)), 3)
    LOGGER.info(
        "节奏量化：notes=%d 偏移均值=%.0fms 合法时值=%.0f%% 和弦锚点=%d",
        len(result),
        stats["mean_shift_ms"],
        stats["legal_duration_ratio"] * 100.0,
        stats["harmony_anchored"],
    )
    return result, stats


def quantizer_enabled() -> bool:
    """是否启用 HMM 量化（``AITSM_RHYTHM_QUANTIZER=grid`` 可退回旧的就近吸附）。"""
    raw = os.environ.get("AITSM_RHYTHM_QUANTIZER", "hmm")
    return str(raw).strip().lower() not in {"grid", "snap", "legacy", "0", "off", "false"}

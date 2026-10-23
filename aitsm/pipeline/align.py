"""歌词-旋律对齐（F9 × F10 交叉层）：把音节挂到旋律音符上。

对齐策略（纯算法，可离线单测）：
1. 歌词行 -> 音节序列（复用 lyrics.line_syllables 的词级时间戳/均分逻辑）；
2. 音符 -> 歌词行：按时间重叠最大者归属，弱起/越界的音符允许半个 beat 的容差；
3. 行内用动态规划做单调对齐，允许四种动作：
   - 单音节 -> 单音符（常规）；
   - 单音节 -> 多音符（一字多音 / melisma）：首音挂歌词，后续音 ``lyric=None``
     并用 ``tie`` 串联（``start`` -> ``continue`` -> ``stop``）；
   - 多音节 -> 单音符（一音多字 / lyric_extend）：音符 ``lyric`` 用连字符连接，
     并标 ``lyric_extend=True``；
   - 跳过音节 / 跳过音符（惩罚项，避免硬凑）。
   代价函数综合“起音时间差 + 结束时间差 - 重叠奖励”，保证结果贴近真实演唱时刻。

保真纪律：宁可留 ``lyric=None`` 也不把歌词硬塞到不相干的音符上；所有未匹配音节
写入 alignment.unmatched 与 warnings 供上层校对面板展示。
"""

from __future__ import annotations

import copy
import logging
import math
from bisect import bisect_left
from typing import Any, Iterable, Sequence

from .lyrics import line_syllables

LOGGER = logging.getLogger(__name__)

__all__ = [
    "align_lyrics_to_melody",
    "align_syllables_to_notes",
    "syllables_from_lyrics",
    "assign_notes_to_lines",
    "apply_alignment",
    "MAX_MELISMA",
    "MAX_SHARE",
]

MAX_MELISMA = 4  # 一个音节最多覆盖几个音符（一字多音）
MAX_SHARE = 3  # 一个音符最多容纳几个音节（一音多字）

_MATCH_START_WEIGHT = 1.2  # 起音时间差权重（最重要）
_MATCH_END_WEIGHT = 0.25  # 结束时间差权重
_MATCH_OVERLAP_WEIGHT = 0.9  # 时间重叠奖励
_SKIP_SYLLABLE_COST = 0.7  # 丢弃一个音节的代价
_SKIP_NOTE_COST = 0.4  # 一个音符不挂歌词的代价
_MELISMA_PENALTY = 0.25  # 一字多音每多一个音的代价（略低于丢弃音符，优先保留旋律）
_SHARE_PENALTY = 0.3  # 一音多字每多一个字的代价（略低于丢弃音节，优先保留歌词）

# --- 音乐约束（Whisper 时间戳只当「软锚点」，不是真值） --------------------------
# Whisper 是语音模型，对歌唱的时间戳有 0.2–1.0 s 的系统性误差，副歌重复段尤其容易
# 整段错位。如果按「时间差越小越对」直接信它，会把错位的歌词焊死。所以：
#  * 时间差超过 _ANCHOR_CAP 后不再增加代价（再远也不更差）——留给音乐结构去决定；
#  * 音符起音落在节拍上给一点奖励（歌唱的咬字几乎总在拍点/拍点上）；
#  * 跨越长休止（换气、间奏）的一字多音给惩罚。
_ANCHOR_CAP = 0.42  # 秒：超过这个偏差后时间证据饱和
_BEAT_BONUS = 0.10  # 起音落在拍点上的奖励
_BEAT_TOLERANCE_RATIO = 0.18  # 拍长比例：|起音-拍点| < 18% 拍长算「在拍上」
_REST_PENALTY = 0.30  # 一字多音跨过长休止的惩罚（高于「这个音不挂词」的 0.4 的一半以上：
#                      谱面上把 a 拖过 2 秒间奏是明显错误，宁可不给那个音挂词）
_REST_THRESHOLD_RATIO = 0.75  # 相邻音间隔超过 0.75 拍算「长休止」
_INF = float("inf")


# ---------------------------------------------------------------------------
# 基础工具
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


def _note_start(note: dict) -> float:
    start = _as_float(note.get("start"))
    if start is None:
        onset = _as_float(note.get("onset"))
        start = onset if onset is not None else 0.0
    return max(0.0, start)


def _note_end(note: dict) -> float:
    duration = _as_float(note.get("dur"))
    start = _note_start(note)
    end = _as_float(note.get("end"))
    if end is None or end <= start:
        end = start + (duration if duration and duration > 0 else 0.1)
    return end


def _syllable_start(syllable: dict) -> float:
    start = _as_float(syllable.get("start"))
    return 0.0 if start is None else max(0.0, start)


def _syllable_end(syllable: dict) -> float:
    start = _syllable_start(syllable)
    end = _as_float(syllable.get("end"))
    if end is None or end <= start:
        end = start + 0.05
    return end


def _beat_period(beats: dict | None) -> float:
    """从 beats dict 推一个 beat 长度（秒），用于容差；不可用时默认 0.5s。"""
    if not isinstance(beats, dict):
        return 0.5
    raw = sorted(
        float(value)
        for value in (beats.get("beats") or [])
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0
    )
    if len(raw) >= 2:
        gaps = [later - earlier for earlier, later in zip(raw, raw[1:]) if later - earlier > 1e-4]
        if gaps:
            return max(0.05, float(sorted(gaps)[len(gaps) // 2]))
    bpm = _as_float(beats.get("bpm"))
    if bpm and bpm > 0:
        return max(0.05, 60.0 / bpm)
    return 0.5


def _match_cost(
    syllable_start: float,
    syllable_end: float,
    note_start: float,
    note_end: float,
    *,
    beat_bonus: float = 0.0,
    rest_count: int = 0,
) -> float:
    """音节与（可能跨多个音的）音符跨度之间的匹配代价，越小越好。

    时间证据是**饱和**的：偏差超过 ``_ANCHOR_CAP`` 后代价不再增长。这条规则把 Whisper
    的时间戳从「真值」降级为「软锚点」——当它整段错位（副歌重复、清唱段落）时，DP 不会
    为了凑时间把歌词塞到错误的音符上，而是交给节拍与休止约束去决定。

    ``beat_bonus`` 与 ``rest_count`` 由调用方按音符预先算好（避免在 DP 内层重复算）。
    """
    start_diff = min(abs(note_start - syllable_start), _ANCHOR_CAP)
    end_diff = min(abs(note_end - syllable_end), _ANCHOR_CAP)
    overlap = max(0.0, min(syllable_end, note_end) - max(syllable_start, note_start))
    duration = max(syllable_end - syllable_start, 1e-3)
    overlap_ratio = min(1.0, overlap / duration)
    return (
        _MATCH_START_WEIGHT * start_diff
        + _MATCH_END_WEIGHT * end_diff
        - _MATCH_OVERLAP_WEIGHT * overlap_ratio
        + beat_bonus
        + _REST_PENALTY * min(2, int(rest_count))
    )


# ---------------------------------------------------------------------------
# 歌词行 -> 音节
# ---------------------------------------------------------------------------


def syllables_from_lyrics(lyrics: Sequence[dict] | None) -> tuple[list[dict], list[dict]]:
    """把歌词行展开成音节流，并返回有效行列表。

    返回 (syllables, lines)：syllables 每项带 ``line``（行下标）与 ``text/start/end``，
    lines 是清洗后仍可用的行（保持输入顺序）。
    """
    syllables: list[dict] = []
    lines: list[dict] = []
    for line in lyrics or []:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        line_start = _as_float(line.get("start"))
        line_end = _as_float(line.get("end"))
        normalized = dict(line)
        normalized["start"] = 0.0 if line_start is None else max(0.0, line_start)
        normalized["end"] = (
            normalized["start"] + 0.3 * max(1, len(text))
            if line_end is None or line_end <= normalized["start"]
            else line_end
        )
        entries = line_syllables(normalized)
        if not entries:
            continue
        line_index = len(lines)
        for entry in entries:
            syllables.append(
                {
                    "text": str(entry.get("text") or "").strip(),
                    "start": _syllable_start(entry),
                    "end": _syllable_end(entry),
                    "line": line_index,
                }
            )
        lines.append(normalized)
    return syllables, lines


# ---------------------------------------------------------------------------
# 音符 -> 歌词行
# ---------------------------------------------------------------------------


def assign_notes_to_lines(
    notes: Sequence[dict],
    lines: Sequence[dict],
    *,
    tolerance: float = 0.25,
) -> list[list[int]]:
    """把音符按时间重叠归属到歌词行；返回每行持有的音符下标列表。"""
    groups: list[list[int]] = [[] for _ in lines]
    if not lines:
        return groups
    windows = [(_syllable_start(line), _syllable_end(line)) for line in lines]
    for note_index, note in enumerate(notes):
        start = _note_start(note)
        end = _note_end(note)
        overlap_line: int | None = None
        overlap_best = 0.0
        distance_line = 0
        distance_best = _INF
        for line_index, (line_start, line_end) in enumerate(windows):
            overlap = max(0.0, min(end, line_end) - max(start, line_start))
            if overlap > overlap_best + 1e-9:
                overlap_best = overlap
                overlap_line = line_index
            distance = 0.0
            if end < line_start:
                distance = line_start - end
            elif start > line_end:
                distance = start - line_end
            if distance < distance_best - 1e-9:
                distance_best = distance
                distance_line = line_index
        chosen: int | None
        if overlap_line is not None:
            chosen = overlap_line
        elif distance_best <= tolerance + 1e-9:
            chosen = distance_line
        else:
            chosen = None
        if chosen is None:
            continue
        groups[chosen].append(note_index)
    for group in groups:
        group.sort(key=lambda index: (_note_start(notes[index]), _note_end(notes[index])))
    return groups


# ---------------------------------------------------------------------------
# 行内 DP 对齐
# ---------------------------------------------------------------------------


def align_syllables_to_notes(
    syllables: Sequence[dict],
    notes: Sequence[dict],
    *,
    max_melisma: int = MAX_MELISMA,
    max_share: int = MAX_SHARE,
    beat_times: Sequence[float] | None = None,
    beat_period: float | None = None,
) -> list[tuple]:
    """单调 DP：把一行音节对齐到一组音符。

    ``beat_times`` / ``beat_period`` 给出节拍位置：音符起音落在拍点上会有奖励，
    跨越长休止的一字多音会被惩罚。不传则退回纯时间匹配（保持向后兼容）。

    返回动作列表（按时间顺序），每个元素是元组：
    - ``("syllable", i, j, q)``：音节 i 覆盖音符 j..j+q-1；
    - ``("share", i, p, j)``：音节 i..i+p-1 共用音符 j；
    - ``("skip_syllable", i)`` / ``("skip_note", j)``：未匹配。
    """
    syllable_count = len(syllables)
    note_count = len(notes)
    if syllable_count == 0 or note_count == 0:
        return [("skip_syllable", index) for index in range(syllable_count)] + [
            ("skip_note", index) for index in range(note_count)
        ]

    max_melisma = max(1, int(max_melisma))
    max_share = max(1, int(max_share))

    syllable_spans = [(_syllable_start(item), _syllable_end(item)) for item in syllables]
    note_spans = [(_note_start(item), _note_end(item)) for item in notes]

    # --- 音乐约束预计算 -----------------------------------------------------
    period = float(beat_period) if beat_period and beat_period > 0 else 0.5
    anchors = sorted(float(value) for value in (beat_times or []))
    tolerance = _BEAT_TOLERANCE_RATIO * period
    beat_bonus: list[float] = []
    for onset, _end in note_spans:
        if anchors and min(abs(onset - anchor) for anchor in anchors) <= tolerance:
            beat_bonus.append(-_BEAT_BONUS)
        else:
            beat_bonus.append(0.0)
    rest_threshold = _REST_THRESHOLD_RATIO * period
    gap_flags: list[int] = []
    for index in range(max(0, note_count - 1)):
        gap = max(0.0, note_spans[index + 1][0] - note_spans[index][1])
        gap_flags.append(1 if gap >= rest_threshold else 0)
    rest_prefix = [0] * (note_count + 1)
    for index, flag in enumerate(gap_flags):
        rest_prefix[index + 1] = rest_prefix[index] + flag
    # 注意 rest_prefix[k] 统计的是「前 k 个音符之间的间隔」，所以音符 j..j+q-1
    # 内部的休止数 = rest_prefix[j+q-1] - rest_prefix[j]

    def _rests_inside(start: int, span: int) -> int:
        last = start + span - 1
        return max(0, rest_prefix[last] - rest_prefix[start])

    # dp[i][j]：已处理 i 个音节、j 个音符的最小代价；back 记录前驱动作
    dp = [[_INF] * (note_count + 1) for _ in range(syllable_count + 1)]
    back: list[list[tuple | None]] = [[None] * (note_count + 1) for _ in range(syllable_count + 1)]
    dp[0][0] = 0.0

    for i in range(syllable_count + 1):
        for j in range(note_count + 1):
            current = dp[i][j]
            if current == _INF:
                continue
            # 跳过音节
            if i < syllable_count:
                cost = current + _SKIP_SYLLABLE_COST
                if cost < dp[i + 1][j]:
                    dp[i + 1][j] = cost
                    back[i + 1][j] = ("skip_syllable", i)
            # 跳过音符
            if j < note_count:
                cost = current + _SKIP_NOTE_COST
                if cost < dp[i][j + 1]:
                    dp[i][j + 1] = cost
                    back[i][j + 1] = ("skip_note", j)
            if i >= syllable_count or j >= note_count:
                continue
            syllable_start, syllable_end = syllable_spans[i]
            # 一字多音：音节 i 覆盖音符 j..j+q-1
            for q in range(1, max_melisma + 1):
                if j + q > note_count:
                    break
                note_start = note_spans[j][0]
                note_end = note_spans[j + q - 1][1]
                cost = (
                    current
                    + _match_cost(
                        syllable_start,
                        syllable_end,
                        note_start,
                        note_end,
                        beat_bonus=beat_bonus[j],
                        rest_count=_rests_inside(j, q),
                    )
                    + _MELISMA_PENALTY * (q - 1)
                )
                if cost < dp[i + 1][j + q]:
                    dp[i + 1][j + q] = cost
                    back[i + 1][j + q] = ("syllable", i, j, q)
            # 一音多字：音节 i..i+p-1 共用音符 j
            for p in range(2, max_share + 1):
                if i + p > syllable_count:
                    break
                span_start = syllable_spans[i][0]
                span_end = syllable_spans[i + p - 1][1]
                cost = (
                    current
                    + _match_cost(
                        span_start,
                        span_end,
                        note_spans[j][0],
                        note_spans[j][1],
                        beat_bonus=beat_bonus[j],
                    )
                    + _SHARE_PENALTY * (p - 1)
                )
                if cost < dp[i + p][j + 1]:
                    dp[i + p][j + 1] = cost
                    back[i + p][j + 1] = ("share", i, p, j)

    # 回溯
    ops: list[tuple] = []
    i, j = syllable_count, note_count
    while i > 0 or j > 0:
        action = back[i][j]
        if action is None:  # pragma: no cover - 理论上不可达
            if i > 0:
                ops.append(("skip_syllable", i - 1))
                i -= 1
            elif j > 0:
                ops.append(("skip_note", j - 1))
                j -= 1
            continue
        ops.append(action)
        kind = action[0]
        if kind == "skip_syllable":
            i -= 1
        elif kind == "skip_note":
            j -= 1
        elif kind == "syllable":
            i -= 1
            j -= action[3]
        else:  # share
            i -= action[2]
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# 应用对齐结果
# ---------------------------------------------------------------------------


def apply_alignment(
    notes: Sequence[dict],
    syllables: Sequence[dict],
    ops: Iterable[tuple],
) -> dict:
    """把 DP 动作写入音符 dict（就地修改），返回统计。"""
    stats = {"matched_syllables": 0, "melisma": 0, "shared": 0, "skipped_syllables": 0}
    for action in ops:
        kind = action[0]
        if kind == "skip_syllable":
            stats["skipped_syllables"] += 1
            continue
        if kind == "skip_note":
            continue
        if kind == "syllable":
            _, syllable_index, note_index, span = action
            note = notes[note_index]
            note["lyric"] = str(syllables[syllable_index]["text"])
            if span > 1:
                note["tie"] = "start"
                for offset in range(1, span):
                    target = notes[note_index + offset]
                    if offset == span - 1:
                        target["tie"] = "stop"
                    else:
                        target["tie"] = "continue"
                    target["lyric"] = None
                stats["melisma"] += 1
            stats["matched_syllables"] += 1
            continue
        # share：多音节共用同一音符
        _, syllable_index, count, note_index = action
        parts = [str(syllables[syllable_index + offset]["text"]) for offset in range(count)]
        note = notes[note_index]
        note["lyric"] = "-".join(parts)
        note["lyric_extend"] = True
        note["lyric_syllables"] = parts
        stats["shared"] += 1
        stats["matched_syllables"] += count
    return stats


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def align_lyrics_to_melody(melody: dict, lyrics: list[dict], beats: dict) -> dict:
    """把歌词音节挂到旋律音符上，返回新的 melody dict（不修改入参）。

    契约::

        align_lyrics_to_melody(melody, lyrics, beats) -> melody dict
        notes[].lyric 已填好；一字多音的后续音 lyric=None 且带 tie。

    副产物：``alignment``（统计 + 未匹配音节）与 ``warnings`` 追加说明。
    """
    result: dict = copy.deepcopy(melody) if isinstance(melody, dict) else {}
    raw_notes = result.get("notes")
    notes: list[dict] = [note for note in raw_notes if isinstance(note, dict)] if isinstance(
        raw_notes, list
    ) else []
    notes.sort(key=lambda note: (_note_start(note), _note_end(note)))
    result["notes"] = notes

    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(item) for item in warnings if item is not None]
    result["warnings"] = warnings

    # 重新对齐前清掉旧的歌词信息，避免重复调用叠加
    for note in notes:
        note["lyric"] = None
        note.pop("tie", None)
        note.pop("lyric_extend", None)
        note.pop("lyric_syllables", None)

    syllables, lines = syllables_from_lyrics(lyrics)
    period = _beat_period(beats)
    beat_times: list[float] = []
    if isinstance(beats, dict):
        for value in beats.get("beats") or []:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                beat_times.append(float(value))
    beat_times.sort()

    alignment: dict = {
        "lines": len(lines),
        "syllables": len(syllables),
        "notes": len(notes),
        "matched_syllables": 0,
        "notes_with_lyric": 0,
        "coverage": 0.0,
        "melisma": 0,
        "shared": 0,
        "unmatched": [],
        "beat_anchors": len(beat_times),
    }
    result["alignment"] = alignment

    if not notes:
        if lines:
            warnings.append("旋律为空，歌词无法对齐")
        return result
    if not syllables:
        warnings.append("无歌词音节，旋律音符 lyric 保持 null")
        return result

    groups = assign_notes_to_lines(notes, lines, tolerance=period * 0.5)
    stats = {"matched_syllables": 0, "melisma": 0, "shared": 0, "skipped_syllables": 0}
    unmatched: list[dict] = []
    empty_lines = 0

    for line_index, line in enumerate(lines):
        line_note_indices = groups[line_index] if line_index < len(groups) else []
        line_syllables = [
            syllable for syllable in syllables if syllable["line"] == line_index
        ]
        if not line_note_indices:
            empty_lines += 1
            unmatched.extend(
                {"text": syllable["text"], "start": round(syllable["start"], 3)}
                for syllable in line_syllables
            )
            continue
        line_notes = [notes[index] for index in line_note_indices]
        ops = align_syllables_to_notes(
            line_syllables,
            line_notes,
            beat_times=beat_times,
            beat_period=period,
        )
        line_stats = apply_alignment(line_notes, line_syllables, ops)
        for key in stats:
            stats[key] += line_stats.get(key, 0)
        for action in ops:
            if action[0] == "skip_syllable":
                syllable = line_syllables[action[1]]
                unmatched.append(
                    {"text": syllable["text"], "start": round(syllable["start"], 3)}
                )

    notes_with_lyric = sum(1 for note in notes if note.get("lyric"))
    alignment.update(
        {
            "matched_syllables": stats["matched_syllables"],
            "notes_with_lyric": notes_with_lyric,
            "coverage": round(notes_with_lyric / len(notes), 3),
            "melisma": stats["melisma"],
            "shared": stats["shared"],
            "unmatched": unmatched[:30],
            "lines_without_notes": empty_lines,
        }
    )

    if empty_lines:
        warnings.append(f"{empty_lines} 行歌词未找到对应旋律片段（可能为伴奏段落或旋律缺失）")
    if unmatched:
        sample = "、".join(item["text"] for item in unmatched[:8])
        warnings.append(
            f"{len(unmatched)} 个歌词音节未能对齐到音符（ASR 可能多识别或旋律缺失）：{sample}"
        )
    if notes_with_lyric == 0:
        warnings.append("全部音符都未挂上歌词，对齐失败，请人工校对歌词时间轴")
    elif alignment["coverage"] < 0.5:
        warnings.append(
            f"仅 {alignment['coverage']:.0%} 的音符挂上歌词，覆盖率偏低，建议校对歌词时间轴"
        )
    LOGGER.info(
        "歌词对齐完成：lines=%d syllables=%d matched=%d notes=%d coverage=%.2f",
        alignment["lines"],
        alignment["syllables"],
        alignment["matched_syllables"],
        alignment["notes"],
        alignment["coverage"],
    )
    return result

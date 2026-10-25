"""渲染层时间轴：把谱面里的「小节内 beat」换算成绝对秒。

``score.json``（``aitsm.score/v1``）里只有 ``bars[].start_sec`` 是绝对的，
和弦、人声、歌词的 ``beat`` 都是**小节内的四分音符拍**。网页播放器与
abcjs 兜底都需要一条「绝对秒」时间轴，本模块负责这次换算。

约定：

* 时间单调递增，任何事件时长不为负；
* 事件时间被夹在小节区间内（``[bar.start, bar.end]``），脏数据不会把
  后续小节推乱；
* 缺失的字段留空（``midi`` 为 ``None``、``lyric`` 为空串），**不猜**。

本模块是纯函数，不读写文件。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["build_timing", "parse_duration"]

_EPS = 1e-9


def _num(value: Any, default: float = 0.0) -> float:
    """尽量把 ``value`` 变成有限浮点数，失败返回 ``default``。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _round(value: float) -> float:
    """统一保留 6 位小数，和 contracts 里的序列化风格一致。"""
    return round(float(value), 6)


def parse_duration(value: Any, default: float | None = None) -> float | None:
    """解析谱面时值。

    谱面时值的单位是四分音符拍，既可能是数字，也可能是 ``"3/4"`` 这样的
    分数字符串（三连音等精确值）。非法值返回 ``default``。
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        if "/" in text:
            numerator, _, denominator = text.partition("/")
            try:
                top = float(numerator)
                bottom = float(denominator)
            except ValueError:
                return default
            if bottom == 0:
                return default
            return max(0.0, top / bottom)
        value = text
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0:
        return default
    return result


def _beats_per_bar(time_signature: Any, default: int = 4) -> int:
    """取拍号分子作为每小节拍数；非法值回退 ``default``。"""
    if isinstance(time_signature, (list, tuple)) and time_signature:
        try:
            numerator = int(time_signature[0])
        except (TypeError, ValueError):
            return default
        return numerator if numerator > 0 else default
    return default


def _as_bars(score: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    bars = score.get("bars")
    if not isinstance(bars, list):
        return []
    return sorted(
        (bar for bar in bars if isinstance(bar, Mapping)),
        key=lambda bar: _num(bar.get("index"), 0.0),
    )


def _bar_start(bar: Mapping[str, Any]) -> float | None:
    value = bar.get("start_sec")
    if value is None:
        return None
    number = _num(value, float("nan"))
    return number if math.isfinite(number) else None


def _bar_end(bar: Mapping[str, Any]) -> float | None:
    value = bar.get("end_sec")
    if value is None:
        return None
    number = _num(value, float("nan"))
    return number if math.isfinite(number) else None


def build_timing(score: Mapping[str, Any]) -> dict[str, Any]:
    """从 ``aitsm.score/v1`` 谱面算出绝对时间轴。

    参数
    ----
    score:
        含 ``bars`` 的谱面 dict（``meta`` 可缺省，缺省时按 120 BPM / 4/4）。

    返回
    ----
    dict: ``{"bars": [...], "notes": [...], "chords": [...], "lyrics": [...]}``：

    * ``bars``: ``{index, start, end}``
    * ``notes``: ``{start, end, midi, lyric, bar}``（``midi`` 可为 ``None``）
    * ``chords``: ``{start, end, symbol, bar}``
    * ``lyrics``: ``{start, end, text, bar}``

    所有时间单位都是秒；各列表按开始时间排序。
    """
    if not isinstance(score, Mapping):
        raise ValueError(f"build_timing 需要 score dict，收到 {type(score).__name__}")

    meta = score.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    base_tempo = _num(meta.get("tempo"), 120.0)
    if base_tempo <= 0:
        base_tempo = 120.0
    default_ts = meta.get("time_signature")

    bars_out: list[dict[str, Any]] = []
    notes_out: list[dict[str, Any]] = []
    chords_out: list[dict[str, Any]] = []
    lyrics_out: list[dict[str, Any]] = []

    cursor = 0.0
    for position, bar in enumerate(_as_bars(score)):
        tempo = _num(bar.get("tempo"), base_tempo)
        if tempo <= 0:
            tempo = base_tempo
        seconds_per_beat = 60.0 / tempo

        start = _bar_start(bar)
        if start is None:
            start = cursor
        # 单调：小节之间不允许回退
        start = max(start, cursor)

        end = _bar_end(bar)
        beats = _beats_per_bar(bar.get("time_signature") or default_ts)
        if end is None or end < start:
            end = start + beats * seconds_per_beat
        end = max(start, end)

        index = int(_num(bar.get("index"), position + 1))
        bars_out.append({"index": index, "start": _round(start), "end": _round(end)})
        cursor = end

        def to_sec(beat: Any, fallback: float = 0.0) -> float:
            """小节内 beat -> 绝对秒（夹在本小节区间内）。"""
            number = _num(beat, float("nan"))
            if not math.isfinite(number):
                number = fallback
            seconds = start + max(0.0, number) * seconds_per_beat
            return min(max(seconds, start), end)

        # ---------------- 和弦 ----------------
        chords_raw = [
            item for item in (bar.get("chords") or []) if isinstance(item, Mapping)
        ]
        chords_raw.sort(key=lambda item: _num(item.get("beat"), 0.0))
        for offset, chord in enumerate(chords_raw):
            symbol = str(chord.get("symbol") or "").strip()
            if not symbol:
                continue
            beat = max(0.0, _num(chord.get("beat"), 0.0))
            duration_beats = parse_duration(chord.get("dur"))
            if duration_beats is None:
                nxt = next(
                    (
                        _num(item.get("beat"), 0.0)
                        for item in chords_raw[offset + 1 :]
                        if _num(item.get("beat"), 0.0) > beat + _EPS
                    ),
                    None,
                )
                duration_beats = max(
                    0.0, (nxt - beat) if nxt is not None else (beats - beat)
                )
            chord_start = to_sec(beat)
            chord_end = min(chord_start + duration_beats * seconds_per_beat, end)
            chords_out.append(
                {
                    "start": _round(chord_start),
                    "end": _round(max(chord_start, chord_end)),
                    "symbol": symbol,
                    "bar": index,
                }
            )

        # ---------------- 人声 / 歌词 ----------------
        vocal_raw = [
            item for item in (bar.get("vocal") or []) if isinstance(item, Mapping)
        ]
        vocal_raw.sort(key=lambda item: _num(item.get("beat"), 0.0))
        for offset, note in enumerate(vocal_raw):
            beat = max(0.0, _num(note.get("beat"), 0.0))
            duration_beats = parse_duration(note.get("dur"))
            if duration_beats is None:
                nxt = next(
                    (
                        _num(item.get("beat"), 0.0)
                        for item in vocal_raw[offset + 1 :]
                        if _num(item.get("beat"), 0.0) > beat + _EPS
                    ),
                    None,
                )
                duration_beats = max(
                    0.0, (nxt - beat) if nxt is not None else (beats - beat)
                )
            note_start = to_sec(beat)
            note_end = min(note_start + duration_beats * seconds_per_beat, end)
            note_end = max(note_start, note_end)

            midi_raw = note.get("midi")
            midi: int | None = None
            if midi_raw is not None:
                number = _num(midi_raw, float("nan"))
                if math.isfinite(number):
                    midi = int(round(number))
            lyric = str(note.get("lyric") or "").strip()

            notes_out.append(
                {
                    "start": _round(note_start),
                    "end": _round(note_end),
                    "midi": midi,
                    "lyric": lyric,
                    "bar": index,
                }
            )
            if lyric:
                lyrics_out.append(
                    {
                        "start": _round(note_start),
                        "end": _round(note_end),
                        "text": lyric,
                        "bar": index,
                    }
                )

    notes_out.sort(key=lambda item: (item["start"], item["end"], item["bar"]))
    chords_out.sort(key=lambda item: (item["start"], item["end"], item["bar"]))
    lyrics_out.sort(key=lambda item: (item["start"], item["end"], item["bar"]))

    return {
        "bars": bars_out,
        "notes": notes_out,
        "chords": chords_out,
        "lyrics": lyrics_out,
    }

"""ABC 记谱：给 abcjs 兜底渲染器的最小可读五线谱。

VexFlow 是主渲染器；当它失败时，网页会把本模块生成的 ABC 交给 abcjs。
因此这里只追求「能读、能对拍」：五线谱主旋律 + 和弦符号 + 逐字歌词，
不追求排版细节。

输入既可以是编配层的 ``aitsm.score/v1`` 谱面（``bars[].vocal`` 等），
也可以是融合层的 ``aitsm.symbolic/v1`` 符号总谱（绝对秒的
``melody`` / ``chords`` + ``beats`` 节拍网格），还可以直接传
:class:`aitsm.contracts.SymbolicScore` 实例。

字符与时长约定：``L:1/4``，时值倍数即四分音符拍数；音高按调号拼写，
调号已经隐含的升降号不重复写。本模块是纯函数，不读写文件。
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping, Sequence

__all__ = ["write_abc"]

_EPS = 1e-6
_MAX_LINE_WIDTH = 78

#: 升号调里被升高的音名顺序
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
#: 降号调里被降低的音名顺序
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")

_SHARP_SPELLING = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_SPELLING = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

_MAJOR_FIFTHS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
_MINOR_FIFTHS = {
    "A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
    "D": -1, "G": -2, "C": -3, "F": -4, "Bb": -5, "Eb": -6, "Ab": -7,
}

#: w: 行里需要避开的字符（空格分音节；连字符/下划线/星号是 ABC 控制符）
_LYRIC_DROP = frozenset(' \t\r\n|[]{}"\\*_-~^')


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clean_text(text: Any, *, limit: int = 160) -> str:
    """单行文本：去掉换行与控制字符，限制长度。"""
    value = str(text or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def _time_signature(value: Any) -> tuple[int, int]:
    """拍号 -> ``(分子, 分母)``；非法值回退 4/4。"""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            numerator, denominator = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return 4, 4
        if numerator > 0 and denominator > 0:
            return numerator, denominator
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 4, 4


def _abc_duration(beats: float) -> str:
    """四分音符拍数 -> ABC 时值后缀（L:1/4 语境）。

    ``1`` 为默认长度，直接省略；``0.5`` 写 ``/2``；``1.5`` 写 ``3/2``；
    ``0.75`` 写 ``3/4``。
    """
    if beats <= _EPS:
        return ""
    value = Fraction(beats).limit_denominator(64)
    if value <= 0:
        return ""
    numerator, denominator = value.numerator, value.denominator
    if denominator == 1:
        return "" if numerator == 1 else str(numerator)
    if numerator == 1:
        return f"/{denominator}"
    return f"{numerator}/{denominator}"


def _abc_lyric(text: Any) -> str:
    """歌词音节清洗；空音节用 ``*`` 占位（w: 行对齐用）。"""
    cleaned = "".join(char for char in str(text or "") if char not in _LYRIC_DROP)
    return cleaned or "*"


def _abc_chord(symbol: Any) -> str:
    """和弦符号清洗（引号内不允许再出现引号）。"""
    value = re.sub(r"\s+", "", str(symbol or ""))
    return value.replace('"', "")


@dataclass(frozen=True)
class _Key:
    """记谱调：ABC 的 ``K:`` 字段与调号隐含的升降。"""

    abc: str
    fifths: int = 0
    mode: str = "major"

    def expected_alter(self, step: str) -> int:
        """该音名在本调号里应带的升降（+1 / -1 / 0）。"""
        if self.fifths > 0:
            return 1 if step in _SHARP_ORDER[: self.fifths] else 0
        if self.fifths < 0:
            return -1 if step in _FLAT_ORDER[: -self.fifths] else 0
        return 0


def _abc_pitch(midi: int, key: _Key) -> str:
    """MIDI 音高 -> ABC 音名（含八度标记与必要的临时记号）。"""
    midi = int(midi)
    spelling = _FLAT_SPELLING if key.fifths < 0 else _SHARP_SPELLING
    name = spelling[midi % 12]
    step = name[0]
    alter = 1 if name.endswith("#") else (-1 if name.endswith("b") else 0)

    expected = key.expected_alter(step)
    if alter == expected:
        accidental = ""
    elif alter == 0:
        accidental = "="  # 调号里有升降，这个音要还原
    elif alter > 0:
        accidental = "^" * alter
    else:
        accidental = "_" * (-alter)

    octave = midi // 12 - 1
    if octave >= 5:
        body = step.lower() + "'" * (octave - 5)
    else:
        body = step.upper() + "," * (4 - octave)
    return accidental + body


def _key_from_score(meta: Mapping[str, Any]) -> _Key:
    """从 ``score.meta.key`` 取记谱调。"""
    raw = meta.get("key")
    raw = raw if isinstance(raw, Mapping) else {}
    name = str(raw.get("play") or raw.get("sounding") or raw.get("detected") or "C").strip()
    mode = str(raw.get("mode") or "").strip().lower()
    match = re.split(r"\s+", name, maxsplit=1)[0]
    tonic_text = match.replace("♯", "#").replace("♭", "b").replace("♮", "")
    if not mode:
        mode = "minor" if tonic_text.endswith("m") or "minor" in name.lower() else "major"
    tonic_match = re.fullmatch(r"([A-Ga-g])([#b]?)(m?)", tonic_text)
    tonic = (tonic_match.group(1).upper() + tonic_match.group(2)) if tonic_match else "C"
    fifths_raw = raw.get("num_sharps_flats", raw.get("fifths"))
    fifths: int | None = None
    if fifths_raw is not None:
        number = _num(fifths_raw, float("nan"))
        if math.isfinite(number):
            fifths = int(round(number))
    if fifths is None:
        table = _MINOR_FIFTHS if mode.startswith("min") else _MAJOR_FIFTHS
        fifths = table.get(tonic, 0)
    fifths = max(-7, min(7, fifths))
    abc = tonic + ("m" if mode.startswith("min") else "")
    return _Key(abc=abc, fifths=fifths, mode="minor" if mode.startswith("min") else "major")


def _key_from_symbolic(data: Mapping[str, Any]) -> _Key:
    raw = data.get("key")
    raw = raw if isinstance(raw, Mapping) else {}
    tonic_text = str(raw.get("tonic") or "C").strip().replace("♯", "#").replace("♭", "b")
    tonic_match = re.fullmatch(r"([A-Ga-g])([#b]?)", tonic_text)
    tonic = (tonic_match.group(1).upper() + tonic_match.group(2)) if tonic_match else "C"
    mode = str(raw.get("mode") or "major").strip().lower()
    is_minor = mode.startswith("min")
    fifths_raw = raw.get("fifths", raw.get("num_sharps_flats"))
    fifths: int | None = None
    if fifths_raw is not None:
        number = _num(fifths_raw, float("nan"))
        if math.isfinite(number):
            fifths = int(round(number))
    if fifths is None:
        fifths = (_MINOR_FIFTHS if is_minor else _MAJOR_FIFTHS).get(tonic, 0)
    fifths = max(-7, min(7, fifths))
    abc = tonic + ("m" if is_minor else "")
    return _Key(abc=abc, fifths=fifths, mode="minor" if is_minor else "major")


# --------------------------------------------------------------------------- #
# 中间模型：一小节的旋律 + 和弦
# --------------------------------------------------------------------------- #


@dataclass
class _Note:
    """小节内旋律事件（rest 时 ``midi`` 为 ``None``）。"""

    beat: float
    dur: float
    midi: int | None = None
    lyric: str = ""


@dataclass
class _Bar:
    """一小节：拍数、旋律事件、和弦符号（beat 相对小节起点）。"""

    beats: float = 4.0
    notes: list[_Note] = field(default_factory=list)
    chords: list[tuple[float, str]] = field(default_factory=list)
    pickup: bool = False


@dataclass
class _Song:
    """``write_abc`` 的内部输入：与来源格式无关。"""

    bars: list[_Bar]
    key: _Key
    title: str = ""
    tempo: float = 0.0
    meter: tuple[int, int] = (4, 4)


# --------------------------------------------------------------------------- #
# 适配器 1：aitsm.score/v1
# --------------------------------------------------------------------------- #


def _score_song(score: Mapping[str, Any]) -> _Song:
    meta_raw = score.get("meta")
    meta = meta_raw if isinstance(meta_raw, Mapping) else {}
    key = _key_from_score(meta)

    raw_bars = [bar for bar in (score.get("bars") or []) if isinstance(bar, Mapping)]
    raw_bars.sort(key=lambda bar: _num(bar.get("index"), 0.0))

    base_tempo = _num(meta.get("tempo"), 0.0)
    default_ts = meta.get("time_signature")
    meter = _time_signature(default_ts) if default_ts is not None else None

    bars: list[_Bar] = []
    for position, bar in enumerate(raw_bars):
        bar_ts = bar.get("time_signature") if bar.get("time_signature") is not None else default_ts
        numerator, _denominator = _time_signature(bar_ts)
        if meter is None:
            meter = (numerator, _denominator)

        tempo = _num(bar.get("tempo"), base_tempo)
        if tempo <= 0:
            tempo = base_tempo

        # 小节实际拍数：优先按真实时间跨度算（弱起小节天然不足一整小节）
        beats = float(numerator)
        if tempo > 0:
            start_raw = bar.get("start_sec")
            end_raw = bar.get("end_sec")
            if start_raw is not None and end_raw is not None:
                start = _num(start_raw, float("nan"))
                end = _num(end_raw, float("nan"))
                if math.isfinite(start) and math.isfinite(end) and end > start:
                    seconds_per_beat = 60.0 / tempo
                    actual = (end - start) / seconds_per_beat
                    if bar.get("pickup") or actual < numerator * 0.75:
                        beats = max(0.25, round(actual * 4) / 4)

        notes: list[_Note] = []
        vocal = [item for item in (bar.get("vocal") or []) if isinstance(item, Mapping)]
        vocal.sort(key=lambda item: _num(item.get("beat"), 0.0))
        for offset, note in enumerate(vocal):
            beat = max(0.0, _num(note.get("beat"), 0.0))
            duration = _parse_dur(note.get("dur"))
            if duration is None:
                nxt = next(
                    (
                        _num(item.get("beat"), 0.0)
                        for item in vocal[offset + 1 :]
                        if _num(item.get("beat"), 0.0) > beat + _EPS
                    ),
                    None,
                )
                duration = max(0.0, (nxt - beat) if nxt is not None else (beats - beat))
            midi_raw = note.get("midi")
            midi: int | None = None
            if midi_raw is not None:
                number = _num(midi_raw, float("nan"))
                if math.isfinite(number):
                    midi = int(round(number))
            notes.append(
                _Note(
                    beat=beat,
                    dur=max(0.0, duration),
                    midi=midi,
                    lyric=str(note.get("lyric") or "").strip(),
                )
            )

        chords: list[tuple[float, str]] = []
        for chord in bar.get("chords") or []:
            if not isinstance(chord, Mapping):
                continue
            symbol = _abc_chord(chord.get("symbol"))
            if not symbol:
                continue
            chords.append((max(0.0, _num(chord.get("beat"), 0.0)), symbol))
        chords.sort(key=lambda item: item[0])

        # 不让越界的事件被裁掉：小节至少覆盖到最后一个事件的结束
        event_end = max(
            [0.0] + [note.beat + note.dur for note in notes] + [beat for beat, _ in chords]
        )
        beats = max(beats, event_end)
        bars.append(_Bar(beats=beats, notes=notes, chords=chords, pickup=bool(bar.get("pickup"))))

    return _Song(
        bars=bars,
        key=key,
        title=_clean_text(meta.get("title")),
        tempo=base_tempo,
        meter=meter or (4, 4),
    )


def _parse_dur(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "/" in text:
            top, _, bottom = text.partition("/")
            try:
                numerator, denominator = float(top), float(bottom)
            except ValueError:
                return None
            if denominator == 0:
                return None
            return max(0.0, numerator / denominator)
        value = text
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


# --------------------------------------------------------------------------- #
# 适配器 2：aitsm.symbolic/v1
# --------------------------------------------------------------------------- #


def _symbolic_song(data: Mapping[str, Any]) -> _Song:
    beats_raw = data.get("beats")
    beats = beats_raw if isinstance(beats_raw, Mapping) else {}
    bpm = _num(beats.get("bpm"), 120.0)
    if bpm <= 0:
        bpm = 120.0
    beat_period = 60.0 / bpm
    numerator, denominator = _time_signature(beats.get("time_signature"))
    bar_seconds = beat_period * numerator

    chords: list[tuple[float, float, str]] = []
    for item in data.get("chords") or []:
        if not isinstance(item, Mapping):
            continue
        symbol = _abc_chord(item.get("symbol"))
        if not symbol:
            continue
        start = _num(item.get("start"), 0.0)
        end = max(start, _num(item.get("end"), start))
        chords.append((start, end, symbol))

    melody: list[tuple[float, float, int | None, str]] = []
    for item in data.get("melody") or []:
        if not isinstance(item, Mapping):
            continue
        start = _num(item.get("start"), 0.0)
        end = max(start, _num(item.get("end"), start))
        midi_raw = item.get("midi")
        midi: int | None = None
        if midi_raw is not None:
            number = _num(midi_raw, float("nan"))
            if math.isfinite(number):
                midi = int(round(number))
        melody.append((start, end, midi, str(item.get("lyric") or "").strip()))

    # 小节边界：优先用 downbeats，缺失时按节拍网格铺
    downbeats = sorted(
        _num(value, float("nan")) for value in (beats.get("downbeats") or [])
    )
    downbeats = [value for value in downbeats if math.isfinite(value)]
    event_end = max(
        [0.0]
        + [end for _, end, _, _ in melody]
        + [end for _, end, _ in chords]
    )
    if len(downbeats) >= 2:
        bounds = list(downbeats)
        if event_end > bounds[-1] + bar_seconds:
            while bounds[-1] + bar_seconds < event_end:
                bounds.append(bounds[-1] + bar_seconds)
        bounds.append(max(bounds[-1] + bar_seconds, event_end))
    elif len(downbeats) == 1:
        origin = downbeats[0]
        count = max(1, int(math.ceil((max(event_end, origin + bar_seconds) - origin) / bar_seconds)))
        bounds = [origin + index * bar_seconds for index in range(count + 1)]
    else:
        origin = min(
            [0.0] + [start for start, _, _, _ in melody] + [start for start, _, _ in chords]
        )
        count = max(1, int(math.ceil((max(event_end, origin + bar_seconds) - origin) / bar_seconds)))
        bounds = [origin + index * bar_seconds for index in range(count + 1)]

    bars = [
        _Bar(
            beats=max(0.25, round((bounds[index + 1] - bounds[index]) / beat_period * 4) / 4),
            pickup=index == 0
            and (bounds[1] - bounds[0]) / beat_period < numerator - 0.5,
        )
        for index in range(len(bounds) - 1)
    ]

    def bar_index_of(moment: float) -> int:
        return max(0, min(len(bars) - 1, bisect_right(bounds, moment) - 1))

    for start, _end, symbol in chords:
        index = bar_index_of(start)
        bar_start = bounds[index]
        bar = bars[index]
        beat = max(0.0, (start - bar_start) / beat_period)
        beat = min(beat, max(0.0, bar.beats - _EPS))
        bar.chords.append((beat, symbol))

    for start, end, midi, lyric in melody:
        index = bar_index_of(start)
        bar_start = bounds[index]
        bar = bars[index]
        beat = max(0.0, (start - bar_start) / beat_period)
        beat = min(beat, max(0.0, bar.beats - _EPS))
        duration = max(0.0, min((end - start) / beat_period, bar.beats - beat))
        bar.notes.append(_Note(beat=beat, dur=duration, midi=midi, lyric=lyric))

    for bar in bars:
        if bar.notes:
            bar.notes.sort(key=lambda note: note.beat)
        if bar.chords:
            bar.chords.sort(key=lambda item: item[0])

    meta_raw = data.get("meta")
    meta = meta_raw if isinstance(meta_raw, Mapping) else {}
    return _Song(
        bars=bars,
        key=_key_from_symbolic(data),
        title=_clean_text(meta.get("title")),
        tempo=bpm,
        meter=(numerator, denominator),
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


def _chord_at(chords: Sequence[tuple[float, str]], beat: float) -> str:
    """取落在 ``beat`` 上的和弦符号（容差 1e-3 拍）。"""
    for position, symbol in chords:
        if abs(position - beat) <= 1e-3:
            return symbol
    return ""


def _covering_note(notes: Sequence[_Note], start: float, end: float) -> _Note | None:
    """覆盖 ``[start, end]`` 整段的旋律事件。"""
    for note in notes:
        if note.beat - 1e-3 <= start and note.beat + note.dur + 1e-3 >= end:
            return note
    return None


def _render_bar(bar: _Bar, key: _Key, lyric_tokens: list[str]) -> str:
    """把一小节渲染成 ABC 音符串，并把歌词音节追加到 ``lyric_tokens``。"""
    beats = max(0.25, bar.beats)
    points = {0.0, beats}
    for note in bar.notes:
        begin = min(max(note.beat, 0.0), beats)
        finish = min(max(note.beat + note.dur, begin), beats)
        points.add(begin)
        points.add(finish)
    for position, _symbol in bar.chords:
        points.add(min(max(position, 0.0), beats))
    ordered = sorted(points)

    tokens: list[str] = []
    for start, end in zip(ordered, ordered[1:]):
        length = end - start
        if length <= _EPS:
            continue
        chord = _chord_at(bar.chords, start)
        prefix = f'"{chord}"' if chord else ""
        note = _covering_note(bar.notes, start, end)
        if note is not None and note.midi is not None:
            pitch = _abc_pitch(note.midi, key)
            continued = start > note.beat + 1e-3
            last = end >= note.beat + note.dur - 1e-3
            token = prefix + pitch + _abc_duration(length)
            if not last:
                token += "-"
            lyric_tokens.append("*" if continued else _abc_lyric(note.lyric))
            tokens.append(token)
        else:
            tokens.append(prefix + "z" + _abc_duration(length))
    return " ".join(tokens)


def _assemble_parts(bars: Sequence[_Bar], key: _Key) -> list[str]:
    """把各小节渲染结果按行宽排版；有歌词时每行后跟一条 ``w:``。"""
    parts: list[tuple[str, list[str]]] = []
    for bar in bars:
        tokens: list[str] = []
        body = _render_bar(bar, key, tokens)
        parts.append((body, tokens))

    has_lyrics = any(token != "*" for _body, tokens in parts for token in tokens)

    lines: list[str] = []
    bodies: list[str] = []
    lyrics: list[str] = []
    width = 0
    for body, tokens in parts:
        extra = len(body) + (3 if bodies else 0)
        if bodies and width + extra > _MAX_LINE_WIDTH:
            lines.append(_join_line(bodies, lyrics, has_lyrics, last=False))
            bodies, lyrics, width = [], [], 0
        bodies.append(body)
        lyrics.extend(tokens)
        width += len(body) + (3 if len(bodies) > 1 else 0)
    if bodies:
        lines.append(_join_line(bodies, lyrics, has_lyrics, last=True))
    return lines


def _join_line(
    bodies: Sequence[str], lyrics: Sequence[str], has_lyrics: bool, *, last: bool
) -> str:
    music = " | ".join(body for body in bodies if body) + (" |]" if last else " |")
    if not has_lyrics:
        return music
    return f"{music}\nw: {' '.join(lyrics)}"


def _as_song(source: Any) -> _Song:
    """把三种受支持的输入统一成 ``_Song``。"""
    data: Any = source
    if not isinstance(data, Mapping):
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
    if not isinstance(data, Mapping):
        raise ValueError(f"write_abc 需要谱面 dict 或 SymbolicScore，收到 {type(source).__name__}")

    if isinstance(data.get("bars"), list):
        return _score_song(data)
    if isinstance(data.get("beats"), Mapping) or isinstance(data.get("melody"), list):
        return _symbolic_song(data)
    raise ValueError("write_abc 无法识别的输入：需要 aitsm.score/v1 的 bars 或 aitsm.symbolic/v1 的 beats")


def write_abc(source: Any, *, title: str | None = None) -> str:
    """把谱面或符号总谱写成可被 abcjs 渲染的 ABC 文本。

    参数
    ----
    source:
        ``aitsm.score/v1`` 谱面 dict、``aitsm.symbolic/v1`` 符号总谱 dict，
        或 :class:`aitsm.contracts.SymbolicScore` 实例。
    title:
        覆盖标题；缺省时用 ``meta.title``。

    返回
    ----
    str: 含 ``X/T/M/L/Q/K`` 头、和弦标注、主旋律与 ``w:`` 歌词行的 ABC 文本。
    """
    song = _as_song(source)
    if title is not None and str(title).strip():
        song.title = _clean_text(title)

    lines = ["X:1"]
    if song.title:
        lines.append(f"T:{song.title}")
    lines.append(f"M:{song.meter[0]}/{song.meter[1]}")
    lines.append("L:1/4")
    if song.tempo > 0:
        lines.append(f"Q:1/4={int(round(song.tempo))}")
    lines.append(f"K:{song.key.abc}")

    if not song.bars:
        lines.append("z4 |]")
        return "\n".join(lines) + "\n"

    lines.extend(_assemble_parts(song.bars, song.key))
    return "\n".join(lines) + "\n"

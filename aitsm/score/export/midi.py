"""自实现标准 MIDI 文件（SMF format 1，PPQ=480）导出 + 最小解析器。

为什么不用 mido：项目依赖要尽量少（Lead 统一装包），而 SMF 的写入其实只是
「变长整数 + 事件流」的字节拼接，自己写 200 行就够，还能顺手实现一个解析器做
往返自检（导出后立刻读回来核对事件数，避免写出「能存不能播」的文件）。

轨道布局
--------
===== ====================== ======================================
轨道  内容                   说明
===== ====================== ======================================
0     速度 / 拍号 / 调号      每个小节起始的 tempo/拍号变化都会写
1     人声主旋律              ch0，program 53（Voice Oohs）
2     吉他扫弦                ch1，program 25（Acoustic Guitar），按 pattern 展开
3     节拍器（可选）           ch9，只写 meta.include_metronome 为真时
===== ====================== ======================================
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..model import (
    bar_length_quarters,
    duration_value,
    fret_to_midi,
    parse_time_signature,
    tuning_to_midis,
)
from . import ExportError

__all__ = ["export_midi", "parse_midi", "PPQ"]

#: 每四分音符 tick 数。
PPQ = 480

_EPS = 1e-6

#: 同一拍的排序权重（小的先写）：保证 note-off 早于 note-on。
_ORDER_META = 0
_ORDER_PROGRAM = 1
_ORDER_NOTE_OFF = 2
_ORDER_NOTE_ON = 3
_ORDER_TEXT = 4

#: 扫弦时相邻弦之间的「扫过」延迟（tick）。12 tick = 1/40 拍，听感像真的扫下去。
STRUM_SPREAD = 12

#: 各轨道默认音色（GM program number，0 基）。
_PROGRAM_VOCAL = 53  # Voice Oohs
_PROGRAM_GUITAR = 25  # Acoustic Guitar (steel)
_PROGRAM_CLICK = 0


# ---------------------------------------------------------------- 基础编码


def _varlen(value: int) -> bytes:
    """MIDI 变长整数编码。"""
    if value < 0:
        raise ValueError(f"变长整数不能为负：{value}")
    buffer = bytearray([value & 0x7F])
    value >>= 7
    while value:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buffer))


def _meta_event(mtype: int, payload: bytes) -> bytes:
    """元事件：``FF <type> <len> <payload>``。"""
    return b"\xff" + bytes([mtype & 0x7F]) + _varlen(len(payload)) + payload


def _tempo_meta(bpm: float) -> bytes:
    microseconds = int(round(60_000_000 / max(float(bpm), 1e-3)))
    microseconds = max(1, min(0xFFFFFF, microseconds))
    return _meta_event(0x51, microseconds.to_bytes(3, "big"))


def _time_signature_meta(numerator: int, denominator: int) -> bytes:
    power = 0
    value = int(denominator)
    while value > 1 and power < 8:
        value //= 2
        power += 1
    return _meta_event(0x58, bytes([max(1, min(255, int(numerator))), power, 24, 8]))


def _key_signature_meta(fifths: int, minor: bool) -> bytes:
    sf = max(-7, min(7, int(fifths)))
    return _meta_event(0x59, struct.pack(">bB", sf, 1 if minor else 0))


def _track_name_meta(name: str) -> bytes:
    return _meta_event(0x03, name.encode("utf-8")[:120])


def _text_meta(text: str) -> bytes:
    return _meta_event(0x05, text.encode("utf-8")[:120])


def _program_change(channel: int, program: int) -> bytes:
    return bytes([0xC0 | (channel & 0x0F), max(0, min(127, int(program)))])


def _note_on(channel: int, pitch: int, velocity: int) -> bytes:
    return bytes([0x90 | (channel & 0x0F), pitch & 0x7F, max(1, min(127, int(velocity)))])


def _note_off(channel: int, pitch: int) -> bytes:
    return bytes([0x80 | (channel & 0x0F), pitch & 0x7F, 64])


# ---------------------------------------------------------------- 轨道容器


class _Track:
    """一条 MIDI 轨道的事件缓冲（绝对 tick + 排序权重）。"""

    def __init__(self, name: str, channel: int | None = None, program: int | None = None) -> None:
        self.name = name
        self.channel = channel
        self.program = program
        self._events: list[tuple[int, int, int, bytes]] = []
        self._seq = 0
        self.note_count = 0

    def add(self, tick: int, order: int, data: bytes) -> None:
        self._events.append((max(0, int(round(tick))), order, self._seq, data))
        self._seq += 1

    def note(self, channel: int, start: int, duration: int, pitch: int, velocity: int) -> None:
        start = max(0, int(round(start)))
        duration = max(1, int(round(duration)))
        self.add(start, _ORDER_NOTE_ON, _note_on(channel, pitch, velocity))
        self.add(start + duration, _ORDER_NOTE_OFF, _note_off(channel, pitch))
        self.note_count += 1

    def to_chunk(self) -> bytes:
        events = sorted(self._events)
        body = bytearray()
        if self.name:
            body += _varlen(0) + _track_name_meta(self.name)
        if self.channel is not None and self.program is not None:
            body += _varlen(0) + _program_change(self.channel, self.program)
        previous = 0
        for tick, _order, _seq, data in events:
            body += _varlen(tick - previous) + data
            previous = tick
        body += _varlen(0) + _meta_event(0x2F, b"")
        return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


# ---------------------------------------------------------------- 上下文


class _Context:
    """从 score.meta / score.voicings 提炼的导出上下文。"""

    def __init__(self, score: dict) -> None:
        meta = score.get("meta") or {}
        self.meta = meta
        key = meta.get("key") or {}
        self.fifths = max(-7, min(7, int(key.get("num_sharps_flats") or 0)))
        self.minor = str(key.get("mode", "major")).lower().startswith("min")
        self.time_signature = parse_time_signature(meta.get("time_signature"), (4, 4))
        tempo = meta.get("tempo")
        try:
            self.tempo = float(tempo) if tempo else 100.0
        except (TypeError, ValueError):
            self.tempo = 100.0
        if not (10.0 <= self.tempo <= 400.0):
            self.tempo = 100.0
        guitar = meta.get("guitar") or {}
        tuning = guitar.get("tuning") or guitar.get("tuning_name") or "standard"
        try:
            self.tuning_midis = tuning_to_midis(tuning)
        except Exception:  # noqa: BLE001 - 调弦不可解析时退回标准调弦
            self.tuning_midis = tuning_to_midis("standard")
        self.voicings: dict[str, dict] = {
            str(key): value for key, value in (score.get("voicings") or {}).items() if isinstance(value, dict)
        }
        self.patterns: dict[str, dict] = {
            str(key): value for key, value in (score.get("strum_patterns") or {}).items() if isinstance(value, dict)
        }
        self.include_metronome = bool(meta.get("include_metronome"))


def _safe_beats(value: Any) -> float:
    try:
        return max(0.0, duration_value(value))
    except (TypeError, ValueError):
        return 0.0


def _safe_dur(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return max(0.0, duration_value(value))
    except (TypeError, ValueError):
        return default


def _bars(score: dict) -> list[dict]:
    return [bar for bar in (score.get("bars") or []) if isinstance(bar, dict)]


def _bar_quarters(bar: dict, ctx: _Context) -> float:
    """小节长度（四分音符拍）；弱起小节取实际内容长度。"""
    signature = parse_time_signature(bar.get("time_signature") or ctx.time_signature, ctx.time_signature)
    full = bar_length_quarters(signature)
    if not bar.get("pickup"):
        return full
    content = 0.0
    for note in bar.get("vocal") or []:
        content = max(content, _safe_beats(note.get("beat")) + _safe_dur(note.get("dur"), 1.0))
    for chord in bar.get("chords") or []:
        content = max(content, _safe_beats(chord.get("beat")) + _safe_dur(chord.get("dur"), full))
    if content <= _EPS:
        return full
    return min(full, max(1.0, round(content * 4) / 4))


def _bar_ticks(bar: dict, ctx: _Context) -> int:
    return max(1, int(round(_bar_quarters(bar, ctx) * PPQ)))


# ---------------------------------------------------------------- 轨道构建


def _build_tempo_track(bars: list[dict], ctx: _Context, warnings: list[str]) -> _Track:
    """轨道 0：速度 / 拍号 / 调号。"""
    track = _Track("Conductor")
    track.add(0, _ORDER_META, _key_signature_meta(ctx.fifths, ctx.minor))
    track.add(0, _ORDER_META, _time_signature_meta(*ctx.time_signature))
    track.add(0, _ORDER_META, _tempo_meta(ctx.tempo))

    cursor = 0
    previous_signature = ctx.time_signature
    previous_tempo = ctx.tempo
    for bar in bars:
        signature = parse_time_signature(bar.get("time_signature") or ctx.time_signature, ctx.time_signature)
        tempo = bar.get("tempo")
        try:
            tempo_value = float(tempo) if tempo else previous_tempo
        except (TypeError, ValueError):
            tempo_value = previous_tempo
        if cursor > 0:
            if signature != previous_signature:
                track.add(cursor, _ORDER_META, _time_signature_meta(*signature))
            if abs(tempo_value - previous_tempo) > 1e-6:
                track.add(cursor, _ORDER_META, _tempo_meta(tempo_value))
        elif abs(tempo_value - previous_tempo) > 1e-6:
            # 第一小节自身覆盖速度时，在 tick 0 重新写一次
            track.add(0, _ORDER_META, _tempo_meta(tempo_value))
        previous_signature = signature
        previous_tempo = tempo_value
        cursor += _bar_ticks(bar, ctx)
    if cursor <= 0:
        warnings.append("乐谱总时长为 0，MIDI 只有一条空的指挥轨")
    return track


def _collect_vocal(bars: list[dict], ctx: _Context, warnings: list[str]) -> list[dict]:
    """把各小节旋律收集为绝对 tick 事件，并合并延音线。"""
    events: list[dict] = []
    cursor = 0
    for bar in bars:
        bar_ticks = _bar_ticks(bar, ctx)
        for note in sorted(
            (note for note in (bar.get("vocal") or []) if isinstance(note, dict)),
            key=lambda item: _safe_beats(item.get("beat")),
        ):
            midi = note.get("midi")
            if note.get("rest") or midi is None:
                continue
            try:
                pitch = int(midi)
            except (TypeError, ValueError):
                continue
            if not (0 <= pitch <= 127):
                warnings.append(f"第 {bar.get('index')} 小节旋律 midi={midi!r} 超出 MIDI 范围，已跳过")
                continue
            start = cursor + int(round(_safe_beats(note.get("beat")) * PPQ))
            duration = max(1, int(round(_safe_dur(note.get("dur"), 1.0) * PPQ)))
            end = min(cursor + bar_ticks, start + duration)
            if end <= start:
                continue
            confidence = note.get("confidence")
            try:
                velocity = 62 if confidence is None else int(round(48 + 32 * float(confidence)))
            except (TypeError, ValueError):
                velocity = 72
            events.append(
                {
                    "start": start,
                    "end": end,
                    "pitch": pitch,
                    "velocity": max(1, min(127, velocity)),
                    "lyric": str(note.get("lyric") or ""),
                    "tie": note.get("tie"),
                }
            )
        cursor += bar_ticks

    # 合并延音线：前一个音 tie=start，后一个音同音高且 tie in (stop, continue)
    merged: list[dict] = []
    for event in events:
        if (
            merged
            and merged[-1]["tie"] in ("start", "continue")
            and merged[-1]["pitch"] == event["pitch"]
            and event["tie"] in ("stop", "continue")
            and abs(merged[-1]["end"] - event["start"]) <= 4
        ):
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["tie"] = event["tie"]
            merged[-1]["lyric"] = merged[-1]["lyric"] or event["lyric"]
            continue
        merged.append(event)
    return merged


def _sanitize_spans(
    spans: Sequence[tuple[int, int, int, Any]],
    warnings: list[str],
    label: str,
) -> list[tuple[int, int, int, Any]]:
    """同一轨同一音高不允许重叠或重复 Note On（否则播放像「打嗝」）。

    * 同一 tick 上的同音高音符（扫弦里两根弦同音、重复展开的 hit）只留一个；
    * 后一个音开始得比前一个音结束还早 -> 把前一个截短到后一个开始处。
    """
    buckets: dict[int, list[tuple[int, int, int, Any]]] = {}
    for start, duration, pitch, payload in spans:
        buckets.setdefault(int(pitch), []).append((max(0, int(start)), max(1, int(duration)), int(pitch), payload))

    dropped = 0
    truncated = 0
    result: list[tuple[int, int, int, Any]] = []
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0])
        kept: list[list[Any]] = []
        for start, duration, pitch, payload in bucket:
            if kept and start == kept[-1][0]:
                dropped += 1
                if duration > kept[-1][1]:  # 保留更长的那个音
                    kept[-1][1] = duration
                    kept[-1][3] = payload
                continue
            if kept and start < kept[-1][0] + kept[-1][1]:
                kept[-1][1] = max(1, start - kept[-1][0])
                truncated += 1
            kept.append([start, duration, pitch, payload])
        result.extend(tuple(item) for item in kept)  # type: ignore[misc]
    if dropped:
        warnings.append(f"{label}：合并了 {dropped} 个同一 tick 上重复的同音高音符")
    if truncated:
        warnings.append(f"{label}：截短了 {truncated} 个互相重叠的同音高音符")
    result.sort(key=lambda item: (item[0], item[2]))
    return result


def _build_vocal_track(events: Sequence[dict], warnings: list[str]) -> _Track:
    """轨道 1：人声旋律（含歌词文本事件）。"""
    track = _Track("Vocal", channel=0, program=_PROGRAM_VOCAL)
    spans = [
        (int(event["start"]), int(event["end"]) - int(event["start"]), int(event["pitch"]), event)
        for event in events
    ]
    for start, duration, pitch, event in _sanitize_spans(spans, warnings, "人声轨"):
        track.note(0, start, duration, pitch, event["velocity"])
        if event.get("lyric"):
            track.add(start, _ORDER_TEXT, _text_meta(event["lyric"]))
    if not events:
        warnings.append("没有旋律音符，MIDI 人声轨为空")
    return track


def _voicing_frets(chord: dict, ctx: _Context) -> list[int] | None:
    """取和弦指法的 6 个品位（低音弦在前）；找不到时返回 None。"""
    voicing_id = chord.get("voicing_id")
    voicing = ctx.voicings.get(str(voicing_id)) if voicing_id else None
    if voicing is None:
        symbol = str(chord.get("symbol") or "").strip()
        for candidate in ctx.voicings.values():
            if str(candidate.get("symbol") or "").strip() == symbol:
                voicing = candidate
                break
    if voicing is None:
        return None
    frets = voicing.get("frets") or []
    result: list[int] = []
    for fret in frets[:6]:
        try:
            result.append(int(fret))
        except (TypeError, ValueError):
            result.append(-1)
    while len(result) < 6:
        result.append(-1)
    return result


def _sounding_strings(frets: Sequence[int], ctx: _Context) -> list[tuple[int, int, int]]:
    """指法 -> [(弦号, 品位, midi)]，低音弦在前（弦号 6..1）。"""
    strings: list[tuple[int, int, int]] = []
    for index, fret in enumerate(frets[:6]):
        if fret < 0:
            continue
        string_number = 6 - index
        midi = fret_to_midi(ctx.tuning_midis, string_number, fret)
        if midi is None:
            continue
        strings.append((string_number, fret, midi))
    return strings


def _pattern_for(bar: dict, ctx: _Context) -> dict | None:
    """取小节使用的节奏型；找不到时退回任意 4/4 型（并保持稳定）。"""
    guitar = bar.get("guitar") or {}
    pattern_id = guitar.get("pattern_id")
    pattern = ctx.patterns.get(str(pattern_id)) if pattern_id else None
    if pattern is not None:
        return pattern
    if not ctx.patterns:
        return None
    # 按 id 排序后取第一个，保证确定性（不能依赖 dict 插入顺序里的随机内容）
    return ctx.patterns[sorted(ctx.patterns)[0]]


def _collect_guitar(
    bars: list[dict],
    ctx: _Context,
    warnings: list[str],
) -> list[tuple[int, int, int, int]]:
    """按 pattern 展开扫弦，返回 [(start_tick, duration, pitch, velocity)]。"""
    notes: list[tuple[int, int, int, int]] = []
    cursor = 0
    missing_voicing: set[str] = set()
    for bar in bars:
        bar_ticks = _bar_ticks(bar, ctx)
        chords = sorted(
            (chord for chord in (bar.get("chords") or []) if isinstance(chord, dict)),
            key=lambda item: _safe_beats(item.get("beat")),
        )
        if not chords:
            cursor += bar_ticks
            continue
        guitar = bar.get("guitar") or {}
        hits = [hit for hit in (guitar.get("hits") or []) if isinstance(hit, dict) and hit.get("strings")]
        pattern = _pattern_for(bar, ctx)
        if not hits and pattern is None:
            warnings.append(f"第 {bar.get('index')} 小节既没有 hits 也没有节奏型，吉他轨跳过该小节")
            cursor += bar_ticks
            continue

        # 预取每个和弦的指法
        voiced: list[tuple[float, dict, list[tuple[int, int, int]]]] = []
        for chord in chords:
            frets = _voicing_frets(chord, ctx)
            if frets is None:
                symbol = str(chord.get("symbol") or "")
                if symbol not in missing_voicing:
                    missing_voicing.add(symbol)
                    warnings.append(f"和弦 {symbol!r} 没有指法，扫弦轨跳过它")
                continue
            strings = _sounding_strings(frets, ctx)
            if not strings:
                continue
            voiced.append((_safe_beats(chord.get("beat")), chord, strings))
        if not voiced:
            cursor += bar_ticks
            continue

        def chord_at(beat: float) -> list[tuple[int, int, int]]:
            active = voiced[0][2]
            for chord_beat, _chord, strings in voiced:
                if chord_beat <= beat + _EPS:
                    active = strings
            return active

        if hits:
            for hit in hits:
                beat = min(_safe_beats(hit.get("beat")), max(0.0, bar_ticks / PPQ))
                start = cursor + int(round(beat * PPQ))
                duration = max(1, int(round(_safe_dur(hit.get("dur"), 0.5) * PPQ)))
                stroke = str(hit.get("stroke") or "down")
                strings = chord_at(beat)
                by_number = {number: midi for number, _fret, midi in strings}
                chosen: list[int] = []
                for pair in hit.get("strings") or []:
                    try:
                        number = int(pair[0])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if number in by_number:
                        chosen.append(by_number[number])
                if not chosen:
                    chosen = [midi for _number, _fret, midi in strings]
                velocity = 84 if hit.get("accent") else 68
                notes.extend(_stroke_notes(start, duration, chosen, stroke, velocity))
        else:
            events = [
                event for event in (pattern.get("events") or []) if isinstance(event, dict)
            ]
            for event in events:
                beat = _safe_beats(event.get("beat"))
                if beat >= bar_ticks / PPQ - _EPS:
                    continue
                start = cursor + int(round(beat * PPQ))
                duration = max(1, int(round(_safe_dur(event.get("dur"), 0.5) * PPQ)))
                stroke = str(event.get("stroke") or "down")
                strings = chord_at(beat)
                midis = [midi for _number, _fret, midi in strings]
                velocity = 88 if event.get("accent") else 66
                notes.extend(_stroke_notes(start, duration, midis, stroke, velocity))
        cursor += bar_ticks
    return notes


def _stroke_notes(
    start: int,
    duration: int,
    midis: Sequence[int],
    stroke: str,
    velocity: int,
) -> list[tuple[int, int, int, int]]:
    """把一次扫弦动作展开成若干（错开几 tick 的）音符。"""
    if not midis:
        return []
    if stroke == "bass":
        selected = [midis[0]]
        return [(start, max(duration * 2, PPQ // 2), selected[0], velocity)]
    if stroke == "pluck":
        selected = list(reversed(list(midis)))[:2] or list(midis)[:1]
    elif stroke == "up":
        selected = list(reversed(list(midis)))
    elif stroke == "mute":
        # 闷音：只用最低两根弦，短促、轻
        selected = list(midis)[:2]
        return [
            (start + index * 4, max(30, duration // 3), midi, max(20, velocity - 40))
            for index, midi in enumerate(selected)
        ]
    else:  # down
        selected = list(midis)
    notes: list[tuple[int, int, int, int]] = []
    for index, midi in enumerate(selected):
        offset = index * STRUM_SPREAD
        # 扫弦弦数多时把时值略微压缩，避免后一根弦拖过下一个音
        length = max(PPQ // 4, duration - offset)
        notes.append((start + offset, length, midi, velocity))
    return notes


def _build_guitar_track(notes: Sequence[tuple[int, int, int, int]], warnings: list[str]) -> _Track:
    track = _Track("Guitar", channel=1, program=_PROGRAM_GUITAR)
    spans = [(int(start), int(duration), int(pitch), int(velocity)) for start, duration, pitch, velocity in notes]
    for start, duration, pitch, velocity in _sanitize_spans(spans, warnings, "吉他轨"):
        track.note(1, start, duration, pitch, velocity)
    if not notes:
        warnings.append("没有吉他音，MIDI 吉他轨为空")
    return track


def _build_click_track(bars: list[dict], ctx: _Context) -> _Track:
    """轨道 3：节拍器（ch9，downbeat 用 77，其余用 76）。"""
    track = _Track("Click", channel=9, program=_PROGRAM_CLICK)
    cursor = 0
    for bar in bars:
        bar_quarters = _bar_quarters(bar, ctx)
        signature = parse_time_signature(bar.get("time_signature") or ctx.time_signature, ctx.time_signature)
        # 每「分析节拍」一下；6/8 这类复拍子按八分音符点数
        beats = int(signature[0]) if signature[1] >= 4 else int(bar_quarters)
        for beat in range(max(1, beats)):
            tick = cursor + int(round(beat * (bar_quarters / max(1, beats)) * PPQ))
            if tick >= cursor + _bar_ticks(bar, ctx):
                break
            track.note(9, tick, 40, 77 if beat == 0 else 76, 100 if beat == 0 else 76)
        cursor += _bar_ticks(bar, ctx)
    return track


# ---------------------------------------------------------------- 组装与写盘


def _assemble(tracks: Sequence[_Track]) -> bytes:
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), PPQ)
    return header + b"".join(track.to_chunk() for track in tracks)


def _write_atomic(out_path: Path, data: bytes) -> None:
    """先写临时文件再原子替换，任何异常都不留下半截文件。"""
    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"无法创建导出目录 {out_path.parent}：{exc}") from exc
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, out_path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExportError(f"写入 MIDI 失败（{out_path}）：{exc}") from exc


# ---------------------------------------------------------------- 解析器（往返自检用）


def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        if pos >= len(data):
            raise ValueError("MIDI 数据意外结束（变长整数）")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def _parse_track(body: bytes) -> dict:
    """解析单条 MTrk 数据块 -> {notes, meta, event_count}。"""
    pos = 0
    tick = 0
    running: int | None = None
    open_notes: dict[tuple[int, int], list[list[int]]] = {}
    notes: list[dict] = []
    meta: list[dict] = []
    events = 0
    while pos < len(body):
        delta, pos = _read_varlen(body, pos)
        tick += delta
        if pos >= len(body):
            break
        status = body[pos]
        if status == 0xFF:  # 元事件
            pos += 1
            if pos >= len(body):
                break
            mtype = body[pos]
            pos += 1
            length, pos = _read_varlen(body, pos)
            payload = body[pos : pos + length]
            pos += length
            meta.append({"tick": tick, "type": mtype, "data": payload})
            events += 1
            if mtype == 0x2F:
                break
            continue
        if status in (0xF0, 0xF7):  # sysex
            pos += 1
            length, pos = _read_varlen(body, pos)
            pos += length
            events += 1
            continue
        if status & 0x80:
            running = status
            pos += 1
        elif running is None:
            raise ValueError("MIDI 数据缺少 running status")
        status = running
        kind = status & 0xF0
        channel = status & 0x0F
        data_length = 1 if kind in (0xC0, 0xD0) else 2
        if pos + data_length > len(body):
            raise ValueError("MIDI 数据意外结束（事件参数）")
        params = body[pos : pos + data_length]
        pos += data_length
        events += 1
        if kind == 0x90 and params[1] > 0:
            open_notes.setdefault((channel, params[0]), []).append([tick, params[1]])
        elif kind in (0x80, 0x90):
            key = (channel, params[0])
            stack = open_notes.get(key)
            if stack:
                start, velocity = stack.pop(0)
                notes.append(
                    {
                        "tick": start,
                        "duration": max(0, tick - start),
                        "channel": channel,
                        "pitch": params[0],
                        "velocity": velocity,
                    }
                )
    notes.sort(key=lambda item: (item["tick"], item["pitch"]))
    return {"notes": notes, "meta": meta, "event_count": events}


def parse_midi(data: bytes) -> dict:
    """解析标准 MIDI 文件，返回轨道 / 音符 / 元事件（用于导出往返校验）。"""
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("不是标准 MIDI 文件（缺少 MThd）")
    header_length = struct.unpack(">I", data[4:8])[0]
    format_type, track_count, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + header_length
    tracks: list[dict] = []
    for index in range(track_count):
        if data[pos : pos + 4] != b"MTrk":
            raise ValueError(f"第 {index} 条轨道缺少 MTrk 标识")
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + length]
        pos += 8 + length
        parsed = _parse_track(body)
        name = ""
        for item in parsed["meta"]:
            if item["type"] == 0x03:
                name = item["data"].decode("utf-8", "replace")
                break
        parsed["name"] = name
        parsed["index"] = index
        tracks.append(parsed)
    return {
        "format": format_type,
        "track_count": track_count,
        "division": division,
        "tracks": tracks,
    }


# ---------------------------------------------------------------- 对外入口


def export_midi(score: dict, out_path: Path) -> dict:
    """把 ``aitsm.score/v1`` 乐谱导出为标准 MIDI 文件（format 1，PPQ=480）。

    返回
    ----
    dict: ``{"format": "midi", "path", "size_bytes", "smf_format", "ppq",
    "tracks", "notes": {...}, "tempo_events", "time_signature_events",
    "duration_sec", "warnings"}``。

    异常
    ----
    :class:`ExportError`: 输入不是合法谱面、小节为空、写盘失败或往返自检不通过。
    """
    if not isinstance(score, dict):
        raise ExportError(f"MIDI 导出需要 score dict，收到 {type(score).__name__}")
    if score.get("schema") != "aitsm.score/v1":
        raise ExportError(f"MIDI 导出需要 schema=aitsm.score/v1，收到 {score.get('schema')!r}")
    bars = _bars(score)
    if not bars:
        raise ExportError("乐谱没有小节（bars 为空），无法导出 MIDI")

    warnings: list[str] = []
    ctx = _Context(score)

    tempo_track = _build_tempo_track(bars, ctx, warnings)
    vocal_events = _collect_vocal(bars, ctx, warnings)
    vocal_track = _build_vocal_track(vocal_events, warnings)
    guitar_notes = _collect_guitar(bars, ctx, warnings)
    guitar_track = _build_guitar_track(guitar_notes, warnings)
    tracks = [tempo_track, vocal_track, guitar_track]
    if ctx.include_metronome:
        click_track = _build_click_track(bars, ctx)
        if click_track.note_count == 0:
            warnings.append("节拍器轨为空，已跳过")
        else:
            tracks.append(click_track)

    data = _assemble(tracks)
    # 往返自检：立刻读回来核对，确保文件不会「能存不能播」
    parsed = parse_midi(data)
    if parsed["format"] != 1 or parsed["track_count"] != len(tracks) or parsed["division"] != PPQ:
        raise ExportError("MIDI 自检失败：文件头信息与预期不符")
    parsed_notes = sum(len(track["notes"]) for track in parsed["tracks"])
    written_notes = sum(track.note_count for track in tracks)
    if parsed_notes != written_notes:
        raise ExportError(f"MIDI 自检失败：写入 {written_notes} 个音符，读回 {parsed_notes} 个")
    _write_atomic(Path(out_path), data)

    total_ticks = sum(_bar_ticks(bar, ctx) for bar in bars)
    duration_sec = round(total_ticks / PPQ * 60.0 / ctx.tempo, 3)
    tempo_events = sum(
        1 for track in parsed["tracks"] for item in track["meta"] if item["type"] == 0x51
    )
    time_signature_events = sum(
        1 for track in parsed["tracks"] for item in track["meta"] if item["type"] == 0x58
    )
    return {
        "format": "midi",
        "path": str(Path(out_path)),
        "size_bytes": len(data),
        "smf_format": 1,
        "ppq": PPQ,
        "tracks": [track.name for track in tracks],
        "notes": {
            "vocal": vocal_track.note_count,
            "guitar": guitar_track.note_count,
            "metronome": tracks[3].note_count if len(tracks) > 3 else 0,
            "total": written_notes,
        },
        "tempo_events": tempo_events,
        "time_signature_events": time_signature_events,
        "tempo_bpm": ctx.tempo,
        "duration_sec": duration_sec,
        "warnings": warnings,
    }

"""融合层：把「人声联合转录」和「伴奏联合转录」对齐成 :class:`SymbolicScore`。

三件事
------
``fuse``
    旋律优先取人声转录的结果；人声没唱到的空白段落用伴奏转录给出的
    ``melody_skeleton`` 补，并把两个引擎的 provenance / warnings 原样透传。
``snap_notes_to_beats``
    只把**音头**吸到最近的节拍网格上（默认容差 60 ms），时值保持不变，
    保证结果按起点单调、``end`` 不早于 ``start``。
``write_performance_midi``
    把 :class:`SymbolicScore` 写成一份可播放的 performance SMF
    （format 1 / PPQ=480；tempo、拍号、调号、段落标记、旋律、和弦）。

重模块（尤其 :mod:`aitsm.score.export.midi`）全部在函数内惰性导入。
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import (
    AccompanimentResult,
    BeatGrid,
    ChordEvent,
    KeySignature,
    LyricLine,
    LyricWord,
    MelodyNote,
    Provenance,
    Section,
    SymbolicScore,
    VocalResult,
    merge_warnings,
)

__all__ = [
    "fuse",
    "snap_notes_to_beats",
    "write_performance_midi",
]

#: 融合层在 provenance 里的引擎名（规则式骨架融合，不是模型）。
FUSION_ENGINE = "rule-skeleton"

#: 至少空这么长（秒）的段落才值得用伴奏骨架补，默认一个小节。
_MIN_FILL_BARS = 1.0


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _warn(
    message: str,
    *,
    level: str = "warn",
    code: str = "",
    stage: str = "fusion",
) -> dict[str, Any]:
    data: dict[str, Any] = {"level": level, "message": message, "stage": stage}
    if code:
        data["code"] = code
    return data


def _engine_warnings(items: Iterable[Any], stage: str, code: str) -> list[dict[str, Any]]:
    """把子引擎的 ``list[str]`` 警告转成 :attr:`SymbolicScore.warnings` 的 dict 形状。"""
    out: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, Mapping):
            data = dict(item)
            data.setdefault("stage", stage)
            data.setdefault("level", "warn")
            if code:
                data.setdefault("code", code)
            out.append(data)
        else:
            text = str(item or "").strip()
            if text:
                out.append(_warn(text, stage=stage, code=code))
    return out


def _bar_length(beats: BeatGrid) -> float:
    return max(1e-6, float(beats.beat_period) * max(1, int(beats.beats_per_bar)))


def _span(
    melody: Sequence[MelodyNote],
    skeleton: Sequence[MelodyNote],
    chords: Sequence[ChordEvent],
    sections: Sequence[Section],
    beats: BeatGrid,
) -> float:
    end = 0.0
    for item in melody:
        end = max(end, float(item.end))
    for item in skeleton:
        end = max(end, float(item.end))
    for item in chords:
        end = max(end, float(item.end))
    for item in sections:
        end = max(end, float(item.end))
    if beats.beats:
        end = max(end, float(beats.beats[-1]) + float(beats.beat_period))
    if beats.downbeats:
        end = max(end, float(beats.downbeats[-1]) + _bar_length(beats))
    return max(0.0, end)


def _gaps(intervals: Sequence[tuple[float, float]], span: float) -> list[tuple[float, float]]:
    """从排好序的覆盖区间里找出 ``[0, span]`` 上的空白段。"""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        start = max(0.0, min(float(start), span))
        end = max(start, min(float(end), span))
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > 1e-6:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if span - cursor > 1e-6:
        gaps.append((cursor, span))
    return gaps


def _overlaps(start: float, end: float, intervals: Sequence[tuple[float, float]]) -> bool:
    for low, high in intervals:
        if start < high - 1e-6 and end > low + 1e-6:
            return True
    return False


def _dedupe_notes(notes: Sequence[MelodyNote]) -> list[MelodyNote]:
    """同一 midi、同一时刻的重复音只留一个（保留置信度高的）。"""
    best: dict[tuple[int, float], MelodyNote] = {}
    for note in notes:
        key = (int(note.midi), round(float(note.start), 3))
        current = best.get(key)
        if current is None or float(note.confidence) > float(current.confidence):
            best[key] = note
    return sorted(best.values(), key=lambda item: (item.start, item.midi, item.end))


def _lyrics_from_notes(notes: Sequence[MelodyNote], beat_period: float) -> list[LyricLine]:
    """从音符自带的 ``lyric`` 反推歌词行（人声引擎没给行结构时的兜底）。"""
    sung = [note for note in sorted(notes, key=lambda item: item.start) if str(note.lyric or "").strip()]
    if not sung:
        return []
    gap_limit = max(1.5, 3.0 * max(1e-3, float(beat_period)))
    lines: list[LyricLine] = []
    current: list[MelodyNote] = []
    for note in sung:
        if current and note.start - current[-1].end > gap_limit:
            lines.append(_line_from_notes(current))
            current = []
        current.append(note)
    if current:
        lines.append(_line_from_notes(current))
    return lines


def _line_from_notes(notes: Sequence[MelodyNote]) -> LyricLine:
    words = [
        LyricWord(start=float(note.start), end=float(note.end), text=str(note.lyric))
        for note in notes
    ]
    return LyricLine(
        start=float(notes[0].start),
        end=max(float(note.end) for note in notes),
        text=" ".join(word.text for word in words),
        words=words,
        confidence=min(float(note.confidence) for note in notes),
    )


# --------------------------------------------------------------------------- #
# 融合
# --------------------------------------------------------------------------- #


def _fill_from_skeleton(
    primary: Sequence[MelodyNote],
    skeleton: Sequence[MelodyNote],
    beats: BeatGrid,
    *,
    min_gap_bars: float = _MIN_FILL_BARS,
) -> tuple[list[MelodyNote], int, float]:
    """用伴奏骨架补「人声完全没唱」的空段。

    返回 ``(补进来的音符, 个数, 补上的秒数)``。只有整段空白（默认 ≥1 小节）
    才会补，避免把人声的换气口填成假音。
    """
    if not skeleton:
        return [], 0, 0.0
    span = _span(primary, skeleton, [], [], beats)
    min_gap = _bar_length(beats) * max(0.25, float(min_gap_bars))
    covered = [(float(note.start), float(note.end)) for note in primary]
    gaps = [gap for gap in _gaps(covered, span) if gap[1] - gap[0] >= min_gap]
    if not gaps:
        return [], 0, 0.0
    filled: list[MelodyNote] = []
    total_seconds = 0.0
    for note in sorted(skeleton, key=lambda item: (item.start, item.midi)):
        start, end = float(note.start), float(note.end)
        if start >= span:
            continue
        if not any(low - 1e-6 <= start and end <= high + 1e-6 for low, high in gaps):
            continue
        if _overlaps(start, end, covered):
            continue
        filled.append(
            MelodyNote(
                start=start,
                end=max(start, end),
                midi=int(note.midi),
                confidence=float(note.confidence) * 0.85,
                lyric=str(note.lyric or ""),
                syllable_index=note.syllable_index,
                source="skeleton",
            )
        )
        covered.append((start, end))
        total_seconds += max(0.0, end - start)
    return filled, len(filled), total_seconds


def fuse(
    vocal: VocalResult,
    accompaniment: AccompanimentResult,
    *,
    meta: Mapping[str, Any] | None = None,
    audio: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> SymbolicScore:
    """把两路联合转录合成一份 :class:`SymbolicScore`。

    * 节拍 / 调性 / 和弦 / 段落来自伴奏转录（人声侧没有这些字段）。
    * 旋律：人声优先；某段人声完全空白（≥1 小节）且伴奏骨架有音时才补，
      补进来的音 ``source="skeleton"``、置信度打折。
    * 歌词：优先用人声给的行结构；没有就从音符的 ``lyric`` 拼回。
    * provenance 依次写 ``vocal_transcription`` / ``accompaniment_transcription``
      / ``fusion`` 三条，引擎与 requested 原样保留（降级信息不丢）。
    """
    if not isinstance(vocal, VocalResult):
        raise TypeError(f"fuse 的 vocal 必须是 VocalResult，收到 {type(vocal).__name__}")
    if not isinstance(accompaniment, AccompanimentResult):
        raise TypeError(
            f"fuse 的 accompaniment 必须是 AccompanimentResult，收到 {type(accompaniment).__name__}"
        )

    beats = accompaniment.beats or BeatGrid(bpm=vocal.bpm or 120.0)
    key = accompaniment.key or KeySignature(tonic="C")
    warnings: list[dict[str, Any]] = []
    warnings.extend(_engine_warnings(vocal.warnings, "vocal_transcription", "vocal_warning"))
    warnings.extend(
        _engine_warnings(accompaniment.warnings, "accompaniment_transcription", "accompaniment_warning")
    )

    vocal_notes = _dedupe_notes(vocal.notes)
    filled, filled_count, filled_seconds = _fill_from_skeleton(
        vocal_notes, accompaniment.melody_skeleton, beats
    )

    melody = _dedupe_notes(list(vocal_notes) + list(filled))

    if not vocal_notes:
        warnings.append(
            _warn(
                "人声转录没有产出音符，旋律完全来自伴奏骨架" if filled_count else "人声转录没有产出音符",
                stage="fusion",
                code="melody_from_skeleton" if filled_count else "no_vocal_melody",
            )
        )
    elif filled_count:
        warnings.append(
            _warn(
                f"用伴奏旋律骨架补齐 {filled_count} 个人声空白音（{filled_seconds:.2f}s）",
                level="info",
                stage="fusion",
                code="skeleton_fill",
            )
        )
    if not melody:
        warnings.append(
            _warn("融合后旋律为空，下游只能得到和弦/节拍骨架", stage="fusion", code="no_melody")
        )

    lyrics = list(vocal.lyrics or [])
    if not lyrics:
        lyrics = _lyrics_from_notes(melody, beats.beat_period)

    span = _span(melody, [], accompaniment.chords, accompaniment.sections, beats)

    vocal_provenance_ok = bool(vocal_notes)
    fusion_detail: dict[str, Any] = {
        "vocal_notes": len(vocal_notes),
        "skeleton_notes": len(accompaniment.melody_skeleton or []),
        "skeleton_filled": filled_count,
        "skeleton_filled_seconds": round(float(filled_seconds), 3),
        "melody_notes": len(melody),
        "chords": len(accompaniment.chords or []),
        "sections": len(accompaniment.sections or []),
        "lyrics": len(lyrics),
        "span_seconds": round(float(span), 3),
    }

    provenance = [
        _provenance(
            layer="vocal_transcription",
            engine=vocal.engine or "none",
            requested=vocal.requested,
            ok=vocal_provenance_ok,
            elapsed_seconds=vocal.elapsed_seconds,
            detail={
                "notes": len(vocal.notes or []),
                "lyrics": len(vocal.lyrics or []),
                "bpm": vocal.bpm,
                "aligned": bool(vocal.alignment),
            },
            warnings=list(vocal.warnings or []),
        ),
        _provenance(
            layer="accompaniment_transcription",
            engine=accompaniment.engine or "none",
            requested=accompaniment.requested,
            ok=bool(accompaniment.chords or accompaniment.sections or beats.beats),
            elapsed_seconds=accompaniment.elapsed_seconds,
            detail={
                "chords": len(accompaniment.chords or []),
                "sections": len(accompaniment.sections or []),
                "bpm": round(float(beats.bpm), 3),
                "key": key.display,
                "skeleton_notes": len(accompaniment.melody_skeleton or []),
            },
            warnings=list(accompaniment.warnings or []),
        ),
        _provenance(
            layer="fusion",
            engine=FUSION_ENGINE,
            requested="fusion",
            ok=bool(melody or accompaniment.chords),
            detail=fusion_detail,
        ),
    ]

    meta_out: dict[str, Any] = dict(meta or {})
    meta_out.setdefault("bpm", round(float(beats.bpm), 3))
    meta_out.setdefault("key", key.display)
    meta_out.setdefault("time_signature", beats.time_signature)
    meta_out.setdefault("duration_seconds", round(float(span), 3))
    if vocal_notes and filled_count:
        melody_source = "fused"
    elif vocal_notes:
        melody_source = "vocal"
    elif melody:
        melody_source = "skeleton"
    else:
        melody_source = "none"
    meta_out.setdefault("melody_source", melody_source)
    if vocal.bpm and "vocal_bpm" not in meta_out:
        meta_out["vocal_bpm"] = round(float(vocal.bpm), 3)

    source_out: dict[str, Any] = dict(source or {})
    source_out.setdefault("vocal_engine", vocal.engine or "none")
    source_out.setdefault("accompaniment_engine", accompaniment.engine or "none")
    if accompaniment.abc:
        source_out.setdefault("abc", accompaniment.abc)

    return SymbolicScore(
        beats=beats,
        key=key,
        chords=list(accompaniment.chords or []),
        sections=list(accompaniment.sections or []),
        melody=melody,
        lyrics=lyrics,
        meta=meta_out,
        audio=dict(audio or {}),
        source=source_out,
        provenance=provenance,
        warnings=merge_warnings(warnings),
    )


def _provenance(
    *,
    layer: str,
    engine: str,
    requested: str,
    ok: bool,
    elapsed_seconds: float | None = None,
    detail: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
) -> Provenance:
    return Provenance(
        layer=layer,
        engine=str(engine),
        requested=str(requested or ""),
        ok=bool(ok),
        elapsed_seconds=elapsed_seconds,
        detail=dict(detail or {}),
        warnings=[str(item) for item in (warnings or [])],
    )


# --------------------------------------------------------------------------- #
# 音头吸附
# --------------------------------------------------------------------------- #


def _grid_points(beatgrid: BeatGrid | None, span: float) -> list[float]:
    """取节拍网格点（beats + downbeats）；网格缺失就按 bpm 铺到 ``span``。"""
    if beatgrid is None:
        return []
    points = {round(float(item), 6) for item in (beatgrid.beats or [])}
    points.update(round(float(item), 6) for item in (beatgrid.downbeats or []))
    if len(points) < 2:
        period = float(beatgrid.beat_period)
        if period <= 0 or span <= 0:
            return sorted(points)
        first = float(beatgrid.first_downbeat_sec or 0.0)
        if first > span:
            first = 0.0
        position = first
        while position <= span + period:
            points.add(round(position, 6))
            position += period
    return sorted(points)


def _nearest(points: Sequence[float], value: float) -> float | None:
    if not points:
        return None
    index = bisect.bisect_left(points, value)
    candidates: list[float] = []
    if index < len(points):
        candidates.append(points[index])
    if index > 0:
        candidates.append(points[index - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item - value))


def snap_notes_to_beats(
    notes: Sequence[MelodyNote],
    beatgrid: BeatGrid | None,
    *,
    tolerance: float = 0.06,
) -> list[MelodyNote]:
    """把每个音的**音头**吸到最近的节拍网格点。

    * 只动 ``start``：容差（默认 60 ms）内才吸，时值（``end - start``）原样保留，
      ``end`` 严格不早于 ``start``。
    * 结果按起点单调递增；两个音被吸到同一格时，后来者保留原始起点（仍晚于前者）。
    * ``source`` 标记为 ``"snapped"``，便于下游知道动过哪些音。
    """
    ordered = sorted(
        (note for note in notes if isinstance(note, MelodyNote)),
        key=lambda item: (float(item.start), float(item.end), int(item.midi)),
    )
    if not ordered:
        return []

    tolerance = max(0.0, float(tolerance))
    span = max(float(note.end) for note in ordered)
    points = _grid_points(beatgrid, span)
    if not points:
        return [
            MelodyNote(
                start=float(note.start),
                end=max(float(note.start), float(note.end)),
                midi=int(note.midi),
                confidence=float(note.confidence),
                lyric=str(note.lyric or ""),
                syllable_index=note.syllable_index,
                source=str(note.source or ""),
            )
            for note in ordered
        ]

    out: list[MelodyNote] = []
    previous_start: float | None = None
    for note in ordered:
        original_start = float(note.start)
        duration = max(0.0, float(note.end) - original_start)
        nearest = _nearest(points, original_start)
        start = nearest if nearest is not None and abs(nearest - original_start) <= tolerance else original_start
        if previous_start is not None and start <= previous_start:
            # 同一网格点上的第二个音：保留原始起点（它本来就更晚），仍保证单调
            start = original_start if original_start > previous_start else previous_start
        moved = abs(start - original_start) > 1e-9
        out.append(
            MelodyNote(
                start=start,
                end=start + duration,
                midi=int(note.midi),
                confidence=float(note.confidence),
                lyric=str(note.lyric or ""),
                syllable_index=note.syllable_index,
                source="snapped" if moved else str(note.source or ""),
            )
        )
        previous_start = start
    return out


# --------------------------------------------------------------------------- #
# performance MIDI
# --------------------------------------------------------------------------- #

#: 和弦符号后缀 -> 相对根音的半音集合（够用就好，认不出的一律大三和弦）。
_CHORD_INTERVALS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("maj13", (0, 4, 7, 11, 14, 21)),
    ("maj11", (0, 4, 7, 11, 14, 17)),
    ("maj9", (0, 4, 7, 11, 14)),
    ("maj7", (0, 4, 7, 11)),
    ("m7b5", (0, 3, 6, 10)),
    ("m7", (0, 3, 7, 10)),
    ("m9", (0, 3, 7, 10, 14)),
    ("m11", (0, 3, 7, 10, 14, 17)),
    ("m13", (0, 3, 7, 10, 14, 21)),
    ("m6", (0, 3, 7, 9)),
    ("dim7", (0, 3, 6, 9)),
    ("dim", (0, 3, 6)),
    ("aug", (0, 4, 8)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
    ("sus", (0, 5, 7)),
    ("add9", (0, 4, 7, 14)),
    ("13", (0, 4, 7, 10, 14, 21)),
    ("11", (0, 4, 7, 10, 14, 17)),
    ("9", (0, 4, 7, 10, 14)),
    ("7", (0, 4, 7, 10)),
    ("6", (0, 4, 7, 9)),
    ("5", (0, 7)),
)


def _chord_pitches(symbol: str) -> list[int]:
    """和弦符号 -> 一个四声部以内的钢琴和弦（中音区）。"""
    from .musicxml_io import parse_chord_symbol

    parsed = parse_chord_symbol(symbol)
    if parsed is None:
        return []
    step = parsed["root_step"]
    alter = parsed["root_alter"]
    from .musicxml_io import _STEP_TO_PC as STEP_TO_PC  # 同包内部表，避免重复定义

    root_pc = (STEP_TO_PC.get(str(step).upper()[:1], 0) + int(alter)) % 12
    suffix = str(parsed["suffix"] or "")
    intervals: tuple[int, ...] = (0, 4, 7)
    lowered = suffix.lower()
    for name, values in _CHORD_INTERVALS:
        if lowered == name or lowered.startswith(name):
            intervals = values
            break
    else:
        if lowered.startswith("m") and not lowered.startswith("maj"):
            intervals = (0, 3, 7)
    root_midi = 60 + root_pc
    if root_midi > 71:
        root_midi -= 12
    return [max(0, min(127, root_midi + interval)) for interval in intervals]


def _tick_seconds(seconds: float, beat_period: float, ppq: int) -> int:
    return max(0, int(round(float(seconds) / max(beat_period, 1e-6) * ppq)))


def write_performance_midi(score: SymbolicScore, path: str | Path) -> dict[str, Any]:
    """把 :class:`SymbolicScore` 写成一份合法、可播放的 performance SMF。

    复用 :mod:`aitsm.score.export.midi` 的内部编码器（惰性导入）：format 1、
    PPQ=480、第 0 轨 tempo/拍号/调号/段落标记，第 1 轨旋律，第 2 轨和弦。
    写盘后立刻读回自检（轨道数 / 音符数 / MThd），失败直接抛异常，不写半截文件。

    返回 ``{"format": "midi", "path", "size_bytes", "smf_format", "ppq",
    "tracks", "notes": {...}, "duration_seconds", "warnings"}``。
    """
    from ..score.export.midi import (  # 惰性导入：只有真的要写 MIDI 时才加载
        PPQ,
        _ORDER_META,
        _ORDER_TEXT,
        _Track,
        _assemble,
        _key_signature_meta,
        _tempo_meta,
        _text_meta,
        _time_signature_meta,
        _write_atomic,
        parse_midi,
    )

    if not isinstance(score, SymbolicScore):
        raise TypeError(
            f"write_performance_midi 需要 SymbolicScore，收到 {type(score).__name__}"
        )

    warnings: list[str] = []
    beat_period = float(score.beats.beat_period)

    tempo_track = _Track("tempo")
    numerator, denominator = _time_signature_parts(score.beats.time_signature, score.beats.beats_per_bar)
    tempo_track.add(0, _ORDER_META, _time_signature_meta(numerator, denominator))
    tempo_track.add(
        0,
        _ORDER_META,
        _key_signature_meta(int(score.key.fifths), score.key.mode == "minor"),
    )
    tempo_track.add(0, _ORDER_META, _tempo_meta(float(score.beats.bpm)))
    title = str(score.meta.get("title") or "").strip()
    if title:
        tempo_track.add(0, _ORDER_TEXT, _text_meta(title))
    for section in score.sections:
        label = str(section.label or section.kind or "").strip()
        if not label:
            continue
        tick = _tick_seconds(section.start, beat_period, PPQ)
        tempo_track.add(tick, _ORDER_TEXT, _text_meta(f"[{label}]"))

    melody_track = _Track("melody", channel=0, program=53)
    for note in sorted(score.melody, key=lambda item: (item.start, item.midi)):
        start = _tick_seconds(note.start, beat_period, PPQ)
        end = _tick_seconds(note.end if note.end > note.start else note.start, beat_period, PPQ)
        duration = max(1, end - start)
        velocity = 88 if float(note.confidence) >= 0.5 else 72
        melody_track.note(0, start, duration, int(note.midi), velocity)
    if not score.melody:
        warnings.append("旋律为空：performance MIDI 只有和弦与节拍信息")

    chord_track = _Track("chords", channel=1, program=0)
    for chord in sorted(score.chords, key=lambda item: item.start):
        pitches = _chord_pitches(str(chord.symbol or ""))
        if not pitches:
            continue
        start = _tick_seconds(chord.start, beat_period, PPQ)
        end = _tick_seconds(max(chord.end, chord.start), beat_period, PPQ)
        duration = max(PPQ // 8, end - start)
        for pitch in pitches:
            chord_track.note(1, start, duration, pitch, 58)

    tracks = [tempo_track, melody_track, chord_track]
    data = _assemble(tracks)

    parsed = parse_midi(data)
    written = sum(track.note_count for track in tracks)
    read_back = sum(len(track["notes"]) for track in parsed["tracks"])
    if parsed["format"] != 1 or parsed["division"] != PPQ:
        raise ValueError("MIDI 自检失败：文件头与预期不符")
    if parsed["track_count"] != len(tracks):
        raise ValueError("MIDI 自检失败：轨道数与预期不符")
    if read_back != written:
        raise ValueError(f"MIDI 自检失败：写入 {written} 个音符，读回 {read_back} 个")

    out_path = Path(path)
    _write_atomic(out_path, data)

    return {
        "format": "midi",
        "path": str(out_path),
        "size_bytes": len(data),
        "smf_format": int(parsed["format"]),
        "ppq": int(parsed["division"]),
        "tracks": [str(track.name) for track in tracks],
        "notes": {
            "melody": melody_track.note_count,
            "chords": chord_track.note_count,
            "total": written,
        },
        "duration_seconds": round(
            max((float(note.end) for note in score.melody), default=0.0), 3
        ),
        "warnings": warnings,
    }


def _time_signature_parts(time_signature: Any, beats_per_bar: Any) -> tuple[int, int]:
    """``"4/4"`` -> ``(4, 4)``；非法时退回四分音符。"""
    text = str(time_signature or "")
    numerator = 0
    denominator = 4
    if "/" in text:
        left, _, right = text.partition("/")
        try:
            numerator = int(left.strip())
            denominator = int(right.strip())
        except ValueError:
            numerator = 0
    if numerator <= 0:
        try:
            numerator = int(beats_per_bar)
        except (TypeError, ValueError):
            numerator = 4
    numerator = max(1, min(255, numerator))
    if denominator <= 0 or (denominator & (denominator - 1)) != 0:
        denominator = 4
    return numerator, denominator

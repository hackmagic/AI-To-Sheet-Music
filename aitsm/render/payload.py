"""渲染载荷：把后端谱面整理成网页渲染器需要的唯一 JSON。

``web/js/musicxml-renderer.js``/VexFlow 侧只需要看这一份 payload：标题、
段落、时间轴、和弦图、ABC 兜底谱与可选 MusicXML。本模块是**纯函数**：
不读文件、不访问网络、不修改入参，缺失字段一律留空，不猜数据。

载荷版本是 ``aitsm.payload/v1``，可被 :func:`json.dumps` 直接序列化
（``NaN``/无穷大统一转 ``None``）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .abc import write_abc
from .timing import build_timing

__all__ = ["PAYLOAD_SCHEMA", "build_payload"]

PAYLOAD_SCHEMA = "aitsm.payload/v1"

_MUSICXML_KEYS = ("musicxml", "music_xml", "musicXml", "MusicXML", "xml")

_EMPTY_TIMING = {"bars": [], "notes": [], "chords": [], "lyrics": []}


def _mapping(source: Any) -> Mapping[str, Any]:
    """把 dict 或带 ``to_dict()`` 的对象统一成只读 Mapping。"""
    if isinstance(source, Mapping):
        return source
    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:  # pragma: no cover - 第三方对象序列化失败
            return {}
        if isinstance(data, Mapping):
            return data
    return {}


def _jsonable(value: Any) -> Any:
    """递归转成 JSON 可序列化值；非有限浮点转 ``None``。"""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    number = _num(value, float("nan"))
    if not math.isfinite(number):
        return None
    return int(round(number))


def _first_text(score: Mapping[str, Any], symbolic: Mapping[str, Any]) -> str:
    """从谱面或符号总谱里找出可选的 MusicXML 文本。"""
    for source in (score, symbolic):
        for key in _MUSICXML_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _voicing_entry(voicing_id: str, voicing: Mapping[str, Any]) -> dict[str, Any]:
    """一份可画的指法图数据（空弦/闷弦/横按都保留原样）。"""
    return {
        "id": voicing_id,
        "symbol": str(voicing.get("symbol") or ""),
        "frets": _jsonable(voicing.get("frets") or []),
        "fingers": _jsonable(voicing.get("fingers") or []),
        "base_fret": _int_or_none(voicing.get("base_fret")),
        "barres": _jsonable(voicing.get("barres") or []),
        "difficulty": _int_or_none(voicing.get("difficulty")),
        "label": str(voicing.get("label") or ""),
    }


def _voicing_index(voicings: Mapping[str, Any]) -> dict[str, str]:
    """和弦符号 -> 指法 id（大小写不敏感，取第一个）。"""
    index: dict[str, str] = {}
    for voicing_id, voicing in voicings.items():
        if not isinstance(voicing, Mapping):
            continue
        symbol = str(voicing.get("symbol") or "").strip().lower()
        if symbol and symbol not in index:
            index[symbol] = str(voicing_id)
    return index


def _chord_diagrams(score: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """指法图：全部 voicing 目录 + 每小节每个和弦对应的图。"""
    raw_voicings = score.get("voicings")
    raw_voicings = raw_voicings if isinstance(raw_voicings, Mapping) else {}
    catalog: dict[str, Any] = {}
    for voicing_id, voicing in raw_voicings.items():
        if isinstance(voicing, Mapping):
            catalog[str(voicing_id)] = _voicing_entry(str(voicing_id), voicing)
    by_symbol = _voicing_index(raw_voicings)

    guitar = meta.get("guitar")
    guitar = guitar if isinstance(guitar, Mapping) else {}
    tuning = _jsonable(guitar.get("tuning") or [])

    bars: list[dict[str, Any]] = []
    chord_count = 0
    for position, bar in enumerate(score.get("bars") or []):
        if not isinstance(bar, Mapping):
            continue
        index = _int_or_none(bar.get("index"))
        if index is None:
            index = position + 1
        diagrams: list[dict[str, Any]] = []
        for chord in bar.get("chords") or []:
            if not isinstance(chord, Mapping):
                continue
            symbol = str(chord.get("symbol") or "").strip()
            voicing_id = str(chord.get("voicing_id") or "").strip()
            if not voicing_id and symbol:
                voicing_id = by_symbol.get(symbol.lower(), "")
            diagram = catalog.get(voicing_id) if voicing_id else None
            diagrams.append(
                {
                    "beat": round(_num(chord.get("beat"), 0.0), 6),
                    "dur": round(_num(chord.get("dur"), 0.0), 6),
                    "symbol": symbol,
                    "voicing_id": voicing_id,
                    "diagram": diagram,
                }
            )
            chord_count += 1
        bars.append({"index": index, "chords": diagrams})

    return {
        "tuning": tuning,
        "capo": _int_or_none(meta.get("capo")) or 0,
        "voicings": catalog,
        "bars": bars,
        "count": chord_count,
    }


def _stats(
    score: Mapping[str, Any],
    timing: Mapping[str, Any],
    diagrams: Mapping[str, Any],
    sections: list[Any],
) -> dict[str, Any]:
    """给界面顶部的计数摘要（全部来自已有数据，不推算音乐内容）。"""
    timing_bars = [bar for bar in (timing.get("bars") or []) if isinstance(bar, Mapping)]
    duration = _num(timing_bars[-1].get("end"), 0.0) if timing_bars else 0.0
    return {
        "bars": len(score.get("bars") or []),
        "sections": len(sections),
        "chords": len(timing.get("chords") or []),
        "notes": len(timing.get("notes") or []),
        "lyrics": len(timing.get("lyrics") or []),
        "voicings": len(diagrams.get("voicings") or {}),
        "duration_sec": round(duration, 6),
    }


def build_payload(
    score: Any,
    *,
    symbolic: Any = None,
    audio_url: str | None = None,
    abc: str | None = None,
) -> dict[str, Any]:
    """构造网页渲染载荷（``aitsm.payload/v1``）。

    参数
    ----
    score:
        ``aitsm.score/v1`` 谱面 dict（或带 ``to_dict()`` 的对象）。
    symbolic:
        可选的 ``aitsm.symbolic/v1`` 符号总谱；用于补 MusicXML 等附加文本。
    audio_url:
        覆盖音频地址；缺省取 ``meta.source.audio_url``。
    abc:
        覆盖 ABC 文本；缺省用 :func:`aitsm.render.abc.write_abc` 生成。

    返回
    ----
    dict: 含 ``schema``/``title``/``meta``/``sections``/``stats``/
    ``musicxml``/``abc``/``timing``/``chord_diagrams``/``audio_url`` 的
    纯数据，可直接 :func:`json.dumps`。
    """
    score_data = _mapping(score)
    symbolic_data = _mapping(symbolic)

    meta_raw = score_data.get("meta")
    meta = _jsonable(meta_raw) if isinstance(meta_raw, Mapping) else {}

    title = str(meta.get("title") or score_data.get("title") or "").strip()

    raw_sections = score_data.get("sections")
    sections = _jsonable(raw_sections) if isinstance(raw_sections, list) else []

    try:
        timing = build_timing(score_data)
    except (TypeError, ValueError, KeyError):
        timing = dict(_EMPTY_TIMING)

    diagrams = _chord_diagrams(score_data, meta)

    if not isinstance(abc, str) or not abc.strip():
        try:
            abc = write_abc(score_data)
        except (TypeError, ValueError, KeyError):
            abc = ""

    if isinstance(audio_url, str) and audio_url.strip():
        resolved_audio = audio_url.strip()
    else:
        source = meta.get("source")
        source = source if isinstance(source, Mapping) else {}
        resolved_audio = str(source.get("audio_url") or "")

    return {
        "schema": PAYLOAD_SCHEMA,
        "title": title,
        "meta": meta,
        "sections": sections,
        "stats": _stats(score_data, timing, diagrams, sections),
        "musicxml": _first_text(score_data, symbolic_data),
        "abc": abc,
        "timing": timing,
        "chord_diagrams": diagrams,
        "audio_url": resolved_audio,
    }

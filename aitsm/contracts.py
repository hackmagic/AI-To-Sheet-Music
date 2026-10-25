"""层间数据契约（v2 架构）。

v2 把整条链路切成五层，每层只通过本模块里的数据类交接，**不允许跨层直接读对方
的中间文件**：

.. code-block:: text

    separation      ──> SeparationResult
    transcribe.vocal        ──> VocalResult
    transcribe.accompaniment──> AccompanimentResult
    fusion                  ──> SymbolicScore  ──> symbolic.json + symbolic.musicxml
    arrangement             ──> score.json (aitsm.score/v1)   # 只吃符号事件，不碰音频
    render                  ──> payload.json

两条硬约束：

1. **时间单位是秒**（绝对音频时间轴），与 ``analysis.json`` 一致；只有在转
   MusicXML 时才换算成「四分音符拍」。保持单一时间单位可以避免下游再猜。
2. **编配层不接触原始音频**：``SymbolicScore`` 是它唯一需要的输入。``audio``
   字段只用于把播放音源地址写进 ``score.json``，不参与任何音乐判断。

``SymbolicScore.to_musicxml()`` / ``from_musicxml()`` 的实现放在
:mod:`aitsm.fusion.musicxml_io`，本模块只在方法里惰性导入，避免循环依赖。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SCHEMA",
    "Provenance",
    "BeatGrid",
    "KeySignature",
    "ChordEvent",
    "Section",
    "MelodyNote",
    "LyricWord",
    "LyricLine",
    "SeparationResult",
    "VocalResult",
    "AccompanimentResult",
    "SymbolicScore",
]

SCHEMA = "aitsm.symbolic/v1"

#: 各层在 provenance 里的标准名字（顺序即执行顺序）
LAYERS: tuple[str, ...] = (
    "separation",
    "vocal_transcription",
    "accompaniment_transcription",
    "fusion",
    "quantization",
)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _s(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return min(1.0, max(0.0, _f(value, default)))


# --------------------------------------------------------------------------- #
# 通用元素
# --------------------------------------------------------------------------- #


@dataclass
class Provenance:
    """一层是怎么算出来的：用了哪个引擎、是否降级、耗时多少。

    ``engine`` 是**实际生效**的实现（如 ``demucs`` / ``vocalparse`` / ``hmm``），
    ``requested`` 是架构首选（如 ``htdemucs_ft`` / ``sheetsage2`` / ``duet``）。
    两者不同就说明发生了降级，必须原样透传到 manifest 与网页质量面板。
    """

    layer: str
    engine: str
    requested: str = ""
    model: str = ""
    version: str = ""
    ok: bool = True
    cached: bool = False
    elapsed_seconds: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "layer": self.layer,
            "engine": self.engine,
            "requested": self.requested,
            "model": self.model,
            "version": self.version,
            "ok": bool(self.ok),
            "cached": bool(self.cached),
        }
        if self.elapsed_seconds is not None:
            data["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        if self.detail:
            data["detail"] = dict(self.detail)
        if self.warnings:
            data["warnings"] = list(self.warnings)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Provenance":
        data = dict(data or {})
        return cls(
            layer=_s(data.get("layer"), "unknown"),
            engine=_s(data.get("engine"), "unknown"),
            requested=_s(data.get("requested")),
            model=_s(data.get("model")),
            version=_s(data.get("version")),
            ok=bool(data.get("ok", True)),
            cached=bool(data.get("cached", False)),
            elapsed_seconds=(
                _f(data["elapsed_seconds"]) if data.get("elapsed_seconds") is not None else None
            ),
            detail=dict(data.get("detail") or {}),
            warnings=[_s(item) for item in (data.get("warnings") or []) if _s(item)],
        )

    @property
    def degraded(self) -> bool:
        return bool(self.requested) and self.requested != self.engine


@dataclass
class BeatGrid:
    """节拍骨架。``beats`` / ``downbeats`` 是绝对秒。"""

    bpm: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beats_per_bar: int = 4
    time_signature: str = "4/4"
    confidence: float = 0.0
    pickup_beats: int = 0
    first_downbeat_sec: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        self.bpm = _f(self.bpm, 120.0)
        if self.bpm <= 0:
            self.bpm = 120.0
        self.beats = sorted(_f(item) for item in (self.beats or []))
        self.downbeats = sorted(_f(item) for item in (self.downbeats or []))
        if not self.downbeats and self.beats:
            step = max(1, _i(self.beats_per_bar, 4))
            self.downbeats = list(self.beats[::step])
        self.beats_per_bar = max(1, _i(self.beats_per_bar, 4))
        self.confidence = _clamp01(self.confidence)
        self.pickup_beats = max(0, _i(self.pickup_beats, 0))
        if not self.first_downbeat_sec and self.downbeats:
            self.first_downbeat_sec = self.downbeats[0]

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def bar_length_seconds(self) -> float:
        return self.beat_period * self.beats_per_bar

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": round(float(self.bpm), 3),
            "beats": [round(item, 6) for item in self.beats],
            "downbeats": [round(item, 6) for item in self.downbeats],
            "beats_per_bar": int(self.beats_per_bar),
            "time_signature": self.time_signature,
            "confidence": round(float(self.confidence), 4),
            "pickup_beats": int(self.pickup_beats),
            "first_downbeat_sec": round(float(self.first_downbeat_sec), 6),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BeatGrid":
        data = dict(data or {})
        beats = data.get("beats") or []
        return cls(
            bpm=_f(data.get("bpm"), 120.0),
            beats=[_f(item) for item in beats],
            downbeats=[_f(item) for item in (data.get("downbeats") or [])],
            beats_per_bar=_i(data.get("beats_per_bar"), 4),
            time_signature=_s(data.get("time_signature"), "4/4"),
            confidence=_f(data.get("confidence")),
            pickup_beats=_i(data.get("pickup_beats")),
            first_downbeat_sec=_f(data.get("first_downbeat_sec")),
            source=_s(data.get("source")),
        )


@dataclass
class KeySignature:
    tonic: str
    mode: str = "major"
    confidence: float = 0.0
    fifths: int = 0

    def __post_init__(self) -> None:
        self.tonic = _s(self.tonic, "C")
        self.mode = _s(self.mode, "major").lower()
        if self.mode not in ("major", "minor"):
            self.mode = "major"
        self.confidence = _clamp01(self.confidence)
        self.fifths = _i(self.fifths)

    @property
    def display(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tonic": self.tonic,
            "mode": self.mode,
            "confidence": round(float(self.confidence), 4),
            "fifths": int(self.fifths),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "KeySignature":
        data = dict(data or {})
        return cls(
            tonic=_s(data.get("tonic"), "C"),
            mode=_s(data.get("mode"), "major"),
            confidence=_f(data.get("confidence")),
            fifths=_i(data.get("fifths")),
        )


@dataclass
class ChordEvent:
    """一个和弦符号在绝对时间轴上的区间。"""

    start: float
    end: float
    symbol: str
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        self.start = _f(self.start)
        self.end = max(self.start, _f(self.end, self.start))
        self.symbol = _s(self.symbol).strip()
        self.confidence = _clamp01(self.confidence)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "symbol": self.symbol,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChordEvent":
        return cls(
            start=_f(data.get("start")),
            end=_f(data.get("end")),
            symbol=_s(data.get("symbol")),
            confidence=_f(data.get("confidence")),
            source=_s(data.get("source")),
        )


@dataclass
class Section:
    start: float
    end: float
    kind: str = "other"
    label: str = ""
    energy: float = 0.5
    repeat_of: str | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.start = _f(self.start)
        self.end = max(self.start, _f(self.end, self.start))
        self.kind = _s(self.kind, "other")
        self.energy = _clamp01(self.energy, 0.5)
        self.confidence = _clamp01(self.confidence)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "kind": self.kind,
            "label": self.label,
            "energy": round(float(self.energy), 4),
            "repeat_of": self.repeat_of,
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Section":
        return cls(
            start=_f(data.get("start")),
            end=_f(data.get("end")),
            kind=_s(data.get("kind"), "other"),
            label=_s(data.get("label")),
            energy=_f(data.get("energy"), 0.5),
            repeat_of=data.get("repeat_of") or None,
            confidence=_f(data.get("confidence")),
        )


@dataclass
class MelodyNote:
    """旋律音。``lyric`` 是该音对应的歌词（一字一音时非空）。"""

    start: float
    end: float
    midi: int
    confidence: float = 0.0
    lyric: str = ""
    syllable_index: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        self.start = _f(self.start)
        self.end = max(self.start, _f(self.end, self.start))
        self.midi = max(0, min(127, _i(self.midi, 60)))
        self.confidence = _clamp01(self.confidence)
        self.lyric = _s(self.lyric)
        if self.syllable_index is not None:
            self.syllable_index = _i(self.syllable_index)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "midi": int(self.midi),
            "confidence": round(float(self.confidence), 4),
            "lyric": self.lyric,
            "source": self.source,
        }
        if self.syllable_index is not None:
            data["syllable_index"] = int(self.syllable_index)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MelodyNote":
        return cls(
            start=_f(data.get("start")),
            end=_f(data.get("end")),
            midi=_i(data.get("midi"), 60),
            confidence=_f(data.get("confidence")),
            lyric=_s(data.get("lyric")),
            syllable_index=(
                _i(data["syllable_index"]) if data.get("syllable_index") is not None else None
            ),
            source=_s(data.get("source")),
        )


@dataclass
class LyricWord:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(_f(self.start), 6), "end": round(_f(self.end), 6), "text": self.text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LyricWord":
        return cls(start=_f(data.get("start")), end=_f(data.get("end")), text=_s(data.get("text")))


@dataclass
class LyricLine:
    start: float
    end: float
    text: str
    words: list[LyricWord] = field(default_factory=list)
    confidence: float = 0.0
    interjection: bool = False

    def __post_init__(self) -> None:
        self.start = _f(self.start)
        self.end = max(self.start, _f(self.end, self.start))
        self.text = _s(self.text)
        self.words = [
            item if isinstance(item, LyricWord) else LyricWord.from_dict(item)
            for item in (self.words or [])
        ]
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "text": self.text,
            "words": [item.to_dict() for item in self.words],
            "confidence": round(float(self.confidence), 4),
            "interjection": bool(self.interjection),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LyricLine":
        return cls(
            start=_f(data.get("start")),
            end=_f(data.get("end")),
            text=_s(data.get("text")),
            words=[LyricWord.from_dict(item) for item in (data.get("words") or []) if isinstance(item, Mapping)],
            confidence=_f(data.get("confidence")),
            interjection=bool(data.get("interjection", False)),
        )


# --------------------------------------------------------------------------- #
# 第 0 层：分离
# --------------------------------------------------------------------------- #


@dataclass
class SeparationResult:
    """分轨结果。``stems`` 的值是文件路径或 None。"""

    engine: str = "none"
    model: str = ""
    device: str = "cpu"
    stems: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cached: bool = False
    elapsed_seconds: float | None = None
    #: 后端额外产物（如 Demucs 的 ``hq_stems``、质量旋钮），只做透传，不解释。
    detail: dict[str, Any] = field(default_factory=dict)

    #: 架构首选模型。engine 与它不同即视为降级。
    REQUESTED = "htdemucs_ft"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "engine": self.engine,
            "requested": self.REQUESTED,
            "model": self.model,
            "device": self.device,
            "stems": {key: value for key, value in self.stems.items()},
            "warnings": list(self.warnings),
            "cached": bool(self.cached),
        }
        if self.detail:
            data["detail"] = dict(self.detail)
        if self.elapsed_seconds is not None:
            data["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        return data

    @property
    def vocals(self) -> str | None:
        return self.stems.get("vocals")

    @property
    def accompaniment(self) -> str | None:
        return self.stems.get("other") or self.stems.get("accompaniment")


# --------------------------------------------------------------------------- #
# 第 1A 层：人声联合转录
# --------------------------------------------------------------------------- #


@dataclass
class VocalResult:
    """人声联合转录：歌词、音高、时值、词-音符对齐一次产出。"""

    notes: list[MelodyNote] = field(default_factory=list)
    lyrics: list[LyricLine] = field(default_factory=list)
    bpm: float | None = None
    engine: str = "none"
    requested: str = "VocalParse"
    alignment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "engine": self.engine,
            "requested": self.requested,
            "bpm": self.bpm,
            "notes": [item.to_dict() for item in self.notes],
            "lyrics": [item.to_dict() for item in self.lyrics],
            "alignment": dict(self.alignment),
            "warnings": list(self.warnings),
        }
        if self.elapsed_seconds is not None:
            data["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "VocalResult":
        data = dict(data or {})
        return cls(
            notes=[MelodyNote.from_dict(item) for item in (data.get("notes") or [])],
            lyrics=[LyricLine.from_dict(item) for item in (data.get("lyrics") or [])],
            bpm=_f(data["bpm"]) if data.get("bpm") is not None else None,
            engine=_s(data.get("engine"), "none"),
            requested=_s(data.get("requested"), "VocalParse"),
            alignment=dict(data.get("alignment") or {}),
            warnings=[_s(item) for item in (data.get("warnings") or [])],
        )


# --------------------------------------------------------------------------- #
# 第 1B 层：伴奏联合转录
# --------------------------------------------------------------------------- #


@dataclass
class AccompanimentResult:
    """伴奏联合转录：和弦、节拍、调性、段落、旋律骨架一次产出。"""

    beats: BeatGrid
    key: KeySignature
    chords: list[ChordEvent] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    melody_skeleton: list[MelodyNote] = field(default_factory=list)
    abc: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    engine: str = "none"
    requested: str = "SheetSage2"
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "engine": self.engine,
            "requested": self.requested,
            "beats": self.beats.to_dict(),
            "key": self.key.to_dict(),
            "chords": [item.to_dict() for item in self.chords],
            "sections": [item.to_dict() for item in self.sections],
            "melody_skeleton": [item.to_dict() for item in self.melody_skeleton],
            "abc": self.abc,
            "event_count": len(self.events),
            "warnings": list(self.warnings),
        }
        if self.elapsed_seconds is not None:
            data["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        return data


# --------------------------------------------------------------------------- #
# 符号总谱：fusion 的产物，arrangement 的输入
# --------------------------------------------------------------------------- #


@dataclass
class SymbolicScore:
    """一份可编配的符号总谱。

    这是 v2 架构的枢纽：fusion 层把「人声联合转录」和「伴奏联合转录」对齐到
    同一时间轴后产出它，arrangement 层只读它（以及它的 MusicXML 序列化），
    完全不需要音频。
    """

    beats: BeatGrid
    key: KeySignature
    chords: list[ChordEvent] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    melody: list[MelodyNote] = field(default_factory=list)
    lyrics: list[LyricLine] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    provenance: list[Provenance] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    schema: str = SCHEMA

    # ---------- 构造 ----------

    @classmethod
    def from_analysis(cls, analysis: Mapping[str, Any]) -> "SymbolicScore":
        """从既有 ``analysis.json`` 构造（过渡期兼容路径）。"""
        analysis = dict(analysis or {})
        beats_raw = dict(analysis.get("beats") or {})
        key_raw = dict(analysis.get("key") or {})
        melody_raw = analysis.get("melody") or {}
        if isinstance(melody_raw, list):  # 也接受纯音符列表
            melody_raw = {"notes": melody_raw}

        beats = BeatGrid(
            bpm=_f(beats_raw.get("bpm"), 120.0),
            beats=[_f(item) for item in (beats_raw.get("beats") or [])],
            downbeats=[_f(item) for item in (beats_raw.get("downbeats") or [])],
            beats_per_bar=_i(beats_raw.get("beats_per_bar"), 4),
            time_signature=_s(beats_raw.get("time_signature"), "4/4"),
            confidence=_f(beats_raw.get("confidence")),
            pickup_beats=_i(beats_raw.get("pickup_beats")),
            first_downbeat_sec=_f(beats_raw.get("first_downbeat_sec")),
            source=_s(beats_raw.get("source"), "analysis"),
        )
        key = KeySignature(
            tonic=_s(key_raw.get("tonic"), "C"),
            mode=_s(key_raw.get("mode"), "major"),
            confidence=_f(key_raw.get("confidence")),
            fifths=_i(key_raw.get("num_sharps_flats"), _i(key_raw.get("fifths"))),
        )

        chords = [
            ChordEvent(
                start=_f(item.get("start")),
                end=_f(item.get("end")),
                symbol=_s(item.get("symbol")),
                confidence=_f(item.get("confidence")),
                source=_s(item.get("source"), "analysis"),
            )
            for item in (analysis.get("chords") or [])
            if isinstance(item, Mapping) and _s(item.get("symbol")).strip()
        ]
        sections = [
            Section(
                start=_f(item.get("start")),
                end=_f(item.get("end")),
                kind=_s(item.get("kind"), "other"),
                label=_s(item.get("label")),
                energy=_f(item.get("energy"), 0.5),
                repeat_of=item.get("repeat_of") or None,
                confidence=_f(item.get("confidence")),
            )
            for item in (analysis.get("sections") or [])
            if isinstance(item, Mapping)
        ]
        melody = [
            MelodyNote(
                start=_f(item.get("start")),
                end=_f(item.get("end")),
                midi=_i(item.get("midi"), 60),
                confidence=_f(item.get("confidence")),
                lyric=_s(item.get("lyric")),
                syllable_index=(
                    _i(item["syllable_index"]) if item.get("syllable_index") is not None else None
                ),
                source=_s(item.get("source"), "analysis"),
            )
            for item in (melody_raw.get("notes") or [])
            if isinstance(item, Mapping) and item.get("midi") is not None
        ]
        lyrics = [
            LyricLine.from_dict(item)
            for item in (analysis.get("lyrics") or [])
            if isinstance(item, Mapping) and _s(item.get("text")).strip()
        ]

        meta = {
            "title": _s(analysis.get("title")),
            "artist": _s(analysis.get("artist")),
            "schema_source": _s(analysis.get("schema")),
        }
        for key_name in ("options", "quality", "runtime"):
            if analysis.get(key_name) is not None:
                meta[key_name] = analysis.get(key_name)

        return cls(
            beats=beats,
            key=key,
            chords=sorted(chords, key=lambda item: (item.start, item.end)),
            sections=sorted(sections, key=lambda item: item.start),
            melody=sorted(melody, key=lambda item: (item.start, item.midi)),
            lyrics=sorted(lyrics, key=lambda item: item.start),
            meta=meta,
            audio=dict(analysis.get("audio") or {}),
            source=dict(analysis.get("source") or {}),
            warnings=[dict(item) for item in (analysis.get("warnings") or []) if isinstance(item, Mapping)],
        )

    # ---------- 序列化 ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "meta": dict(self.meta),
            "audio": dict(self.audio),
            "source": dict(self.source),
            "beats": self.beats.to_dict(),
            "key": self.key.to_dict(),
            "chords": [item.to_dict() for item in self.chords],
            "sections": [item.to_dict() for item in self.sections],
            "melody": [item.to_dict() for item in self.melody],
            "lyrics": [item.to_dict() for item in self.lyrics],
            "provenance": [item.to_dict() for item in self.provenance],
            "warnings": [dict(item) for item in self.warnings],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SymbolicScore":
        data = dict(data or {})
        if not data.get("beats"):
            raise ValueError("symbolic score 缺少 beats")
        return cls(
            beats=BeatGrid.from_dict(data.get("beats")),
            key=KeySignature.from_dict(data.get("key")),
            chords=[ChordEvent.from_dict(item) for item in (data.get("chords") or [])],
            sections=[Section.from_dict(item) for item in (data.get("sections") or [])],
            melody=[MelodyNote.from_dict(item) for item in (data.get("melody") or [])],
            lyrics=[LyricLine.from_dict(item) for item in (data.get("lyrics") or [])],
            meta=dict(data.get("meta") or {}),
            audio=dict(data.get("audio") or {}),
            source=dict(data.get("source") or {}),
            provenance=[Provenance.from_dict(item) for item in (data.get("provenance") or [])],
            warnings=[dict(item) for item in (data.get("warnings") or [])],
            schema=_s(data.get("schema"), SCHEMA),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # ---------- 兼容视图：喂给既有 arrange ----------

    def to_analysis(self) -> dict[str, Any]:
        """转成 ``arrange(analysis, options)`` 认识的形状。

        目的只有一个：让编配层在 v2 下继续用一份稳定的输入。转换**不引入新事实**，
        缺失的字段就留空，绝不猜。
        """
        analysis: dict[str, Any] = {
            "schema": _s(self.meta.get("schema_source"), "aitsm.analysis/v1"),
            "source": dict(self.source),
            "audio": dict(self.audio),
            "beats": self.beats.to_dict(),
            "key": {
                **self.key.to_dict(),
                "num_sharps_flats": int(self.key.fifths),
                "name": self.key.display,
                "display_name": self.key.display,
                "tonic_pc": _pc(self.key.tonic),
            },
            "chords": [item.to_dict() for item in self.chords],
            "sections": [item.to_dict() for item in self.sections],
            "lyrics": [item.to_dict() for item in self.lyrics],
            "melody": {
                "notes": [item.to_dict() for item in self.melody],
                "available": bool(self.melody),
                "engine": _engine_of(self.provenance, "vocal_transcription"),
                "warnings": [],
            },
            "warnings": [dict(item) for item in self.warnings],
            "symbolic": {
                "schema": self.schema,
                "provenance": [item.to_dict() for item in self.provenance],
            },
            "tools": dict(self.meta.get("tools") or {}),
        }
        if self.meta.get("quality") is not None:
            analysis["quality"] = self.meta["quality"]
        return analysis

    # ---------- MusicXML（惰性委托给 fusion 层） ----------

    def to_musicxml(self, *, include_lyrics: bool = True) -> str:
        from .fusion import musicxml_io

        return musicxml_io.write_musicxml(self, include_lyrics=include_lyrics)

    @classmethod
    def from_musicxml(cls, text: str, *, sidecar: Mapping[str, Any] | None = None) -> "SymbolicScore":
        from .fusion import musicxml_io

        return musicxml_io.read_musicxml(text, sidecar=sidecar)

    # ---------- 磁盘 ----------

    @staticmethod
    def filenames(stem: str = "symbolic") -> tuple[str, str]:
        return f"{stem}.json", f"{stem}.musicxml"

    def save(self, directory: str | Path, *, stem: str = "symbolic") -> dict[str, str]:
        """写出 ``symbolic.json`` + ``symbolic.musicxml``，返回两条路径。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_name, xml_name = self.filenames(stem)
        json_path = directory / json_name
        xml_path = directory / xml_name
        json_path.write_text(self.to_json(), encoding="utf-8")
        xml_path.write_text(self.to_musicxml(), encoding="utf-8")
        return {"json": str(json_path), "musicxml": str(xml_path)}

    @classmethod
    def load(cls, directory: str | Path, *, stem: str = "symbolic") -> "SymbolicScore":
        """优先 MusicXML（符号契约）+ sidecar（绝对时间/置信度），退化到纯 JSON。"""
        directory = Path(directory)
        json_name, xml_name = cls.filenames(stem)
        json_path = directory / json_name
        xml_path = directory / xml_name
        sidecar: dict[str, Any] | None = None
        if json_path.is_file():
            sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        if xml_path.is_file():
            text = xml_path.read_text(encoding="utf-8")
            return cls.from_musicxml(text, sidecar=sidecar)
        if sidecar is None:
            raise FileNotFoundError(f"找不到符号总谱: {json_path} / {xml_path}")
        return cls.from_dict(sidecar)

    # ---------- 便捷查询 ----------

    def chord_at(self, seconds: float) -> ChordEvent | None:
        for item in self.chords:
            if item.start - 1e-6 <= seconds < item.end + 1e-6:
                return item
        return None

    def section_at(self, seconds: float) -> Section | None:
        for item in self.sections:
            if item.start - 1e-6 <= seconds < item.end + 1e-6:
                return item
        return None

    def notes_between(self, start: float, end: float) -> list[MelodyNote]:
        return [item for item in self.melody if item.end > start and item.start < end]

    def get_provenance(self, layer: str) -> Provenance | None:
        for item in self.provenance:
            if item.layer == layer:
                return item
        return None

    @property
    def duration(self) -> float:
        candidates = [item.end for item in self.chords]
        candidates += [item.end for item in self.sections]
        candidates += [item.end for item in self.melody]
        candidates += [item.end for item in self.lyrics]
        if self.beats.beats:
            candidates.append(self.beats.beats[-1])
        return max(candidates) if candidates else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "bpm": round(self.beats.bpm, 3),
            "time_signature": self.beats.time_signature,
            "key": self.key.display,
            "bars": len(self.beats.downbeats) or max(0, len(self.beats.beats) // self.beats.beats_per_bar),
            "chords": len(self.chords),
            "sections": len(self.sections),
            "melody_notes": len(self.melody),
            "lyric_lines": len(self.lyrics),
            "lyric_notes": sum(1 for item in self.melody if item.lyric),
            "duration": round(self.duration, 3),
        }


# --------------------------------------------------------------------------- #
# 私有工具
# --------------------------------------------------------------------------- #


_PC: dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}


def _pc(name: str) -> int:
    text = _s(name).strip()
    if not text:
        return 0
    key = text[0].upper() + text[1:].replace("♯", "#").replace("♭", "b")
    return _PC.get(key.upper(), 0)


def _engine_of(provenance: Iterable[Provenance], layer: str) -> str:
    for item in provenance:
        if item.layer == layer:
            return item.engine
    return ""


def merge_warnings(*groups: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """按 (level, code, stage, message) 去重合并告警，保持出现顺序。"""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            key = (
                record.get("level"),
                record.get("code"),
                record.get("stage"),
                record.get("message"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged

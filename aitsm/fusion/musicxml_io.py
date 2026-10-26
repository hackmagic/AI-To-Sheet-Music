"""SymbolicScore <-> MusicXML 4.0（partwise）。

设计要点
--------
* 只依赖标准库 :mod:`xml.etree.ElementTree`：不引入 music21，不自定义命名空间。
* 时间轴：:class:`aitsm.contracts.SymbolicScore` 用**秒**，MusicXML 用**四分音符**。
  写入时按 tempo（``beats.bpm``）把秒换算成四分音符，再用固定 ``DIVISIONS`` 的
  **整数 tick** 表达，避免浮点误差破坏「每小节时值总和 == 拍号」。
* 小节按 ``beats.downbeats`` 切；第一个 downbeat 之前有材料时记为弱起小节
  （``<measure implicit="yes">``），弱起小节的时值按实际材料长度写。
* 额外信息全部用**标准元素**承载，不塞私货：

  ============  ==================================================
  信息          MusicXML 元素
  ============  ==================================================
  和弦          ``<harmony>`` / ``<kind>`` / ``<bass>``
  歌词          ``<lyric number="1"><text>``
  段落          ``<direction><direction-type><rehearsal>``
  速度          ``<direction><direction-type>`` + ``<sound tempo>``
  调号/拍号      ``<attributes><key>`` / ``<time>``
  跨小节音符     ``<tie type="start">`` + ``<notations><tied>``
  ============  ==================================================

* 读取时若给了 ``sidecar``（``SymbolicScore.to_dict()`` 形状），用它的绝对时间与
  置信度覆盖 MusicXML 里被谱面量化吸掉的部分；缺失时按 tempo map 推算。

公开契约::

    write_musicxml(score, *, include_lyrics=True) -> str
    read_musicxml(text, *, sidecar=None) -> SymbolicScore
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import (
    BeatGrid,
    ChordEvent,
    KeySignature,
    LyricLine,
    LyricWord,
    MelodyNote,
    Provenance,
    SCHEMA,
    Section,
    SymbolicScore,
    merge_warnings,
)

__all__ = [
    "DIVISIONS",
    "MUSICXML_VERSION",
    "format_chord_symbol",
    "parse_chord_symbol",
    "read_musicxml",
    "write_musicxml",
]

#: 每个四分音符的 divisions。24 同时覆盖 1/4、1/8、1/16、附点时值（0.375 拍 = 9）
#: 与三连音（1/3 拍 = 8），不需要 float 时值也能保证每小节总长精确。
DIVISIONS = 24

MUSICXML_VERSION = "4.0"

_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)

_EPS = 1e-9

#: 从 MusicXML 读回来的音符没有「识别置信度」这个概念，统一给一个中性值；
#: 真实值由 sidecar 覆盖。
_DEFAULT_CONFIDENCE = 0.5

_STEP_TO_PC: dict[str, int] = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

#: fifths -> 大调主音 / 小调主音
_MAJOR_TONIC: dict[int, str] = {
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
    -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb",
}
_MINOR_TONIC: dict[int, str] = {
    0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
    -1: "D", -2: "G", -3: "C", -4: "F", -5: "Bb", -6: "Eb", -7: "Ab",
}

#: MusicXML ``<kind>`` -> 和弦符号后缀（写回时优先用 ``<kind text>``，这里只是兜底）
_KIND_TO_SUFFIX: dict[str, str] = {
    "major": "",
    "minor": "m",
    "dominant": "7",
    "major-seventh": "maj7",
    "minor-seventh": "m7",
    "diminished": "dim",
    "diminished-seventh": "dim7",
    "augmented": "aug",
    "suspended-second": "sus2",
    "suspended-fourth": "sus4",
    "major-sixth": "6",
    "minor-sixth": "m6",
    "half-diminished": "m7b5",
    "dominant-ninth": "9",
    "major-ninth": "maj9",
    "minor-ninth": "m9",
    "dominant-11th": "11",
    "major-11th": "maj11",
    "minor-11th": "m11",
    "dominant-13th": "13",
    "major-13th": "maj13",
    "minor-13th": "m13",
    "power": "5",
}

#: ``<kind>none</kind>`` 的内部哨兵：表示「和弦到此结束」，写出/读回共用。
_NO_CHORD = "@none"

_SUFFIX_TO_KIND: dict[str, str] = {
    "": "major",
    "m": "minor",
    "min": "minor",
    "7": "dominant",
    "dom7": "dominant",
    "maj7": "major-seventh",
    "M7": "major-seventh",
    "Δ7": "major-seventh",
    "m7": "minor-seventh",
    "min7": "minor-seventh",
    "dim": "diminished",
    "°": "diminished",
    "o": "diminished",
    "dim7": "diminished-seventh",
    "°7": "diminished-seventh",
    "aug": "augmented",
    "+": "augmented",
    "sus2": "suspended-second",
    "sus4": "suspended-fourth",
    "sus": "suspended-fourth",
    "6": "major-sixth",
    "m6": "minor-sixth",
    "m7b5": "half-diminished",
    "ø": "half-diminished",
    "ø7": "half-diminished",
    "9": "dominant-ninth",
    "maj9": "major-ninth",
    "m9": "minor-ninth",
    "11": "dominant-11th",
    "maj11": "major-11th",
    "m11": "minor-11th",
    "13": "dominant-13th",
    "maj13": "major-13th",
    "m13": "minor-13th",
    "5": "power",
}

_CHORD_RE = re.compile(
    r"^\s*([A-Ga-g])([#b♯♭]*)(.*?)(?:\s*/\s*([A-Ga-g])([#b♯♭]*))?\s*$"
)

#: 时值（四分音符）-> (``<type>``, 附点数量)
_TYPE_BY_QUARTERS: tuple[tuple[float, str, int], ...] = (
    (4.0, "whole", 0),
    (3.0, "half", 1),
    (2.0, "half", 0),
    (1.5, "quarter", 1),
    (1.0, "quarter", 0),
    (0.75, "eighth", 1),
    (0.5, "eighth", 0),
    (0.375, "16th", 1),
    (0.25, "16th", 0),
    (0.125, "32nd", 0),
)

_TIE = "tie"


# --------------------------------------------------------------------------- #
# 音名 / 和弦符号
# --------------------------------------------------------------------------- #


def _name_to_step_alter(name: str) -> tuple[str, int]:
    """``"C#"`` -> ``("C", 1)``，``"Bb"`` -> ``("B", -1)``。"""
    text = (name or "").strip()
    if not text or text[0].upper() not in _STEP_TO_PC:
        return "C", 0
    step = text[0].upper()
    alter = 0
    for char in text[1:]:
        if char in "#♯":
            alter += 1
        elif char in "b♭":
            alter -= 1
    return step, alter


def _step_alter_to_name(step: str, alter: int) -> str:
    step = (step or "C").strip().upper()[:1]
    if step not in _STEP_TO_PC:
        step = "C"
    alter = int(round(alter))
    if alter > 0:
        return step + "#" * alter
    if alter < 0:
        return step + "b" * (-alter)
    return step


def parse_chord_symbol(symbol: str) -> dict[str, Any] | None:
    """把 ``"Am7"`` / ``"G/B"`` 拆成根音 + 后缀 + 低音。

    返回 ``{"root_step", "root_alter", "suffix", "bass_step", "bass_alter"}``；
    解析不了返回 ``None``（调用方原样跳过，不猜）。
    """
    text = str(symbol or "").strip()
    if not text:
        return None
    match = _CHORD_RE.match(text)
    if not match:
        return None
    root_step = match.group(1).upper()
    root_alter = match.group(2).count("#") - match.group(2).count("b")
    root_alter += match.group(2).count("♯") - match.group(2).count("♭")
    suffix = match.group(3).strip()
    bass_step = (match.group(4) or "").upper() or None
    bass_alter = 0
    if bass_step:
        bass_alter = match.group(5).count("#") - match.group(5).count("b")
        bass_alter += match.group(5).count("♯") - match.group(5).count("♭")
    return {
        "root_step": root_step,
        "root_alter": root_alter,
        "suffix": suffix,
        "bass_step": bass_step,
        "bass_alter": bass_alter,
    }


def format_chord_symbol(
    root_step: str,
    root_alter: int = 0,
    suffix: str = "",
    bass_step: str | None = None,
    bass_alter: int = 0,
) -> str:
    """根音 + 后缀 + （可选）斜杠低音 -> 和弦符号字符串。"""
    text = _step_alter_to_name(root_step, root_alter) + str(suffix or "")
    if bass_step:
        text += "/" + _step_alter_to_name(bass_step, bass_alter)
    return text


# --------------------------------------------------------------------------- #
# 写：SymbolicScore -> MusicXML
# --------------------------------------------------------------------------- #


def _time_signature_parts(text: Any, beats_per_bar: Any) -> tuple[int, int]:
    """``"4/4"`` -> ``(4, 4)``；非法时退回 ``(beats_per_bar, 4)``。"""
    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(text or ""))
    numerator = 0
    denominator = 0
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
    if numerator <= 0:
        try:
            numerator = int(beats_per_bar)
        except (TypeError, ValueError):
            numerator = 4
    numerator = max(1, numerator)
    # MusicXML 的 beat-type 必须是 2 的幂；不是就退回四分音符
    if denominator <= 0 or denominator > 32 or (denominator & (denominator - 1)) != 0:
        denominator = 4
    return numerator, denominator


def _spell_pitch(midi: int, fifths: int) -> tuple[str, int, int]:
    """MIDI -> ``(step, alter, octave)``，按调号选升/降号拼写。"""
    value = max(0, min(127, int(round(midi))))
    name = (_FLAT_NAMES if fifths < 0 else _SHARP_NAMES)[value % 12]
    step, alter = _name_to_step_alter(name)
    octave = value // 12 - 1
    return step, alter, octave


def _pitch_to_midi(step: str, alter: float, octave: int) -> int:
    base = _STEP_TO_PC.get((step or "C").strip().upper()[:1], 0)
    return max(0, min(127, (int(octave) + 1) * 12 + base + int(round(alter))))


def _duration_type(quarters: float) -> tuple[str, int] | None:
    for value, name, dots in _TYPE_BY_QUARTERS:
        if abs(quarters - value) <= 1e-6:
            return name, dots
    return None


@dataclass
class _Layout:
    """秒 <-> divisions 的换算，以及小节骨架（全部用整数 divisions）。"""

    beat_period: float
    divisions: int
    bar_len_div: int
    beat_unit_q: float
    pickup_div: int
    bar_starts: list[int] = field(default_factory=list)
    pickup_lens: int = 0
    bar_lens: list[int] = field(default_factory=list)
    total_div: int = 0
    origin: float = 0.0
    numerator: int = 4
    denominator: int = 4

    # ---- 构造 ----

    @classmethod
    def from_score(cls, score: SymbolicScore) -> "_Layout":
        numerator, denominator = _time_signature_parts(
            score.beats.time_signature, score.beats.beats_per_bar
        )
        beat_unit_q = 4.0 / denominator
        bar_len_q = numerator * beat_unit_q
        beat_period = 60.0 / max(float(score.beats.bpm or 0.0), 1e-6)
        bar_len_div = max(1, int(round(bar_len_q * DIVISIONS)))
        bar_len_sec = bar_len_div / DIVISIONS * beat_period

        downbeats = sorted(float(item) for item in (score.beats.downbeats or []))
        first_downbeat = downbeats[0] if downbeats else 0.0

        # 弱起小节 = 从 0 到第一个 downbeat 的那段（可以是任意长度：前奏留白、
        # 半个小节、甚至 >1 小节）。写成 ``implicit="yes"`` 的小节，读回时靠这个
        # 属性还原 first_downbeat，保证往返不丢对齐信息。超过一整小节的引入段
        # 拆成「若干个整小节 + 最后一个残缺小节」，避免出现 sum > 拍号 的小节。
        pickup_div = 0
        pickup_lens: list[int] = []
        origin = 0.0
        if first_downbeat > 1e-3:
            raw = max(0, int(round(first_downbeat / beat_period * DIVISIONS)))
            remainder = raw % bar_len_div
            tolerance = max(1, DIVISIONS // 8)
            if remainder and min(remainder, bar_len_div - remainder) <= tolerance:
                # 检测误差（downbeat 时间与 tempo 网格差几毫秒）吸附到小节线
                if remainder <= tolerance:
                    raw -= remainder
                else:
                    raw += bar_len_div - remainder
            if raw > 0:
                full_bars, tail = divmod(raw, bar_len_div)
                pickup_lens = [bar_len_div] * full_bars + ([tail] if tail else [])
                pickup_div = raw

        layout = cls(
            beat_period=beat_period,
            divisions=DIVISIONS,
            bar_len_div=bar_len_div,
            beat_unit_q=beat_unit_q,
            pickup_div=pickup_div,
            origin=origin,
            numerator=numerator,
            denominator=denominator,
        )

        content_end = 0
        for item in score.melody:
            content_end = max(content_end, layout.div(item.end))
        for item in score.chords:
            content_end = max(content_end, layout.div(item.end))
        for item in score.sections:
            content_end = max(content_end, layout.div(item.end))
        if score.beats.beats:
            content_end = max(content_end, layout.div(score.beats.beats[-1]))
        content_end = max(content_end, 0)

        total_div = max(content_end, pickup_div + bar_len_div)
        relative = max(1, total_div - pickup_div)
        bars = max(1, int(math.ceil(relative / bar_len_div)))
        starts: list[int] = []
        lens: list[int] = []
        cursor = 0
        for length in pickup_lens:
            starts.append(cursor)
            lens.append(length)
            cursor += length
        for index in range(bars):
            starts.append(pickup_div + index * bar_len_div)
            lens.append(bar_len_div)
        layout.bar_starts = starts
        layout.bar_lens = lens
        layout.pickup_lens = len(pickup_lens)
        layout.total_div = starts[-1] + lens[-1]
        return layout

    # ---- 换算 ----

    def div(self, seconds: float) -> int:
        """绝对秒 -> 整数 divisions（相对谱面起点）。"""
        return int(round((float(seconds) - self.origin) / self.beat_period * self.divisions))

    def quarters(self, seconds: float) -> float:
        return (float(seconds) - self.origin) / self.beat_period

    def locate(self, position: int) -> tuple[int, int]:
        """divisions 位置 -> ``(小节序号, 小节内偏移)``。"""
        for index in range(len(self.bar_starts) - 1, -1, -1):
            if position >= self.bar_starts[index]:
                return index, position - self.bar_starts[index]
        return 0, position - self.bar_starts[0]


def _segments(layout: _Layout, start_div: int, end_div: int) -> list[tuple[int, int, int]]:
    """把一个跨小节音符切成 ``[(小节序号, 小节内偏移, 长度)]``。"""
    cursor = max(0, int(start_div))
    end = max(int(end_div), cursor + 1)
    end = min(end, layout.total_div)
    pieces: list[tuple[int, int, int]] = []
    guard = 0
    while cursor < end and guard < 10_000:
        guard += 1
        index, offset = layout.locate(cursor)
        room = layout.bar_lens[index] - offset
        if room <= 0:  # 异常数据：跳到下一小节头
            index += 1
            if index >= len(layout.bar_starts):
                break
            offset = 0
            room = layout.bar_lens[index]
        take = min(end - cursor, room)
        if take <= 0:
            break
        pieces.append((index, offset, take))
        cursor += take
    return pieces


@dataclass
class _Item:
    """小节内的一个元素（音符或休止）。"""

    offset: int
    duration: int
    kind: str = "note"  # note | rest
    midi: int = 60
    lyric: str = ""
    tie_start: bool = False
    tie_stop: bool = False


def _monophonic(
    score: SymbolicScore, layout: _Layout
) -> list[tuple[int, int, MelodyNote]]:
    """把旋律压成单声部：后音起点不早于前音终点（重叠时截断前音）。"""
    reduced: list[tuple[int, int, MelodyNote]] = []
    cursor = 0
    for note in sorted(score.melody, key=lambda item: (item.start, item.midi, item.end)):
        start = max(layout.div(note.start), 0)
        end = max(layout.div(note.end), start + 1)
        if start >= layout.total_div:
            continue
        start = max(start, cursor)
        if start >= layout.total_div:
            continue
        end = max(min(end, layout.total_div), start + 1)
        end = min(end, layout.total_div)
        if end <= start:
            continue
        reduced.append((start, end, note))
        cursor = end
    return reduced


def _place_harmony(
    measures: list[dict[str, Any]], layout: "_Layout", position: int, symbol: str
) -> None:
    """把和弦挂到「覆盖该 divisions 位置」的音符上，用相对 ``<offset>`` 定位。"""
    if position < 0 or position >= layout.total_div:
        return
    index, offset = layout.locate(position)
    measure = measures[index]
    target = 0
    for position_index, item in enumerate(measure["items"]):
        if item.offset <= offset:
            target = position_index
        else:
            break
    measure["harmonies"].append((target, offset - measure["items"][target].offset, symbol))


def _build_measures(score: SymbolicScore, layout: _Layout) -> list[dict[str, Any]]:
    """构造每小节要写出的元素（音符 / 休止 / 和弦 / 段落）。"""
    measures: list[dict[str, Any]] = []
    for index, (start, length) in enumerate(zip(layout.bar_starts, layout.bar_lens)):
        measures.append(
            {
                "number": str(index + 1),
                "implicit": index < layout.pickup_lens,
                "sections": [],
                "harmonies": [],
                "items": [],
                "_notes": [],
            }
        )

    for start_div, end_div, note in _monophonic(score, layout):
        pieces = _segments(layout, start_div, end_div)
        for position, (index, offset, take) in enumerate(pieces):
            measures[index]["_notes"].append(
                _Item(
                    offset=offset,
                    duration=take,
                    kind="note",
                    midi=int(note.midi),
                    lyric=str(note.lyric or ""),
                    tie_start=position < len(pieces) - 1,
                    tie_stop=position > 0,
                )
            )

    # 休止符补齐：每小节从头填到小节尾，保证时值总和恒等于拍号
    for index, measure in enumerate(measures):
        bar_len = layout.bar_lens[index]
        items = sorted(measure["_notes"], key=lambda item: (item.offset, item.duration))
        filled: list[_Item] = []
        cursor = 0
        for item in items:
            offset = item.offset
            if offset < cursor:  # 理论上单声部化后不会发生，防御性截断
                continue
            if offset > cursor:
                filled.append(_Item(offset=cursor, duration=offset - cursor, kind="rest"))
            filled.append(item)
            cursor = offset + item.duration
        if cursor < bar_len:
            filled.append(_Item(offset=cursor, duration=bar_len - cursor, kind="rest"))
        measure["items"] = filled
        if sum(entry.duration for entry in filled) != bar_len:  # 内部自检
            raise ValueError(
                f"小节 {index + 1} 时值总和不等于拍号："
                f"{sum(entry.duration for entry in filled)} != {bar_len}"
            )
        measure.pop("_notes", None)

    # 和弦：挂到「覆盖该时刻」的元素上，用 <offset> 定位
    for chord in sorted(score.chords, key=lambda item: item.start):
        symbol = str(chord.symbol or "").strip()
        if not symbol:
            continue
        _place_harmony(measures, layout, layout.div(chord.start), symbol)

    # 最后一个和弦的结束时刻用 <kind>none</kind> 显式收尾，
    # 否则读回时只能把它的时值一路拖到全曲结束。
    if score.chords:
        last_end = max(float(chord.end) for chord in score.chords)
        position = layout.div(last_end)
        if 0 < position < layout.total_div - 1:
            _place_harmony(measures, layout, position, _NO_CHORD)

    # 段落：放在小节头，非小节线起点用 <offset> 补
    for section in sorted(score.sections, key=lambda item: item.start):
        label = str(section.label or "").strip() or str(section.kind or "").strip()
        if not label:
            continue
        position = layout.div(section.start)
        if position < 0 or position >= layout.total_div:
            continue
        index, offset = layout.locate(position)
        measures[index]["sections"].append((offset, label))

    return measures


def _append_attributes(parent: ET.Element, score: SymbolicScore, layout: _Layout) -> None:
    attributes = ET.SubElement(parent, "attributes")
    ET.SubElement(attributes, "divisions").text = str(layout.divisions)
    key_el = ET.SubElement(attributes, "key")
    ET.SubElement(key_el, "fifths").text = str(max(-7, min(7, int(score.key.fifths))))
    ET.SubElement(key_el, "mode").text = "minor" if score.key.mode == "minor" else "major"
    time_el = ET.SubElement(attributes, "time")
    ET.SubElement(time_el, "beats").text = str(layout.numerator)
    ET.SubElement(time_el, "beat-type").text = str(layout.denominator)
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"


def _append_tempo(parent: ET.Element, bpm: float) -> None:
    tempo = round(float(bpm), 3)
    if not math.isfinite(tempo) or tempo <= 0:
        tempo = 120.0
    direction = ET.SubElement(parent, "direction", {"placement": "above"})
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = _trim_number(tempo)
    ET.SubElement(direction, "sound", {"tempo": _trim_number(tempo)})


def _append_section(parent: ET.Element, offset: int, label: str) -> None:
    direction = ET.SubElement(parent, "direction", {"placement": "above"})
    direction_type = ET.SubElement(direction, "direction-type")
    ET.SubElement(direction_type, "rehearsal").text = label
    if offset:
        ET.SubElement(direction, "offset").text = str(int(offset))


def _append_harmony(parent: ET.Element, offset: int, symbol: str) -> None:
    if symbol == _NO_CHORD:
        harmony = ET.SubElement(parent, "harmony")
        ET.SubElement(harmony, "kind").text = "none"
        if offset:
            ET.SubElement(harmony, "offset").text = str(int(offset))
        return
    parsed = parse_chord_symbol(symbol)
    if parsed is None:
        return
    harmony = ET.SubElement(parent, "harmony")
    root = ET.SubElement(harmony, "root")
    ET.SubElement(root, "root-step").text = parsed["root_step"]
    if parsed["root_alter"]:
        ET.SubElement(root, "root-alter").text = str(parsed["root_alter"])
    suffix = parsed["suffix"]
    kind_value = _SUFFIX_TO_KIND.get(suffix, "other")
    kind = ET.SubElement(harmony, "kind")
    kind.text = kind_value
    if suffix:
        kind.set("text", suffix)
    if parsed["bass_step"]:
        bass = ET.SubElement(harmony, "bass")
        ET.SubElement(bass, "bass-step").text = parsed["bass_step"]
        if parsed["bass_alter"]:
            ET.SubElement(bass, "bass-alter").text = str(parsed["bass_alter"])
    if offset:
        ET.SubElement(harmony, "offset").text = str(int(offset))


def _append_note(parent: ET.Element, item: _Item, layout: _Layout, fifths: int, include_lyrics: bool) -> None:
    note = ET.SubElement(parent, "note")
    if item.kind == "rest":
        ET.SubElement(note, "rest")
    else:
        step, alter, octave = _spell_pitch(item.midi, fifths)
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = step
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
        ET.SubElement(pitch, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(int(item.duration))
    if item.tie_stop:
        ET.SubElement(note, _TIE, {"type": "stop"})
    if item.tie_start:
        ET.SubElement(note, _TIE, {"type": "start"})
    ET.SubElement(note, "voice").text = "1"
    type_info = _duration_type(item.duration / layout.divisions)
    if type_info is not None:
        ET.SubElement(note, "type").text = type_info[0]
        for _ in range(type_info[1]):
            ET.SubElement(note, "dot")
    if item.tie_start or item.tie_stop:
        notations = ET.SubElement(note, "notations")
        if item.tie_stop:
            ET.SubElement(notations, "tied", {"type": "stop"})
        if item.tie_start:
            ET.SubElement(notations, "tied", {"type": "start"})
    if item.kind == "note" and item.lyric and include_lyrics:
        lyric = ET.SubElement(note, "lyric", {"number": "1"})
        ET.SubElement(lyric, "syllabic").text = "single"
        ET.SubElement(lyric, "text").text = item.lyric


def _trim_number(value: float) -> str:
    return f"{float(value):g}"


def _serialize(root: ET.Element) -> str:
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - Python < 3.9
        pass
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{_DOCTYPE}\n{body}\n'


def write_musicxml(score: SymbolicScore, *, include_lyrics: bool = True) -> str:
    """把 :class:`SymbolicScore` 写成 MusicXML 4.0 partwise 文本。

    * 单声部（旋律）；同一声部内的重叠音符按开始时间截断。
    * 跨小节的音符自动拆成带 ``<tie>`` 的多个音符。
    * 每小节用休止符补齐，时值总和恒等于拍号（弱起小节除外）。
    """
    if not isinstance(score, SymbolicScore):
        raise TypeError(f"write_musicxml 需要 SymbolicScore，收到 {type(score).__name__}")

    layout = _Layout.from_score(score)
    measures = _build_measures(score, layout)

    root = ET.Element("score-partwise", {"version": MUSICXML_VERSION})
    title = str(score.meta.get("title") or "").strip()
    if title:
        ET.SubElement(root, "movement-title").text = title
    artist = str(score.meta.get("artist") or "").strip()
    identification = ET.SubElement(root, "identification")
    if artist:
        ET.SubElement(identification, "creator", {"type": "composer"}).text = artist
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "aitsm-fusion"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    ET.SubElement(score_part, "part-name").text = "Melody"
    part = ET.SubElement(root, "part", {"id": "P1"})

    fifths = max(-7, min(7, int(score.key.fifths)))
    for index, measure_data in enumerate(measures):
        measure = ET.SubElement(part, "measure", {"number": measure_data["number"]})
        if measure_data["implicit"]:
            measure.set("implicit", "yes")
        if index == 0:
            _append_attributes(measure, score, layout)
            _append_tempo(measure, score.beats.bpm)
        for offset, label in measure_data["sections"]:
            _append_section(measure, offset, label)
        harmony_at: dict[int, list[tuple[int, str]]] = {}
        for target, offset, symbol in measure_data["harmonies"]:
            harmony_at.setdefault(target, []).append((offset, symbol))
        for item_index, item in enumerate(measure_data["items"]):
            for offset, symbol in harmony_at.get(item_index, []):
                _append_harmony(measure, offset, symbol)
            _append_note(measure, item, layout, fifths, include_lyrics)

    return _serialize(root)


# --------------------------------------------------------------------------- #
# 读：MusicXML -> SymbolicScore
# --------------------------------------------------------------------------- #


def _local(tag: Any) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local(child.tag) == name]


def _first(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local(child.tag) == name:
            return child
    return None


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return str(element.text).strip()


def _number(element: ET.Element | None, default: float = 0.0) -> float:
    text = _text(element)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


class _TempoMap:
    """``[(四分音符位置, bpm)]`` 的分段速度表（秒 = 累积 + 段内线性）。"""

    def __init__(self) -> None:
        self._segments: list[tuple[float, float, float]] = []  # (start_q, bpm, start_sec)
        self._last_q = 0.0

    def add(self, position_q: float, bpm: float) -> None:
        value = float(bpm)
        if not math.isfinite(value) or value <= 0:
            return
        position = max(0.0, float(position_q))
        if self._segments and abs(position - self._segments[-1][0]) < 1e-9:
            self._segments[-1] = (self._segments[-1][0], value, self._segments[-1][2])
            return
        start_sec = self.seconds(position)
        self._segments.append((position, value, start_sec))
        self._last_q = max(self._last_q, position)

    def seconds(self, position_q: float) -> float:
        position = float(position_q)
        if not self._segments:
            return position * 0.5  # 默认 120 BPM
        chosen = self._segments[0]
        for segment in self._segments:
            if segment[0] <= position + 1e-9:
                chosen = segment
            else:
                break
        start_q, bpm, start_sec = chosen
        return start_sec + (position - start_q) * 60.0 / bpm

    @property
    def initial_bpm(self) -> float:
        return self._segments[0][1] if self._segments else 120.0


def _read_harmony(element: ET.Element) -> str:
    root = _first(element, "root")
    root_step = _text(_first(root, "root-step")) if root is not None else "C"
    root_alter = int(round(_number(_first(root, "root-alter"), 0.0))) if root is not None else 0
    kind_el = _first(element, "kind")
    suffix = ""
    if kind_el is not None:
        text = (kind_el.get("text") or "").strip()
        if (kind_el.text or "").strip() == "none" and not text:
            return _NO_CHORD
        if text:
            suffix = text
        else:
            suffix = _KIND_TO_SUFFIX.get(_text(kind_el), "")
    bass = _first(element, "bass")
    bass_step = None
    bass_alter = 0
    if bass is not None:
        bass_step = _text(_first(bass, "bass-step")) or None
        bass_alter = int(round(_number(_first(bass, "bass-alter"), 0.0)))
    return format_chord_symbol(root_step or "C", root_alter, suffix, bass_step, bass_alter)


def _read_note_parts(note_el: ET.Element) -> dict[str, Any]:
    pitch_el = _first(note_el, "pitch")
    data: dict[str, Any] = {"rest": pitch_el is None}
    if pitch_el is not None:
        step = _text(_first(pitch_el, "step")) or "C"
        alter = _number(_first(pitch_el, "alter"), 0.0)
        octave = int(round(_number(_first(pitch_el, "octave"), 4.0)))
        data["midi"] = _pitch_to_midi(step, alter, octave)
    ties = {item.get("type") for item in _children(note_el, _TIE)}
    data["tie_start"] = "start" in ties
    data["tie_stop"] = "stop" in ties
    lyric_text = ""
    for lyric in _children(note_el, "lyric"):
        if lyric.get("number") not in (None, "1"):
            continue
        text_el = _first(lyric, "text")
        candidate = _text(text_el).replace("\r", " ").replace("\n", " ")
        if candidate:
            lyric_text = candidate
            break
    data["lyric"] = lyric_text
    return data


def _guess_section_kind(label: str) -> str:
    text = str(label or "").lower()
    for hint, kind in (
        ("intro", "intro"),
        ("verse", "verse"),
        ("chorus", "chorus"),
        ("refrain", "chorus"),
        ("bridge", "bridge"),
        ("outro", "outro"),
        ("solo", "solo"),
        ("interlude", "interlude"),
        ("break", "break"),
    ):
        if hint in text:
            return kind
    return "other"


def read_musicxml(text: str, *, sidecar: Mapping[str, Any] | None = None) -> SymbolicScore:
    """从 MusicXML 文本还原 :class:`SymbolicScore`。

    ``sidecar`` 为 ``SymbolicScore.to_dict()`` 形状时，用其绝对时间与置信度覆盖
    MusicXML 里被谱面量化吸掉的信息（节拍网格、调号、音符起止、和弦起止、
    段落、歌词文本、meta/audio/source/provenance/warnings）。
    """
    if isinstance(text, bytes):
        raw = text
    else:
        raw = str(text or "")
    if not raw.strip():
        raise ValueError("read_musicxml 收到空的 MusicXML 文本")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"MusicXML 不是合法 XML：{exc}") from exc

    part = _first(root, "part")
    if part is None:
        raise ValueError("MusicXML 缺少 <part> 元素")

    divisions = DIVISIONS
    fifths = 0
    mode = "major"
    numerator = 4
    denominator = 4

    tempo_map = _TempoMap()
    notes_raw: list[dict[str, Any]] = []
    harmonies_raw: list[tuple[float, str]] = []
    sections_raw: list[tuple[float, str]] = []
    measure_starts_q: list[float] = []
    measure_lens_q: list[float] = []
    measure_implicit: list[bool] = []
    pending_ties: dict[int, dict[str, Any]] = {}

    measure_start_q = 0.0
    for measure in _children(part, "measure"):
        measure_implicit.append(str(measure.get("implicit") or "").strip().lower() in ("yes", "true", "1"))
        cursor_q = 0.0
        measure_max_q = 0.0
        previous_start_q: float | None = None
        for element in list(measure):
            tag = _local(element.tag)
            if tag == "attributes":
                value = _first(element, "divisions")
                if value is not None and _number(value) > 0:
                    divisions = max(1, int(round(_number(value))))
                key_el = _first(element, "key")
                if key_el is not None:
                    fifths_el = _first(key_el, "fifths")
                    if fifths_el is not None:
                        fifths = max(-7, min(7, int(round(_number(fifths_el)))))
                    mode_el = _first(key_el, "mode")
                    if mode_el is not None and _text(mode_el):
                        mode = "minor" if _text(mode_el).lower().startswith("min") else "major"
                time_el = _first(element, "time")
                if time_el is not None:
                    beats_el = _first(time_el, "beats")
                    beat_type_el = _first(time_el, "beat-type")
                    if beats_el is not None and _number(beats_el) > 0:
                        numerator = max(1, int(round(_number(beats_el))))
                    if beat_type_el is not None and _number(beat_type_el) > 0:
                        denominator = max(1, int(round(_number(beat_type_el))))
                continue
            if tag == "backup":
                cursor_q -= _number(_first(element, "duration")) / divisions
                continue
            if tag == "forward":
                cursor_q += _number(_first(element, "duration")) / divisions
                measure_max_q = max(measure_max_q, cursor_q)
                continue
            if tag == "direction":
                for node in element.iter():
                    if _local(node.tag) == "sound" and node.get("tempo"):
                        try:
                            tempo_map.add(measure_start_q + cursor_q, float(node.get("tempo")))
                        except (TypeError, ValueError):
                            pass
                label = ""
                for node in element.iter():
                    local = _local(node.tag)
                    if local in ("rehearsal", "words") and _text(node):
                        label = _text(node)
                        break
                if label:
                    offset_q = _number(_first(element, "offset")) / divisions
                    sections_raw.append((measure_start_q + cursor_q + offset_q, label))
                continue
            if tag == "harmony":
                offset_q = _number(_first(element, "offset")) / divisions
                symbol = _read_harmony(element)
                if symbol:
                    harmonies_raw.append((measure_start_q + cursor_q + offset_q, symbol))
                continue
            if tag != "note":
                continue

            duration_q = _number(_first(element, "duration")) / divisions
            if _first(element, "rest") is not None:
                if _first(element, "chord") is None:
                    cursor_q += duration_q
                    measure_max_q = max(measure_max_q, cursor_q)
                continue

            parts = _read_note_parts(element)
            midi = int(parts.get("midi", 60))
            if _first(element, "chord") is not None and previous_start_q is not None:
                start_q = previous_start_q
            else:
                start_q = cursor_q
            absolute_q = measure_start_q + start_q
            end_q = absolute_q + duration_q
            if parts["tie_stop"] and midi in pending_ties:
                record = pending_ties[midi]
                record["end_q"] = max(float(record["end_q"]), end_q)
                if not parts["tie_start"]:
                    del pending_ties[midi]
            else:
                record = {
                    "start_q": absolute_q,
                    "end_q": end_q,
                    "midi": midi,
                    "lyric": str(parts.get("lyric") or ""),
                }
                notes_raw.append(record)
                if parts["tie_start"]:
                    pending_ties[midi] = record
                elif midi in pending_ties:
                    del pending_ties[midi]
            previous_start_q = start_q
            if _first(element, "chord") is None:
                cursor_q += duration_q
                measure_max_q = max(measure_max_q, cursor_q)

        measure_starts_q.append(measure_start_q)
        measure_len_q = max(measure_max_q, 0.0)
        measure_lens_q.append(measure_len_q)
        measure_start_q += measure_len_q

    total_q = measure_start_q

    def seconds(position_q: float) -> float:
        return tempo_map.seconds(position_q)

    beat_unit_q = 4.0 / max(1, denominator)
    bar_len_q = numerator * beat_unit_q
    downbeats = [seconds(item) for item in measure_starts_q]
    beats: list[float] = []
    for start_q, length_q in zip(measure_starts_q, measure_lens_q):
        step_index = 0
        while step_index * beat_unit_q < length_q - 1e-9:
            beats.append(seconds(start_q + step_index * beat_unit_q))
            step_index += 1
    beats = sorted({round(item, 6) for item in beats})

    pickup_beats = 0
    first_downbeat_sec = downbeats[0] if downbeats else 0.0
    # 前导 implicit 小节 = 弱起/前奏区（可能拆成多个小节，甚至含整小节）
    leading = 0
    leading_len_q = 0.0
    for flag, length_q in zip(measure_implicit, measure_lens_q):
        if not flag:
            break
        leading += 1
        leading_len_q += length_q
    has_pickup = leading > 0
    if not has_pickup and downbeats and measure_lens_q and measure_lens_q[0] < bar_len_q - 1e-6:
        has_pickup = True
        leading = 1
        leading_len_q = measure_lens_q[0]
    if has_pickup and len(downbeats) > leading:
        pickup_beats = max(0, int(round(leading_len_q / beat_unit_q)))
        first_downbeat_sec = downbeats[leading]
        # 弱起小节自己的起点不是 downbeat，不能留在 downbeats 里，
        # 否则二次写入会把弱起当成正拍、整篇错位。
        downbeats = downbeats[leading:]

    beats_grid = BeatGrid(
        bpm=tempo_map.initial_bpm,
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=numerator,
        time_signature=f"{numerator}/{denominator}",
        confidence=_DEFAULT_CONFIDENCE,
        pickup_beats=pickup_beats,
        first_downbeat_sec=first_downbeat_sec,
        source="musicxml",
    )

    tonic = (_MINOR_TONIC if mode == "minor" else _MAJOR_TONIC).get(fifths, "C")
    key = KeySignature(tonic=tonic, mode=mode, confidence=_DEFAULT_CONFIDENCE, fifths=fifths)

    melody: list[MelodyNote] = []
    for record in sorted(notes_raw, key=lambda item: (item["start_q"], item["midi"])):
        start = seconds(record["start_q"])
        end = max(start, seconds(record["end_q"]))
        melody.append(
            MelodyNote(
                start=start,
                end=end,
                midi=int(record["midi"]),
                confidence=_DEFAULT_CONFIDENCE,
                lyric=str(record.get("lyric") or ""),
                source="musicxml",
            )
        )

    total_seconds = seconds(total_q)

    chords: list[ChordEvent] = []
    harmonies_sorted = sorted(harmonies_raw, key=lambda item: item[0])
    current: ChordEvent | None = None
    for position_q, symbol in harmonies_sorted:
        start = seconds(position_q)
        if symbol == _NO_CHORD:
            if current is not None:
                current.end = max(current.start, start)
                chords.append(current)
                current = None
            continue
        if current is not None:
            current.end = max(current.start, start)
            chords.append(current)
        current = ChordEvent(
            start=start,
            end=total_seconds,
            symbol=symbol,
            confidence=_DEFAULT_CONFIDENCE,
            source="musicxml",
        )
    if current is not None:
        chords.append(current)

    sections: list[Section] = []
    sections_sorted = sorted(sections_raw, key=lambda item: item[0])
    for index, (position_q, label) in enumerate(sections_sorted):
        start = seconds(position_q)
        end = seconds(sections_sorted[index + 1][0]) if index + 1 < len(sections_sorted) else total_seconds
        sections.append(
            Section(
                start=start,
                end=max(start, end),
                kind=_guess_section_kind(label),
                label=label,
                confidence=_DEFAULT_CONFIDENCE,
            )
        )

    lyrics = _group_lyrics(melody, beat_unit_q * 60.0 / max(tempo_map.initial_bpm, 1e-6))

    score = SymbolicScore(
        beats=beats_grid,
        key=key,
        chords=chords,
        sections=sections,
        melody=melody,
        lyrics=lyrics,
        meta={"title": _text(_first(root, "movement-title"))},
        source={"format": "musicxml", "version": root.get("version", MUSICXML_VERSION)},
        schema=SCHEMA,
    )
    if sidecar:
        _apply_sidecar(score, sidecar)
    return score


def _group_lyrics(melody: Sequence[MelodyNote], beat_seconds: float) -> list[LyricLine]:
    """把逐音节的 ``<lyric>`` 按时间相近度拼回歌词行。"""
    sung = [note for note in melody if str(note.lyric or "").strip()]
    if not sung:
        return []
    gap_limit = max(1.5, 2.0 * max(beat_seconds, 1e-3))
    lines: list[LyricLine] = []
    current: list[MelodyNote] = []
    for note in sung:
        if current and note.start - current[-1].end > gap_limit:
            lines.append(_make_lyric_line(current))
            current = []
        current.append(note)
    if current:
        lines.append(_make_lyric_line(current))
    return lines


def _make_lyric_line(notes: Sequence[MelodyNote]) -> LyricLine:
    words = [
        LyricWord(start=note.start, end=note.end, text=str(note.lyric))
        for note in notes
    ]
    return LyricLine(
        start=notes[0].start,
        end=max(note.end for note in notes),
        text=" ".join(word.text for word in words),
        words=words,
        confidence=_DEFAULT_CONFIDENCE,
    )


# --------------------------------------------------------------------------- #
# sidecar 覆盖
# --------------------------------------------------------------------------- #


def _match_sidecar(
    items: Sequence[Any],
    raws: Sequence[Mapping[str, Any]],
    *,
    time_key: str = "start",
    tolerance: float = 0.15,
) -> list[tuple[Any, Mapping[str, Any]]]:
    """按「最近起点」把 XML 元素与 sidecar 记录配对（不重复使用）。"""
    pairs: list[tuple[Any, Mapping[str, Any]]] = []
    used: set[int] = set()
    for item in items:
        best: tuple[float, int] | None = None
        for index, raw in enumerate(raws):
            if index in used:
                continue
            try:
                distance = abs(float(raw.get(time_key, 0.0)) - float(item.start))
            except (TypeError, ValueError):
                continue
            if best is None or distance < best[0]:
                best = (distance, index)
        if best is None or best[0] > tolerance:
            continue
        used.add(best[1])
        pairs.append((item, raws[best[1]]))
    return pairs


def _apply_sidecar(score: SymbolicScore, sidecar: Mapping[str, Any]) -> None:
    data = dict(sidecar or {})
    if not data:
        return

    if data.get("beats"):
        score.beats = BeatGrid.from_dict(data["beats"])
    if data.get("key"):
        score.key = KeySignature.from_dict(data["key"])

    if "meta" in data:
        score.meta = {**_as_dict(data.get("meta")), **score.meta}
    for name in ("audio", "source"):
        if data.get(name):
            merged = _as_dict(getattr(score, name))
            merged.update(_as_dict(data.get(name)))
            setattr(score, name, merged)
    if data.get("provenance"):
        score.provenance = [Provenance.from_dict(item) for item in data["provenance"]]
    if data.get("warnings"):
        score.warnings = merge_warnings(score.warnings, data.get("warnings"))
    if data.get("schema"):
        score.schema = str(data["schema"])

    # ---- 音符：只覆盖绝对时间与置信度（midi / 歌词以 MusicXML 为准） ----
    side_notes = [item for item in (data.get("melody") or []) if isinstance(item, Mapping)]
    if side_notes:
        if len(side_notes) == len(score.melody):
            pairs = list(zip(score.melody, side_notes))
        else:
            pairs = _match_sidecar(score.melody, side_notes)
        for note, raw in pairs:
            try:
                note.start = float(raw.get("start", note.start))
                note.end = max(note.start, float(raw.get("end", note.end)))
            except (TypeError, ValueError):
                pass
            if raw.get("confidence") is not None:
                try:
                    note.confidence = min(1.0, max(0.0, float(raw["confidence"])))
                except (TypeError, ValueError):
                    pass
        score.melody.sort(key=lambda item: (item.start, item.midi))

    # ---- 和弦：符号以 MusicXML 为准，时间/置信度用 sidecar ----
    side_chords = [item for item in (data.get("chords") or []) if isinstance(item, Mapping)]
    if side_chords:
        pairs = (
            list(zip(score.chords, side_chords))
            if len(side_chords) == len(score.chords)
            else _match_sidecar(score.chords, side_chords)
        )
        for chord, raw in pairs:
            try:
                chord.start = float(raw.get("start", chord.start))
                chord.end = max(chord.start, float(raw.get("end", chord.end)))
            except (TypeError, ValueError):
                pass
            if raw.get("confidence") is not None:
                try:
                    chord.confidence = min(1.0, max(0.0, float(raw["confidence"])))
                except (TypeError, ValueError):
                    pass
        score.chords.sort(key=lambda item: (item.start, item.end))

    # ---- 段落：标签以 MusicXML 为准，时间/置信度用 sidecar ----
    side_sections = [item for item in (data.get("sections") or []) if isinstance(item, Mapping)]
    if side_sections:
        pairs = (
            list(zip(score.sections, side_sections))
            if len(side_sections) == len(score.sections)
            else _match_sidecar(score.sections, side_sections)
        )
        for section, raw in pairs:
            try:
                section.start = float(raw.get("start", section.start))
                section.end = max(section.start, float(raw.get("end", section.end)))
            except (TypeError, ValueError):
                pass
            for name in ("kind", "label"):
                value = raw.get(name)
                if value:
                    setattr(section, name, str(value))
            if raw.get("confidence") is not None:
                try:
                    section.confidence = min(1.0, max(0.0, float(raw["confidence"])))
                except (TypeError, ValueError):
                    pass
        score.sections.sort(key=lambda item: item.start)

    # ---- 歌词：sidecar 有完整行结构（含 words/text），直接整体覆盖 ----
    if data.get("lyrics"):
        score.lyrics = [
            LyricLine.from_dict(item) for item in data["lyrics"] if isinstance(item, Mapping)
        ]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}

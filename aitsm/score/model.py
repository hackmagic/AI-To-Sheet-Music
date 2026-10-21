"""score 数据模型与通用工具。

本模块是全套「弹唱谱编配引擎」共享的地基，只依赖标准库：

- 音名 / 音高（MIDI 号）与八度换算；
- 调号（五度圈升降数）与按调号拼写音名；
- 和弦符号解析（含斜杠低音、七和弦、挂留、加音等）与移调；
- 时值表示（四分音符拍；三连音用 ``"1/3"`` 这样的分数字符串）；
- 小节长度 / 拍号工具；
- ``score.json`` 的读写。

约定（与 ``schemas/score.schema.json`` 一致）：

- 谱面时间单位为**四分音符拍**，小节内相对（beat 0 = 小节线）；
- 音高一律为 MIDI 号，且为**记谱音高**（capo 已从发声音高扣除）；
- ``frets`` 数组按**低音弦到高音弦**排列（下标 0 = 6 弦，下标 5 = 1 弦）；
- ``strings`` 表示法里的弦号 ``1`` = 最高音弦（1 弦）。

中文注释、英文标识符、类型标注，全部纯标准库实现。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "SHARP_NAMES",
    "FLAT_NAMES",
    "STANDARD_TUNING",
    "DROP_D_TUNING",
    "STRING_NAMES",
    "ChordSymbol",
    "SECTION_LABELS",
    "SECTION_KINDS",
    "note_to_pc",
    "pc_to_name",
    "pc_to_name_for_fifths",
    "key_fifths",
    "fifths_for",
    "spell_midi",
    "midi_to_name",
    "parse_note_name",
    "parse_chord_symbol",
    "chord_intervals",
    "chord_pitch_classes",
    "transpose_chord_symbol",
    "format_chord_symbol",
    "duration_value",
    "format_duration",
    "bar_length_quarters",
    "parse_time_signature",
    "tuning_to_midis",
    "fret_to_midi",
    "midi_to_fret_candidates",
    "load_score",
    "dump_score",
]

# --------------------------------------------------------------------------
# 音名与音高
# --------------------------------------------------------------------------

#: 升号拼写（分析结果统一用升号，见 analysis.schema.json 的 key.tonic 说明）
SHARP_NAMES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
#: 降号拼写（平调调号下显示用）
FLAT_NAMES: tuple[str, ...] = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

#: 标准调弦，低到高（E2 A2 D3 G3 B3 E4）
STANDARD_TUNING: tuple[int, ...] = (40, 45, 50, 55, 59, 64)
#: Drop D 调弦，低到高（D2 A2 D3 G3 B3 E4）
DROP_D_TUNING: tuple[int, ...] = (38, 45, 50, 55, 59, 64)

#: 弦号（1 = 最高音弦）到音名的展示名，用于指法图与提示文案
STRING_NAMES: dict[int, str] = {1: "1弦(e)", 2: "2弦(B)", 3: "3弦(G)", 4: "4弦(D)", 5: "5弦(A)", 6: "6弦(E)"}

TUNINGS: dict[str, tuple[int, ...]] = {
    "standard": STANDARD_TUNING,
    "drop_d": DROP_D_TUNING,
}

_NOTE_ALIASES: dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1, "D♭": 1, "C♯": 1,
    "D": 2,
    "D#": 3, "Eb": 3, "E♭": 3, "D♯": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5,
    "F#": 6, "Gb": 6, "G♭": 6, "F♯": 6,
    "G": 7,
    "G#": 8, "Ab": 8, "A♭": 8, "G♯": 8,
    "A": 9,
    "A#": 10, "Bb": 10, "B♭": 10, "A♯": 10,
    "B": 11, "Cb": 11,
}

#: 大调主音 -> 调号升降数（五度圈，正为升）
_MAJOR_FIFTHS: dict[int, int] = {
    0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7,
    5: -1, 10: -2, 3: -3, 8: -4,
}
#: 小调主音 -> 调号升降数（关系大调推导）
_MINOR_FIFTHS: dict[int, int] = {
    9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: 6, 10: 7,
    2: -1, 7: -2, 0: -3, 5: -4,
}

#: 段落 kind -> 中文 label
SECTION_LABELS: dict[str, str] = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "预副歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "interlude": "间奏",
    "solo": "独奏",
    "outro": "尾奏",
    "other": "其他",
}
SECTION_KINDS: tuple[str, ...] = tuple(SECTION_LABELS)


def note_to_pc(name: str) -> int:
    """音名 -> 音级（0=C ... 11=B）。无法识别时抛 ``ValueError``。"""
    if not isinstance(name, str):
        raise ValueError(f"音名必须是字符串: {name!r}")
    text = name.strip()
    if not text:
        raise ValueError("音名不能为空")
    key = text[0].upper() + text[1:]
    if key in _NOTE_ALIASES:
        return _NOTE_ALIASES[key]
    raise ValueError(f"无法识别的音名: {name!r}")


def parse_note_name(name: str) -> tuple[int, int | None]:
    """解析带八度的音名（如 ``"C#4"``），返回 ``(pc, octave)``；无八度时 octave 为 None。"""
    match = re.fullmatch(r"\s*([A-Ga-g][#b♯♭]?)\s*(-?\d+)?\s*", str(name or ""))
    if not match:
        raise ValueError(f"无法识别的音名: {name!r}")
    pc = note_to_pc(match.group(1))
    octave = int(match.group(2)) if match.group(2) is not None else None
    return pc, octave


def pc_to_name(pc: int, prefer_flats: bool = False) -> str:
    """音级 -> 音名。``prefer_flats`` 为真时用降号拼写。"""
    return FLAT_NAMES[pc % 12] if prefer_flats else SHARP_NAMES[pc % 12]


def key_fifths(tonic_pc: int, mode: str = "major") -> int:
    """主音音级 + 调式 -> 调号升降数（-7..7，正为升号）。"""
    table = _MINOR_FIFTHS if str(mode).lower().startswith("min") else _MAJOR_FIFTHS
    return table[tonic_pc % 12]


def fifths_for(tonic: str, mode: str = "major") -> int:
    """音名主音 + 调式 -> 调号升降数。"""
    return key_fifths(note_to_pc(tonic), mode)


def pc_to_name_for_fifths(pc: int, fifths: int) -> str:
    """按调号方向拼写音级：降号调用降号，其余用升号。"""
    return pc_to_name(pc, prefer_flats=fifths < 0)


def spell_midi(midi: int, fifths: int = 0) -> tuple[str, int, int]:
    """MIDI 音高 -> ``(step, alter, octave)``，按调号方向选择等音拼写。

    ``step`` 为 ``C``..``B``，``alter`` 为 -1/0/1，可直接写入 MusicXML ``<pitch>``。
    """
    midi = int(midi)
    pc = midi % 12
    octave = midi // 12 - 1
    name = pc_to_name(pc, prefer_flats=fifths < 0)
    step = name[0]
    alter = 1 if name.endswith("#") else (-1 if name.endswith("b") else 0)
    return step, alter, octave


def midi_to_name(midi: int, prefer_flats: bool = False) -> str:
    """MIDI 音高 -> 带八度的音名（如 ``C4`` = 60）。"""
    midi = int(midi)
    return f"{pc_to_name(midi % 12, prefer_flats)}{midi // 12 - 1}"


# --------------------------------------------------------------------------
# 和弦符号
# --------------------------------------------------------------------------

#: 和弦性质 -> 相对根音的半音集合（14 = 九音）
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "5": (0, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "aug": (0, 4, 8),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "69": (0, 4, 7, 9, 14),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7sus4": (0, 5, 7, 10),
}

#: 判定「指法合法」时必须出现的关键音级（相对根音，取模 12）：
#: 三和弦需根音 + 三音 + 五音；挂留用二度/四度替代三音；五和弦只要根音 + 五音。
QUALITY_ESSENTIAL: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "5": (0, 7),
    "7": (0, 4, 10),
    "maj7": (0, 4, 11),
    "m7": (0, 3, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6),
    "m7b5": (0, 3, 6),
    "aug": (0, 4, 8),
    "6": (0, 4, 7),
    "m6": (0, 3, 7),
    "69": (0, 4, 7),
    "9": (0, 4, 10),
    "maj9": (0, 4, 11),
    "m9": (0, 3, 10),
    "add9": (0, 4, 7),
    "madd9": (0, 3, 7),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7sus4": (0, 5, 10),
}

#: 和弦后缀的规范化别名（小写、去空格后查表）
_SUFFIX_ALIASES: dict[str, str] = {
    "": "maj",
    "maj": "maj",
    "major": "maj",
    "M": "maj",
    "Δ": "maj",
    "m": "min",
    "min": "min",
    "minor": "min",
    "-": "min",
    "5": "5",
    "power": "5",
    "7": "7",
    "dom7": "7",
    "maj7": "maj7",
    "M7": "maj7",
    "Δ7": "maj7",
    "major7": "maj7",
    "m7": "m7",
    "min7": "m7",
    "-7": "m7",
    "minor7": "m7",
    "dim": "dim",
    "dim7": "dim7",
    "o": "dim",
    "°": "dim",
    "o7": "dim7",
    "°7": "dim7",
    "m7b5": "m7b5",
    "ø": "m7b5",
    "ø7": "m7b5",
    "aug": "aug",
    "+": "aug",
    "aug7": "aug",
    "6": "6",
    "add6": "6",
    "m6": "m6",
    "min6": "m6",
    "-6": "m6",
    "69": "69",
    "6/9": "69",
    "9": "9",
    "maj9": "maj9",
    "M9": "maj9",
    "m9": "m9",
    "min9": "m9",
    "-9": "m9",
    "add9": "add9",
    "madd9": "madd9",
    "m(add9)": "madd9",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus": "sus4",
    "7sus4": "7sus4",
    "7sus": "7sus4",
}

_CHORD_RE = re.compile(
    r"^\s*([A-Ga-g][#b♯♭]?)\s*"          # 根音
    r"([^/]*?)"                           # 和弦性质
    r"(?:\s*/\s*([A-Ga-g][#b♯♭]?))?\s*$"  # 可选斜杠低音
)

#: 规范化的和弦性质 -> 默认后缀（用于重建和弦符号）
_QUALITY_SUFFIX: dict[str, str] = {
    "maj": "",
    "min": "m",
    "5": "5",
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "dim": "dim",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "aug": "aug",
    "6": "6",
    "m6": "m6",
    "69": "69",
    "9": "9",
    "maj9": "maj9",
    "m9": "m9",
    "add9": "add9",
    "madd9": "madd9",
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
}


@dataclass(frozen=True)
class ChordSymbol:
    """解析后的和弦符号。

    - ``symbol``：规范化后的书写形式（如 ``C``、``Am7``、``G/B``）；
    - ``root_pc``/``bass_pc``：根音与斜杠低音的音级（无斜杠时 bass_pc 为 None）；
    - ``quality``：规范化性质键（见 :data:`QUALITY_INTERVALS`）；
    - ``intervals``：相对根音的半音集合；
    - ``unsupported``：后缀无法识别、已退化按大三和弦处理时为 True。
    """

    symbol: str
    root_pc: int
    quality: str
    bass_pc: int | None = None
    unsupported: bool = False
    raw: str = ""

    @property
    def is_minor(self) -> bool:
        return self.quality.startswith("m") and not self.quality.startswith("maj")

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITY_INTERVALS[self.quality]

    @property
    def essential(self) -> tuple[int, ...]:
        """判定指法合法性的关键音级（相对根音）。"""
        return QUALITY_ESSENTIAL[self.quality]

    @property
    def pitch_classes(self) -> frozenset[int]:
        """和弦包含的音级集合（含斜杠低音）。"""
        pcs = {(self.root_pc + interval) % 12 for interval in self.intervals}
        if self.bass_pc is not None:
            pcs.add(self.bass_pc % 12)
        return frozenset(pcs)

    @property
    def required_pcs(self) -> frozenset[int]:
        """关键音级绝对化后的集合。"""
        pcs = {(self.root_pc + interval) % 12 for interval in self.essential}
        if self.bass_pc is not None:
            pcs.add(self.bass_pc % 12)
        return frozenset(pcs)


def chord_intervals(quality: str) -> tuple[int, ...]:
    """和弦性质 -> 相对根音的半音集合；未知性质退化为大三和弦。"""
    return QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["maj"])


def chord_pitch_classes(symbol: str) -> frozenset[int]:
    """和弦符号 -> 包含的音级集合（解析失败时抛 ``ValueError``）。"""
    return parse_chord_symbol(symbol).pitch_classes


def parse_chord_symbol(symbol: str) -> ChordSymbol:
    """解析和弦符号。

    支持：``C`` ``Cm`` ``C7`` ``Cmaj7`` ``Cm7`` ``Csus2`` ``Csus4`` ``Cdim``
    ``Caug`` ``C6`` ``Cm6`` ``C9`` ``Cadd9`` ``Cmadd9`` ``C5`` ``C7sus4``
    ``Cm7b5`` ``Cdim7`` 以及斜杠和弦 ``C/E`` ``G/B`` ``D/F#``。
    后缀无法识别时退化为大三和弦，并把 ``unsupported`` 置 True（调用方负责告警，
    不静默编造音响）。
    """
    raw = str(symbol or "").strip().replace("♯", "#").replace("♭", "b")
    if not raw:
        raise ValueError("和弦符号不能为空")
    match = _CHORD_RE.match(raw)
    if not match:
        raise ValueError(f"无法解析的和弦符号: {symbol!r}")
    root_text, suffix_text, bass_text = match.group(1), match.group(2), match.group(3)
    root_pc = note_to_pc(root_text[0].upper() + root_text[1:])
    unsupported = False
    quality = "maj"
    suffix = (suffix_text or "").strip()
    if suffix:
        # 逐级退化：先试完整后缀，再试去掉常见修饰记号后的核心后缀
        candidate = suffix
        for _ in range(3):
            lookup = _SUFFIX_ALIASES.get(candidate)
            if lookup is None:
                # 大小写不敏感再试一次（如 "MAJ7"、"MIN7"）
                lookup = _SUFFIX_ALIASES.get(candidate.capitalize())
            if lookup is not None:
                quality = lookup
                break
            # 去掉括号内容 / 尾部 alter（如 "7#9"、"add9(no3)"）
            stripped = re.sub(r"\([^)]*\)", "", candidate)
            stripped = re.sub(r"(?:b|#)\d+$", "", stripped)
            stripped = stripped.replace(" ", "")
            if stripped == candidate or not stripped:
                unsupported = True
                quality = "maj"
                break
            candidate = stripped
    bass_pc: int | None = None
    if bass_text:
        bass_pc = note_to_pc(bass_text[0].upper() + bass_text[1:])
        if bass_pc == root_pc:
            bass_pc = None
    normalized = format_chord_symbol(root_pc, quality, bass_pc)
    return ChordSymbol(
        symbol=normalized,
        root_pc=root_pc,
        quality=quality,
        bass_pc=bass_pc,
        unsupported=unsupported,
        raw=raw,
    )


def format_chord_symbol(
    root_pc: int,
    quality: str = "maj",
    bass_pc: int | None = None,
    *,
    prefer_flats: bool = False,
) -> str:
    """用音级与性质重建和弦符号。"""
    root = pc_to_name(root_pc, prefer_flats)
    suffix = _QUALITY_SUFFIX.get(quality, "")
    text = f"{root}{suffix}"
    if bass_pc is not None and bass_pc % 12 != root_pc % 12:
        text += f"/{pc_to_name(bass_pc, prefer_flats)}"
    return text


def transpose_chord_symbol(
    symbol: str,
    semitones: int,
    *,
    prefer_flats: bool = False,
) -> str:
    """和弦符号整体移调 ``semitones`` 半音（斜杠低音一并移动）。"""
    chord = parse_chord_symbol(symbol)
    if not semitones:
        return format_chord_symbol(
            chord.root_pc, chord.quality, chord.bass_pc, prefer_flats=prefer_flats
        )
    return format_chord_symbol(
        (chord.root_pc + semitones) % 12,
        chord.quality,
        None if chord.bass_pc is None else (chord.bass_pc + semitones) % 12,
        prefer_flats=prefer_flats,
    )


# --------------------------------------------------------------------------
# 时值 / 拍号
# --------------------------------------------------------------------------

#: 谱面时间的最小网格（四分音符拍的分数），1/4 = 十六分音符
_GRID_EPS = 1e-6


def duration_value(duration: Any) -> float:
    """时值 -> float（四分音符拍）。支持 ``number`` 与 ``"1/3"`` 分数字符串。"""
    if duration is None:
        return 0.0
    if isinstance(duration, bool):
        raise ValueError(f"非法时值: {duration!r}")
    if isinstance(duration, (int, float)):
        return float(duration)
    if isinstance(duration, str):
        text = duration.strip()
        if not text:
            return 0.0
        if "/" in text:
            num_text, den_text = text.split("/", 1)
            den = float(den_text)
            if den == 0:
                raise ValueError(f"非法时值: {duration!r}")
            return float(num_text) / den
        return float(text)
    raise ValueError(f"非法时值: {duration!r}")


def format_duration(value: float) -> float | str:
    """float 时值 -> JSON 友好的表示：三连音类写成 ``"n/3"``，其余写 float。

    只在「无法用有限小数精确表示」或「明显是三连音网格」时返回分数字符串，
    保证 ``score.json`` 中的时值可被前端与导出器无损还原。
    """
    value = float(value)
    if abs(value - round(value)) < _GRID_EPS:
        return float(int(round(value)))
    # 三连音网格：value * 3 接近整数（1/3、1/6、2/3 ...）
    scaled = value * 3
    if abs(scaled - round(scaled)) < _GRID_EPS:
        numerator = int(round(scaled))
        fraction = Fraction(numerator, 3)
        # 能约分成 1/2、1/4 等有限小数时仍用 float，方便前端计算
        if fraction.denominator in (1, 2, 4, 8):
            return float(value)
        return f"{fraction.numerator}/{fraction.denominator}"
    return round(value, 6)


def parse_time_signature(value: Any, default: tuple[int, int] = (4, 4)) -> tuple[int, int]:
    """解析拍号：``"4/4"`` / ``[4, 4]`` / ``(4, 4)`` -> ``(4, 4)``。"""
    if value is None:
        return default
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
        if not match:
            return default
        return int(match.group(1)), int(match.group(2))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            numerator, denominator = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return default
        if numerator >= 1 and denominator >= 1:
            return numerator, denominator
    return default


def bar_length_quarters(time_signature: Sequence[int]) -> float:
    """拍号 -> 每小节长度（四分音符拍）。4/4 -> 4.0，3/4 -> 3.0，6/8 -> 3.0。"""
    numerator, denominator = int(time_signature[0]), int(time_signature[1])
    if numerator < 1 or denominator < 1:
        raise ValueError(f"非法拍号: {time_signature!r}")
    return numerator * 4.0 / denominator


def quarter_beats_per_beat(beats_per_bar: int, time_signature: Sequence[int]) -> float:
    """「分析节拍」相对于四分音符的倍数。

    4/4 的 beat 是四分音符 -> 1.0；6/8 若 ``beats_per_bar=6`` 则每个 beat 是八分音符
    -> 0.5；若 ``beats_per_bar=2``（按附点四分打拍）则每个 beat 是 1.5 个四分音符。
    """
    numerator, denominator = int(time_signature[0]), int(time_signature[1])
    bar_len = bar_length_quarters((numerator, denominator))
    beats = int(beats_per_bar) if beats_per_bar else numerator
    if beats <= 0:
        beats = numerator
    return bar_len / beats


# --------------------------------------------------------------------------
# 调弦 / 弦品换算
# --------------------------------------------------------------------------

def tuning_to_midis(name_or_list: Any) -> list[int]:
    """调弦名或显式音高数组 -> 低到高的 6 个空弦 MIDI 音高。"""
    if isinstance(name_or_list, str):
        key = name_or_list.strip().lower().replace("-", "_").replace(" ", "_")
        if key in TUNINGS:
            return list(TUNINGS[key])
        raise ValueError(f"未知调弦: {name_or_list!r}")
    if isinstance(name_or_list, (list, tuple)) and len(name_or_list) == 6:
        return [int(pitch) for pitch in name_or_list]
    raise ValueError(f"非法调弦定义: {name_or_list!r}")


def fret_to_midi(tuning: Sequence[int], string_number: int, fret: int | None) -> int | None:
    """弦号（1 = 最高音弦）+ 品位 -> MIDI 音高；闷弦 / 不弹返回 None。"""
    if fret is None or fret < 0:
        return None
    tuning = list(tuning)
    if not 1 <= int(string_number) <= len(tuning):
        raise ValueError(f"弦号越界: {string_number!r}")
    index = len(tuning) - int(string_number)
    return int(tuning[index]) + int(fret)


def midi_to_fret_candidates(
    midi: int,
    tuning: Sequence[int] = STANDARD_TUNING,
    *,
    max_fret: int = 15,
    min_fret: int = 0,
    max_results: int = 6,
) -> list[tuple[int, int]]:
    """MIDI 音高 -> 候选 ``(弦号, 品位)``，按品位从低到高排序（弦号 1 = 最高音弦）。"""
    tuning = list(tuning)
    results: list[tuple[int, int]] = []
    for index, open_pitch in enumerate(tuning):
        fret = int(midi) - int(open_pitch)
        if min_fret <= fret <= max_fret:
            results.append((len(tuning) - index, fret))
    results.sort(key=lambda item: (item[1], item[0]))
    return results[:max_results]


# --------------------------------------------------------------------------
# score.json 读写
# --------------------------------------------------------------------------

def load_score(path: str | Path) -> dict:
    """读取 score.json。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_score(score: dict, path: str | Path, *, indent: int = 2) -> Path:
    """写出 score.json（UTF-8、不转义中文）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(score, ensure_ascii=False, indent=indent, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:
    """numpy 标量等非原生类型的兜底序列化。"""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"无法序列化的类型: {type(value)!r}")


def clamp(value: float, low: float, high: float) -> float:
    """把数值夹到 ``[low, high]`` 区间。"""
    return max(low, min(high, value))


def quantize(value: float, grid: float) -> float:
    """把数值量化到最近的 ``grid`` 倍数（浮点噪声安全）。"""
    if grid <= 0:
        return float(value)
    steps = round(float(value) / grid)
    return round(steps * grid, 6)


def format_beats(value: float) -> float | str:
    """拍位置格式化：整数去掉小数点，三连音位置写成 ``"n/3"``。"""
    result = format_duration(value)
    return result


def iter_bars(score: dict) -> Iterable[dict]:
    """按顺序遍历小节。"""
    return score.get("bars") or []


def ensure_float(value: Any, default: float = 0.0) -> float:
    """安全转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_finite_number(value: Any) -> bool:
    """是否为有限数值（排除 bool / NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))

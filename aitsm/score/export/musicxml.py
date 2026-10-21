"""手写 MusicXML 4.0 导出器（不依赖 music21 / MuseScore CLI）。

输出结构
--------
``score-partwise`` 两个声部：

* ``P1`` **主旋律**（treble）：调号 / 拍号 / 速度 / 段落排号 / ``<harmony>`` 和弦符号 /
  带 ``<lyric>`` 的歌词；
* ``P2`` **吉他 TAB**（``<clef><sign>TAB</sign>`` + ``<staff-details>``）：
  每个和弦按所选指法写成一组同时发声的音，并带 ``<technical><string><fret>``。

记谱约定
--------
整份谱**都写在「变调夹位」上**（``meta.key.play``）：旋律音高与和弦音节已扣掉 capo，
TAB 品位也是相对琴枕的（与真人看 capo 谱的习惯一致），``<capo>`` 写进
``<staff-details>`` 供 MuseScore 显示。这样谱面内部自洽，不会出现「旋律与 TAB 差 capo」。
"""

from __future__ import annotations

import datetime as _dt
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..model import (
    bar_length_quarters,
    duration_value,
    fret_to_midi,
    parse_chord_symbol,
    parse_time_signature,
    spell_midi,
    tuning_to_midis,
)
from . import ExportError

__all__ = ["export_musicxml", "DIVISIONS"]

#: 每个四分音符的 divisions。480 能被 2/3/4/5/6/8 整除，三连音（160）也不会有余数。
DIVISIONS = 480

#: 默认每行小节数（``meta.bars_per_system`` 可覆盖）。真谱上 4 小节一行最容易读。
DEFAULT_BARS_PER_SYSTEM = 4

_EPS = 1e-6
_MUSICXML_VERSION = "4.0"
_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)

#: 和弦性质 -> (MusicXML kind, 谱面显示后缀)
_KIND_MAP: dict[str, tuple[str, str]] = {
    "maj": ("major", ""),
    "min": ("minor", "m"),
    "5": ("power", "5"),
    "7": ("dominant", "7"),
    "maj7": ("major-seventh", "maj7"),
    "m7": ("minor-seventh", "m7"),
    "dim": ("diminished", "dim"),
    "dim7": ("diminished-seventh", "dim7"),
    "m7b5": ("half-diminished", "m7b5"),
    "aug": ("augmented", "aug"),
    "6": ("major-sixth", "6"),
    "m6": ("minor-sixth", "m6"),
    "69": ("major-sixth", "6/9"),
    "9": ("dominant-ninth", "9"),
    "maj9": ("major-ninth", "maj9"),
    "m9": ("minor-ninth", "m9"),
    "add9": ("major", "add9"),
    "madd9": ("minor", "madd9"),
    "sus2": ("suspended-second", "sus2"),
    "sus4": ("suspended-fourth", "sus4"),
    "7sus4": ("suspended-fourth", "7sus4"),
}

#: 需要额外 ``<degree>`` 的性质：quality -> [(音级, 升降, 类型)]
_EXTRA_DEGREES: dict[str, list[tuple[int, int, str]]] = {
    "69": [(9, 0, "add")],
    "add9": [(9, 0, "add")],
    "madd9": [(9, 0, "add")],
    "7sus4": [(7, -1, "add")],
}

#: (四分音符时值, MusicXML type, 附点数)
_NOTE_VALUES: tuple[tuple[float, str, int], ...] = (
    (8.0, "breve", 0),
    (6.0, "whole", 1),
    (4.0, "whole", 0),
    (3.0, "half", 1),
    (2.0, "half", 0),
    (1.5, "quarter", 1),
    (1.0, "quarter", 0),
    (0.75, "eighth", 1),
    (0.5, "eighth", 0),
    (0.375, "16th", 1),
    (0.25, "16th", 0),
    (0.1875, "32nd", 1),
    (0.125, "32nd", 0),
    (0.0625, "64th", 0),
)


# ---------------------------------------------------------------- 类型与时值换算


def _ticks(dur_quarters: float) -> int:
    """四分音符时值 -> divisions 刻度（最小 1）。"""
    return max(1, int(round(float(dur_quarters) * DIVISIONS)))


def _triplet_ratio(dur_quarters: float) -> tuple[int, int] | None:
    """判断是否为三连音时值（1/3、2/3、4/3、8/3 拍），返回 (3, 2)。"""
    for numerator in (1, 2, 4, 8):
        if abs(dur_quarters - numerator / 3.0) < 1e-6:
            return (3, 2)
    return None


def _note_type(dur_quarters: float) -> tuple[str, int, tuple[int, int] | None]:
    """时值 -> (type 文本, 附点数, 三连音比例)。"""
    ratio = _triplet_ratio(dur_quarters)
    target = dur_quarters * 1.5 if ratio else dur_quarters
    best = min(_NOTE_VALUES, key=lambda item: abs(item[0] - target))
    return best[1], best[2], ratio


def _append_pitch(parent: ET.Element, step: str, alter: int, octave: int) -> None:
    pitch = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(int(alter))
    ET.SubElement(pitch, "octave").text = str(int(octave))


def _append_duration(parent: ET.Element, dur_quarters: float) -> None:
    """写 ``<duration>``（必须是 note 内第二个元素，紧跟 pitch/rest/chord 之后）。"""
    ET.SubElement(parent, "duration").text = str(_ticks(dur_quarters))


def _append_note_type(parent: ET.Element, dur_quarters: float) -> None:
    """写 ``<type>`` + 附点 + 三连音标记。

    MusicXML 的 note 子元素有**固定顺序**：``duration`` → ``tie`` → ``voice`` →
    ``type`` → ``dot`` → ``time-modification`` → ``notations`` → ``lyric``。
    所以 type 必须在 voice 之后单独写，不能和 duration 一起写。
    """
    note_type, dots, ratio = _note_type(dur_quarters)
    ET.SubElement(parent, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(parent, "dot")
    if ratio is not None:
        mod = ET.SubElement(parent, "time-modification")
        ET.SubElement(mod, "actual-notes").text = str(ratio[0])
        ET.SubElement(mod, "normal-notes").text = str(ratio[1])


def _append_rest(parent: ET.Element, dur_quarters: float, *, voice: int = 1) -> None:
    if dur_quarters <= _EPS:
        return
    note = ET.SubElement(parent, "note")
    ET.SubElement(note, "rest")
    _append_duration(note, dur_quarters)
    ET.SubElement(note, "voice").text = str(voice)
    _append_note_type(note, dur_quarters)



def _vocal_tie_maps(bars: Sequence[dict]) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], bool]]:
    """计算可写出的延音线 / 一字多音延长标记。

    MusicXML 的 ``<tie>`` 只能连接**同音高**的音；analysis 里的 ``tie`` 在
    一字多音（melisma）时音高会变，直接写出会让 MuseScore 画出断掉的延音线。
    这里按音高验证：同音高才保留 tie，否则改成 ``<lyric><extend/>``。
    """
    flat: list[tuple[int, int, dict]] = []
    for bar in bars:
        bar_index = int(bar.get("index") or 0)
        for position, (_beat, note) in enumerate(_vocal_events(bar)):
            if note.get("midi") is None:
                continue
            flat.append((bar_index, position, note))
    tie_map: dict[tuple[int, int], str] = {}
    extend_map: dict[tuple[int, int], bool] = {}
    for position, (bar_index, note_position, note) in enumerate(flat):
        midi = note.get("midi")
        previous = flat[position - 1][2] if position > 0 else None
        following = flat[position + 1][2] if position + 1 < len(flat) else None
        tie = note.get("tie")
        keep: str | None = None
        if tie == "start" and following is not None and following.get("midi") == midi:
            keep = "start"
        elif tie in ("stop", "continue") and previous is not None and previous.get("midi") == midi:
            keep = str(tie)
        if keep is not None:
            tie_map[(bar_index, note_position)] = keep
        if note.get("lyric") is None:
            previous_has_lyric = bool(
                previous is not None
                and previous.get("midi") is not None
                and (previous.get("lyric") or previous.get("lyric_extend"))
            )
            if note.get("lyric_extend") or previous_has_lyric:
                extend_map[(bar_index, note_position)] = True
    return tie_map, extend_map


def _append_notations(note: ET.Element, *, tie: str | None = None, tab: tuple[int, int] | None = None) -> None:
    """写 ``<notations>``：延音线 + TAB 弦品。``tab`` 为 (弦号, 品位)。"""
    if tie is None and tab is None:
        return
    notations = ET.SubElement(note, "notations")
    if tie in ("start", "stop", "continue"):
        ET.SubElement(notations, "tied", {"type": tie})
    if tab is not None:
        technical = ET.SubElement(notations, "technical")
        string_number, fret = tab
        ET.SubElement(technical, "string").text = str(int(string_number))
        ET.SubElement(technical, "fret").text = str(int(fret))


# ---------------------------------------------------------------- 乐谱上下文


class _Context:
    """从 score.meta 提炼出的导出上下文（调号、速度、调弦……）。"""

    def __init__(self, score: dict) -> None:
        meta = score.get("meta") or {}
        self.meta = meta
        key = meta.get("key") or {}
        self.fifths = int(key.get("num_sharps_flats") or 0)
        self.fifths = max(-7, min(7, self.fifths))
        self.mode = "minor" if str(key.get("mode", "major")).lower().startswith("min") else "major"
        self.time_signature = parse_time_signature(meta.get("time_signature"), (4, 4))
        tempo = meta.get("tempo")
        try:
            self.tempo = float(tempo) if tempo else None
        except (TypeError, ValueError):
            self.tempo = None
        if self.tempo is not None and not (1.0 <= self.tempo <= 400.0):
            self.tempo = None
        self.title = str(meta.get("title") or "未命名")
        self.artist = str(meta.get("artist") or "")
        self.capo = int(meta.get("capo") or 0)
        guitar = meta.get("guitar") or {}
        tuning = guitar.get("tuning") or guitar.get("tuning_name") or "standard"
        try:
            self.tuning_midis = tuning_to_midis(tuning)
        except Exception:  # noqa: BLE001 - 调弦无法解析时退回标准调弦
            self.tuning_midis = tuning_to_midis("standard")
        self.voicings: dict[str, dict] = {
            str(key): value for key, value in (score.get("voicings") or {}).items() if isinstance(value, dict)
        }
        self.include_tab = bool(guitar) and meta.get("export_tab", True) is not False
        # 延音线 / 一字多音的合法性映射（按小节号 + 小节内事件下标索引）
        self.tie_map, self.extend_map = _vocal_tie_maps(_bars(score))
        self.section_starts: dict[int, dict] = {}
        for section in score.get("sections") or []:
            try:
                self.section_starts[int(section.get("start_bar") or 0)] = section
            except (TypeError, ValueError):
                continue


def _bars(score: dict) -> list[dict]:
    bars = [bar for bar in (score.get("bars") or []) if isinstance(bar, dict)]
    return bars


def _bar_length(bar: dict, ctx: _Context) -> float:
    """小节应有长度（四分音符拍）；弱起小节取实际内容长度。

    弱起（pickup）小节不能按完整拍号补满，否则 MusicXML 里会多出空拍、
    和后面的小节错位。它也不能被「跨满小节」的和弦时值撑满：上游可能给弱起
    小节的最后一个和弦写 ``dur=4``（按完整小节算的），这里必须忽略这种时值。
    """
    signature = parse_time_signature(bar.get("time_signature") or ctx.time_signature, ctx.time_signature)
    full = bar_length_quarters(signature)
    if not bar.get("pickup"):
        return full

    content = 0.0
    for note in bar.get("vocal") or []:
        if not isinstance(note, dict):
            continue
        content = max(content, _safe_beats(note.get("beat")) + _safe_dur(note.get("dur"), 1.0))
    guitar = bar.get("guitar") or {}
    if content <= _EPS:
        # 没有旋律时才看扫弦：有旋律的弱起小节长度由「唱进来的第一拍」决定
        for hit in guitar.get("hits") or []:
            if not isinstance(hit, dict):
                continue
            content = max(content, _safe_beats(hit.get("beat")) + _safe_dur(hit.get("dur"), 0.5))
    if content <= _EPS:
        # 没有旋律/扫弦可参考时，只能看和弦起音：跨满小节的时值不可信，按一拍计
        onsets = 0.0
        for chord in bar.get("chords") or []:
            if not isinstance(chord, dict):
                continue
            start = _safe_beats(chord.get("beat"))
            dur = _safe_dur(chord.get("dur"), full)
            if dur >= full - 1e-6:
                dur = 1.0
            onsets = max(onsets, start + dur)
        content = onsets
    if content <= _EPS:
        return full
    # 弱起小节至少要有一个四分音符，且不超过完整小节
    return min(full, max(1.0, round(content * 4) / 4))


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


# ---------------------------------------------------------------- 版面（换行）


def _bar_number(bar: dict, position: int) -> int:
    """小节号：优先用 bar.index，缺失/非法时退回顺序号。"""
    try:
        index = int(bar.get("index") or position)
    except (TypeError, ValueError):
        return position
    return index


def _bars_per_system(score: dict) -> int:
    """每行小节数（``meta.bars_per_system`` 可覆盖；0 表示只按段落换行）。"""
    meta = score.get("meta") or {}
    value = meta.get("bars_per_system")
    if value is None:
        return DEFAULT_BARS_PER_SYSTEM
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_BARS_PER_SYSTEM


def _chunk_sizes(count: int, target: int) -> list[int]:
    """把 ``count`` 个小节切成每行不超过 ``target`` 的块，且不留单小节的孤行。"""
    if count <= 0:
        return []
    if target < 2 or count <= target:
        return [count]
    sizes: list[int] = []
    remaining = count
    while remaining > target:
        size = target
        if remaining - size == 1:  # 最后一行会只剩 1 小节 -> 挪一个过去
            size -= 1
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes


def _section_runs(bars: list[dict], score: dict) -> list[list[int]]:
    """把小节按段落切成连续的「位置」区间，未被段落覆盖的小节并入前一段落。"""
    ranges: list[tuple[int, int]] = []
    for section in score.get("sections") or []:
        if not isinstance(section, dict):
            continue
        try:
            start = int(section.get("start_bar") or 0)
            end = int(section.get("end_bar") or start)
        except (TypeError, ValueError):
            continue
        start = max(1, start)
        end = min(len(bars), max(start, end))
        if start <= end:
            ranges.append((start, end))

    runs: list[list[int]] = []
    previous_key: int | None = None
    for position, bar in enumerate(bars, start=1):
        index = _bar_number(bar, position)
        key: int | None = None
        for number, (start, end) in enumerate(ranges):
            if start <= index <= end:
                key = number
                break
        if not runs:
            runs.append([position])
        elif key is None or key == previous_key:
            runs[-1].append(position)  # 段落空洞/同段落：继续当前行组
        else:
            runs.append([position])
        if key is not None:
            previous_key = key
    return runs


def _system_breaks(bars: list[dict], score: dict) -> set[int]:
    """需要强制换行的小节位置（1-based）。

    规则：段落开头一定换行；每行最多 ``bars_per_system`` 小节（默认 4）；
    单小节孤行会被并入相邻行；``bar.new_system`` / ``bar.new_page`` 也生效。
    """
    target = _bars_per_system(score)
    breaks: set[int] = set()
    for run_index, run in enumerate(_section_runs(bars, score)):
        if target < 2:
            # 只按段落换行：段落第一小节换行，段内不自动断行
            if run_index > 0:
                breaks.add(run[0])
            continue
        cursor = 0
        for chunk_index, size in enumerate(_chunk_sizes(len(run), target)):
            if run_index > 0 or chunk_index > 0:
                breaks.add(run[cursor])
            cursor += size
    for position, bar in enumerate(bars, start=1):
        if position > 1 and (bar.get("new_system") or bar.get("new_page")):
            breaks.add(position)
    return breaks


# ---------------------------------------------------------------- 和弦符号


def _append_harmony(parent: ET.Element, chord: dict, ctx: _Context) -> bool:
    """写一个 ``<harmony>``；符号无法解析时返回 False（由调用方告警）。"""
    symbol = str(chord.get("symbol") or "").strip()
    if not symbol:
        return False
    try:
        parsed = parse_chord_symbol(symbol)
    except ValueError:
        return False
    kind, text = _KIND_MAP.get(parsed.quality, ("major", ""))
    harmony = ET.SubElement(parent, "harmony")
    # 注意：MusicXML 的 <harmony> 没有 text 属性（只有 <kind> 有），
    # 和弦显示文本通过 <kind text="..."> 给出。
    root = ET.SubElement(harmony, "root")
    step, alter, _octave = spell_midi(60 + parsed.root_pc, ctx.fifths)
    ET.SubElement(root, "root-step").text = step
    if alter:
        ET.SubElement(root, "root-alter").text = str(int(alter))
    ET.SubElement(harmony, "kind", {"text": text or symbol}).text = kind
    if parsed.bass_pc is not None and parsed.bass_pc % 12 != parsed.root_pc % 12:
        bass = ET.SubElement(harmony, "bass")
        step, alter, _octave = spell_midi(60 + parsed.bass_pc, ctx.fifths)
        ET.SubElement(bass, "bass-step").text = step
        if alter:
            ET.SubElement(bass, "bass-alter").text = str(int(alter))
    for value, alter, dtype in _EXTRA_DEGREES.get(parsed.quality, []):
        degree = ET.SubElement(harmony, "degree")
        ET.SubElement(degree, "degree-value").text = str(value)
        ET.SubElement(degree, "degree-alter").text = str(alter)
        ET.SubElement(degree, "degree-type").text = dtype
    return True


# ---------------------------------------------------------------- 属性 / 方向


def _append_attributes(
    measure: ET.Element,
    *,
    ctx: _Context,
    include_time: bool,
    include_key: bool,
    tab: bool,
    time_signature: Sequence[int],
    first: bool = False,
) -> None:
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    if include_key:
        key = ET.SubElement(attributes, "key")
        ET.SubElement(key, "fifths").text = str(ctx.fifths)
        ET.SubElement(key, "mode").text = ctx.mode
    if include_time:
        time = ET.SubElement(attributes, "time")
        beats, beat_type = int(time_signature[0]), int(time_signature[1])
        ET.SubElement(time, "beats").text = str(beats)
        ET.SubElement(time, "beat-type").text = str(beat_type)
    if tab:
        clef = ET.SubElement(attributes, "clef")
        ET.SubElement(clef, "sign").text = "TAB"
        ET.SubElement(clef, "line").text = "5"
        # 调弦信息只需要在首小节声明一次；每小节重复 88 遍虽合法但文件冗余
        if not first:
            return
        details = ET.SubElement(attributes, "staff-details")
        ET.SubElement(details, "staff-lines").text = "6"
        # line 1 = 最高音弦（1 弦），因此从调弦数组尾部往前取
        for line in range(1, 7):
            midi = ctx.tuning_midis[6 - line]
            step, alter, octave = spell_midi(midi, ctx.fifths)
            tuning = ET.SubElement(details, "staff-tuning", {"line": str(line)})
            ET.SubElement(tuning, "tuning-step").text = step
            if alter:
                ET.SubElement(tuning, "tuning-alter").text = str(int(alter))
            ET.SubElement(tuning, "tuning-octave").text = str(int(octave))
        if ctx.capo > 0:
            ET.SubElement(details, "capo").text = str(int(ctx.capo))
    else:
        clef = ET.SubElement(attributes, "clef")
        ET.SubElement(clef, "sign").text = "G"
        ET.SubElement(clef, "line").text = "2"


def _append_tempo(
    measure: ET.Element,
    bpm: float,
    *,
    visible: bool = True,
    label_bpm: float | None = None,
) -> None:
    """写速度标记。

    ``visible=False`` 时只保留不可见的 ``<sound tempo>``（用于第二谱表），
    避免在 MuseScore 里两行谱各挂一个「♩=88」而显得重复。
    ``label_bpm`` 用于让谱面显示的速度与 ``<sound>`` 里的精确值不同
    （例如谱面统一标「♩=92」，而小节里保留 92.28 这类精确值给回放）。
    """
    if visible:
        direction = ET.SubElement(measure, "direction", {"placement": "above"})
        dtype = ET.SubElement(direction, "direction-type")
        metronome = ET.SubElement(dtype, "metronome")
        ET.SubElement(metronome, "beat-unit").text = "quarter"
        # 谱面显示用整数（「♩=88」而不是「♩=87.591」）
        shown = round(float(label_bpm if label_bpm is not None else bpm))
        ET.SubElement(metronome, "per-minute").text = f"{shown:d}"
        ET.SubElement(direction, "sound", {"tempo": f"{bpm:g}"})
        return
    direction = ET.SubElement(measure, "direction", {"placement": "above"})
    # XSD 要求 direction 里 <direction-type> 必须在 <sound> 之前，
    # 所以不可见的速度也用 metronome 承载，只是加 print-object="no" 不打印。
    dtype = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(dtype, "metronome", {"print-object": "no"})
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = f"{round(float(bpm)):d}"
    ET.SubElement(direction, "sound", {"tempo": f"{bpm:g}"})


def _append_rehearsal(measure: ET.Element, label: str) -> None:
    if not label:
        return
    direction = ET.SubElement(measure, "direction", {"placement": "above"})
    dtype = ET.SubElement(direction, "direction-type")
    ET.SubElement(dtype, "rehearsal").text = label


# ---------------------------------------------------------------- 声部内容


def _vocal_events(bar: dict) -> list[tuple[float, dict]]:
    """小节内的旋律事件（按拍排序）。"""
    events: list[tuple[float, dict]] = []
    for note in bar.get("vocal") or []:
        if not isinstance(note, dict):
            continue
        events.append((_safe_beats(note.get("beat")), note))
    events.sort(key=lambda item: item[0])
    return events


def _chord_events(bar: dict) -> list[tuple[float, dict]]:
    events: list[tuple[float, dict]] = []
    for chord in bar.get("chords") or []:
        if not isinstance(chord, dict):
            continue
        events.append((_safe_beats(chord.get("beat")), chord))
    events.sort(key=lambda item: item[0])
    return events


def _emit_vocal_measure(
    measure: ET.Element,
    bar: dict,
    ctx: _Context,
    warnings: list[str],
    counts: dict[str, int],
) -> None:
    """主旋律小节：和弦符号与音符按拍位置交织，空隙补休止。"""
    notes = _vocal_events(bar)
    chords = _chord_events(bar)
    bar_len = _bar_length(bar, ctx)
    cursor = 0.0
    index_note = index_chord = 0
    guard = 0
    while (index_note < len(notes) or index_chord < len(chords)) and guard < 10000:
        guard += 1
        next_note = notes[index_note][0] if index_note < len(notes) else None
        next_chord = chords[index_chord][0] if index_chord < len(chords) else None
        positions = [value for value in (next_note, next_chord) if value is not None]
        if not positions:
            break
        target = min(positions)
        if target > cursor + _EPS:
            _append_rest(measure, target - cursor)
            cursor = target
        if next_chord is not None and next_chord <= cursor + _EPS:
            if _append_harmony(measure, chords[index_chord][1], ctx):
                counts["harmonies"] += 1
            else:
                symbol = chords[index_chord][1].get("symbol")
                warnings.append(f"第 {bar.get('index')} 小节的和弦符号 {symbol!r} 无法解析，已跳过 <harmony>")
            index_chord += 1
            continue
        if next_note is not None and next_note <= cursor + _EPS:
            note = notes[index_note][1]
            dur = _safe_dur(note.get("dur"), 1.0)
            if cursor + dur > bar_len + 1e-6:
                clipped = max(bar_len - cursor, 0.25)
                warnings.append(
                    f"第 {bar.get('index')} 小节旋律超出小节长度（{dur:g} -> {clipped:g} 拍），已截断"
                )
                dur = clipped
            midi = note.get("midi")
            if note.get("rest") or midi is None:
                _append_rest(measure, dur)
            else:
                element = ET.SubElement(measure, "note")
                step, alter, octave = spell_midi(int(midi), ctx.fifths)
                _append_pitch(element, step, alter, octave)
                _append_duration(element, dur)
                key = (int(bar.get("index") or 0), index_note)
                tie = ctx.tie_map.get(key)
                if tie:
                    ET.SubElement(element, "tie", {"type": tie})
                ET.SubElement(element, "voice").text = "1"
                _append_note_type(element, dur)
                _append_notations(element, tie=tie)
                lyric = str(note.get("lyric") or "").strip()
                if lyric:
                    lyric_element = ET.SubElement(element, "lyric", {"number": "1"})
                    ET.SubElement(lyric_element, "syllabic").text = "single"
                    ET.SubElement(lyric_element, "text").text = lyric
                elif ctx.extend_map.get(key):
                    # 一字多音：后续音符不重复写词，用 <extend/> 表示延长
                    lyric_element = ET.SubElement(element, "lyric", {"number": "1"})
                    ET.SubElement(lyric_element, "extend")
                counts["notes"] += 1
            cursor += dur
            index_note += 1
            continue
        # 数据异常（位置落在已发出内容中间）：丢弃该事件并告警
        if next_note is not None and next_note < cursor - _EPS:
            warnings.append(f"第 {bar.get('index')} 小节有重叠的旋律事件（beat={next_note:g:.3f}），已跳过")
            index_note += 1
        elif next_chord is not None and next_chord < cursor - _EPS:
            warnings.append(f"第 {bar.get('index')} 小节有重叠的和弦事件（beat={next_chord:g:.3f}），已跳过")
            index_chord += 1
        else:  # 理论上不可达，防止死循环
            break
    if cursor < bar_len - _EPS:
        _append_rest(measure, bar_len - cursor)


def _voicing_for(chord: dict, ctx: _Context) -> dict | None:
    """取和弦对应的指法（先查 voicing_id，再退回符号查表）。"""
    voicing_id = chord.get("voicing_id")
    if voicing_id and voicing_id in ctx.voicings:
        return ctx.voicings[voicing_id]
    symbol = str(chord.get("symbol") or "").strip()
    for candidate in ctx.voicings.values():
        if str(candidate.get("symbol") or "").strip() == symbol:
            return candidate
    return None


def _append_tab_chord(
    measure: ET.Element,
    voicing: dict,
    dur: float,
    ctx: _Context,
    counts: dict[str, int],
) -> int:
    """把指法写成一串同时发声的 TAB 音（低音弦在前）。返回写入的音数。"""
    frets = voicing.get("frets") or []
    written = 0
    for index, fret in enumerate(frets[:6]):
        if fret is None or int(fret) < 0:
            continue
        string_number = 6 - index  # 数组索引 0 = 6 弦（最低音弦）
        midi = fret_to_midi(ctx.tuning_midis, string_number, int(fret))
        if midi is None:
            continue
        note = ET.SubElement(measure, "note")
        if written > 0:
            ET.SubElement(note, "chord")
        step, alter, octave = spell_midi(midi, ctx.fifths)
        _append_pitch(note, step, alter, octave)
        _append_duration(note, dur)
        ET.SubElement(note, "voice").text = "5"
        _append_note_type(note, dur)
        _append_notations(note, tab=(string_number, int(fret)))
        written += 1
    counts["tab_notes"] += written
    return written


def _emit_guitar_measure(
    measure: ET.Element,
    bar: dict,
    ctx: _Context,
    warnings: list[str],
    counts: dict[str, int],
) -> None:
    """TAB 小节：每个和弦以指法整块发声，空隙补休止。"""
    chords = _chord_events(bar)
    bar_len = _bar_length(bar, ctx)
    cursor = 0.0
    for index, (start, chord) in enumerate(chords):
        start = min(start, bar_len)
        if start > cursor + _EPS:
            _append_rest(measure, start - cursor, voice=5)
            cursor = start
        next_start = chords[index + 1][0] if index + 1 < len(chords) else bar_len
        dur = _safe_dur(chord.get("dur"), next_start - start)
        dur = min(dur, max(next_start - start, 0.0), max(bar_len - start, 0.0))
        if dur <= _EPS:
            continue
        voicing = _voicing_for(chord, ctx)
        if voicing is None:
            warnings.append(f"第 {bar.get('index')} 小节的和弦 {chord.get('symbol')!r} 没有指法，TAB 留空")
            _append_rest(measure, dur, voice=5)
            cursor = start + dur
            continue
        if _append_tab_chord(measure, voicing, dur, ctx, counts) == 0:
            warnings.append(f"第 {bar.get('index')} 小节的和弦 {chord.get('symbol')!r} 指法全闷音，TAB 留空")
            _append_rest(measure, dur, voice=5)
        cursor = start + dur
    if cursor < bar_len - _EPS:
        _append_rest(measure, bar_len - cursor, voice=5)


# ---------------------------------------------------------------- 组装


def _build_score(score: dict, warnings: list[str]) -> tuple[ET.Element, dict[str, int]]:
    ctx = _Context(score)
    bars = _bars(score)
    counts = {"notes": 0, "harmonies": 0, "tab_notes": 0}

    root = ET.Element("score-partwise", {"version": _MUSICXML_VERSION})
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = ctx.title
    ET.SubElement(root, "movement-title").text = ctx.title
    identification = ET.SubElement(root, "identification")
    if ctx.artist:
        ET.SubElement(identification, "creator", {"type": "composer"}).text = ctx.artist
    ET.SubElement(identification, "rights").text = "Generated by AI-To-Sheet-Music"
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "AI-To-Sheet-Music score-engine"
    ET.SubElement(encoding, "encoding-date").text = _dt.date.today().isoformat()

    part_list = ET.SubElement(root, "part-list")
    for part_id, name, midi_program in (
        ("P1", "Vocal 主旋律", 53),
        ("P2", "Guitar TAB", 25),
    ):
        score_part = ET.SubElement(part_list, "score-part", {"id": part_id})
        ET.SubElement(score_part, "part-name").text = name
        score_instrument = ET.SubElement(score_part, "score-instrument", {"id": f"{part_id}-I1"})
        ET.SubElement(score_instrument, "instrument-name").text = name
        midi_instrument = ET.SubElement(score_part, "midi-instrument", {"id": f"{part_id}-I1"})
        ET.SubElement(midi_instrument, "midi-channel").text = "1" if part_id == "P1" else "2"
        ET.SubElement(midi_instrument, "midi-program").text = str(midi_program)
        ET.SubElement(midi_instrument, "volume").text = "80"

    vocal_part = ET.SubElement(root, "part", {"id": "P1"})
    tab_part = ET.SubElement(root, "part", {"id": "P2"})

    previous_signature = None
    previous_tempo = None
    # 上一次「显示出来」的速度值：谱面只标有意义的速度变化，逐小节的
    # 细微抖动（83/85/86/88…）只写进不可见的 sound，否则整页都是 ♩= 标记
    shown_tempo = None
    system_breaks = _system_breaks(bars, score)
    for position, bar in enumerate(bars, start=1):
        signature = parse_time_signature(bar.get("time_signature") or ctx.time_signature, ctx.time_signature)
        attributes_needed = position == 1 or signature != previous_signature
        number = str(position)
        attributes = {"number": number}
        if bar.get("pickup") and position == 1:
            attributes["implicit"] = "yes"
        vocal_measure = ET.SubElement(vocal_part, "measure", attributes)
        tab_measure = ET.SubElement(tab_part, "measure", dict(attributes))

        try:
            bar_index = int(bar.get("index") or position)
        except (TypeError, ValueError):
            bar_index = position
        section = ctx.section_starts.get(bar_index)

        # 人看谱的可读性：每行固定 4 小节（默认）、每个段落开头强制新系统，
        # 且不留单小节孤行；段落标签用 <rehearsal>（MuseScore 会画成方框标记）。
        new_page = bool(bar.get("new_page"))
        new_system = position in system_breaks
        if position > 1:
            for measure in (vocal_measure, tab_measure):
                if new_page:
                    ET.SubElement(measure, "print", {"new-page": "yes"})
                elif new_system:
                    ET.SubElement(measure, "print", {"new-system": "yes"})

        _append_attributes(
            vocal_measure,
            ctx=ctx,
            include_time=attributes_needed,
            include_key=position == 1,
            tab=False,
            time_signature=signature,
            first=position == 1,
        )
        if ctx.include_tab:
            _append_attributes(
                tab_measure,
                ctx=ctx,
                include_time=attributes_needed,
                include_key=position == 1,
                tab=True,
                time_signature=signature,
                first=position == 1,
            )

        tempo = bar.get("tempo")
        try:
            tempo_value = float(tempo) if tempo else ctx.tempo
        except (TypeError, ValueError):
            tempo_value = ctx.tempo
        if tempo_value and (position == 1 or tempo_value != previous_tempo):
            # 只在速度变化 ≥15% 时显示标记（首小节总是显示）。
            # 节拍检测给出的逐小节速度（83/85/88/92/101）多是拟合噪声，
            # 全部标出来会让整页都是 ♩= 标记；精确值仍写进 <sound tempo> 供回放。
            show = (
                shown_tempo is None
                or abs(tempo_value - shown_tempo) / max(float(shown_tempo), 1e-6) >= 0.15
            )
            _append_tempo(
                vocal_measure,
                tempo_value,
                visible=show,
                label_bpm=ctx.tempo if shown_tempo is None and ctx.tempo else None,
            )
            if show:
                shown_tempo = tempo_value
            if ctx.include_tab:
                # 第二谱表只放不可见的 sound 事件，避免 MuseScore 里速度标记重复
                _append_tempo(tab_measure, tempo_value, visible=False)
        previous_tempo = tempo_value

        if section:
            label = str(section.get("label") or section.get("kind") or "")
            # 段落标记只放在第一谱表上方；两行谱各放一个会在 MuseScore 里重复显示
            _append_rehearsal(vocal_measure, label)

        _emit_vocal_measure(vocal_measure, bar, ctx, warnings, counts)
        if ctx.include_tab:
            _emit_guitar_measure(tab_measure, bar, ctx, warnings, counts)
        else:
            _append_rest(tab_measure, _bar_length(bar, ctx), voice=5)

        previous_signature = signature

    ET.indent(root, space="  ")
    return root, counts


def _serialize(root: ET.Element) -> bytes:
    body = ET.tostring(root, encoding="utf-8")
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"{_DOCTYPE}\n"
    ).encode("utf-8")
    return header + body + b"\n"


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
        raise ExportError(f"写入 MusicXML 失败（{out_path}）：{exc}") from exc


def _self_check(data: bytes, counts: dict[str, int], measure_count: int) -> None:
    """解析回来核对结构，防止写出「看起来像 XML 但打不开」的文件。"""
    try:
        parsed = ET.fromstring(data)
    except ET.ParseError as exc:  # pragma: no cover - 正常不应触发
        raise ExportError(f"MusicXML 自检失败：生成的 XML 无法解析（{exc}）") from exc
    if parsed.tag != "score-partwise":
        raise ExportError(f"MusicXML 自检失败：根节点应为 score-partwise，实际为 {parsed.tag}")
    parts = parsed.findall("part")
    if len(parts) < 1:
        raise ExportError("MusicXML 自检失败：没有 part 节点")
    measures = sum(len(part.findall("measure")) for part in parts)
    if measures < measure_count:
        raise ExportError(f"MusicXML 自检失败：小节数应为 {measure_count}，实际写出 {measures}")
    notes = sum(len(part.findall(".//note")) for part in parts)
    if notes < counts["notes"]:
        raise ExportError(f"MusicXML 自检失败：音符数应为 {counts['notes']}，实际写出 {notes}")
    harmonies = sum(len(part.findall(".//harmony")) for part in parts)
    if harmonies < counts["harmonies"]:
        raise ExportError(f"MusicXML 自检失败：和弦符号数应为 {counts['harmonies']}，实际写出 {harmonies}")


def export_musicxml(score: dict, out_path: Path) -> dict:
    """把 ``aitsm.score/v1`` 乐谱导出为 MusicXML 4.0 文件。

    参数
    ----
    score:
        ``aitsm.score/v1`` 谱面 dict（必须含 ``bars``）。
    out_path:
        输出文件路径（.musicxml / .xml）。

    返回
    ----
    dict: ``{"format": "musicxml", "path", "size_bytes", "measures", "parts",
    "notes", "harmonies", "tab_notes", "capo", "include_tab", "warnings"}``。

    异常
    ----
    :class:`ExportError`: 输入不是合法谱面、小节为空、写盘失败或结构自检不通过。
    """
    if not isinstance(score, dict):
        raise ExportError(f"MusicXML 导出需要 score dict，收到 {type(score).__name__}")
    if score.get("schema") != "aitsm.score/v1":
        raise ExportError(f"MusicXML 导出需要 schema=aitsm.score/v1，收到 {score.get('schema')!r}")
    bars = _bars(score)
    if not bars:
        raise ExportError("乐谱没有小节（bars 为空），无法导出 MusicXML")

    warnings: list[str] = []
    root, counts = _build_score(score, warnings)
    data = _serialize(root)
    _self_check(data, counts, len(bars))
    _write_atomic(Path(out_path), data)

    meta = score.get("meta") or {}
    bars = _bars(score)
    return {
        "format": "musicxml",
        "version": _MUSICXML_VERSION,
        "path": str(Path(out_path)),
        "size_bytes": len(data),
        "measures": len(bars),
        "systems": len(_system_breaks(bars, score)) + 1,
        "parts": 2,
        "notes": counts["notes"],
        "harmonies": counts["harmonies"],
        "tab_notes": counts["tab_notes"],
        "capo": int(meta.get("capo") or 0),
        "include_tab": bool(_Context(score).include_tab),
        "warnings": warnings,
    }

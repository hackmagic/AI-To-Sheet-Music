"""层 1B：伴奏联合转录适配器（SheetSage2 优先，音频管线显式降级）。

本模块是 :class:`aitsm.contracts.AccompanimentResult` 的唯一生产线，职责只有两件：

1. **优先走研究模型**：环境变量 ``AITSM_SHEETSAGE2_DIR`` 指向本地权重目录、且
   ``transformers`` 可导入时，用 SheetSage2 做联合转录，再把它返回的
   ``abc`` / ``midi`` / ``events`` 规范成契约类型。
2. **不可用时如实降级**：复用 v1 已经调好的节拍/调性/和弦/结构四个阶段
   （``aitsm.pipeline.{beats,keydet,chords,structure}``），绝不重写算法，也绝不
   联网下载权重。降级原因写进 ``warnings``，``engine`` 记 ``beats+chroma``，
   ``requested`` 保留 ``SheetSage2``，让下游一眼看出「请求了什么、实际用了什么」。

三条硬约束：

* 只产出 :mod:`aitsm.contracts` 里的类型，内部中间结构可以随便定；
* 重模块（``transformers`` / ``torch`` / ``librosa`` / ``pretty_midi`` /
  ``soundfile``）一律**函数内惰性导入**，``import aitsm.transcribe.accompaniment``
  不能拉起任何神经网络或音频栈；
* 任何一步失败都要显式降级或抛错，**不允许**返回看起来成功、实际没有内容的
  结果（例如空和弦却标 ``ok``）。

对外函数：

* :func:`is_available` —— 研究模型链路是否真的可用（只看本地，不联网）。
* :func:`transcribe_with_sheetsage` —— 只试研究模型；不可用/失败返回 ``None``。
* :func:`transcribe_accompaniment` —— 主入口，返回 ``AccompanimentResult``。
* :func:`parse_events` —— 把任意形状的 events 规范成 ``AccompanimentResult``。
* :func:`normalize_events` —— 上面那个函数的中间层，返回可复用的规范化 dict。
* :func:`abc_from_result` —— 生成人类可读 ABC（调试用，允许空旋律）。
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from aitsm.contracts import (
    AccompanimentResult,
    BeatGrid,
    ChordEvent,
    KeySignature,
    MelodyNote,
    Section,
)

__all__ = [
    "ENGINE_FALLBACK",
    "ENGINE_SHEETSAGE2",
    "LAYER",
    "REQUESTED_ENGINE",
    "SHEETSAGE2_DIR_ENV",
    "abc_from_result",
    "is_available",
    "normalize_events",
    "parse_events",
    "sheetsage2_dir",
    "transcribe_accompaniment",
    "transcribe_with_sheetsage",
]

#: provenance 里的层名（与 contracts.LAYERS 一致）
LAYER = "accompaniment_transcription"
#: 研究模型真正跑通时的 engine 名
ENGINE_SHEETSAGE2 = "sheetsage2"
#: 降级到 v1 音频管线时的 engine 名
ENGINE_FALLBACK = "beats+chroma"
#: 架构首选的模型名（写进 requested，便于质量面板识别降级）
REQUESTED_ENGINE = "SheetSage2"
#: 本地权重目录的环境变量（绝不联网下载）
SHEETSAGE2_DIR_ENV = "AITSM_SHEETSAGE2_DIR"

#: 进度回调：兼容 ``progress(stage, percent, message)`` 与 ``progress(dict)``
ProgressCallback = Callable[..., Any]

_EPS = 1e-6
#: ABC 输出固定用八分音符作最小单位（L:1/8），够表达常见流行歌节奏
_ABC_UNITS_PER_BEAT = 2
#: 没有 end 的和弦事件，兜底补多长（秒）
_DEFAULT_CHORD_SECONDS = 1.0
#: ABC 最多渲染多少小节，防止异常事件把调试输出撑爆
_MAX_ABC_BARS = 512

_CHORD_LABELS = {"chord", "chords", "harmony", "harmonies"}
_SECTION_LABELS = {"section", "sections", "segment", "structure"}
_NOTE_LABELS = {"note", "notes", "melody", "melody_note"}
_BEAT_LABELS = {"beat", "beats", "tempo", "pulse"}
_DOWNBEAT_LABELS = {"downbeat", "downbeats", "bar", "bars", "bar_start"}

_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _f(value: Any, default: float = 0.0) -> float:
    """宽容地转 float（无穷/NaN 视为缺省）。"""
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


def _raw_text(value: Any) -> str:
    """保留换行的文本字段（ABC 对尾部换行敏感，不能随手 strip）。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _s(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _emit(progress: ProgressCallback | None, stage: str, percent: float, message: str) -> None:
    """调用进度回调；两种签名都兼容，回调自身报错不能影响主流程。"""
    if progress is None:
        return
    try:
        progress(stage, float(percent), message)
        return
    except TypeError:
        pass
    except Exception:  # 回调自己炸了不该拖垮转录
        return
    try:
        progress({"stage": stage, "percent": float(percent), "message": message})
    except Exception:
        return


def _dedupe(items: Iterable[Any]) -> list[str]:
    """去重并保持首次出现顺序（告警会从多个来源汇聚，重复没意义）。"""
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """按优先级取第一个「非 None」的值；空容器留作后备，继续往后找。"""
    fallback: Any = None
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict, set, str)) and not value:
            if fallback is None:
                fallback = value
            continue
        return value
    return fallback


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value]
    return [value]


def _times(values: Any) -> list[float]:
    """把 [1.0, {"time": 2.0}, ...] 统一成升序秒数列表。"""
    output: list[float] = []
    for item in _as_list(values):
        if isinstance(item, Mapping):
            raw = _first_present(item, ("time", "start", "onset", "beat", "t"))
        else:
            raw = item
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(round(max(0.0, value), 6))
    return sorted(output)


def _median_interval(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    intervals = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if not intervals:
        return 0.0
    return intervals[len(intervals) // 2]


def _time_signature(value: Any) -> str:
    """``"3/4"`` / ``{"time_signature": "6/8"}`` → 合法拍号字符串。"""
    if isinstance(value, Mapping):
        value = _first_present(value, ("time_signature", "meter", "signature", "timesig"))
    text = _s(value)
    match = re.fullmatch(r"(\d{1,2})\s*/\s*(\d{1,2})", text)
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if numerator > 0 and denominator > 0:
            return f"{numerator}/{denominator}"
    return "4/4"


def _beats_per_bar(signature: str) -> int:
    try:
        return max(1, int(str(signature).split("/")[0]))
    except (ValueError, IndexError):
        return 4


def _note_name(pitch_class: int, *, prefer_flat: bool = False) -> str:
    """音级 → 音名；复用 keydet 的表，避免两处拼写规则漂移。"""
    from aitsm.pipeline.keydet import note_name

    return note_name(int(pitch_class) % 12, prefer_flat=prefer_flat)


def _key_fifths(tonic: str, mode: str) -> int:
    """按主音名 + 调式算调号数量（正数升号、负数降号）。"""
    match = re.fullmatch(r"([A-Ga-g])([#b]{0,2})", str(tonic).strip())
    if not match:
        return 0
    pitch_class = _PITCH_CLASS[match.group(1).upper()]
    for accidental in match.group(2):
        pitch_class += 1 if accidental == "#" else -1
    from aitsm.pipeline.keydet import key_signature_count

    return int(key_signature_count(pitch_class % 12, mode))


# --------------------------------------------------------------------------- #
# SheetSage2 可用性（只看本地，绝不联网）
# --------------------------------------------------------------------------- #


def _import_transformers() -> Any | None:
    """惰性导入 transformers；缺失或损坏都返回 None（不抛异常）。"""
    try:
        return importlib.import_module("transformers")
    except Exception:
        return None


def sheetsage2_dir() -> Path | None:
    """``AITSM_SHEETSAGE2_DIR`` 指向的本地权重目录；无效则 None。"""
    raw = os.environ.get(SHEETSAGE2_DIR_ENV, "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
    except (OSError, ValueError):
        return None
    return path if path.is_dir() else None


def is_available() -> bool:
    """SheetSage2 链路是否可用：``transformers`` 可导入且本地权重目录存在。

    只做本地探测，不触发任何下载。
    """
    return _import_transformers() is not None and sheetsage2_dir() is not None


def _unavailable_reason() -> str:
    """一句话说清为什么没用上 SheetSage2（写进 warnings）。"""
    problems: list[str] = []
    if _import_transformers() is None:
        problems.append("未安装 transformers")
    if sheetsage2_dir() is None:
        problems.append(f"{SHEETSAGE2_DIR_ENV} 未指向有效权重目录")
    detail = "、".join(problems) if problems else "本地条件不满足"
    return f"SheetSage2 未启用（{detail}），改用音频管线 {ENGINE_FALLBACK} 完成伴奏转录。"


def _resolve_device(device: str) -> str:
    """``auto`` → 有 CUDA 用 CUDA，否则 CPU（本环境通常是 CPU-only torch）。"""
    text = _s(device, "auto").lower()
    if text and text != "auto":
        return text
    try:
        import torch

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:  # torch 缺失或没有 CUDA 都退回 CPU
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# 事件规范化
# --------------------------------------------------------------------------- #


def _looks_like_chord(item: Mapping[str, Any]) -> bool:
    if any(key in item for key in ("symbol", "chord", "harmony", "root_pc")):
        return True
    label = _first_present(item, ("label", "name"))
    return bool(label) and "start" in item and not any(key in item for key in ("midi", "pitch", "note"))


def _looks_like_note(item: Mapping[str, Any]) -> bool:
    return any(key in item for key in ("midi", "pitch", "note")) and any(
        key in item for key in ("start", "time", "onset")
    )


def _pitch_to_midi(value: Any) -> int | None:
    """``60`` / ``"C4"`` / ``"F#3"`` → MIDI 音高；无法识别返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _f(value, -1.0)
        if 0.0 <= number <= 127.0:
            return int(round(number))
        return None
    text = _s(value)
    match = re.fullmatch(r"([A-Ga-g])([#b]{0,2})(-?\d{1,2})?", text)
    if not match:
        return None
    pitch_class = _PITCH_CLASS[match.group(1).upper()]
    for accidental in match.group(2):
        pitch_class += 1 if accidental == "#" else -1
    octave = int(match.group(3)) if match.group(3) is not None else 4
    midi = (octave + 1) * 12 + pitch_class
    return max(0, min(127, midi))


def _key_from(value: Any) -> KeySignature:
    """键性字段 → :class:`KeySignature`；接受 dict 与 ``"C major"`` / ``"Am"`` 两种写法。"""
    if isinstance(value, Mapping):
        mode = _s(value.get("mode"), "major").lower()
        mode = "minor" if mode in ("minor", "min", "m", "小的", "小调") else "major"
        tonic = _s(value.get("tonic"))
        if not tonic and value.get("tonic_pc") is not None:
            tonic = _note_name(_i(value.get("tonic_pc")))
        tonic = tonic or "C"
        confidence = _f(value.get("confidence"))
        fifths = value.get("fifths", value.get("num_sharps_flats"))
        fifths = _i(fifths, _key_fifths(tonic, mode)) if fifths is not None else _key_fifths(tonic, mode)
        return KeySignature(tonic=tonic, mode=mode, confidence=confidence, fifths=fifths)
    text = _s(value)
    if not text:
        return KeySignature(tonic="C")
    match = re.fullmatch(r"([A-Ga-g])([#b]?)\s*(.*)", text)
    if not match:
        return KeySignature(tonic="C")
    tonic = match.group(1).upper() + match.group(2)
    tail = match.group(3).strip().lower()
    mode = "minor" if tail.startswith("min") or tail in ("m", "minor") else "major"
    return KeySignature(tonic=tonic, mode=mode, fifths=_key_fifths(tonic, mode))


def _beats_from(
    value: Any,
    *,
    downbeats: Any = None,
    tempo: Any = None,
    meter: Any = None,
) -> BeatGrid:
    """节拍字段 → :class:`BeatGrid`；也接受 pipeline ``analyze_beats`` 原样 dict。"""
    confidence = 0.0
    source = "events"
    beats_per_bar = 0
    pickup_beats = 0
    first_downbeat = 0.0
    if isinstance(value, Mapping):
        mapping = value
        times = _times(_first_present(mapping, ("beats", "beat_times", "times", "pulse")))
        if not _times(downbeats):
            downbeats = _first_present(mapping, ("downbeats", "bar_starts", "downbeat"))
        if tempo is None:
            tempo = _first_present(mapping, ("bpm_display", "bpm", "tempo"))
        if meter is None:
            meter = _first_present(mapping, ("time_signature", "meter"))
        confidence = _f(mapping.get("confidence"))
        source = _s(mapping.get("source"), "pipeline") or "pipeline"
        beats_per_bar = _i(mapping.get("beats_per_bar"))
        pickup_beats = max(0, _i(mapping.get("pickup_beats")))
        first_downbeat = _f(mapping.get("first_downbeat_sec"))
    else:
        times = _times(value)

    downbeat_times = _times(downbeats)
    signature = _time_signature(meter)
    bpb = beats_per_bar if beats_per_bar > 0 else _beats_per_bar(signature)
    bpm = _f(tempo)
    if bpm <= 0:
        interval = _median_interval(times)
        bpm = 60.0 / interval if interval > 0 else 120.0
    return BeatGrid(
        bpm=bpm,
        beats=times,
        downbeats=downbeat_times,
        beats_per_bar=bpb,
        time_signature=signature,
        confidence=confidence,
        pickup_beats=pickup_beats,
        first_downbeat_sec=first_downbeat,
        source=source,
    )


def _chords_from(values: Any) -> list[ChordEvent]:
    """和弦事件 → :class:`ChordEvent` 列表（缺 end 时先留空，由 :func:`_close_chords` 补）。"""
    chords: list[ChordEvent] = []
    for item in _as_list(values):
        if isinstance(item, Mapping):
            start_raw = _first_present(item, ("start", "time", "onset", "at"))
            symbol = _s(_first_present(item, ("symbol", "chord", "label", "name", "harmony")))
            if start_raw is None or not symbol:
                continue
            start = _f(start_raw)
            end_raw = _first_present(item, ("end", "stop"))
            duration = _f(_first_present(item, ("duration", "dur")))
            if end_raw is not None:
                end = _f(end_raw, start)
            elif duration > 0:
                end = start + duration
            else:
                end = start
            chords.append(
                ChordEvent(
                    start=max(0.0, start),
                    end=max(0.0, end),
                    symbol=symbol,
                    confidence=_f(_first_present(item, ("confidence", "score", "prob"))),
                    source=_s(item.get("source"), "events") or "events",
                )
            )
        elif isinstance(item, str) and item.strip():
            # 只有和弦符号、没有时间：无法定位，按「每小节一个」推测风险太高，直接忽略。
            continue
    chords.sort(key=lambda chord: (chord.start, chord.end, chord.symbol))
    return chords


def _close_chords(chords: list[ChordEvent], span: float) -> list[ChordEvent]:
    """补齐/收敛和弦区间：缺 end 补到下一个和弦起点，重叠的截断到下一个起点。"""
    closed: list[ChordEvent] = []
    for index, chord in enumerate(chords):
        next_start = chords[index + 1].start if index + 1 < len(chords) else None
        if chord.end <= chord.start + _EPS:
            if next_start is not None and next_start > chord.start + _EPS:
                chord.end = next_start
            else:
                chord.end = max(chord.start + _DEFAULT_CHORD_SECONDS, span, chord.start + _EPS)
        if next_start is not None and chord.end > next_start + _EPS:
            chord.end = max(chord.start, next_start)
        if chord.end <= chord.start + _EPS:
            # 零宽度事件没有信息量，去掉
            continue
        closed.append(chord)
    return closed


def _sections_from(values: Any) -> list[Section]:
    sections: list[Section] = []
    for item in _as_list(values):
        if not isinstance(item, Mapping):
            continue
        start_raw = _first_present(item, ("start", "time", "onset"))
        if start_raw is None:
            continue
        start = _f(start_raw)
        end_raw = _first_present(item, ("end", "stop"))
        duration = _f(_first_present(item, ("duration", "dur")))
        end = _f(end_raw, start + duration) if end_raw is not None else start + duration
        sections.append(
            Section(
                start=max(0.0, start),
                end=max(0.0, end),
                kind=_s(item.get("kind"), "other") or "other",
                label=_s(item.get("label")),
                energy=_f(item.get("energy"), 0.5),
                repeat_of=_s(item.get("repeat_of")) or None,
                confidence=_f(_first_present(item, ("confidence", "score"))),
            )
        )
    sections.sort(key=lambda section: (section.start, section.end))
    return sections


def _close_sections(sections: list[Section], span: float) -> list[Section]:
    closed: list[Section] = []
    for index, section in enumerate(sections):
        next_start = sections[index + 1].start if index + 1 < len(sections) else None
        if section.end <= section.start + _EPS:
            section.end = (
                next_start if next_start is not None and next_start > section.start + _EPS else max(span, section.start + _DEFAULT_CHORD_SECONDS)
            )
        if next_start is not None and section.end > next_start + _EPS:
            section.end = max(section.start, next_start)
        if section.end <= section.start + _EPS:
            continue
        closed.append(section)
    return closed


def _notes_from(values: Any) -> list[MelodyNote]:
    notes: list[MelodyNote] = []
    for item in _as_list(values):
        if not isinstance(item, Mapping):
            continue
        midi = _pitch_to_midi(
            _first_present(item, ("midi", "pitch", "note", "midi_pitch"))
        )
        if midi is None:
            continue
        start_raw = _first_present(item, ("start", "time", "onset"))
        if start_raw is None:
            continue
        start = max(0.0, _f(start_raw))
        end_raw = _first_present(item, ("end", "stop"))
        duration = _f(_first_present(item, ("duration", "dur")))
        end = _f(end_raw, start + duration) if end_raw is not None else start + duration
        notes.append(
            MelodyNote(
                start=start,
                end=max(start, end),
                midi=midi,
                confidence=_f(_first_present(item, ("confidence", "score", "prob"))),
                lyric=_s(item.get("lyric")),
                source=_s(item.get("source"), "events") or "events",
            )
        )
    notes.sort(key=lambda note: (note.start, note.end, note.midi))
    return notes


def _span(beats: BeatGrid, chords: list[ChordEvent], sections: list[Section], notes: list[MelodyNote]) -> float:
    """整段时间轴长度：取所有事件结束点与节拍外推的最大值。"""
    ends: list[float] = []
    if beats.beats:
        ends.append(beats.beats[-1] + beats.beat_period)
    ends.extend(chord.end for chord in chords)
    ends.extend(section.end for section in sections)
    ends.extend(note.end for note in notes)
    return max([value for value in ends if math.isfinite(value)] + [0.0])


def _flatten_events(
    beats: BeatGrid,
    chords: list[ChordEvent],
    sections: list[Section],
    notes: list[MelodyNote],
    artifacts: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """所有元素压平成 ``AccompanimentResult.events``（可直接 JSON 化）。"""
    events: list[dict[str, Any]] = []
    for value in beats.downbeats:
        events.append({"type": "downbeat", "time": round(value, 6)})
    for value in beats.beats:
        events.append({"type": "beat", "time": round(value, 6)})
    for chord in chords:
        events.append({"type": "chord", **chord.to_dict()})
    for section in sections:
        events.append({"type": "section", **section.to_dict()})
    for note in notes:
        events.append({"type": "note", **note.to_dict()})
    for artifact in artifacts:
        events.append({"type": "artifact", **dict(artifact)})
    return events


def normalize_events(events: Any) -> dict[str, Any]:
    """把「联合转录输出」规范成中间结构（供 :func:`parse_events` 与降级路径复用）。

    接受两种形状：

    * ``dict``：带 ``beats`` / ``downbeats`` / ``chords`` / ``sections`` /
      ``notes`` / ``key`` / ``tempo`` / ``time_signature`` / ``abc`` / ``midi`` /
      ``events`` 等键（也兼容 ``beat_times`` / ``harmonies`` / ``segments`` /
      ``melody`` 这些别名，以及 SheetSage2 的嵌套返回）；
    * ``list``：混排的事件条目，条目用 ``type`` / ``kind`` / ``event`` 标注类型，
      没有标注时按字段猜（有 ``symbol`` 是和弦、有 ``midi`` 是音符、只有数字是拍点）。

    返回 dict：
    ``{"beats": BeatGrid, "key": KeySignature, "chords": [ChordEvent],
    "sections": [Section], "melody": [MelodyNote], "abc": str, "midi": str,
    "events": [dict], "warnings": [str]}``。
    """
    if events is None:
        data: dict[str, Any] = {}
        items: list[Any] = []
    elif isinstance(events, Mapping):
        data = dict(events)
        nested = _first_present(data, ("result", "output", "data", "prediction"))
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: value for key, value in data.items() if key not in merged})
            data = merged
        items = list(_as_list(_first_present(data, ("events", "items", "tracks", "annotations"))))
    elif isinstance(events, (list, tuple, set)):
        data = {}
        items = [item for item in events]
    else:
        raise TypeError(f"events 只接受 dict 或 list，收到 {type(events).__name__}")

    beat_values: list[Any] = list(_as_list(_first_present(data, ("beats", "beat_times", "beat"))))
    downbeat_values: list[Any] = list(_as_list(_first_present(data, ("downbeats", "bar_starts", "downbeat"))))
    chord_values: list[Any] = list(_as_list(_first_present(data, ("chords", "chord", "harmonies", "chord_events"))))
    section_values: list[Any] = list(_as_list(_first_present(data, ("sections", "segments", "structure", "section_events"))))
    note_values: list[Any] = list(_as_list(_first_present(data, ("notes", "melody", "melody_skeleton", "melody_notes"))))
    artifacts: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, Mapping):
            label = ""
            for key in ("type", "kind", "event", "category"):
                if item.get(key):
                    label = _s(item.get(key)).lower()
                    break
            if label in _CHORD_LABELS or (not label and _looks_like_chord(item)):
                chord_values.append(item)
            elif label in _SECTION_LABELS:
                section_values.append(item)
            elif label in _NOTE_LABELS or (not label and _looks_like_note(item)):
                note_values.append(item)
            elif label in _DOWNBEAT_LABELS:
                downbeat_values.append(item)
            elif label in _BEAT_LABELS:
                beat_values.append(item)
            elif label in ("abc", "text"):
                data.setdefault("abc", item.get("text") or item.get("abc") or "")
            elif label in ("midi", "artifact", "file"):
                artifacts.append({key: value for key, value in item.items() if key not in ("type", "kind", "event")})
            elif _looks_like_chord(item):
                chord_values.append(item)
            elif _looks_like_note(item):
                note_values.append(item)
            elif isinstance(item.get("beats"), (list, tuple)):
                beat_values.append(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            beat_values.append(item)

    # 拍点字段本身也可能是 pipeline 的整块 dict（含 bpm/downbeats/…）
    beats_source: Any = beat_values
    if len(beat_values) == 1 and isinstance(beat_values[0], Mapping) and "beats" in beat_values[0]:
        beats_source = beat_values[0]
    beats = _beats_from(
        beats_source,
        downbeats=downbeat_values,
        tempo=_first_present(data, ("bpm", "tempo", "tempo_bpm")),
        meter=_first_present(data, ("time_signature", "meter", "timesig")),
    )
    key = _key_from(_first_present(data, ("key", "tonality", "key_signature")))
    chords = _chords_from(chord_values)
    sections = _sections_from(section_values)
    notes = _notes_from(note_values)
    span = _span(beats, chords, sections, notes)
    chords = _close_chords(chords, span)
    sections = _close_sections(sections, span)
    if sections:
        sections = _merge_section_tail(sections, span)

    abc_text = _raw_text(_first_present(data, ("abc", "abc_text", "abc_notation")))
    midi_path = _s(_first_present(data, ("midi", "midi_path", "midi_file")))
    warnings: list[str] = []
    return {
        "beats": beats,
        "key": key,
        "chords": chords,
        "sections": sections,
        "melody": notes,
        "abc": abc_text,
        "midi": midi_path,
        "events": _flatten_events(beats, chords, sections, notes, artifacts),
        "warnings": warnings,
    }


def _merge_section_tail(sections: list[Section], span: float) -> list[Section]:
    """最后一段如果没盖到结尾（events 里常见），把它补到 span。"""
    last = sections[-1]
    if span > last.end + _EPS:
        last.end = max(last.end, span)
    return sections


def _result_from_payload(
    payload: Mapping[str, Any],
    *,
    engine: str,
    requested: str,
    warnings: Iterable[Any] = (),
) -> AccompanimentResult:
    """中间结构 → :class:`AccompanimentResult`（保证 abc 一定非空）。"""
    result = AccompanimentResult(
        beats=payload["beats"],
        key=payload["key"],
        chords=list(payload["chords"]),
        sections=list(payload["sections"]),
        melody_skeleton=list(payload["melody"]),
        abc=_raw_text(payload.get("abc")),
        events=list(payload["events"]),
        engine=engine,
        requested=requested,
        warnings=_dedupe(list(warnings)),
    )
    if not result.abc.strip():
        result.abc = abc_from_result(result)
    return result


def parse_events(events: Any) -> AccompanimentResult:
    """把任意形状的联合转录输出规范成 :class:`AccompanimentResult`。

    与 :func:`normalize_events` 的区别：这个函数直接返回契约类型，方便调用方
    复用它处理缓存文件、测试 fixture，或 SheetSage2 的原始返回。
    """
    payload = normalize_events(events)
    return _result_from_payload(
        payload,
        engine="events",
        requested="events",
        warnings=payload.get("warnings") or (),
    )


# --------------------------------------------------------------------------- #
# ABC 生成（人类可读的调试输出）
# --------------------------------------------------------------------------- #


def _abc_pitch(midi: int, *, prefer_flat: bool) -> str:
    """MIDI → ABC 音名（``C4`` → ``C``、``C5`` → ``c``、``C3`` → ``C,``）。"""
    value = max(0, min(127, _i(midi, 60)))
    name = _note_name(value % 12, prefer_flat=prefer_flat)
    letter = name[0]
    accidental = ""
    if len(name) > 1 and name[1] in "#b":
        accidental = "^" if name[1] == "#" else "_"
    octave = value // 12 - 1
    if octave >= 5:
        body = letter.lower()
        octave_marks = "'" * (octave - 5)
    else:
        body = letter.upper()
        octave_marks = "," * max(0, 4 - octave)
    return f"{accidental}{body}{octave_marks}"


def _abc_units(seconds: float, unit: float) -> int:
    """秒数 → ABC 时值单位（至少 1 个单位，避免零时值）。"""
    if unit <= _EPS:
        return 1
    return max(1, int(round(max(0.0, seconds) / unit)))


def _multiplier(units: int) -> str:
    return "" if units <= 1 else str(int(units))


def _rest(units: int) -> str:
    return "z" + _multiplier(units)


def _chord_label_at(chords: list[ChordEvent], seconds: float) -> str:
    """该时刻「正在响」的和弦符号（沿用最近一次换和弦的结果）。"""
    label = ""
    for chord in chords:
        if chord.start <= seconds + _EPS:
            label = chord.symbol
        else:
            break
    return label


def _result_span(result: AccompanimentResult) -> float:
    return _span(result.beats, result.chords, result.sections, result.melody_skeleton)


def _abc_bar_starts(beats: BeatGrid, span: float) -> tuple[list[float], float]:
    """按小节线切分时间轴，返回 (小节起点, 小节长度)；无证据时按 4/4 等距铺满。"""
    bar_seconds = max(_EPS, beats.bar_length_seconds())
    starts: list[float] = []
    downbeats = sorted(value for value in beats.downbeats if value >= -_EPS)
    if downbeats:
        anchor = downbeats[0]
        if anchor > bar_seconds * 0.25:
            starts.append(0.0)
        cursor = anchor
        while cursor < span - bar_seconds * 0.35 and len(starts) < _MAX_ABC_BARS:
            starts.append(round(cursor, 6))
            cursor += bar_seconds
    elif span > 0:
        cursor = 0.0
        while cursor < span - bar_seconds * 0.35 and len(starts) < _MAX_ABC_BARS:
            starts.append(round(cursor, 6))
            cursor += bar_seconds
    if not starts and span > 0:
        starts = [0.0]
    return starts, bar_seconds


def _render_abc_bar(
    bar_start: float,
    bar_end: float,
    chords: list[ChordEvent],
    notes: list[MelodyNote],
    *,
    unit: float,
    prefer_flat: bool,
) -> list[str]:
    """渲染一个小节：和弦标记挂在其后的音符/休止符上，保证 ABC 合法。"""
    tokens: list[str] = []
    cursor = bar_start
    last_label = ""
    index = 0
    chord_onsets = sorted(
        {max(chord.start, bar_start) for chord in chords if chord.start < bar_end and chord.end > bar_start}
    )
    guard = 0
    while cursor < bar_end - unit * 0.25 and guard < 512:
        guard += 1
        label = _chord_label_at(chords, cursor)
        if label and label != last_label:
            tokens.append(f'"{label}"')
        last_label = label
        note = notes[index] if index < len(notes) else None
        if note is not None and note.start <= cursor + unit * 0.5:
            start = max(cursor, note.start)
            gap = start - cursor
            if gap > unit * 0.5:
                units = _abc_units(gap, unit)
                tokens.append(_rest(units))
                cursor += units * unit
                continue
            end = min(max(note.end, start + unit * 0.5), bar_end)
            units = _abc_units(end - start, unit)
            tokens.append(_abc_pitch(note.midi, prefer_flat=prefer_flat) + _multiplier(units))
            cursor = start + units * unit
            index += 1
            continue
        targets = [bar_end]
        if note is not None:
            targets.append(max(cursor, note.start))
        targets.extend(onset for onset in chord_onsets if onset > cursor + unit * 0.25)
        gap_end = min(targets)
        if gap_end <= cursor + unit * 0.25:
            gap_end = min(bar_end, cursor + unit)
        units = _abc_units(gap_end - cursor, unit)
        tokens.append(_rest(units))
        cursor += units * unit
    if not tokens:
        tokens.append(_rest(_abc_units(bar_end - bar_start, unit)))
    return tokens


def abc_from_result(result: AccompanimentResult) -> str:
    """把结果渲染成人类可读的 ABC（调试输出，不是交付谱面）。

    * 头部固定含 ``X:`` / ``T:`` / ``M:`` / ``L:`` / ``Q:`` / ``K:``；
    * 和弦以 ``"C"`` 形式的注解挂在音符或休止符前，没有旋律时表现为「整小节休止
      + 和弦标记」，仍然能读出和声进行；
    * 旋律为空是允许的（伴奏层本来就没有可靠旋律），但头一定存在。
    """
    beats = result.beats
    key = result.key
    period = beats.beat_period if beats.beat_period > 0 else 0.5
    bpb = max(1, int(beats.beats_per_bar))
    bpm = int(round(beats.bpm)) if beats.bpm > 0 else 120
    signature = _time_signature(beats.time_signature) if beats.time_signature else f"{bpb}/4"
    prefer_flat = int(key.fifths) < 0
    span = _result_span(result)

    lines: list[str] = [
        "X:1",
        f"T:Accompaniment skeleton ({result.engine or 'none'})",
        "C:AI-To-Sheet-Music",
        f"M:{signature}",
        "L:1/8",
        f"Q:1/4={max(1, bpm)}",
        f"K:{key.tonic}{'' if key.mode == 'major' else 'm'}",
    ]
    if span <= _EPS:
        return "\n".join(lines) + "\n"

    bar_starts, bar_seconds = _abc_bar_starts(beats, span)
    unit = period / _ABC_UNITS_PER_BEAT
    chords = sorted(result.chords, key=lambda chord: chord.start)
    notes = sorted(result.melody_skeleton, key=lambda note: (note.start, note.end))
    line: list[str] = []
    for index, bar_start in enumerate(bar_starts):
        bar_end = bar_starts[index + 1] if index + 1 < len(bar_starts) else bar_start + bar_seconds
        local_chords = [
            chord for chord in chords if chord.start < bar_end and chord.end > bar_start - _EPS
        ]
        local_notes = [
            note for note in notes if bar_start - _EPS <= note.start < bar_end
        ]
        tokens = _render_abc_bar(
            bar_start,
            bar_end,
            local_chords,
            local_notes,
            unit=unit,
            prefer_flat=prefer_flat,
        )
        line.append(" ".join(tokens) + " |")
        if len(line) >= 4:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 降级路径：复用 v1 音频管线
# --------------------------------------------------------------------------- #


def _pipeline_beats(source: str) -> dict[str, Any]:
    from aitsm.pipeline import beats as beats_module

    return beats_module.analyze_beats(source)


def _pipeline_key(source: str, chords: Any = None) -> dict[str, Any]:
    from aitsm.pipeline import keydet as keydet_module

    return keydet_module.detect_key(source, chords=list(chords) if chords else None)


def _pipeline_chords(
    source: str, beats: dict[str, Any], *, key: Any = None, total_seconds: float | None = None
) -> list[dict[str, Any]]:
    from aitsm.pipeline import chords as chords_module

    return chords_module.recognize_chords(
        source,
        beats,
        key=dict(key) if isinstance(key, Mapping) else None,
        total_seconds=total_seconds,
    )


def _pipeline_structure(
    source: str, beats: dict[str, Any], chords: list[dict[str, Any]], total_seconds: float
) -> list[dict[str, Any]]:
    from aitsm.pipeline import structure as structure_module

    return structure_module.analyze_structure(source, beats, chords, total_seconds)


def _pipeline_layout(total_seconds: float, beats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """证据不足时的通用曲式兜底，直接复用 pipeline 的实现（不重写）。"""
    from aitsm.pipeline import structure as structure_module

    return structure_module.fallback_layout(total_seconds, beats)


def _probe_duration(source: str) -> float | None:
    """读音频时长；文件不存在或读不了就返回 None（不抛异常）。"""
    try:
        path = Path(source)
        if not path.is_file():
            return None
        import soundfile  # 惰性导入：只有真音频才需要

        info = soundfile.info(str(path))
        frames = float(getattr(info, "frames", 0) or 0)
        rate = float(getattr(info, "samplerate", 0) or 0)
        if frames > 0 and rate > 0:
            return frames / rate
    except Exception:
        return None
    return None


def _estimate_total(beats_dict: Mapping[str, Any], chords: Any = None, sections: Any = None) -> float:
    """没有音频时长时，用拍点外推 + 事件结束点估计总时长。"""
    times = [float(value) for value in (beats_dict.get("beats") or []) if _f(value) > 0]
    period = _median_interval(sorted(times))
    span = max(times) + period if times else 0.0
    for item in list(_as_list(chords)) + list(_as_list(sections)):
        if isinstance(item, Mapping):
            span = max(span, _f(item.get("end"), _f(item.get("start"))))
    return span


def _fallback_transcription(
    source: str,
    *,
    progress: ProgressCallback | None,
    warnings: list[str],
    device: str,
) -> AccompanimentResult:
    """音频管线降级：beats → key → chords → (key 复核) → structure，全部复用 v1。"""
    _emit(progress, "accompaniment.beats", 30.0, "降级路径：检测节拍与拍号")
    try:
        beats_dict = _pipeline_beats(source)
    except Exception as exc:
        raise RuntimeError(
            f"伴奏转录降级失败：节拍分析不可用（{type(exc).__name__}: {exc}）。"
            "请确认音频文件可读且依赖完整。"
        ) from exc
    if not isinstance(beats_dict, Mapping):
        raise RuntimeError(f"伴奏转录降级失败：节拍分析返回了 {type(beats_dict).__name__}，应为 dict。")
    warnings.extend(_dedupe(_as_list(beats_dict.get("warnings"))))

    total_seconds = _probe_duration(source) or 0.0
    if total_seconds <= 0:
        total_seconds = _estimate_total(beats_dict)
    if total_seconds <= 0:
        total_seconds = 1.0
        warnings.append("无法确定音频时长，按 1 秒兜底渲染（节拍/和弦结果可能不完整）。")

    _emit(progress, "accompaniment.key", 45.0, "降级路径：估计调性")
    key_dict: dict[str, Any] = {}
    try:
        key_dict = _pipeline_key(source) or {}
        warnings.extend(_dedupe(_as_list(key_dict.get("warnings"))))
    except Exception as exc:
        warnings.append(f"调性估计失败，按 C 大调继续：{type(exc).__name__}: {exc}")

    _emit(progress, "accompaniment.chords", 60.0, "降级路径：识别和弦")
    chords: list[dict[str, Any]] = []
    try:
        chords = list(_pipeline_chords(source, dict(beats_dict), key=key_dict or None, total_seconds=total_seconds))
    except Exception as exc:
        warnings.append(f"和弦识别失败，伴奏结果没有和弦：{type(exc).__name__}: {exc}")

    if chords:
        # 与 run.py 一致：和弦证据能消除关系大小调歧义，值得再复核一次调性。
        _emit(progress, "accompaniment.key_refine", 70.0, "降级路径：用和弦证据复核调性")
        try:
            refined = _pipeline_key(source, chords=chords)
            if isinstance(refined, Mapping) and refined:
                key_dict = {**key_dict, **dict(refined)}
                warnings.extend(_dedupe(_as_list(refined.get("warnings"))))
        except Exception as exc:
            warnings.append(f"调性复核失败，沿用初判：{type(exc).__name__}: {exc}")

    total_seconds = max(total_seconds, _estimate_total(beats_dict, chords))
    _emit(progress, "accompaniment.structure", 85.0, "降级路径：切分歌曲结构")
    sections: list[dict[str, Any]] = []
    try:
        sections = list(_pipeline_structure(source, dict(beats_dict), chords, total_seconds))
    except Exception as exc:
        warnings.append(f"结构分析失败，使用通用曲式兜底：{type(exc).__name__}: {exc}")
        try:
            sections = list(_pipeline_layout(total_seconds, dict(beats_dict)))
        except Exception as layout_exc:  # pragma: no cover - 兜底也失败时放弃段落
            sections = []
            warnings.append(f"通用曲式兜底也失败，段落留空：{type(layout_exc).__name__}: {layout_exc}")

    payload = normalize_events(
        {
            "beats": dict(beats_dict),
            "key": key_dict,
            "chords": chords,
            "sections": sections,
        }
    )
    result = _result_from_payload(
        payload,
        engine=ENGINE_FALLBACK,
        requested=REQUESTED_ENGINE,
        warnings=warnings,
    )
    return result


# --------------------------------------------------------------------------- #
# SheetSage2 路径
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _find_output_file(out_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    try:
        candidates = sorted(
            path for path in out_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _read_midi_notes(path: Path) -> tuple[list[MelodyNote], str]:
    """用 pretty_midi 解析 MIDI 音符；没装就返回空列表 + 一条原因。"""
    try:
        import pretty_midi
    except Exception:
        return [], "pretty_midi 未安装，SheetSage2 的 MIDI 未解析为音符（文件路径保留在 events 里）。"
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        return [], f"MIDI 解析失败：{type(exc).__name__}: {exc}"
    notes: list[MelodyNote] = []
    for instrument in getattr(midi, "instruments", []) or []:
        for note in getattr(instrument, "notes", []) or []:
            notes.append(
                MelodyNote(
                    start=float(note.start),
                    end=max(float(note.start), float(note.end)),
                    midi=int(note.pitch),
                    confidence=min(1.0, max(0.0, float(getattr(note, "velocity", 0) or 0) / 127.0)),
                    source="sheetsage2-midi",
                )
            )
    notes.sort(key=lambda item: (item.start, item.end, item.midi))
    return notes, ""


def _load_sheetsage2_model(model_dir: Path, *, device: str, warnings: list[str]) -> tuple[Any, Path] | None:
    """加载 SheetSage2。

    两条路径，都**只读本地目录**：

    1. **本地包导入**（首选）：权重仓库自带完整 ``modeling_sheetsage2.py``，
       复制成 ``<cache>/pkgs/ss2/sheetsage2`` 后以普通包导入。这条路不依赖
       ``trust_remote_code``——实测 transformers 4.57 的模块缓存布局与本权重
       目录名不匹配，会误报「连不上 Hub」。
    2. ``AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only)``
       作为兜底：只在 1 不可用时尝试。

    返回 ``(model, package_dir)``；失败返回 ``None`` 并写清原因。
    """
    import shutil

    cache_root = Path(os.environ.get("AITSM_SHEETSAGE2_PKG_DIR") or (Path(work_home()) / "pkgs" / "ss2"))
    package_dir = cache_root / "sheetsage2"
    try:
        if not (package_dir / "modeling_sheetsage2.py").is_file():
            package_dir.parent.mkdir(parents=True, exist_ok=True)
            if package_dir.is_dir():
                shutil.rmtree(package_dir)
            shutil.copytree(model_dir, package_dir)
    except OSError as exc:
        warnings.append(f"SheetSage2 代码复制失败（{exc}），尝试 trust_remote_code 路径。")
        package_dir = None  # type: ignore[assignment]

    if package_dir is not None and (package_dir / "modeling_sheetsage2.py").is_file():
        try:
            if str(package_dir.parent) not in sys.path:
                sys.path.insert(0, str(package_dir.parent))
            from sheetsage2.configuration_sheetsage2 import SheetSage2Config  # type: ignore
            from sheetsage2.modeling_sheetsage2 import SheetSage2Model  # type: ignore

            config = SheetSage2Config.from_pretrained(str(model_dir))
            model = SheetSage2Model.from_pretrained(str(model_dir), config=config)
            model = model.to(_resolve_device(device)) if hasattr(model, "to") else model
            model = model.eval() if hasattr(model, "eval") else model
            return model, package_dir
        except Exception as exc:
            warnings.append(
                f"SheetSage2 本地包导入失败（{type(exc).__name__}: {exc}），尝试 trust_remote_code 路径。"
            )

    transformers = _import_transformers()
    if transformers is None:
        return None
    try:
        auto_model = getattr(transformers, "AutoModel")
        model = auto_model.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        model = model.to(_resolve_device(device)) if hasattr(model, "to") else model
        model = model.eval() if hasattr(model, "eval") else model
        return model, package_dir
    except Exception as exc:
        warnings.append(
            f"SheetSage2 权重加载失败，降级到音频管线：{type(exc).__name__}: {exc}"
        )
        return None


def work_home() -> Path:
    """SheetSage2 代码副本的落盘位置（默认 ``.cache/hf``，与 HF_HOME 保持一致）。"""
    raw = str(os.environ.get("HF_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "huggingface"


def _read_lab_rows(path: Path) -> list[list[str]]:
    """读 SheetSage2 的 ``*.lab``：每行 ``start\\tend\\tvalue...``（空白分隔）。"""
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        fields = [item for item in line.strip().split("\t") if item != ""]
        if not fields:
            fields = line.split()
        if len(fields) >= 2:
            rows.append(fields)
    return rows


def _lab_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chord_symbol_from_lab(value: str) -> str:
    """把 ``"B:sus4(b7)"`` / ``"D:maj/5"`` 转成 ``"B7sus4"`` / ``"D/A"``。

    SheetSage2 用 ``N`` 表示「这一段没有和弦」（no-chord），必须返回空串
    让调用方跳过——它不是一个和弦符号。
    """
    text = str(value or "").strip()
    if not text:
        return ""
    # 根音本身就是 no-chord 标记（如 "N:"）时同样跳过
    if text.upper() in {"N", "NONE", "NO_CHORD", "NC", "X"} or text.split(":")[0].strip().upper() in {
        "N", "NONE", "NC", "X"
    }:
        return ""
    if ":" not in text:
        return text
    root, _, quality = text.partition(":")
    quality = quality.strip()
    bass = ""
    if "/" in quality:
        quality, _, bass_part = quality.partition("/")
        bass = bass_part.strip()
    quality = quality.strip()
    # 质量名 -> 习惯写法
    mapping = {
        "maj": "",
        "min": "m",
        "maj7": "maj7",
        "min7": "m7",
        "sus4": "sus4",
        "sus2": "sus2",
        "sus4(b7)": "7sus4",
        "dim": "dim",
        "aug": "aug",
        "maj/5": "",
        "min/5": "m",
    }
    suffix = mapping.get(quality, quality)
    symbol = f"{root.strip()}{suffix}"
    if bass:
        # bass 是音级序号（1 起），需要换算成音名
        try:
            degree = int(bass)
        except ValueError:
            return f"{symbol}/{bass}"
        symbol = f"{symbol}/{_degree_to_pitch_class(root.strip(), degree)}"
    return symbol


_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _degree_to_pitch_class(root: str, degree: int) -> str:
    """大调音级（1=主音）-> 音名；超出 1–7 时按半音直算。"""
    base = _SEMITONES.get(root[:1].upper(), 0)
    if root[1:2] in ("#", "♯"):
        base += 1
    elif root[1:2] in ("b", "♭"):
        base -= 1
    intervals = (0, 2, 4, 5, 7, 9, 11)
    if 1 <= degree <= 7:
        pc = (base + intervals[degree - 1]) % 12
    else:
        pc = (base + degree - 1) % 12
    return _PC_NAMES[pc]


def _events_from_sheetsage2_dir(out_dir: Path, warnings: list[str]) -> dict[str, Any] | None:
    """从 SheetSage2 的产物目录读取 lab 文件，组装成 ``normalize_events`` 的输入。

    实测产物：``beat.lab``（start, index, num, den）、``downbeat.lab``、
    ``chord.lab``（start, end, ``Root:quality[/bass]``）、``key.lab``（start, end, ``B:minor``）、
    ``structure.lab``、``melody_vocal.mid`` / ``melody_instrumental.mid``、``score.abc``。
    """
    beat_rows = _read_lab_rows(out_dir / "beat.lab")
    downbeat_rows = _read_lab_rows(out_dir / "downbeat.lab")
    chord_rows = _read_lab_rows(out_dir / "chord.lab")
    key_rows = _read_lab_rows(out_dir / "key.lab")
    structure_rows = _read_lab_rows(out_dir / "structure.lab")
    if not (beat_rows or chord_rows or key_rows):
        return None

    beats = [_lab_float(row[0]) for row in beat_rows]
    beats = [value for value in beats if value >= 0]
    downbeats = [_lab_float(row[0]) for row in downbeat_rows]
    downbeats = [value for value in downbeats if value >= 0]

    beats_per_bar = 4
    time_signature = "4/4"
    if beat_rows and len(beat_rows[0]) >= 4:
        try:
            beats_per_bar = int(beat_rows[0][2])
            denominator = int(beat_rows[0][3])
            time_signature = f"{beats_per_bar}/{denominator}"
        except (TypeError, ValueError):
            pass

    bpm = 0.0
    if len(beats) >= 2:
        deltas = [beats[index + 1] - beats[index] for index in range(len(beats) - 1)]
        deltas = [value for value in deltas if value > 1e-6]
        if deltas:
            deltas.sort()
            median = deltas[len(deltas) // 2]
            bpm = round(60.0 / median, 3)

    tonic, mode = "C", "major"
    if key_rows:
        first = key_rows[0]
        value = first[2] if len(first) >= 3 else first[-1]
        if ":" in value:
            tonic_part, _, mode_part = value.partition(":")
            tonic = tonic_part.strip() or "C"
            mode = "minor" if mode_part.strip().lower().startswith("min") else "major"
        else:
            tonic = value.strip() or "C"

    chords = []
    for row in chord_rows:
        if len(row) < 3:
            continue
        start, end = _lab_float(row[0]), _lab_float(row[1])
        symbol = _chord_symbol_from_lab(row[2])
        if symbol and end > start:
            chords.append({"start": start, "end": end, "symbol": symbol, "confidence": 0.9})

    sections = []
    for row in structure_rows:
        if len(row) < 3:
            continue
        start, end = _lab_float(row[0]), _lab_float(row[1])
        kind = str(row[2]).strip().lower() or "other"
        if end > start:
            sections.append({"start": start, "end": end, "kind": kind, "label": kind})

    abc_text = ""
    abc_path = out_dir / "score.abc"
    if abc_path.is_file():
        abc_text = _read_text(abc_path)

    notes: list[dict[str, Any]] = []
    for name in ("melody_vocal.mid", "melody_instrumental.mid", "melody.mid"):
        midi_path = out_dir / name
        if midi_path.is_file():
            parsed, reason = _read_midi_notes(midi_path)
            if reason:
                warnings.append(f"{name}: {reason}")
            # normalize_events 的 _notes_from 只认 Mapping，这里转成 dict；
            # 传 MelodyNote 对象会被静默丢掉（实测踩过）。
            notes.extend(
                {
                    "start": note.start,
                    "end": note.end,
                    "midi": note.midi,
                    "confidence": note.confidence,
                    "source": f"sheetsage2:{name}",
                }
                for note in parsed
            )
            if notes:
                break

    return {
        "beats": {
            "bpm": bpm,
            "beats": beats,
            "downbeats": downbeats or beats[:: max(1, beats_per_bar)],
            "beats_per_bar": beats_per_bar,
            "time_signature": time_signature,
            "confidence": 0.9,
            "source": "sheetsage2",
        },
        "key": {"tonic": tonic, "mode": mode, "confidence": 0.9},
        "chords": chords,
        "sections": sections,
        "melody": notes,
        "abc": abc_text,
    }


def _run_sheetsage2(
    source: str,
    work_dir: Path,
    *,
    device: str,
    warnings: list[str],
) -> AccompanimentResult | None:
    """尝试用 SheetSage2 转录；任何环节不可用/失败都返回 None（由调用方降级）。"""
    model_dir = sheetsage2_dir()
    if model_dir is None:
        warnings.append(_unavailable_reason())
        return None

    out_dir = Path(work_dir) / "sheetsage2"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.append(f"SheetSage2 输出目录不可写（{exc}），降级到音频管线。")
        return None

    target_device = _resolve_device(device)
    loaded = _load_sheetsage2_model(model_dir, device=target_device, warnings=warnings)
    if loaded is None:
        return None
    model, _package_dir = loaded

    try:
        output = model.transcribe(
            str(source), output_dir=str(out_dir), dtype="fp32", preset="default"
        )
    except TypeError:
        # 某些权重版本的 transcribe 不接受这些关键字，逐步退回
        try:
            output = model.transcribe(str(source), output_dir=str(out_dir))
        except TypeError:
            try:
                output = model.transcribe(str(source))
            except Exception as exc:
                warnings.append(f"SheetSage2 推理失败，降级到音频管线：{type(exc).__name__}: {exc}")
                return None
        except Exception as exc:
            warnings.append(f"SheetSage2 推理失败，降级到音频管线：{type(exc).__name__}: {exc}")
            return None
    except Exception as exc:
        warnings.append(f"SheetSage2 推理失败，降级到音频管线：{type(exc).__name__}: {exc}")
        return None

    # ---- 规范化 abc / midi / events ----
    artifacts: list[dict[str, Any]] = []
    payload: dict[str, Any]
    if isinstance(output, Mapping):
        payload = normalize_events(output)
    elif isinstance(output, (list, tuple)):
        payload = normalize_events(list(output))
    elif isinstance(output, (str, Path)):
        text = _read_text(Path(output)) if Path(output).is_file() else str(output)
        payload = normalize_events({"abc": text} if text else {})
    else:
        payload = normalize_events({})

    # SheetSage2 的真实产物落在 out_dir 的 *.lab / score.abc / melody_*.mid 里，
    # transcribe() 的返回值本身可能很稀疏。lab 文件是权威时间轴，优先用它。
    lab_payload = _events_from_sheetsage2_dir(out_dir, warnings)
    if lab_payload is not None:
        normalized_lab = normalize_events(lab_payload)
        # 合并：lab 提供节拍/调性/和弦/段落，transcribe 返回值提供 ABC/MIDI 路径
        if lab_payload.get("beats") and lab_payload["beats"].get("beats"):
            normalized_lab["beats"] = normalized_lab["beats"]
        if not normalized_lab.get("abc") and payload.get("abc"):
            normalized_lab["abc"] = payload["abc"]
        if not normalized_lab.get("midi") and payload.get("midi"):
            normalized_lab["midi"] = payload["midi"]
        if not normalized_lab.get("melody") and payload.get("melody"):
            normalized_lab["melody"] = payload["melody"]
        payload = normalized_lab

    abc_text = _raw_text(payload.get("abc"))
    if not abc_text:
        abc_file = _find_output_file(out_dir, (".abc", ".txt"))
        if abc_file is not None:
            abc_text = _read_text(abc_file)
            if abc_text:
                artifacts.append({"kind": "abc", "path": str(abc_file)})

    # SheetSage2 的 transcribe() 可能把 MIDI 直接以 bytes 返回（实测如此），
    # 而不是写盘路径。bytes 要落盘成文件才能被 pretty_midi 解析。
    midi_value = payload.get("midi")
    if isinstance(midi_value, (bytes, bytearray)) and midi_value[:4] == b"MThd":
        midi_file = out_dir / "sheetsage2-returned.mid"
        try:
            midi_file.write_bytes(bytes(midi_value))
            midi_path = str(midi_file)
        except OSError as exc:
            warnings.append(f"SheetSage2 返回的 MIDI 无法落盘（{exc}），跳过 MIDI 解析。")
            midi_path = ""
    else:
        midi_path = _s(midi_value)
    if not midi_path:
        # 优先旋律轨（vocal 再 instrumental），chords.mid 只含和弦音，不能当旋律
        for name in ("melody_vocal.mid", "melody_instrumental.mid", "melody.mid", "chords.mid"):
            candidate = out_dir / name
            if candidate.is_file():
                midi_path = str(candidate)
                break
    if not midi_path:
        midi_file = _find_output_file(out_dir, (".mid", ".midi"))
        midi_path = str(midi_file) if midi_file is not None else ""
    notes: list[MelodyNote] = list(payload.get("melody") or [])
    if midi_path:
        artifacts.append({"kind": "midi", "path": midi_path})
        if not notes:
            parsed, reason = _read_midi_notes(Path(midi_path))
            if reason:
                warnings.append(reason)
            notes = parsed

    has_content = bool(
        payload.get("chords") or payload.get("sections") or notes or abc_text or midi_path
        or payload["beats"].beats
    )
    if not has_content:
        warnings.append("SheetSage2 未返回可用事件（和弦/节拍/旋律均为空），降级到音频管线。")
        return None

    payload["abc"] = abc_text
    payload["melody"] = notes
    payload["events"] = _flatten_events(
        payload["beats"],
        payload["chords"],
        payload["sections"],
        notes,
        artifacts,
    )
    result = _result_from_payload(
        payload,
        engine=ENGINE_SHEETSAGE2,
        requested=REQUESTED_ENGINE,
        warnings=warnings,
    )
    return result


def transcribe_with_sheetsage(
    audio_path: Any,
    *,
    work_dir: Any = ".",
    device: str = "auto",
    warnings: list[str] | None = None,
) -> AccompanimentResult | None:
    """只尝试 SheetSage2：本地条件不满足或推理失败都返回 ``None``，**从不抛异常冒充成功**。

    这是给上层/测试直接调用的薄封装，语义与 :func:`transcribe_accompaniment` 里的
    首选分支一致；返回 ``None`` 就代表调用方必须自己走降级路径。

    参数
    ----
    audio_path:
        音频路径（伴奏分轨或整曲混音均可）。
    work_dir:
        SheetSage2 的输出目录。
    device:
        ``"auto"`` / ``"cpu"`` / ``"cuda"``。
    warnings:
        可选列表，降级/失败原因会追加进去（原地修改），便于调用方记录 provenance。
    """
    sink: list[str] = warnings if warnings is not None else []
    if not is_available():
        reason = _unavailable_reason()
        if reason not in sink:
            sink.append(reason)
        return None
    work = Path(work_dir) if work_dir else Path(".")
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sink.append(f"SheetSage2 输出目录不可写（{exc}），无法转录。")
        return None
    return _run_sheetsage2(_s(audio_path), work, device=device, warnings=sink)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def _pick_source(audio_path: Any, full_mix: Any) -> tuple[str, list[str]]:
    """伴奏 stem 优先、整曲混音兜底；都不可用时给出明确告警后仍然尝试。"""
    warnings: list[str] = []
    stem = _s(audio_path)
    mix = _s(full_mix)
    stem_ok = bool(stem) and Path(stem).is_file()
    mix_ok = bool(mix) and Path(mix).is_file()
    if stem_ok:
        return stem, warnings
    if mix_ok:
        if stem:
            warnings.append(f"伴奏分轨不可用（{stem}），改用整曲混音 {mix} 分析，和弦精度可能下降。")
        else:
            warnings.append(f"未提供伴奏分轨，使用整曲混音 {mix} 分析，和弦精度可能下降。")
        return mix, warnings
    if stem:
        warnings.append(f"音频文件不存在：{stem}；仍按该路径尝试分析，失败会如实报错。")
        return stem, warnings
    if mix:
        warnings.append(f"音频文件不存在：{mix}；仍按该路径尝试分析，失败会如实报错。")
        return mix, warnings
    raise ValueError("伴奏转录需要音频路径：audio_path 与 full_mix 都为空。")


def transcribe_accompaniment(
    audio_path: Any,
    *,
    work_dir: Any,
    full_mix: Any = None,
    device: str = "auto",
    progress: ProgressCallback | None = None,
) -> AccompanimentResult:
    """伴奏联合转录主入口：先试 SheetSage2，不可用则显式降级到音频管线。

    参数
    ----
    audio_path:
        伴奏分轨路径（首选）。允许不存在，但会写告警并在降级时如实报错。
    work_dir:
        工作目录（SheetSage2 的输出落在这里）。不可写时只记告警，不影响返回。
    full_mix:
        整曲混音路径，伴奏分轨不可用时兜底。
    device:
        ``"auto"`` / ``"cpu"`` / ``"cuda"``；只对 SheetSage2 生效。
    progress:
        进度回调，兼容 ``progress(stage, percent, message)`` 与 ``progress(dict)``。

    返回
    ----
    :class:`aitsm.contracts.AccompanimentResult`：``engine`` 为
    ``"sheetsage2"`` 或 ``"beats+chroma"``，``requested`` 恒为 ``"SheetSage2"``，
    ``warnings`` 说明降级原因，``elapsed_seconds`` 一定有值。
    """
    started = time.perf_counter()
    warnings: list[str] = []
    work = Path(work_dir) if work_dir else Path(".")
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.append(f"工作目录不可用（{exc}），SheetSage2 无法落盘，可能影响结果。")

    source, source_warnings = _pick_source(audio_path, full_mix)
    warnings.extend(source_warnings)
    _emit(progress, "accompaniment", 2.0, "伴奏联合转录：准备输入音频")

    result: AccompanimentResult | None = None
    if is_available():
        _emit(progress, "accompaniment.sheetsage2", 10.0, "SheetSage2 本地权重可用，开始推理")
        result = _run_sheetsage2(source, work, device=device, warnings=warnings)
        if result is not None:
            _emit(
                progress,
                "accompaniment.sheetsage2",
                80.0,
                f"SheetSage2 转录完成（{len(result.chords)} 个和弦）",
            )
    else:
        warnings.append(_unavailable_reason())

    if result is None:
        _emit(progress, "accompaniment", 25.0, "SheetSage2 不可用，降级到音频管线")
        result = _fallback_transcription(
            source,
            progress=progress,
            warnings=warnings,
            device=device,
        )

    merged = _dedupe(list(result.warnings) + warnings)
    result.warnings = merged
    result.elapsed_seconds = round(time.perf_counter() - started, 3)
    _emit(
        progress,
        "accompaniment",
        100.0,
        f"伴奏转录完成（{result.engine}，{len(result.chords)} 个和弦 / {len(result.sections)} 段）",
    )
    return result

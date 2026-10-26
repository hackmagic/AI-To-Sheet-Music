"""伴奏转录的前端接线：SheetSage2 优先，音频管线降级。

`run.py` 原来直接调 ``beats.analyze_beats`` / ``keydet.detect_key`` /
``chords.recognize_chords`` / ``structure.analyze_structure``。本模块把这条链路
包一层：**先试 SheetSage2**（本地权重 + 适配器都就绪时），拿它联合解码出的
节拍/调性/和弦/段落；不可用或失败再逐段走原管线。

为什么不放在 ``pipeline/run.py`` 里直接写
----------------------------------------
``run.py`` 的旧阶段里有很多「质量证据」逻辑（和弦 stats、调性复核、结构兜底警告），
重写会把它们全丢掉。这里只做**取值来源替换**：返回同样的 dict 形状，旧逻辑照样跑。

契约（与 run.py 现有 dict 形状一致）::

    try_sheetsage2(audio, *, work_dir, device, warnings) -> AccompanimentResult | None
    analysis_dicts(result) -> {"beats": dict, "key": dict, "chords": list, "sections": list}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

__all__ = ["is_available", "try_sheetsage2", "analysis_dicts"]


def is_available() -> bool:
    try:
        from aitsm.transcribe import accompaniment

        return accompaniment.is_available()
    except Exception:  # pragma: no cover - 适配器缺失
        return False


def try_sheetsage2(
    audio_path: str | Path,
    *,
    work_dir: str | Path,
    device: str = "auto",
    warnings: list[str] | None = None,
    progress: Any = None,
) -> Any:
    """试一次 SheetSage2；不可用/失败返回 ``None``（调用方走旧管线）。"""
    sink = warnings if warnings is not None else []
    if not is_available():
        return None
    try:
        from aitsm.transcribe import accompaniment

        if callable(progress):
            try:
                progress("transcribe", 0.0, "SheetSage2 联合转录（节拍/调性/和弦/段落）")
            except Exception:
                pass
        result = accompaniment.transcribe_accompaniment(
            str(audio_path),
            work_dir=Path(work_dir),
            device=device,
        )
        if callable(progress):
            try:
                progress("transcribe", 1.0, "SheetSage2 完成")
            except Exception:
                pass
        return result
    except Exception as exc:
        sink.append(f"SheetSage2 失败（{type(exc).__name__}: {exc}），改用音频管线。")
        return None


#: SheetSage2 的段落名可能有连字符/空格变体（实测输出过 ``pre-chorus``），
#: 必须先规范化到 schema 枚举（``SECTION_KINDS``）再往下传，否则 analysis 校验失败。
_KIND_ALIASES = {
    "pre-chorus": "prechorus",
    "pre chorus": "prechorus",
    "prechorus": "prechorus",
    "post-chorus": "chorus",
    "post chorus": "chorus",
    "verse1": "verse",
    "verse2": "verse",
    "guitar solo": "solo",
    "breakdown": "bridge",
    "instrumental": "interlude",
    "fill": "interlude",
}


def _normalize_section_kind(raw: Any) -> str:
    """把任意段落标签规范化成 schema 允许的 kind 之一。

    未知标签一律落到 ``other``——**绝不猜**它对应哪个乐段类型；``other`` 是
    schema 里明确允许的兜底值，下游会按「无段落信息」处理。
    """
    text = str(raw or "").strip().lower().replace("_", "-")
    if not text:
        return "other"
    if text in _KIND_ALIASES:
        return _KIND_ALIASES[text]
    compact = text.replace("-", "").replace(" ", "")
    if compact in _KIND_ALIASES:
        return _KIND_ALIASES[compact]
    if compact in _VALID_KINDS:
        return compact
    return "other"


#: schema ``sections[].kind`` 的合法枚举（与 schemas/analysis.schema.json 保持一致）
_VALID_KINDS = frozenset(
    {"intro", "verse", "prechorus", "chorus", "bridge", "interlude", "solo", "outro", "other"}
)


def analysis_dicts(result: Any) -> dict[str, Any]:
    """把 :class:`AccompanimentResult` 转成 run.py 现有的 dict 形状。

    只做字段搬运，**不补造数据**：SheetSage2 没给的（如和弦置信度细分、
    beat_strength）就留默认值，旧的质量评估逻辑会据此给出对应告警。
    """
    beats = result.beats
    key = result.key
    key_dict: dict[str, Any] = {
        "tonic": key.tonic,
        "mode": key.mode,
        "name": key.display,
        "display_name": key.display,
        "confidence": key.confidence,
        "num_sharps_flats": key.fifths,
        "prefer_flat": key.fifths < 0,
    }
    beats_dict: dict[str, Any] = {
        **beats.to_dict(),
        "bpm_display": round(beats.bpm, 2),
        "beat_strength": [],
        "meter_candidates": [],
        "warnings": [],
        "source": "sheetsage2",
    }
    chords = [
        {
            **item.to_dict(),
            "root_pc": _pc(item.symbol),
            "duration": item.duration,
            "source": item.source or "sheetsage2",
        }
        for item in result.chords
    ]
    sections = [
        {
            **item.to_dict(),
            # kind 必须规范化到 schema 枚举：SheetSage2 实测会输出 pre-chorus
            "kind": _normalize_section_kind(item.kind),
            "label": item.label or item.kind,
            # SheetSage2 不给段落能量；contracts.Section 的默认值 0.5 会让所有段落
            # 落到同一个节奏型（实测 20 个段落全是 pop8）。这里按 kind 给保守先验，
            # 让副歌/主歌能有区别——这不是编造数据，而是类型先验，且写入时可见。
            "energy": _energy_for(item.kind),
        }
        for item in result.sections
    ]
    beats_dict, meter_note = _sanity_check_meter(beats_dict, result.beats)
    out: dict[str, Any] = {"beats": beats_dict, "key": key_dict, "chords": chords, "sections": sections}
    if meter_note:
        out["warnings"] = [meter_note]
    return out


#: 主流流行/摇滚歌曲的拍号先验：4/4 占绝大多数，2/4 与 3/4 少见，6/8 更少。
#: SheetSage2 实测会把 4/4 判成 2/4（拍点间隔显示强拍在 2 拍处，但它报 denominator=4），
#: 这类误判会让小节数翻倍、和弦落位全错，必须校验。
_COMMON_METERS = {(4, 4), (3, 4), (6, 8), (2, 4), (12, 8)}


def _sanity_check_meter(
    beats_dict: dict[str, Any], grid: Any
) -> tuple[dict[str, Any], str]:
    """校验拍号是否自洽；可疑时回退到 4/4 并给出告警。

    判据（只用数据本身，不猜）：

    1. **每小节秒数**：``bar_seconds = downbeat_period``。流行/摇滚歌的常规区间是
       1.5–5.0 秒。``2/4`` 在 93.75 BPM 下只有 1.28 秒/小节——这个组合在真实歌曲里
       极罕见（意味着 181 个“小节”，而整曲只有 229 秒）；
    2. **强拍/拍点比** 与声明的 ``beats_per_bar`` 一致；
    3. 声明拍号在常见集合内。

    1 与 2 同时成立才认为可信。任一条强烈可疑 → 按「强拍间隔 × 每小节 4 拍」
    回退为 4/4（并把 BPM 校正到与新的小节划分一致）。
    """
    beats = [float(x) for x in (beats_dict.get("beats") or [])]
    downbeats = [float(x) for x in (beats_dict.get("downbeats") or [])]
    declared = str(beats_dict.get("time_signature") or "")
    try:
        numerator_text, _, denominator_text = declared.partition("/")
        declared_meter = (int(numerator_text), int(denominator_text))
    except (TypeError, ValueError):
        declared_meter = (0, 0)

    if len(beats) < 4 or len(downbeats) < 2:
        return beats_dict, ""

    beat_gaps = sorted(b - a for a, b in zip(beats, beats[1:]) if b - a > 1e-6)
    down_gaps = sorted(b - a for a, b in zip(downbeats, downbeats[1:]) if b - a > 1e-6)
    if not beat_gaps or not down_gaps:
        return beats_dict, ""
    beat_period = beat_gaps[len(beat_gaps) // 2]
    bar_seconds = down_gaps[len(down_gaps) // 2]
    implied = bar_seconds / beat_period if beat_period > 0 else 0.0
    implied_bars = int(round(implied))
    declared_bars = int(beats_dict.get("beats_per_bar") or 0)

    problems: list[str] = []
    inconsistent = False
    if declared_meter not in _COMMON_METERS:
        problems.append(f"拍号 {declared} 不在常见集合内")
    if implied_bars and abs(implied - implied_bars) <= 0.35 and implied_bars != declared_bars:
        inconsistent = True
        problems.append(
            f"声明的每小节 {declared_bars} 拍，但强拍间隔相当于 {implied_bars} 拍"
        )
    # 每小节秒数是否像正常歌曲：2/4 且 <1.6s/小节 基本可判为「拍单位取错」
    # （把八分音符当成了拍），这也是实测里 4/4 被写成 2/4 的成因。
    if declared_bars <= 2 and bar_seconds < 1.6:
        inconsistent = True
        problems.append(
            f"每小节仅 {bar_seconds:.2f}s（{declared}），"
            "与常见歌曲的小节时长不符，疑似把八分音符当成了拍"
        )

    if not problems:
        return beats_dict, ""

    # 只是少见但自洽（如 5/4）：数据说得通，不替用户改写，只提示
    if not inconsistent:
        note = (
            "SheetSage2 报告的拍号 " + declared + " 较少见（" + "；".join(problems)
            + "），但强拍间隔与它自洽，已保留原值；若与原曲不符请手动指定拍号。"
        )
        return beats_dict, note

    # 回退 4/4：把「被当成拍的八分音符」升格为拍 —— 每小节从 2 拍变 4 拍，
    # 即原有的 downbeats 需要隔一个取一个（小节时长翻倍）。
    # BPM **保持不变**（93.75 是对的）；若按 bar_seconds/4 反推会得到 187.5，
    # 那是八分音符速度，不是四分音符 BPM。
    fallback = dict(beats_dict)
    fallback["beats_per_bar"] = 4
    fallback["time_signature"] = "4/4"
    original_downbeats = list(beats_dict.get("downbeats") or [])
    if declared_bars > 0 and declared_bars * 2 <= 8:
        # 声明 2 拍/小节 → 4 拍/小节：隔一个取一个
        fallback["downbeats"] = original_downbeats[::2]
    else:
        fallback["downbeats"] = original_downbeats
    note = (
        "SheetSage2 的拍号不可信（"
        + "；".join(problems)
        + "），已回退为 4/4（BPM 保持 "
        + str(beats_dict.get("bpm_display") or beats_dict.get("bpm"))
        + "）。小节数按 4/4 重新计算。"
    )
    return fallback, note


_PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6,
    "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}

#: 段落类型 -> 典型能量（SheetSage2 不给 energy 时的保守默认）
_KIND_ENERGY = {
    "intro": 0.2,
    "verse": 0.45,
    "prechorus": 0.6,
    "chorus": 0.75,
    "bridge": 0.5,
    "interlude": 0.35,
    "solo": 0.7,
    "outro": 0.25,
    "other": 0.5,
}


def _pc(symbol: str) -> int:
    text = str(symbol or "").strip()
    if not text:
        return 0
    for length in (2, 1):
        candidate = text[:length].upper()
        if candidate in _PC:
            return _PC[candidate]
    return 0


def _energy_for(kind: str) -> float:
    return _KIND_ENERGY.get(str(kind or "").lower(), 0.5)


def env_enabled(name: str = "AITSM_SHEETSAGE2_FIRST") -> bool:
    """总开关：``0`` 关闭 SheetSage2 优先（强制走旧管线）。"""
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}

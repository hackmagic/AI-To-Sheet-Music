"""F9：分析流水线编排。

按 ingest → separate → beats → key → chords → structure → lyrics/melody 顺序执行，
把各阶段结果组装成 ``analysis.json``，并把耗时、引擎版本、降级原因写进
``manifest.json``。

用法（调试）：
    .venv/Scripts/python -m aitsm.pipeline.run "<音频/视频>" --out work/analysis-v1
    .venv/Scripts/python -m aitsm.pipeline.run "<音频>" --out work/analysis-60s --start 0 --duration 60
"""

from __future__ import annotations

import os

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from aitsm import media
from aitsm.pipeline import beats as beats_mod
from aitsm.pipeline import chords as chords_mod
from aitsm.pipeline import ingest as ingest_mod
from aitsm.pipeline import keydet as keydet_mod
from aitsm.pipeline import separate as separate_mod
from aitsm.pipeline import structure as structure_mod

ANALYSIS_SCHEMA_ID = "aitsm.analysis/v1"
MANIFEST_SCHEMA_ID = "aitsm.manifest/v1"

ProgressCallback = Callable[..., None]


def repo_root() -> Path:
    """仓库根目录（aitsm/pipeline/run.py → parents[2]）。"""
    return Path(__file__).resolve().parents[2]


def schema_path() -> Path:
    return repo_root() / "schemas" / "analysis.schema.json"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _emit(progress: ProgressCallback | None, stage: str, percent: float, message: str) -> None:
    """兼容两种 progress 回调签名：progress(stage, percent, message) 与 progress(dict)。"""
    if progress is None:
        return
    try:
        progress(stage, float(percent), message)
        return
    except TypeError:
        pass
    except Exception:  # 回调自身报错不能影响主流程
        return
    try:
        progress({"stage": stage, "percent": float(percent), "message": message})
    except Exception:
        return


def _warn(stage: str, message: str, *, level: str = "warn", code: str | None = None) -> dict[str, Any]:
    warning: dict[str, Any] = {"level": level, "message": str(message), "stage": stage}
    if code:
        warning["code"] = code
    return warning


# 各阶段的告警 code 规则：子模块历史上只回字符串，这里统一补 code，
# 让前端质量面板可以按 code 分组（Leads 要求：每条 warning 都必须有 code）。
_WARNING_CODE_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "ingest": (
        ("重采样", "audio.resampled"),
        ("视频流", "ingest.video_stream"),
        ("未识别的扩展名", "ingest.unknown_extension"),
        ("片段", "ingest.trimmed"),
        ("内嵌封面", "ingest.attached_pic"),
    ),
    "separate": (
        ("复用", "separate.cache_reused"),
        ("降级", "separate.fallback"),
        ("失败", "separate.failed"),
        ("不可用", "separate.unavailable"),
        ("关闭了分轨", "separate.disabled"),
    ),
    "beats": (
        ("谐波", "beats.hpss_failed"),
        ("起音", "beats.no_onsets"),
        ("拍点跟踪", "beats.sparse_beats"),
        ("拍点数量", "beats.low_confidence"),
        ("速度置信度", "beats.low_confidence"),
        ("八度", "beats.tempo_octave"),
        ("6/8 候选", "beats.meter_demoted"),
        ("不可靠", "beats.low_confidence"),
    ),
    "key": (
        ("关系大小调", "key.relative_ambiguity"),
        ("未提供和弦", "key.chroma_only"),
        ("chroma", "key.low_chroma"),
        ("谐波分离失败", "key.hpss_failed"),
        ("音频为空", "key.empty_audio"),
    ),
    "key_refine": (
        ("关系大小调", "key.relative_ambiguity"),
        ("修正", "key.refined"),
    ),
    "chords": (
        ("未识别到任何和弦", "chords.no_chords"),
        ("置信度", "chords.low_confidence"),
        ("覆盖", "chords.coverage_gap"),
    ),
    "structure": (
        ("只切出 1 个段落", "structure.degenerate"),
        ("兜底", "structure.fallback"),
        ("不够细", "structure.coarse"),
        ("段落过长", "structure.coarse"),
        ("失败", "structure.failed"),
    ),
    "lyrics": (
        ("没有人声轨", "lyrics.no_vocals"),
        ("模块不可用", "lyrics.module_missing"),
        ("失败", "lyrics.failed"),
        ("未能识别出歌词", "lyrics.empty"),
        ("跳过", "lyrics.skipped"),
        ("整曲音频", "lyrics.vocals_fallback_mix"),
    ),
    "melody": (
        ("没有人声轨", "melody.no_vocals"),
        ("模块不可用", "melody.module_missing"),
        ("失败", "melody.failed"),
        ("对齐失败", "melody.align_failed"),
        ("整曲音频", "melody.vocals_fallback_mix"),
    ),
}


def _guess_warning_code(stage: str, message: str) -> str:
    """按关键字给字符串告警补一个稳定 code；匹配不到用 `<stage>.warning`。"""
    text = str(message)
    for keyword, code in _WARNING_CODE_RULES.get(stage, ()):
        if keyword in text:
            return code
    return f"{stage}.warning"


def _normalize_warnings(stage: str, items: Any) -> list[dict[str, Any]]:
    """把子模块的告警（字符串或 dict）统一成 {level, code, message, stage}。

    子模块历史上返回纯字符串，且往往缺 code；这里统一补齐，保证每条 warning
    都有 code（前端质量面板按 code 分组）。
    """
    output: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            warning = dict(item)
            warning.setdefault("stage", stage)
            warning.setdefault("level", "warn")
            warning["message"] = str(warning.get("message") or "")
            if not warning.get("code"):
                warning["code"] = _guess_warning_code(stage, warning["message"])
        else:
            message = str(item)
            warning = _warn(stage, message, code=_guess_warning_code(stage, message))
        output.append(warning)
    return output


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"}
    for name, label in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("demucs", "demucs"),
        ("torch", "torch"),
        ("jsonschema", "jsonschema"),
    ):
        try:
            module = __import__(name)
            versions[label] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[label] = "missing"
    try:
        versions["ffmpeg"] = media.ffmpeg_version()
    except Exception:
        versions["ffmpeg"] = "unknown"
    return versions


# 质量自检口径：各阶段 0..1，overall 是核心四段（节拍/调性/和弦/结构）的加权。
_QUALITY_WEIGHTS = {"beats": 0.20, "key": 0.20, "chords": 0.30, "structure": 0.30}
_CHORDS_LOW_CONFIDENCE = 0.55
# 单段超过 16 小节（4/4）视为「结构可能切得不够细」
_MAX_SECTION_BARS = 16


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bar_seconds(beats: dict[str, Any]) -> float:
    """一小节多少秒（用 BPM 与拍号估算，用于判断段落是否过长）。"""
    bpm = float(beats.get("bpm_display") or beats.get("bpm") or 0.0)
    per_bar = int(beats.get("beats_per_bar") or 4)
    if bpm <= 0:
        return 0.0
    return 60.0 / bpm * max(1, per_bar)


def _find_coarse_sections(
    sections: list[dict[str, Any]], beats: dict[str, Any],
) -> list[tuple[float, float, str]]:
    """返回超过 16 小节的段落（供 structure.coarse 告警用）。"""
    bar_seconds = _bar_seconds(beats)
    if bar_seconds <= 0 or not sections:
        return []
    limit = bar_seconds * _MAX_SECTION_BARS
    coarse: list[tuple[float, float, str]] = []
    for section in sections:
        length = float(section.get("end") or 0.0) - float(section.get("start") or 0.0)
        if length > limit:
            coarse.append((
                float(section.get("start") or 0.0),
                float(section.get("end") or 0.0),
                str(section.get("label") or section.get("kind") or "段落"),
            ))
    return coarse


def _section_chord_profile(
    section: dict[str, Any], chords: list[dict[str, Any]],
) -> dict[str, float]:
    """按持续时长统计段落内各和弦符号的占比（用于判断段落是否真的不同）。"""
    start = float(section.get("start") or 0.0)
    end = float(section.get("end") or 0.0)
    total = max(end - start, 1e-6)
    weights: dict[str, float] = {}
    for chord in chords:
        overlap = min(end, float(chord.get("end") or 0.0)) - max(start, float(chord.get("start") or 0.0))
        if overlap <= 0:
            continue
        symbol = str(chord.get("symbol") or "")
        if not symbol:
            continue
        weights[symbol] = weights.get(symbol, 0.0) + overlap / total
    return weights


def _chord_profile_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """两个和弦占比向量的余弦相似度。"""
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    norm_left = sum(value * value for value in left.values()) ** 0.5
    norm_right = sum(value * value for value in right.values()) ** 0.5
    if norm_left <= 1e-9 or norm_right <= 1e-9:
        return 0.0
    return float(dot / (norm_left * norm_right))


def _bar_symbols(chords: list[dict[str, Any]], beats: dict[str, Any]) -> list[str]:
    """把和弦事件落到小节网格上，返回逐小节的主导和弦符号。"""
    bar_seconds = _bar_seconds(beats)
    if bar_seconds <= 0 or not chords:
        return []
    downbeats = beats.get("downbeats") or []
    origin = float(downbeats[0]) if downbeats else 0.0
    weights: dict[int, dict[str, float]] = {}
    for chord in chords:
        symbol = str(chord.get("symbol") or "")
        if not symbol:
            continue
        start = float(chord.get("start") or 0.0)
        end = float(chord.get("end") or 0.0)
        index = int(round((start - origin) / bar_seconds))
        weights.setdefault(index, {})
        weights[index][symbol] = weights[index].get(symbol, 0.0) + max(end - start, 1e-6)
    if not weights:
        return []
    lowest = min(weights)
    highest = max(weights)
    symbols: list[str] = []
    for index in range(lowest, highest + 1):
        bucket = weights.get(index)
        if not bucket:
            symbols.append("")
            continue
        symbols.append(max(bucket.items(), key=lambda item: item[1])[0])
    return symbols


def _chord_loop_period(
    chords: list[dict[str, Any]], beats: dict[str, Any],
    *, max_period: int = 8, min_ratio: float = 0.80,
) -> int | None:
    """检测整曲是否就是一条短和弦循环（2~8 小节）不断重复。

    命中说明「段落之间没有和弦层面的差异」，任何切分算法都切不出真实段落。
    """
    symbols = _bar_symbols(chords, beats)
    if len(symbols) < 8:
        return None
    for period in range(2, max_period + 1):
        if len(symbols) < period * 2:
            continue
        compared = 0
        matched = 0
        for index in range(period, len(symbols)):
            if not symbols[index] or not symbols[index - period]:
                continue
            compared += 1
            if symbols[index] == symbols[index - period]:
                matched += 1
        if compared >= max(4, period) and matched / compared >= min_ratio:
            return period
    return None


def _looks_uniform_structure(
    sections: list[dict[str, Any]], chords: list[dict[str, Any]],
    *, beats: dict[str, Any] | None = None,
    energy_spread: float = 0.12, similarity: float = 0.75,
) -> bool:
    """段落之间几乎没有能量/和弦差异时返回 True（整曲同一循环 / 合成测试曲）。

    这种情况任何切分算法都切不出真实段落，如实提示比编造边界更诚实。
    """
    if len(sections) < 3 or not chords:
        return False
    if beats is not None and _chord_loop_period(chords, beats) is not None:
        return True
    energies = [float(section.get("energy") or 0.0) for section in sections]
    if max(energies) - min(energies) > energy_spread:
        return False
    profiles = [_section_chord_profile(section, chords) for section in sections]
    if any(not profile for profile in profiles):
        return False
    for left, right in zip(profiles, profiles[1:]):
        if _chord_profile_similarity(left, right) < similarity:
            return False
    return True


def assess_analysis_quality(analysis: dict[str, Any]) -> dict[str, Any]:
    """自检：给每个阶段一个 0..1 的分数，并列出可操作的问题。

    口径说明（与各模块内部的相对分不同，这里是「能不能直接用」的绝对判断）：
    - beats：置信度 + 拍点覆盖率 + downbeat 完备度
    - key：调性置信度（关系大小调歧义已按行业惯例处理，不再扣分）
    - chords：平均置信度 + 时间覆盖率 + 调内和弦占比
    - structure：段落置信度均值 + 时间覆盖率 + 段落长度合理性
    """
    duration = float((analysis.get("audio") or {}).get("duration") or 0.0)
    issues: list[dict[str, Any]] = []
    stages: dict[str, float] = {}

    def add_issue(level: str, code: str, message: str, target: str) -> None:
        issues.append({"level": level, "code": code, "message": message, "target": target})

    # ---- 节拍 ----
    beats = analysis.get("beats") or {}
    beat_times = [float(value) for value in (beats.get("beats") or [])]
    bpm = float(beats.get("bpm") or 0.0)
    expected_beats = duration / (60.0 / bpm) if bpm > 0 and duration > 0 else 0.0
    beat_coverage = _clamp01(len(beat_times) / expected_beats) if expected_beats > 0 else 0.0
    per_bar = max(1, int(beats.get("beats_per_bar") or 4))
    expected_downbeats = max(1.0, len(beat_times) / per_bar)
    downbeat_ratio = _clamp01(len(beats.get("downbeats") or []) / expected_downbeats)
    stages["beats"] = round(_clamp01(
        0.55 * float(beats.get("confidence") or 0.0) + 0.25 * beat_coverage + 0.20 * downbeat_ratio
    ), 3)
    if stages["beats"] < 0.6:
        add_issue("warn", "beats.low_confidence",
                  f"节拍置信度偏低（{stages['beats']:.2f}），BPM 或拍号可能需要人工确认。", "整曲节拍")
    if beat_coverage < 0.9:
        add_issue("warn", "beats.sparse_beats",
                  f"拍点只覆盖了 {beat_coverage:.0%} 的时长，后半段可能没有拍点。", "整曲节拍")

    # ---- 调性 ----
    key = analysis.get("key") or {}
    stages["key"] = round(_clamp01(float(key.get("confidence") or 0.0)), 3)
    if stages["key"] < 0.6:
        code = "key.relative_ambiguity" if key.get("relative_major_swap") else "key.low_confidence"
        add_issue("warn", code,
                  f"调性置信度偏低（{stages['key']:.2f}），关系大小调/同主音调可能需要人工确认。", "整曲调性")

    # ---- 和弦 ----
    chords = analysis.get("chords") or []
    if chords:
        mean_confidence = _mean([float(item.get("confidence") or 0.0) for item in chords])
        covered = sum(float(item.get("end") or 0.0) - float(item.get("start") or 0.0) for item in chords)
        chord_coverage = _clamp01(covered / duration) if duration > 0 else 0.0
        tonic_pc = int(key.get("tonic_pc") or 0) % 12
        scale = {(tonic_pc + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
        diatonic = _mean([
            1.0 if int(item.get("root_pc") or 0) % 12 in scale else 0.0 for item in chords
        ])
        stages["chords"] = round(_clamp01(
            0.60 * mean_confidence + 0.25 * chord_coverage + 0.15 * diatonic
        ), 3)
        if mean_confidence < 0.6:
            add_issue("warn", "chords.low_confidence",
                      f"和弦平均置信度 {mean_confidence:.2f}，转位/七和弦建议人工校对。", "整曲和弦")
        if chord_coverage < 0.9:
            add_issue("warn", "chords.coverage_gap",
                      f"和弦只覆盖了 {chord_coverage:.0%} 的时长，存在未被识别的空洞。", "整曲和弦")
    else:
        stages["chords"] = 0.0
        add_issue("error", "chords.no_chords", "没有识别出任何和弦。", "整曲和弦")

    # ---- 结构 ----
    sections = analysis.get("sections") or []
    if sections:
        mean_section_confidence = _mean([float(item.get("confidence") or 0.0) for item in sections])
        last_end = float(sections[-1].get("end") or 0.0)
        section_coverage = _clamp01(last_end / duration) if duration > 0 else 0.0
        lengths = [float(item.get("end") or 0.0) - float(item.get("start") or 0.0) for item in sections]
        lengths_ok = _mean([1.0 if length >= 2.0 else 0.0 for length in lengths])
        stages["structure"] = round(_clamp01(
            0.50 * mean_section_confidence + 0.30 * section_coverage + 0.20 * lengths_ok
        ), 3)
        if len(sections) < 3:
            add_issue("warn", "structure.too_few_sections",
                      f"只切出 {len(sections)} 个段落，结构信息有限。", "整曲结构")
        coarse = _find_coarse_sections(sections, beats)
        for start, end, label in coarse[:3]:
            add_issue("info", "structure.coarse",
                      f"{label}（{start:.1f}s-{end:.1f}s）超过 {_MAX_SECTION_BARS} 小节，"
                      "可能还能细分，但连续重复同一和弦进行对弹唱无影响。", label)
    else:
        stages["structure"] = 0.0
        add_issue("warn", "structure.missing", "没有段落信息。", "整曲结构")

    # ---- 旋律 / 歌词（不参与 overall，仅提示） ----
    melody = analysis.get("melody") or {}
    notes = melody.get("notes") or []
    if notes:
        median_confidence = sorted(float(note.get("confidence") or 0.0) for note in notes)[len(notes) // 2]
        stages["melody"] = round(_clamp01(median_confidence), 3)
        if median_confidence < 0.5:
            add_issue("warn", "melody.low_confidence",
                      f"旋律音符置信度中位数 {median_confidence:.2f}，旋律线可能需要人工校对。", "整曲旋律")
    else:
        stages["melody"] = 0.0
        add_issue("info", "melody.empty", "未提取到旋律音符（纯伴奏或人声过弱）。", "整曲旋律")

    lyrics = analysis.get("lyrics") or []
    stages["lyrics"] = 1.0 if lyrics else 0.0
    if not lyrics:
        add_issue("info", "lyrics.empty", "没有识别出歌词。", "整曲歌词")

    overall = sum(stages.get(name, 0.0) * weight for name, weight in _QUALITY_WEIGHTS.items())
    return {
        "overall": round(_clamp01(overall), 3),
        "stages": stages,
        "issues": issues,
    }


def validate_analysis(analysis: dict[str, Any]) -> list[str]:
    """用 schemas/analysis.schema.json 校验，返回错误列表（空列表表示通过）。"""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema 未安装，跳过 schema 校验"]
    path = schema_path()
    if not path.is_file():
        return [f"找不到 schema 文件：{path}"]
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"schema 读取失败：{exc}"]
    validator = jsonschema.Draft202012Validator(schema) if "2020-12" in str(schema.get("$schema", "")) else jsonschema.Draft7Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(analysis), key=lambda item: list(item.path)):
        location = "/".join(str(part) for part in error.path) or "(root)"
        errors.append(f"{location}: {error.message}")
    return errors


def _json_default(value: Any) -> Any:
    """让 numpy 标量也能写进 JSON（兜底，正常路径已全部转成 Python 原生类型）。"""
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------------- #
# 可选阶段：歌词 / 旋律（由 melody-lyrics 队友实现，缺失时跳过）
# --------------------------------------------------------------------------- #

def _transcribe_lyrics(
    vocals_wav: str | None,
    duration: float,
    warnings: list[dict[str, Any]],
    *,
    language: str = "auto",
) -> dict[str, Any] | None:
    if not vocals_wav:
        warnings.append(_warn("lyrics", "没有人声轨，跳过歌词识别。", level="info", code="lyrics.no_vocals"))
        return None
    try:
        from aitsm.pipeline import lyrics as lyrics_mod
    except ImportError as exc:
        warnings.append(_warn("lyrics", f"歌词模块不可用（{exc}），已跳过。", code="lyrics.module_missing"))
        return None
    try:
        result = lyrics_mod.transcribe_lyrics(vocals_wav, duration=duration, language=language)
    except Exception as exc:  # 不因歌词失败影响主链路
        warnings.append(_warn("lyrics", f"歌词识别失败：{type(exc).__name__}: {exc}", level="error", code="lyrics.failed"))
        return None
    if result is None:
        return {"available": False, "lines": [], "engine": "none"}
    # 契约：transcribe_lyrics 返回歌词行列表；也兼容返回 {"lines": [...]} 的实现。
    lines: list[Any] = []
    if isinstance(result, list):
        lines = [line for line in result if isinstance(line, dict)]
    elif isinstance(result, dict):
        candidate = result.get("lines")
        if isinstance(candidate, list):
            lines = [line for line in candidate if isinstance(line, dict)]
    payload: dict[str, Any] = {
        "available": bool(lines),
        "lines": lines,
        "engine": "whisper",
        "language": language,
    }
    # transcribe_lyrics 只回传歌词行；引擎型号/设备从模块的 last_run_info() 取，
    # 否则 manifest 里只剩 "whisper"，无法解释 small 与 medium 的 CER 差异。
    run_info: dict[str, Any] = {}
    try:
        getter = getattr(lyrics_mod, "last_run_info", None)
        if callable(getter):
            run_info = getter() or {}
    except Exception:  # 记录元信息失败不能影响主链路
        run_info = {}
    for key in ("engine", "model", "device"):
        value = run_info.get(key)
        if value:
            payload[key] = value
    if not lines:
        warnings.append(_warn("lyrics", "未能识别出歌词（可能无演唱、whisper 缺失或语言不符）。",
                              level="info", code="lyrics.empty"))
    return payload


def _transcribe_melody(
    vocals_wav: str | None,
    beats: dict[str, Any],
    duration: float,
    lyrics: dict[str, Any] | None,
    warnings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not vocals_wav:
        warnings.append(_warn("melody", "没有人声轨，跳过旋律识别。", level="info", code="lyrics.no_vocals"))
        return None
    try:
        from aitsm.pipeline import melody as melody_mod
        from aitsm.pipeline.align import align_lyrics_to_melody
    except ImportError as exc:
        warnings.append(_warn("melody", f"旋律模块不可用（{exc}），已跳过。", code="melody.module_missing"))
        return None
    try:
        melody = melody_mod.transcribe_melody(vocals_wav, beats, duration=duration)
    except Exception as exc:
        warnings.append(_warn("melody", f"旋律识别失败：{type(exc).__name__}: {exc}", level="error", code="melody.failed"))
        return None
    if melody is None:
        return None
    if not isinstance(melody, dict):
        return {"notes": list(melody)}
    lyric_lines = lyrics.get("lines") if isinstance(lyrics, dict) else lyrics
    if lyric_lines:
        try:
            aligned = align_lyrics_to_melody(melody, lyric_lines, beats)
            if isinstance(aligned, dict):
                melody = {**melody, **aligned}
            elif isinstance(aligned, list):
                melody = {**melody, "notes": aligned}
        except Exception as exc:
            warnings.append(_warn("melody", f"歌词-旋律对齐失败：{type(exc).__name__}: {exc}", code="melody.align_failed"))
    # 下游（score 层）按 available 判断旋律是否可用
    melody = dict(melody)
    melody.setdefault("available", bool(melody.get("notes")))
    melody.setdefault("engine", str(melody.get("engine") or "unknown"))
    return melody


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_pipeline(
    source: Path | str,
    work_dir: Path | str,
    options: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """执行完整分析，写出 analysis.json / manifest.json，返回 analysis 字典。"""
    pipeline_started = time.monotonic()
    options = dict(options or {})
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    start_sec = _as_float(options.get("start_sec", options.get("start_seconds", 0.0)), 0.0)
    duration_sec = _as_float(
        options.get("duration_sec", options.get("duration_seconds", options.get("duration"))),
        0.0,
    )
    duration_opt = duration_sec if duration_sec > 0 else None
    device = str(options.get("device") or "auto")
    want_separation = bool(options.get("separate", True)) and not bool(options.get("no_separate", False))

    def stage_begin(name: str) -> float:
        return time.monotonic()

    def stage_end(name: str, started: float, *, engine: str | None = None, ok: bool = True,
                  extra: dict[str, Any] | None = None) -> None:
        elapsed = round(time.monotonic() - started, 3)
        timings[name] = elapsed
        record: dict[str, Any] = {"name": name, "elapsed_seconds": elapsed, "ok": ok}
        if engine:
            record["engine"] = engine
        if extra:
            record.update(extra)
        stage_records.append(record)

    # ---------------- 1. ingest ----------------
    _emit(progress, "ingest", 1.0, "解码音频、生成分析与播放资产")
    started = stage_begin("ingest")
    try:
        ingested = ingest_mod.ingest(source, work, max_seconds=duration_opt, start_seconds=start_sec)
    except Exception as exc:
        stage_end("ingest", started, ok=False)
        raise
    warnings.extend(_normalize_warnings("ingest", ingested.get("warnings")))
    total_seconds = float(ingested["duration"])
    analysis_wav = str(ingested["source_wav"])
    stage_end("ingest", started, engine="ffmpeg",
              extra={"duration": total_seconds, "ffmpeg": ingested.get("tools", {}).get("ffmpeg")})
    _emit(progress, "ingest", 8.0, "解码完成")

    # ---------------- 2. separate ----------------
    stems_info: dict[str, Any] = {"engine": "none"}
    stems_paths: dict[str, str] = {}
    if want_separation:
        _emit(progress, "separate", 10.0, "分离人声/鼓/贝斯/伴奏")
        started = stage_begin("separate")
        try:
            separated = separate_mod.separate(analysis_wav, work, device=device)
            stems_paths = dict(separated.get("stems") or {})
            stems_info = {
                "engine": separated.get("engine") or "none",
                "model": separated.get("model"),
                "device": separated.get("device"),
                "cached": separated.get("cached", False),
                **{stem: path for stem, path in stems_paths.items()},
                "hq_stems": separated.get("hq_stems") or {},
            }
            warnings.extend(_normalize_warnings("separate", separated.get("warnings")))
            stage_end("separate", started, engine=stems_info.get("engine"),
                      extra={"device": separated.get("device"), "cached": separated.get("cached", False)})
            _emit(progress, "separate", 62.0, f"分轨完成（{stems_info.get('engine')}）")
        except Exception as exc:
            stage_end("separate", started, ok=False)
            warnings.append(_warn("separate", f"分轨失败，后续使用混合音频：{type(exc).__name__}: {exc}",
                                  level="error", code="separate.failed"))
            _emit(progress, "separate", 62.0, "分轨失败，继续用混合音频")
    else:
        warnings.append(_warn("separate", "options 关闭了分轨，和弦/歌词精度可能下降。", level="info", code="separate.disabled"))
        _emit(progress, "separate", 62.0, "已跳过分轨")

    # ---------------- 3. beats ----------------
    _emit(progress, "beats", 64.0, "检测节拍与拍号")
    started = stage_begin("beats")
    beats = beats_mod.analyze_beats(analysis_wav)
    warnings.extend(_normalize_warnings("beats", beats.get("warnings")))
    stage_end("beats", started, engine="librosa",
              extra={"bpm": beats.get("bpm"), "time_signature": beats.get("time_signature")})
    _emit(progress, "beats", 72.0, f"{beats.get('bpm')} BPM / {beats.get('time_signature')}")

    # ---------------- 4. key（先纯音频，后续用和弦证据复核） ----------------
    _emit(progress, "key", 73.0, "估计调性")
    started = stage_begin("key")
    key = keydet_mod.detect_key(analysis_wav)
    warnings.extend(_normalize_warnings("key", key.get("warnings")))
    stage_end("key", started, engine="krumhansl")
    _emit(progress, "key", 78.0, key.get("name") or f"{key.get('tonic')} {key.get('mode')}")

    # ---------------- 5. chords ----------------
    _emit(progress, "chords", 79.0, "识别和弦")
    started = stage_begin("chords")
    chords = chords_mod.recognize_chords(
        analysis_wav, beats, key=key, total_seconds=total_seconds,
        bass_wav=stems_paths.get("bass"),
    )
    stage_end("chords", started, engine="chroma-template+viterbi", extra={"count": len(chords)})
    if not chords:
        warnings.append(_warn("chords", "未识别到任何和弦。", level="warn", code="chords.no_chords"))
    elif len(chords) >= 3:
        mean_confidence = sum(float(item.get("confidence") or 0.0) for item in chords) / len(chords)
        if mean_confidence < _CHORDS_LOW_CONFIDENCE:
            warnings.append(_warn(
                "chords",
                f"和弦整体置信度偏低（平均 {mean_confidence:.2f} < {_CHORDS_LOW_CONFIDENCE:.2f}），"
                "转位/七和弦可能需要人工校对。",
                code="chords.low_confidence",
            ))
    _emit(progress, "chords", 90.0, f"识别到 {len(chords)} 个和弦事件")

    # ---------------- 5b. 调性复核（和弦证据能消除关系大小调歧义） ----------------
    if chords:
        started = stage_begin("key_refine")
        try:
            refined = keydet_mod.detect_key(analysis_wav, chords=chords)
            warnings.extend(_normalize_warnings("key_refine", refined.get("warnings")))
            if (refined.get("tonic_pc"), refined.get("mode")) != (key.get("tonic_pc"), key.get("mode")):
                warnings.append(_warn(
                    "key_refine",
                    f"调性经和弦证据修正：{key.get('name')} → {refined.get('name')}",
                    level="info", code="key.refined",
                ))
            key = refined
            stage_end("key_refine", started, engine="krumhansl+chord-functions")
        except Exception as exc:
            stage_end("key_refine", started, ok=False)
            warnings.append(_warn("key_refine", f"调性复核失败，沿用初判：{exc}", code="key.refine_failed"))

    # ---------------- 6. structure ----------------
    _emit(progress, "structure", 91.0, "切分歌曲结构")
    started = stage_begin("structure")
    try:
        sections = structure_mod.analyze_structure(analysis_wav, beats, chords, total_seconds)
    except Exception as exc:
        sections = structure_mod.fallback_layout(total_seconds, beats)
        warnings.append(_warn("structure", f"结构分析失败，使用通用曲式兜底：{type(exc).__name__}: {exc}",
                              level="error", code="structure.failed"))
    stage_end("structure", started, engine="ssm+novelty", extra={"count": len(sections)})
    if len(sections) <= 1:
        warnings.append(_warn("structure", "只切出 1 个段落，结构证据不足。", code="structure.degenerate"))
    coarse_sections = _find_coarse_sections(sections, beats)
    if coarse_sections:
        labels = "、".join(label for _start, _end, label in coarse_sections[:4])
        warnings.append(_warn(
            "structure",
            f"段落可能切得不够细：{labels} 超过 {_MAX_SECTION_BARS} 小节（谱面连续重复同一和弦进行，"
            "对弹唱无影响，仅提示结构信息可能不够细）。",
            level="info", code="structure.coarse",
        ))
    if _looks_uniform_structure(sections, chords, beats=beats):
        warnings.append(_warn(
            "structure",
            "全曲各段落的能量与和弦进行高度一致，未检测到明显段落边界"
            "（可能是整曲同一循环，或编曲缺少段间对比）；段落标注仅供参考。",
            level="info", code="structure.uniform",
        ))
    _emit(progress, "structure", 96.0, f"切出 {len(sections)} 个段落")

    # ---------------- 7. lyrics / melody（可选，队友实现） ----------------
    # 没有人声分轨时降级用整曲音频，保证歌词/旋律链不断（由下游自己写告警）。
    vocal_source = stems_paths.get("vocals")
    if not vocal_source:
        vocal_source = str(analysis_wav)
        warnings.append(_warn(
            "lyrics",
            "没有可用的 vocals 分轨，改用整曲音频做人声识别，歌词与旋律准确率会下降。",
            level="info", code="lyrics.vocals_fallback_mix",
        ))
    _emit(progress, "lyrics", 96.5, "识别歌词与旋律")
    started = stage_begin("lyrics")
    lyric_language = str(options.get("language") or "auto")
    # 模型大小通过环境变量传导（音频层不直接依赖 ASR 实现细节）：
    # CLI/API 传 whisper_model 时临时覆盖，跑完恢复，避免污染同进程的其它任务。
    whisper_model = str(options.get("whisper_model") or "").strip()
    previous_model = os.environ.get("AITSM_WHISPER_MODEL")
    if whisper_model:
        os.environ["AITSM_WHISPER_MODEL"] = whisper_model
    skip_asr = bool(options.get("skip_lyrics")) or os.environ.get("AITSM_SKIP_ASR", "").lower() in {"1", "true", "yes"}
    if skip_asr:
        # ASR 引擎不可用（例如模型需要联网下载）时，重试会白等好几分钟，
        # 这里给调试/离线场景一个明确的跳过开关。
        warnings.append(_warn("lyrics", "按配置跳过歌词识别（skip_lyrics/AITSM_SKIP_ASR）。",
                              level="info", code="lyrics.skipped"))
        lyrics = {"available": False, "lines": [], "engine": "none", "language": lyric_language}
    else:
        try:
            lyrics = _transcribe_lyrics(vocal_source, total_seconds, warnings, language=lyric_language)
        finally:
            if whisper_model:
                if previous_model is None:
                    os.environ.pop("AITSM_WHISPER_MODEL", None)
                else:
                    os.environ["AITSM_WHISPER_MODEL"] = previous_model
    stage_end("lyrics", started,
              engine=(lyrics or {}).get("engine", "none"),
              ok=bool(lyrics and lyrics.get("lines")))
    started = stage_begin("melody")
    melody = _transcribe_melody(vocal_source, beats, total_seconds, lyrics, warnings)
    stage_end("melody", started,
              engine=(melody or {}).get("engine", "none"),
              ok=bool(melody and melody.get("notes")))
    _emit(progress, "melody", 98.0, "歌词与旋律处理完成")

    # ---------------- 8. 组装 analysis ----------------
    _emit(progress, "assemble", 99.0, "组装并校验 analysis.json")
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA_ID,
        "created_at": _now_iso(),
        "source": {
            "original_name": ingested["original_name"],
            "sha256": ingested["sha256"],
            "path": str(Path(source).resolve()),
            "size_bytes": ingested["size_bytes"],
            "source_duration": ingested["source_duration"],
            "start_seconds": ingested.get("start_seconds", 0.0),
        },
        "audio": {
            "analysis_wav": analysis_wav,
            "playback_audio": ingested["playback_wav"],
            "mix_wav": ingested.get("mix_wav"),
            "duration": round(total_seconds, 4),
            "sample_rate": ingested["sample_rate"],
            "channels": ingested["channels"],
            "playback_codec": ingested.get("playback_codec"),
            "probe": ingested.get("probe"),
        },
        "stems": stems_info,
        "beats": beats,
        "key": key,
        "chords": chords,
        "sections": sections,
        # 契约（melody-lyrics 已确认）：lyrics 是数组，melody 是对象。
        "lyrics": (lyrics or {}).get("lines") or [],
        "melody": melody if melody is not None else {"available": False, "notes": [], "warnings": []},
        "tools": {
            **_tool_versions(),
            **({"demucs_model": stems_info.get("model")} if stems_info.get("model") else {}),
            "lyrics": {
                "available": bool(lyrics and lyrics.get("lines")),
                "engine": (lyrics or {}).get("engine", "none"),
                # 型号与设备决定了 CER 的量级（small 0.64 / medium 0.50），必须落盘
                "model": (lyrics or {}).get("model"),
                "device": (lyrics or {}).get("device"),
                "line_count": len((lyrics or {}).get("lines") or []),
            },
            "melody": {
                "available": bool(melody and melody.get("notes")),
                "engine": (melody or {}).get("engine", "none"),
                "note_count": len((melody or {}).get("notes") or []),
            },
        },
        "warnings": warnings,
        "meta": {
            "total_elapsed_seconds": None,
            "stage_timings": timings,
            "options": options,
            "engines": {
                "separation": stems_info.get("engine"),
                "beats": "librosa.beat",
                "key": "krumhansl+chords",
                "chords": "chroma-template+viterbi",
                "structure": "ssm+novelty",
            },
        },
    }

    # 分析质量自检（score-engine 的 quality 字段与前端质量面板都用这一份口径）
    analysis["quality"] = assess_analysis_quality(analysis)
    analysis["meta"]["field_notes"] = {
        "chords[].bars": "该和弦事件持续了几个小节（按 BPM 与拍号估算，四舍五入到 0.01 小节）。",
        "chords[].bar_aligned": "和弦起点是否落在小节线上（容差 8% 小节长度）。"
                                "false 表示和弦从反拍/半小节开始，属于正常现象，不是识别错误，"
                                "score-engine 不应据此降级。",
        "quality": "assess_analysis_quality 的自检结果：overall 0..1，stages 为各阶段分，"
                   "issues 为需要人工关注的点（level 为 info/warn/error）。",
    }

    schema_errors = validate_analysis(analysis)
    if schema_errors:
        warnings.append(_warn(
            "assemble",
            f"analysis 未通过 schema 校验（{len(schema_errors)} 处）：" + "；".join(schema_errors[:5]),
            level="error", code="assemble.schema_invalid",
        ))
        analysis["warnings"] = warnings

    total_elapsed = round(time.monotonic() - pipeline_started, 3)
    analysis["meta"]["total_elapsed_seconds"] = total_elapsed

    analysis_path = work / "analysis.json"
    _write_json(analysis_path, analysis)
    analysis_sha = media.sha256_file(analysis_path)

    # ---------------- 9. manifest ----------------
    manifest = {
        "schema": MANIFEST_SCHEMA_ID,
        "created_at": _now_iso(),
        "input": {
            "path": str(Path(source).resolve()),
            "original_name": ingested["original_name"],
            "sha256": ingested["sha256"],
            "size_bytes": ingested["size_bytes"],
            "duration": ingested["source_duration"],
            "start_seconds": ingested.get("start_seconds", 0.0),
            "analyzed_duration": total_seconds,
        },
        "options": options,
        "stages": stage_records,
        "stage_timings": timings,
        "degraded": [
            {"stage": item.get("stage"), "level": item.get("level"),
             "code": item.get("code"), "message": item.get("message")}
            for item in warnings if item.get("level") in ("warn", "error")
        ],
        "outputs": {
            "analysis": str(analysis_path),
            "analysis_sha256": analysis_sha,
            "manifest": str(work / "manifest.json"),
            "playback_audio": ingested["playback_wav"],
            "stems": stems_paths,
        },
        "tools": analysis["tools"],
        "quality": analysis.get("quality"),
        "schema_validation": {"valid": not schema_errors, "errors": schema_errors[:20]},
        "total_elapsed_seconds": total_elapsed,
    }
    manifest_path = work / "manifest.json"
    _write_json(manifest_path, manifest)

    analysis["meta"]["analysis_json"] = str(analysis_path)
    analysis["meta"]["manifest_json"] = str(manifest_path)
    _emit(progress, "done", 100.0, "分析完成")
    return analysis


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 调试入口
# --------------------------------------------------------------------------- #

def _print_summary(analysis: dict[str, Any]) -> None:
    beats = analysis.get("beats") or {}
    key = analysis.get("key") or {}
    chords = analysis.get("chords") or []
    sections = analysis.get("sections") or []
    print("=" * 78)
    print(f"文件：{analysis['source']['original_name']}")
    print(f"时长：{analysis['audio']['duration']:.2f}s  分析耗时：{analysis['meta']['total_elapsed_seconds']:.1f}s")
    print(f"速度：{beats.get('bpm')} BPM（显示 {beats.get('bpm_display')}） 拍号：{beats.get('time_signature')} "
          f"拍点 {len(beats.get('beats') or [])} 个 置信度 {beats.get('confidence')}")
    print(f"调性：{key.get('name')}（{key.get('tonic')} {key.get('mode')}，调号 {key.get('num_sharps_flats')}，"
          f"置信度 {key.get('confidence')}）")
    print(f"和弦：{len(chords)} 个事件，前 8 个：")
    for chord in chords[:8]:
        print(f"  {chord['start']:7.2f} - {chord['end']:7.2f}  {chord['symbol']:<10} conf={chord['confidence']}")
    print(f"结构：{len(sections)} 段")
    for section in sections:
        print(f"  {section['start']:7.2f} - {section['end']:7.2f}  {section['label']:<8} "
              f"kind={section['kind']:<10} energy={section['energy']:.2f} repeat_of={section['repeat_of']}")
    stems = analysis.get("stems") or {}
    engine = stems.get("engine") or "none"
    detail = "，".join(str(value) for value in (stems.get("model"), stems.get("device")) if value)
    print(f"分轨：{engine}" + (f"（{detail}）" if detail else "（未启用）"))
    quality = analysis.get("quality") or {}
    if quality.get("stages"):
        stage_text = " ".join(f"{name}={value}" for name, value in quality["stages"].items())
        print(f"质量自检：overall={quality.get('overall')}  {stage_text}")
        # 与 warnings 里已出现的同 code 项去重，避免 CLI 重复刷同一句话
        reported = {warning.get("code") for warning in (analysis.get("warnings") or [])}
        for issue in (quality.get("issues") or [])[:8]:
            if issue.get("code") in reported:
                continue
            print(f"  [{issue.get('level')}] {issue.get('code')}: {issue.get('message')} → {issue.get('target')}")
    warning_list = analysis.get("warnings") or []
    if warning_list:
        print(f"警告 {len(warning_list)} 条：")
        for warning in warning_list[:12]:
            print(f"  [{warning.get('level')}] {warning.get('code')} ({warning.get('stage')}): {warning.get('message')}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-To-Sheet-Music 音频分析流水线（调试入口）")
    parser.add_argument("source", help="输入音频/视频文件")
    parser.add_argument("--out", default="work/analysis", help="输出目录（默认 work/analysis）")
    parser.add_argument("--start", type=float, default=0.0, help="起始秒数（截取片段）")
    parser.add_argument("--duration", type=float, default=0.0, help="分析时长（秒，0 表示整首）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="分轨设备")
    parser.add_argument("--no-separate", action="store_true", help="跳过分轨（快速链路验证）")
    parser.add_argument("--json", action="store_true", help="只输出 analysis.json 路径")
    args = parser.parse_args(argv)

    options = {
        "start_sec": args.start,
        "duration_sec": args.duration if args.duration > 0 else None,
        "device": args.device,
        "separate": not args.no_separate,
    }

    def progress(stage: str, percent: float, message: str) -> None:
        print(f"[{percent:6.1f}%] {stage:<10} {message}", flush=True)

    try:
        analysis = run_pipeline(Path(args.source), Path(args.out), options, progress=progress)
    except Exception:
        traceback.print_exc()
        return 1
    if args.json:
        print(analysis["meta"]["analysis_json"])
    else:
        _print_summary(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

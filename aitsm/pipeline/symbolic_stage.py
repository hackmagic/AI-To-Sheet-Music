"""v2 符号契约阶段：把 legacy ``analysis.json`` 收敛成 :class:`SymbolicScore`。

架构上这是一层**过渡适配**：VocalParse / SheetSage2 权重就位后，
:mod:`aitsm.transcribe` 的适配器会直接产出 ``VocalResult`` / ``AccompanimentResult``，
本模块的 ``*_from_analysis`` 就不再被调用（但 fusion / arrangement / render 的输入
形状始终不变）。

设计约束：

- 本阶段**不重复任何 DSP**，只做形状转换与 provenance 记录；
- fusion 不可用（例如 MusicXML 写盘未就绪）时降级为 ``SymbolicScore.from_analysis``，
  符号契约仍然落盘，绝不让主流水线失败；
- 量化失败只降级，不改写音符总数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from aitsm.contracts import (
    AccompanimentResult,
    SymbolicScore,
    VocalResult,
)

__all__ = ["build_symbolic", "accompaniment_from_analysis", "vocal_from_analysis", "build_layers"]

#: 架构首选引擎（落进 provenance，与实际 engine 对比即知降级）
REQUESTED_VOCAL = "VocalParse"
REQUESTED_ACCOMPANIMENT = "SheetSage2"


def _tools(analysis: Mapping[str, Any]) -> dict[str, Any]:
    tools = analysis.get("tools") or {}
    return tools if isinstance(tools, dict) else {}


def _engines(analysis: Mapping[str, Any]) -> dict[str, Any]:
    meta = analysis.get("meta") or {}
    engines = meta.get("engines") or {}
    return engines if isinstance(engines, dict) else {}


def vocal_from_analysis(analysis: Mapping[str, Any]) -> VocalResult:
    """把 analysis 的歌词 + 旋律收成 :class:`VocalResult`（假装是 VocalParse 产出）。"""
    score = SymbolicScore.from_analysis(analysis)
    tools = _tools(analysis)
    melody_tool = tools.get("melody") or {}
    lyrics_tool = tools.get("lyrics") or {}
    engine = str(melody_tool.get("engine") or lyrics_tool.get("engine") or "none")
    warnings: list[str] = []
    if engine not in {"vocalparse", "VocalParse"}:
        warnings.append(
            f"架构首选 {REQUESTED_VOCAL}（联合解码歌词+音高+时值），本机实际使用 {engine}；"
            "歌词与音高是两个独立模型的结果，字-音对齐精度低于联合解码。"
        )
    if not score.melody:
        warnings.append("没有旋律音：VocalParse 与降级链路都未产出音符。")
    melody_raw = analysis.get("melody") or {}
    alignment = (melody_raw.get("alignment") if isinstance(melody_raw, dict) else None) or {}
    return VocalResult(
        notes=list(score.melody),
        lyrics=list(score.lyrics),
        bpm=score.beats.bpm,
        engine=engine,
        requested=REQUESTED_VOCAL,
        alignment=dict(alignment) if isinstance(alignment, dict) else {},
        warnings=warnings,
    )


def accompaniment_from_analysis(analysis: Mapping[str, Any]) -> AccompanimentResult:
    """把 analysis 的节拍/调性/和弦/段落收成 :class:`AccompanimentResult`。

    引擎名必须**如实取自 analysis**：run.py 在走 SheetSage2 时会把
    ``tools.sheetsage2`` 写成 ``{"engine": "sheetsage2", ...}``，这里读它；
    读不到才退回 ``chroma-template+viterbi``。否则下游会把「真的跑了研究模型」
    误判成降级。
    """
    score = SymbolicScore.from_analysis(analysis)
    tools = _tools(analysis)
    sage = tools.get("sheetsage2") if isinstance(tools, dict) else None
    sage_engine = str(sage.get("engine") or "").strip() if isinstance(sage, Mapping) else ""
    engines = _engines(analysis)
    if sage_engine and sage_engine != "none":
        engine = sage_engine
        warnings: list[str] = []
    else:
        engine = str(engines.get("chords") or "beats+chroma")
        warnings = [
            f"架构首选 {REQUESTED_ACCOMPANIMENT}（联合解码旋律/和弦/节拍/调性/段落），"
            f"本机实际使用 {engine} + librosa.beat + krumhansl；节拍与和弦是分阶段独立估计，"
            "段落边界与终止式不会互相印证。"
        ]
    if not score.sections:
        warnings.append("没有段落标签：编配层只能按全曲统一风格处理。")
    return AccompanimentResult(
        beats=score.beats,
        key=score.key,
        chords=list(score.chords),
        sections=list(score.sections),
        melody_skeleton=[],
        events=[
            {"kind": "beat", "time": value} for value in score.beats.beats[:1]
        ],
        engine=engine,
        requested=REQUESTED_ACCOMPANIMENT,
        warnings=warnings,
    )


def build_layers(analysis: Mapping[str, Any]) -> tuple[VocalResult, AccompanimentResult]:
    return vocal_from_analysis(analysis), accompaniment_from_analysis(analysis)


def _fuse(vocal: VocalResult, accompaniment: AccompanimentResult,
          analysis: Mapping[str, Any], warnings: list[str]) -> SymbolicScore | None:
    try:
        from aitsm.fusion import skeleton
    except Exception as exc:  # pragma: no cover - fusion 缺席时的兜底
        warnings.append(f"fusion 层不可用（{type(exc).__name__}: {exc}），改用契约直转。")
        return None
    try:
        return skeleton.fuse(
            vocal,
            accompaniment,
            meta=dict(analysis.get("meta") or {}),
            audio=dict(analysis.get("audio") or {}),
            source=dict(analysis.get("source") or {}),
        )
    except Exception as exc:
        warnings.append(f"fusion.fuse 失败（{type(exc).__name__}: {exc}），改用契约直转。")
        return None


def _quantize(score: SymbolicScore, work: Path, warnings: list[str]) -> SymbolicScore:
    try:
        from aitsm.fusion import quantize as quantize_mod
    except Exception as exc:  # pragma: no cover
        warnings.append(f"quantize 层不可用（{type(exc).__name__}: {exc}），保留原始时值。")
        return score
    try:
        quantized, provenance = quantize_mod.quantize_symbolic(score, work_dir=work)
    except Exception as exc:
        warnings.append(f"量化失败（{type(exc).__name__}: {exc}），保留原始时值。")
        return score
    score.provenance.append(provenance)
    return quantized


def build_symbolic(
    analysis: Mapping[str, Any],
    work_dir: str | Path,
    *,
    quantize: bool = True,
) -> dict[str, Any]:
    """构造并落盘符号契约。返回写进 ``analysis["symbolic"]`` / manifest 的摘要。"""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    vocal, accompaniment = build_layers(analysis)
    score = _fuse(vocal, accompaniment, analysis, warnings)
    if score is None:
        score = SymbolicScore.from_analysis(analysis)
        # 契约直转也要如实记录「哪一层是降级过的」
        from aitsm.contracts import Provenance

        score.provenance.insert(0, Provenance(
            layer="fusion",
            engine="contract-direct",
            requested="fusion.skeleton",
            ok=False,
            detail={"reason": "skeleton.fuse unavailable"},
            warnings=list(warnings),
        ))
    if quantize:
        score = _quantize(score, work, warnings)

    paths: dict[str, Any] = {}
    try:
        saved = score.save(work)
        if isinstance(saved, dict):
            paths.update({key: str(value) for key, value in saved.items() if value})
    except Exception as exc:
        warnings.append(f"符号契约落盘失败（{type(exc).__name__}: {exc}）。")
        try:
            json_path = work / "symbolic.json"
            json_path.write_text(score.to_json(), encoding="utf-8")
            paths["json"] = str(json_path)
        except Exception:  # pragma: no cover - 磁盘问题无能为力
            pass

    stats: dict[str, Any] = {}
    try:
        stats = dict(score.stats())
    except Exception:  # pragma: no cover
        stats = {"notes": len(score.melody), "chords": len(score.chords)}

    result: dict[str, Any] = {
        "available": bool(score.melody or score.chords),
        "schema": score.schema,
        "stats": stats,
        "paths": paths,
        "provenance": [item.to_dict() for item in score.provenance],
        "warnings": warnings,
    }
    result["degraded_layers"] = [
        item.layer for item in score.provenance if _is_degraded(item)
    ]
    return result


def _is_degraded(item: Any) -> bool:
    """按「架构首选 vs 实际实现」判定降级，而不是简单比较字符串。

    ``fusion`` 层的 ``requested`` 是能力名（fusion），引擎名天然不同，
    不能算降级；其余各层的可用别名列在 ``_ENGINE_ALIASES``。
    """
    if not bool(getattr(item, "ok", True)):
        return True
    layer = str(getattr(item, "layer", "") or "")
    if layer == "fusion":
        return False
    requested = str(getattr(item, "requested", "") or "").strip().lower()
    engine = str(getattr(item, "engine", "") or "").strip().lower()
    if not requested or not engine:
        return False
    allowed = _ENGINE_ALIASES.get(requested)
    if allowed is not None:
        return engine not in allowed
    return engine != requested


#: 「架构首选引擎名」→ 视为不降级的实际引擎名集合
_ENGINE_ALIASES: dict[str, set[str]] = {
    "htdemucs_ft": {"htdemucs_ft", "demucs-onnx"},
    "vocalparse": {"vocalparse"},
    "sheetsage2": {"sheetsage2"},
    "duet": {"duet"},
}

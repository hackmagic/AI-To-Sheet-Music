"""节奏量化层：优先 DuET，本地不可用时显式降级到 metrical HMM。

设计原则
--------
* **能用才用**：DuET 需要 ``transformers`` + 本地权重目录（``AITSM_DUET_DIR``）
  + 适配器模块 ``aitsm.fusion.duet``；缺任何一样都直接降级，不偷偷尝试联网。
* **绝不假装成功**：``Provenance.requested`` 恒为 ``"duet"``，``engine`` 写实际
  生效的实现（``"duet"`` / ``"hmm"``）；降级原因进 ``detail["fallback_reason"]``
  和 ``SymbolicScore.warnings``（``code="duet_unavailable"``）。
* **结果必须可记谱**：HMM 输出的时值是网格步长的整数倍，不保证落在
  :func:`aitsm.pipeline.quantize.legal_durations` 里，所以再跑一遍「合法时值」
  归一化（最小改动，绝不移动音头）。

重模块（``transformers`` / ``aitsm.pipeline.quantize``）全部惰性导入。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Mapping

from ..contracts import MelodyNote, Provenance, SymbolicScore, merge_warnings

__all__ = [
    "DEFAULT_DEVICE",
    "DUET_ADAPTER_MODULE",
    "DUET_ENV",
    "ENGINE_DUET",
    "ENGINE_HMM",
    "is_available",
    "probe",
    "quantize_symbolic",
]

DEFAULT_DEVICE = "auto"
DUET_ENV = "AITSM_DUET_DIR"
DUET_ADAPTER_MODULE = "aitsm.fusion.duet"
ENGINE_DUET = "duet"
ENGINE_HMM = "hmm"

#: 要求的量化层首选引擎（写进 provenance.requested，降级时保持不变）。
REQUESTED_ENGINE = ENGINE_DUET

#: 每个四分音符的 HMM 网格数（4 = 十六分）。
_HMM_SUBDIVISION = 4

#: 判断「时值合法」时允许的浮点误差（四分音符单位）。
_LEGAL_EPS = 1e-6


# --------------------------------------------------------------------------- #
# DuET 可用性探测
# --------------------------------------------------------------------------- #


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _resolve_device(device: str | None) -> str:
    """把 ``"auto"`` 解析成实际设备名；没有 CUDA 就是 CPU。"""
    requested = str(device or DEFAULT_DEVICE).strip().lower()
    if requested in ("", "auto"):
        return "cpu"
    if requested.startswith("cuda"):
        try:
            import torch  # noqa: F401  （惰性：只有用户显式要 CUDA 才碰 torch）

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:  # pragma: no cover - torch 存在但初始化失败
            return "cpu"
        return requested
    return requested


def probe(device: str | None = DEFAULT_DEVICE) -> dict[str, Any]:
    """检查 DuET 是否真的可用（不发网络请求、不加载权重）。

    返回 ``{"available": bool, "reason": str, "checks": {...}, "device": str}``。
    """
    duet_dir = str(os.environ.get(DUET_ENV) or "").strip()
    checks: dict[str, Any] = {
        "transformers": _find_spec("transformers"),
        "duet_dir": duet_dir,
        "duet_dir_exists": bool(duet_dir) and Path(duet_dir).expanduser().is_dir(),
        "adapter": _find_spec(DUET_ADAPTER_MODULE),
        "device": _resolve_device(device),
    }

    reason = ""
    if not checks["transformers"]:
        reason = (
            "未安装 transformers（本机离线，不能 pip install / 下载权重），"
            "DuET 不可用，已降级到 metrical HMM"
        )
    elif not checks["duet_dir"]:
        reason = f"未配置 {DUET_ENV}（DuET 本地权重目录），无法离线加载模型，已降级到 metrical HMM"
    elif not checks["duet_dir_exists"]:
        reason = f"{DUET_ENV}={duet_dir} 不存在，DuET 权重缺失，已降级到 metrical HMM"
    elif not checks["adapter"]:
        reason = f"缺少 DuET 适配器 {DUET_ADAPTER_MODULE}（需提供 quantize_melody 入口），已降级到 metrical HMM"

    checks["available"] = not reason
    checks["reason"] = reason
    return checks


def is_available(device: str | None = DEFAULT_DEVICE) -> bool:
    """DuET 是否可用（True 才会走模型路径）。"""
    return bool(probe(device)["available"])


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def quantize_symbolic(
    score: SymbolicScore,
    *,
    work_dir: str | Path | None = None,
    device: str | None = DEFAULT_DEVICE,
) -> tuple[SymbolicScore, Provenance]:
    """把总谱的旋律音量化到节拍网格与合法时值。

    参数:
        score: 融合后的总谱（不会原地修改，返回新的 :class:`SymbolicScore`）。
        work_dir: DuET 适配器可用的临时目录（HMM 路径不使用）。
        device: ``"auto"`` / ``"cpu"`` / ``"cuda"``。

    返回:
        ``(new_score, provenance)``；``provenance.layer == "quantization"``，
        ``requested`` 恒为 ``"duet"``，``engine`` 是实际生效的实现。
    """
    if not isinstance(score, SymbolicScore):
        raise TypeError(f"quantize_symbolic 需要 SymbolicScore，收到 {type(score).__name__}")

    started = time.perf_counter()
    state = probe(device)
    resolved_device = str(state["device"])
    warnings: list[dict[str, Any]] = []

    out = SymbolicScore.from_dict(score.to_dict())
    engine = ENGINE_HMM
    detail: dict[str, Any] = {
        "device": resolved_device,
        "requested_engine": REQUESTED_ENGINE,
        "subdivision": _HMM_SUBDIVISION,
    }

    if state["available"]:
        try:
            engine, quantized, model_detail = _quantize_with_duet(
                out, work_dir=work_dir, device=resolved_device
            )
            detail.update(model_detail)
        except Exception as exc:  # 模型侧任何异常都不许炸掉整条流水线
            engine = ENGINE_HMM
            reason = f"DuET 推理失败（{type(exc).__name__}: {exc}），已降级到 metrical HMM"
            detail["fallback_reason"] = reason
            warnings.append(_warn(reason, code="duet_failed"))
            detail.update(
                _quantize_with_hmm(out)
            )
        else:
            out.melody = quantized
    else:
        detail["fallback_reason"] = state["reason"]
        warnings.append(_warn(state["reason"], code="duet_unavailable", level="info"))
        detail.update(_quantize_with_hmm(out))

    elapsed = time.perf_counter() - started
    legal_ratio = _legal_duration_ratio(out.melody, float(out.beats.beat_period))
    detail.setdefault("legal_duration_ratio", round(legal_ratio, 4))
    detail["notes"] = len(out.melody)
    detail["engine"] = engine

    provenance = Provenance(
        layer="quantization",
        engine=engine,
        requested=REQUESTED_ENGINE,
        ok=True,
        elapsed_seconds=elapsed,
        detail=_clean_detail(detail),
        warnings=[item["message"] for item in warnings if item.get("level") in ("warn", "error")],
    )

    out.provenance = [
        item for item in out.provenance if str(item.layer) != "quantization"
    ] + [provenance]
    out.warnings = merge_warnings(out.warnings, warnings)
    out.source["quantizer"] = engine
    out.meta["quantization_engine"] = engine
    out.meta["quantization_legal_ratio"] = round(legal_ratio, 4)
    out.meta["quantization_degraded"] = engine != REQUESTED_ENGINE
    return out, provenance


# --------------------------------------------------------------------------- #
# DuET 路径
# --------------------------------------------------------------------------- #


def _quantize_with_duet(
    score: SymbolicScore,
    *,
    work_dir: str | Path | None,
    device: str,
) -> tuple[str, list[MelodyNote], dict[str, Any]]:
    """调用本地 DuET 适配器。只有 :func:`probe` 通过才会走到这里。"""
    module = importlib.import_module(DUET_ADAPTER_MODULE)
    entry = getattr(module, "quantize_melody", None)
    if not callable(entry):  # pragma: no cover - probe 已保证有适配器
        raise RuntimeError(f"{DUET_ADAPTER_MODULE} 缺少 quantize_melody()")

    raw = entry(
        notes=[
            {
                "start": float(note.start),
                "end": float(note.end),
                "midi": int(note.midi),
                "confidence": float(note.confidence),
                "lyric": str(note.lyric or ""),
            }
            for note in score.melody
        ],
        beats=score.beats.to_dict(),
        key=score.key.to_dict(),
        work_dir=str(work_dir) if work_dir else None,
        device=device,
        duet_dir=str(os.environ.get(DUET_ENV) or ""),
    )
    if not isinstance(raw, (list, tuple)):
        raise ValueError("DuET 适配器返回值必须是音符列表")

    quantized: list[MelodyNote] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("DuET 适配器返回的元素必须是 dict")
        original = score.melody[index] if index < len(score.melody) else None
        start = float(item.get("start", getattr(original, "start", 0.0)))
        end = max(start, float(item.get("end", start)))
        midi = int(item.get("midi", getattr(original, "midi", 60)))
        quantized.append(
            MelodyNote(
                start=start,
                end=end,
                midi=max(0, min(127, midi)),
                confidence=float(item.get("confidence", getattr(original, "confidence", 0.0))),
                lyric=str(item.get("lyric", getattr(original, "lyric", "") or "")),
                syllable_index=getattr(original, "syllable_index", None),
                source="duet",
            )
        )
    quantized = _legalize(quantized, float(score.beats.beat_period))
    return ENGINE_DUET, quantized, {"adapter": DUET_ADAPTER_MODULE}


# --------------------------------------------------------------------------- #
# HMM 路径
# --------------------------------------------------------------------------- #


def _quantize_with_hmm(score: SymbolicScore) -> dict[str, Any]:
    """用 :func:`aitsm.pipeline.quantize.quantize_notes_hmm` 量化旋律。

    原地更新 ``score.melody``，返回可直接并进 provenance.detail 的统计。
    """
    if not score.melody:
        return {"stats": {"notes": 0}, "snapped": 0, "grid_points": 0}

    from ..pipeline.quantize import quantize_notes_hmm  # 惰性：会拉 numpy

    beat_period = float(score.beats.beat_period)
    until = max(float(score.duration), max(float(note.end) for note in score.melody) + beat_period)

    payload = [
        {
            "start": float(note.start),
            "end": float(note.end),
            "midi": int(note.midi),
            "confidence": float(note.confidence),
            "lyric": str(note.lyric or ""),
            "syllable_index": note.syllable_index,
            "source": str(note.source or "hmm"),
        }
        for note in score.melody
    ]
    horizon = [float(chord.start) for chord in score.chords]

    try:
        raw_notes, stats = quantize_notes_hmm(
            payload,
            score.beats.to_dict(),
            harmony_onsets=horizon or None,
            until=until,
            subdivision=_HMM_SUBDIVISION,
        )
    except Exception as exc:  # HMM 也不许炸掉流水线：原样保留并显式告警
        return {
            "stats": {"notes": len(payload), "engine": "none"},
            "snapped": 0,
            "grid_points": 0,
            "quantizer_error": f"{type(exc).__name__}: {exc}",
        }

    quantized: list[MelodyNote] = []
    for index, note in enumerate(score.melody):
        entry = raw_notes[index] if index < len(raw_notes) else payload[index]
        start = float(entry.get("start", payload[index]["start"]))
        end = float(entry.get("end", payload[index]["end"]))
        quantized.append(
            MelodyNote(
                start=start,
                end=max(start, end),
                midi=int(note.midi),
                confidence=float(note.confidence),
                lyric=str(note.lyric or ""),
                syllable_index=note.syllable_index,
                source=ENGINE_HMM,
            )
        )
    quantized = _legalize(quantized, beat_period)
    score.melody = sorted(quantized, key=lambda item: (item.start, item.midi))
    return {
        "stats": dict(stats),
        "snapped": int(stats.get("snapped", 0)),
        "kept_raw": int(stats.get("kept_raw", 0)),
        "grid_points": int(stats.get("grid_points", 0)),
        "harmony_anchored": int(stats.get("harmony_anchored", 0)),
        "rests": int(stats.get("rests", 0)),
        "mean_shift_ms": round(float(stats.get("mean_shift_ms", 0.0)), 3),
        "max_shift_ms": round(float(stats.get("max_shift_ms", 0.0)), 3),
    }


# --------------------------------------------------------------------------- #
# 合法时值归一化
# --------------------------------------------------------------------------- #


def _legal_values() -> tuple[float, ...]:
    from ..pipeline.quantize import legal_durations

    return tuple(sorted(float(item) for item in legal_durations()))


def _nearest_legal(value: float, legal: tuple[float, ...]) -> float:
    return min(legal, key=lambda item: (abs(item - value), -item))


def _legalize(notes: list[MelodyNote], beat_period: float) -> list[MelodyNote]:
    """把每个音的时值吸到最近的合法值（四分音符单位），音头不动。

    合法值来自 :func:`aitsm.pipeline.quantize.legal_durations`。若吸完之后会
    盖住下一个音，就退到「不超过剩余空间的合法值」；空间连最短合法时值都放不下
    时保留最短合法值（宁可轻微重叠，也不写出不可记谱的时值）。
    """
    if not notes:
        return []
    if beat_period <= 0:
        return list(notes)
    legal = _legal_values()
    shortest = legal[0]

    ordered = sorted(notes, key=lambda item: (item.start, item.midi))
    out: list[MelodyNote] = []
    for index, note in enumerate(ordered):
        start = float(note.start)
        duration_q = max(0.0, float(note.end) - start) / beat_period
        target = _nearest_legal(duration_q, legal)

        next_start = float(ordered[index + 1].start) if index + 1 < len(ordered) else None
        if next_start is not None and next_start > start:
            room_q = (next_start - start) / beat_period - 1e-6
            if room_q >= shortest:
                target = min(target, max(item for item in legal if item <= room_q))
            else:
                target = shortest

        out.append(
            MelodyNote(
                start=start,
                end=start + target * beat_period,
                midi=int(note.midi),
                confidence=float(note.confidence),
                lyric=str(note.lyric or ""),
                syllable_index=note.syllable_index,
                source=str(note.source or ""),
            )
        )
    return out


def _legal_duration_ratio(notes: list[MelodyNote], beat_period: float) -> float:
    """合法时值占比（1.0 = 全部可记谱）。"""
    if not notes or beat_period <= 0:
        return 0.0
    legal = _legal_values()
    legal_count = 0
    for note in notes:
        duration_q = max(0.0, float(note.end) - float(note.start)) / beat_period
        if any(abs(duration_q - item) <= 1e-4 for item in legal):
            legal_count += 1
    return legal_count / len(notes)


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #


def _warn(message: str, *, code: str = "", level: str = "warn") -> dict[str, Any]:
    data: dict[str, Any] = {"level": level, "message": str(message), "stage": "quantization"}
    if code:
        data["code"] = code
    return data


def _clean_detail(detail: Mapping[str, Any]) -> dict[str, Any]:
    """去掉 None / 空容器，避免 provenance detail 里塞噪声。"""
    cleaned: dict[str, Any] = {}
    for key, value in detail.items():
        if value is None or value == "" or value == {} or value == []:
            continue
        cleaned[str(key)] = value
    return cleaned

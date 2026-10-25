"""编配层：把 ``SymbolicScore`` 汇总成一份确定的编配决策。

职责边界
--------
- 输入只允许 :class:`aitsm.contracts.SymbolicScore` 或它的**纯数据摘要**
  （``to_dict()`` 形状 / 至少含 ``chords``、``sections``）；本模块不读音频、
  不 import ``aitsm.pipeline``；
- 指法由 DP 决定（:mod:`aitsm.arrangement.fingering_dp`），TART 只做重排
  （:mod:`aitsm.arrangement.tart_rerank`），节奏型由规则决定
  （:mod:`aitsm.arrangement.strumming_rules`）；
- 大模型顾问（:mod:`aitsm.arrangement.llm_advisor`）的意见只放进
  ``advisor`` 字段，**不覆盖**最终选择，保证同一输入离线/在线得到同一张谱面
  的确定性主干。

输出 :class:`ArrangementPlan` 是纯数据，可安全 JSON 化给后续渲染层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aitsm.contracts import BeatGrid, ChordEvent, KeySignature, MelodyNote, Section, SymbolicScore
from aitsm.score.model import STANDARD_TUNING

from .fingering_dp import (
    COST_WEIGHTS,
    VoicingChoice,
    optimize_progression,
    plan_cost,
)
from .llm_advisor import AdvisorResult, LLMAdvisor
from .strumming_rules import StrumDecision, recommend
from .tart_rerank import active_engine as tart_active_engine, rerank

__all__ = ["ArrangementPlan", "PLAN_SCHEMA", "build_plan", "arrange", "score_summary"]

#: 决策结构版本
PLAN_SCHEMA = "aitsm.arrangement/v1"


@dataclass
class ArrangementPlan:
    """编配决策：和弦指法 + 段落节奏型 + 顾问建议（建议不参与最终决策）。"""

    choices: list[VoicingChoice] = field(default_factory=list)
    strumming: list[StrumDecision] = field(default_factory=list)
    key: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0
    engines: dict[str, str] = field(default_factory=dict)
    advisor: dict[str, Any] = field(default_factory=lambda: {"engine": "off"})
    warnings: list[str] = field(default_factory=list)
    schema: str = PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key": dict(self.key),
            "meta": dict(self.meta),
            "choices": [item.to_dict() for item in self.choices],
            "strumming": [item.to_dict() for item in self.strumming],
            "total_cost": round(float(self.total_cost), 4),
            "engines": dict(self.engines),
            "advisor": self.advisor,
            "warnings": list(self.warnings),
        }


def _coerce_score(score: SymbolicScore | Mapping[str, Any]) -> SymbolicScore:
    """接受 SymbolicScore 或其纯数据摘要；不猜音频相关字段。"""
    if isinstance(score, SymbolicScore):
        return score
    if not isinstance(score, Mapping):
        raise TypeError("build_plan 只接受 SymbolicScore 或其纯数据摘要")
    data = dict(score)
    if data.get("beats"):
        return SymbolicScore.from_dict(data)
    # 纯数据摘要：只保留契约字段，节拍用中性占位（不读音频、不推断 BPM）
    meta = dict(data.get("meta") or {})
    time_signature = str(meta.get("time_signature") or "4/4")
    beats_per_bar = int(meta.get("beats_per_bar") or (time_signature.split("/")[0] or 4))
    beats = BeatGrid(
        bpm=float(meta.get("bpm") or 120.0),
        beats=[],
        downbeats=[],
        beats_per_bar=max(1, beats_per_bar),
        time_signature=time_signature,
    )
    chords = [
        item if isinstance(item, ChordEvent) else ChordEvent.from_dict(item)
        for item in (data.get("chords") or [])
    ]
    sections = [
        item if isinstance(item, Section) else Section.from_dict(item)
        for item in (data.get("sections") or [])
    ]
    melody = [
        item if isinstance(item, MelodyNote) else MelodyNote.from_dict(item)
        for item in (data.get("melody") or [])
    ]
    return SymbolicScore(
        beats=beats,
        key=KeySignature.from_dict(data.get("key")),
        chords=chords,
        sections=sections,
        melody=melody,
        meta=meta,
    )


def score_summary(score: SymbolicScore | Mapping[str, Any]) -> dict[str, Any]:
    """给日志/测试用的轻量摘要（不含 melody/audio 细节）。"""
    parsed = _coerce_score(score)
    return {
        "title": str(parsed.meta.get("title") or ""),
        "key": parsed.key.display,
        "time_signature": parsed.beats.time_signature,
        "bpm": round(float(parsed.beats.bpm), 3),
        "chord_count": len(parsed.chords),
        "section_count": len(parsed.sections),
        "melody_note_count": len(parsed.melody),
        "duration": round(parsed.duration(), 3),
    }


def _advisor_payload(
    advisor: LLMAdvisor,
    chords: Sequence[ChordEvent],
    choices: Sequence[VoicingChoice],
    sections: Sequence[Section],
    *,
    key: KeySignature,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """调用顾问的三类建议；任何失败都只是警告，不阻塞编配。"""
    voicing_result: AdvisorResult = advisor.suggest_voicings(chords, key=key, style=str(meta.get("style") or "pop"))
    review_result: AdvisorResult = advisor.review_plan(choices, key=key, meta=meta)
    strum_result: AdvisorResult = advisor.suggest_strumming(sections, meta)
    engines = {voicing_result.engine, review_result.engine, strum_result.engine}
    warnings = (
        list(voicing_result.warnings) + list(review_result.warnings) + list(strum_result.warnings)
    )
    return {
        "engine": "llm" if "llm" in engines else "rules",
        "advisory": True,
        "voicings": voicing_result.to_dict(),
        "review": review_result.to_dict(),
        "strumming": strum_result.to_dict(),
        "warnings": warnings,
    }


def build_plan(
    score: SymbolicScore | Mapping[str, Any],
    *,
    options: Mapping[str, Any] | None = None,
    advisor: LLMAdvisor | None = None,
    meta: Mapping[str, Any] | None = None,
    max_candidates: int = 8,
    weights: Mapping[str, float] | None = None,
    tuning: Sequence[int] = STANDARD_TUNING,
) -> ArrangementPlan:
    """生成确定性的编配决策。

    参数
    ----
    score:
        :class:`SymbolicScore` 或纯数据摘要（含 ``chords``/``sections`` 即可）。
    options:
        兼容旧契约的选项 dict，支持 ``style`` / ``difficulty`` /
        ``time_signature`` / ``bpm`` / ``title`` / ``max_candidates`` /
        ``weights`` / ``tuning`` / ``advisor``。关键字参数优先。
    advisor:
        可选的大模型顾问；为 ``None`` 时不发起任何 LLM 调用。顾问输出只放在
        ``plan.advisor`` 中供人工/后续流程参考。
    meta:
        覆盖项：``style`` / ``difficulty`` / ``time_signature`` / ``bpm``。
    weights、max_candidates、tuning:
        透传给 DP / 重排 / 候选池。

    返回
    ----
    :class:`ArrangementPlan`；空输入不会抛异常，只会在 ``warnings`` 里记录。
    """
    parsed = _coerce_score(score)
    option_map = dict(options or {})
    if advisor is None and isinstance(option_map.get("advisor"), LLMAdvisor):
        advisor = option_map["advisor"]
    if "max_candidates" in option_map:
        try:
            max_candidates = int(option_map["max_candidates"])
        except (TypeError, ValueError):
            pass
    if "weights" in option_map and weights is None:
        weights = option_map["weights"]
    if "tuning" in option_map and tuning is STANDARD_TUNING:
        tuning = option_map["tuning"]

    meta_map = dict(parsed.meta or {})
    for key in ("style", "difficulty", "time_signature", "bpm", "title"):
        if key in option_map and option_map[key] is not None:
            meta_map[key] = option_map[key]
    meta_map.update({str(k): v for k, v in dict(meta or {}).items()})
    if "time_signature" not in meta_map:
        meta_map["time_signature"] = parsed.beats.time_signature
    if "style" not in meta_map:
        meta_map["style"] = "pop"
    if "difficulty" not in meta_map:
        meta_map["difficulty"] = "intermediate"

    weight_map = dict(COST_WEIGHTS)
    override = dict(weights or {})
    weight_map.update({k: float(v) for k, v in override.items() if k in weight_map})

    warnings: list[str] = []
    engines: dict[str, str] = {"fingering": "dp", "rerank": tart_active_engine(), "strumming": "rules"}

    try:
        choices = optimize_progression(
            parsed.chords,
            tuning=tuning,
            melody_notes=parsed.melody,
            key=parsed.key,
            max_candidates=max_candidates,
            weights=weight_map,
        )
    except Exception as exc:  # DP 自身出问题也不能让整个编配崩掉
        choices = []
        warnings.append(f"指法 DP 失败（{type(exc).__name__}: {exc}）")

    if choices and len(choices) == len(parsed.chords):
        try:
            choices = rerank(
                parsed.chords,
                choices,
                tuning=tuning,
                max_candidates=max_candidates,
                melody_notes=parsed.melody,
                weights=weight_map,
            )
            if choices:
                engines["rerank"] = str(choices[0].engine or engines["rerank"])
        except Exception as exc:  # 重排失败保留 DP 结果
            warnings.append(f"TART 重排失败（{type(exc).__name__}: {exc}），保留 DP 结果")

    if not parsed.chords:
        warnings.append("谱面没有和弦，编配结果为空")
    unplayable = [choice for choice in choices if not choice.frets]
    if unplayable:
        warnings.append(f"{len(unplayable)} 个和弦没有合法指法，已用空指法占位")

    strumming: list[StrumDecision] = []
    if parsed.sections:
        try:
            strumming = recommend(parsed.sections, meta_map)
        except Exception as exc:  # 节奏型选型失败只降级为空
            warnings.append(f"节奏型选型失败（{type(exc).__name__}: {exc}）")
    else:
        warnings.append("谱面没有段落信息，节奏型为空")

    advisor_payload: dict[str, Any] = {"engine": "off"}
    if advisor is not None:
        try:
            advisor_payload = _advisor_payload(
                advisor, parsed.chords, choices, parsed.sections, key=parsed.key, meta=meta_map
            )
            warnings.extend(advisor_payload.get("warnings") or [])
        except Exception as exc:  # 顾问层承诺不抛，这里再兜一层
            advisor_payload = {"engine": "rules", "advisory": True, "error": str(exc)}
            warnings.append(f"顾问调用失败（{type(exc).__name__}: {exc}）")

    try:
        total = plan_cost(choices, weights=weight_map)
    except Exception:
        total = 0.0

    return ArrangementPlan(
        choices=choices,
        strumming=strumming,
        key=parsed.key.to_dict(),
        meta={
            "style": meta_map.get("style", "pop"),
            "difficulty": meta_map.get("difficulty", "intermediate"),
            "time_signature": meta_map.get("time_signature", "4/4"),
            "bpm": round(float(meta_map.get("bpm") or parsed.beats.bpm), 3),
            "title": str(meta_map.get("title") or ""),
            "max_candidates": int(max_candidates),
            "weights": dict(weight_map),
        },
        total_cost=total,
        engines=engines,
        advisor=advisor_payload,
        warnings=warnings,
    )


#: :func:`build_plan` 的别名，便于上层以「编配」语义调用
arrange = build_plan

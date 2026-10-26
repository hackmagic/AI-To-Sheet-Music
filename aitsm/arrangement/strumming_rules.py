"""编配层：按段落能量选节奏型（确定性规则）。

本模块不重新发明节奏型库，只做三件事：

1. 把 :class:`aitsm.contracts.Section`（或它的纯数据摘要）整理成
   :func:`aitsm.score.strum.pick_pattern` 认识的 dict；
2. 调用节奏型库选型（同一输入永远同一输出，无随机数）；
3. 补上人类可读的 ``reason`` 与 ``confidence``，供质量面板/调试输出使用。

选型原则（详见 ``aitsm.score.strum.pick_pattern``）：副歌能量高 → 更密的扫弦；
前奏/主歌 → 可退到分解；3/4 拍只用三拍子的型；``meta["difficulty"]`` 限制上限。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aitsm.contracts import Section
from aitsm.score.strum import get_pattern, pick_pattern

__all__ = ["StrumDecision", "KIND_LABELS", "recommend"]

#: 段落类型的中文名（只用于 reason 文案）
KIND_LABELS: dict[str, str] = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "预副歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "outro": "尾声",
    "interlude": "间奏",
    "other": "其他段落",
}

#: 摘要里没有显式 energy 时的段落默认能量（与编排习惯一致）
_KIND_ENERGY: dict[str, float] = {
    "intro": 0.25,
    "verse": 0.45,
    "prechorus": 0.6,
    "chorus": 0.85,
    "bridge": 0.5,
    "outro": 0.3,
    "interlude": 0.4,
    "other": 0.5,
}


@dataclass
class StrumDecision:
    """一个段落的节奏型建议（只是建议，最终谱面仍由确定性管线决定）。"""

    section: dict[str, Any]
    pattern_id: str
    reason: str
    confidence: float
    engine: str = "rules"
    pattern_name: str = ""
    pattern_kind: str = ""
    energy: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": dict(self.section),
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_kind": self.pattern_kind,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "energy": round(float(self.energy), 4),
            "engine": self.engine,
        }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _section_energy(section: Mapping[str, Any]) -> float:
    """段落能量：显式 energy > kind 默认 > 0.5。"""
    if isinstance(section.get("energy"), (int, float)):
        return max(0.0, min(1.0, float(section["energy"])))
    kind = str(section.get("kind") or "").strip().lower()
    return _KIND_ENERGY.get(kind, 0.5)


def _snapshot(raw: Any) -> dict[str, Any]:
    """把 :class:`Section` / dict 规整成 ``{kind,label,start,end,energy}``。"""
    if isinstance(raw, Section):
        data = raw.to_dict()
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        data = {}
    kind = str(data.get("kind") or "other").strip().lower() or "other"
    return {
        "kind": kind,
        "label": str(data.get("label") or data.get("name") or ""),
        "start": round(_float(data.get("start"), 0.0), 6),
        "end": round(_float(data.get("end"), 0.0), 6),
        "energy": _section_energy({**data, "kind": kind}),
    }


def _display_label(snapshot: Mapping[str, Any]) -> str:
    label = str(snapshot.get("label") or "").strip()
    if label:
        return label
    kind = str(snapshot.get("kind") or "other").strip().lower()
    return KIND_LABELS.get(kind, kind or KIND_LABELS["other"])


def _confidence(section_energy: float, pattern: Mapping[str, Any]) -> float:
    """能量匹配度 → 置信度（0.35~0.95，确定性）。"""
    pattern_energy = _float(pattern.get("energy"), 0.5)
    distance = abs(pattern_energy - section_energy)
    confidence = 1.0 - min(1.0, distance)
    if pattern.get("kind") == "arpeggio" and section_energy <= 0.35:
        confidence += 0.05
    if pattern.get("subdivision") == "sixteenth" and section_energy >= 0.75:
        confidence += 0.05
    return round(max(0.35, min(0.95, confidence)), 4)


def recommend(
    sections: Sequence[Section | Mapping[str, Any]] | None,
    meta: Mapping[str, Any] | None = None,
) -> list[StrumDecision]:
    """为每个段落推荐节奏型。

    参数
    ----
    sections:
        :class:`aitsm.contracts.Section` 列表，或它们的纯数据摘要列表。
    meta:
        选型上下文：``time_signature`` / ``difficulty`` / ``style`` / ``tempo``。

    返回
    ----
    list[StrumDecision]：按输入顺序一一对应；空输入返回空列表（由上层告警）。
    """
    meta_map = dict(meta or {})
    out: list[StrumDecision] = []
    for raw in sections or ():
        snapshot = _snapshot(raw)
        pattern_id = pick_pattern(dict(snapshot), meta_map)
        pattern = get_pattern(pattern_id) or {}
        if not pattern:  # 理论上不会发生：pick_pattern 只返回库内 id
            pattern_id = "pop8"
            pattern = get_pattern(pattern_id) or {}
        pattern_name = str(pattern.get("name") or pattern_id)
        pattern_kind = str(pattern.get("kind") or "strum")
        energy = float(snapshot["energy"])
        pattern_energy = _float(pattern.get("energy"), 0.5)
        label = _display_label(snapshot)
        reason = (
            f"{label}能量 {energy:.2f}，"
            f"选「{pattern_name}」（能量 {pattern_energy:.2f}，"
            f"难度 {int(pattern.get('difficulty') or 1)}，"
            f"{'扫弦' if pattern_kind == 'strum' else '分解'}）"
        )
        out.append(
            StrumDecision(
                section=snapshot,
                pattern_id=pattern_id,
                reason=reason,
                confidence=_confidence(energy, pattern),
                engine="rules",
                pattern_name=pattern_name,
                pattern_kind=pattern_kind,
                energy=energy,
            )
        )
    return out

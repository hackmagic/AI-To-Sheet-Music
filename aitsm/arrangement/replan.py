"""LLM 评审 → 成本权重调整 → 重跑 DP 的闭环。

为什么要有这一层
----------------
`llm_advisor` 原本只把建议写进 ``ArrangementPlan.advisor`` 供人看，**不参与求解**。
那不是闭环：大模型说了什么，谱面完全不变。

这里实现提示中要求的三段回路：

    DP 生成候选  →  LLM 输出结构化评审（JSON）  →  调整成本权重  →  重跑 DP

四条硬约束（都是为了不让 LLM 变成不可控的黑盒）：

1. **只输出结构化 JSON**，字段受 schema 约束；无法解析就当没提过；
2. **只能调权重，不能指定弦品**——LLM 不许直接写 `frets`。它给的是
   「换把流畅度更重要一点」「多用空弦」这类**方向性调整**；
3. **调整幅度有上限**（`:data:`MAX_WEIGHT_DELTA`），且权重必须保持为正，
   避免一次过激建议把 DP 带偏；
4. **两条路径都跑、取成本更低者**。LLM 的建议只是候选，不是命令——
   调权重后重解得来的方案更差，就保留原始结果，并在报告里写明「建议未被采纳」。

离线（未配置 `AITSM_LLM_URL`）时本模块**完全不介入**，行为与之前一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from aitsm.contracts import ChordEvent, MelodyNote

from .fingering_dp import COST_WEIGHTS, VoicingChoice, optimize_progression, plan_cost

__all__ = [
    "WEIGHT_KEYS",
    "MAX_WEIGHT_DELTA",
    "COST_ADJUSTMENT_SCHEMA",
    "ReplanResult",
    "build_prompt",
    "parse_adjustment",
    "apply_adjustment",
    "replan_with_advice",
]

#: LLM 允许调整的权重键（与 DP 的 COST_WEIGHTS 对齐）
WEIGHT_KEYS: tuple[str, ...] = (
    "position_shift",
    "hand_span",
    "open_string",
    "switch",
    "melody_cover",
)

#: 单次调整的相对幅度上限：权重最多变成原来的 50%~150%。
#: 为什么要卡：LLM 说「换把非常重要」时可能给出 ±10 的绝对量，
#: 那样 DP 会退化成「全曲只用一个把位」，反而更难弹。
MAX_WEIGHT_DELTA = 0.5

#: LLM 结构化输出的 schema。**只允许调权重 + 给理由**，不允许指定弦品。
COST_ADJUSTMENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["cost_adjustments"],
    "properties": {
        "cost_adjustments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # 每个权重是 [-1, 1] 的相对调整（-1 = 减半，+1 = 翻 1.5 倍）
                key: {"type": "number", "minimum": -1, "maximum": 1} for key in WEIGHT_KEYS
            },
        },
        "review": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "reason": {"type": "string"},
    },
    "additionalProperties": True,
}


@dataclass
class ReplanResult:
    """一次「按 LLM 建议重解」的结果。"""

    choices: list[VoicingChoice]
    cost: float
    weights: dict[str, float]
    #: 是否采纳了重解结果（重解成本更低才采纳）
    applied: bool = False
    #: LLM 建议的原始调整（已裁剪到 ±MAX_WEIGHT_DELTA 以内）
    requested: dict[str, float] = field(default_factory=dict)
    #: 实际生效的权重变化
    delta: dict[str, float] = field(default_factory=dict)
    baseline_cost: float = 0.0
    baseline_weights: dict[str, float] = field(default_factory=dict)
    source: str = "llm"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "source": self.source,
            "cost": round(float(self.cost), 4),
            "baseline_cost": round(float(self.baseline_cost), 4),
            "weights": {key: round(float(value), 4) for key, value in self.weights.items()},
            "baseline_weights": {
                key: round(float(value), 4) for key, value in self.baseline_weights.items()
            },
            "requested": {key: round(float(value), 4) for key, value in self.requested.items()},
            "delta": {key: round(float(value), 4) for key, value in self.delta.items()},
            "warnings": list(self.warnings),
        }


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def build_prompt(
    chords: Sequence[ChordEvent],
    choices: Sequence[VoicingChoice],
    *,
    weights: Mapping[str, float] | None = None,
) -> str:
    """拼出要求「只调权重」的提示词。

    刻意**不**把候选弦品全量贴进去：LLM 一看到具体 frets 就会试图改写它们，
    而我们的约束是「只调权重」。这里只给节奏级别的统计（把位跨度、空弦比例）。
    """
    weight_map = {**COST_WEIGHTS, **{k: v for k, v in dict(weights or {}).items() if k in COST_WEIGHTS}}
    base_frets = [int(item.base_fret or 0) for item in choices]
    spans = [max(base_frets) - min(base_frets)] if base_frets else [0]
    open_counts = sum(
        1 for item in choices for fret in (item.frets or []) if int(fret) == 0
    )
    total_strings = sum(len(item.frets or []) for item in choices) or 1
    lines = [
        "你是吉他编配评审。只评估**可演奏性**，并给出成本权重调整建议。",
        "严格只返回 JSON，不要解释，不要给出弦品或品位。格式：",
        '{"cost_adjustments":{"position_shift":0.2,"open_string":0.1},'
        '"review":{"score":0.8,"issues":[],"suggestions":[]},"reason":"..."}',
        "",
        f"和弦数：{len(chords)}；把位跨度：{spans[0]} 品；空弦占比：{open_counts / total_strings:.2f}",
        f"当前权重：{ {k: round(v, 2) for k, v in weight_map.items()} }",
        "",
        "可用权重键（正值表示更看重）：",
        "- position_shift：换把距离（越大越倾向少换把）",
        "- hand_span：手跨（越大越倾向窄跨度）",
        "- open_string：空弦利用（越大越倾向多用空弦）",
        "- switch：和弦切换成本（越大越倾向少动手指）",
        "- melody_cover：旋律覆盖（越大越要求旋律音在指法内）",
        "",
        "每个调整取值 [-1, 1]：-1 表示减半，+1 表示放大到 1.5 倍；正负都要给理由。",
    ]
    return "\n".join(lines)


def parse_adjustment(payload: Any) -> tuple[dict[str, float], list[str]]:
    """从 LLM 输出里抽出权重调整；返回 ``(调整, 告警)``。

    容忍三种形状：

    1. ``{"cost_adjustments": {...}}``（review 解析后 data 的顶层，最常见）；
    2. ``{"review": {"cost_adjustments": {...}}}``（模型把调整嵌在 review 里）；
    3. 直接 ``{权重: 值}``。

    非法值（非数字、超范围、未知键）一律丢弃并记告警——**不猜**。
    """
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return {}, ["LLM 输出不是对象，忽略"]
    raw = payload.get("cost_adjustments")
    if not isinstance(raw, Mapping):
        nested = payload.get("review")
        if isinstance(nested, Mapping) and isinstance(nested.get("cost_adjustments"), Mapping):
            raw = nested["cost_adjustments"]
    if not isinstance(raw, Mapping) and any(key in payload for key in WEIGHT_KEYS):
        raw = payload
    if not isinstance(raw, Mapping):
        return {}, ["LLM 输出里没有 cost_adjustments，忽略"]

    adjustments: dict[str, float] = {}
    for key, value in raw.items():
        if key not in WEIGHT_KEYS:
            warnings.append(f"未知权重键 {key!r}，已忽略")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            warnings.append(f"权重 {key} 的值不是数字，已忽略")
            continue
        if number != number or abs(number) == float("inf"):  # NaN / inf
            warnings.append(f"权重 {key} 的值非法，已忽略")
            continue
        if not -1.0 <= number <= 1.0:
            warnings.append(f"权重 {key} 超出 [-1,1]，已裁剪")
        adjustments[key] = _clamp(number)
    return adjustments, warnings


def apply_adjustment(
    adjustments: Mapping[str, float],
    *,
    base_weights: Mapping[str, float] | None = None,
    max_delta: float = MAX_WEIGHT_DELTA,
) -> tuple[dict[str, float], dict[str, float]]:
    """把 ``[-1, 1]`` 的相对调整变成实际权重；返回 ``(新权重, 实际变化)``。

    映射：``new = base * (1 + clamp(adj) * max_delta)``，再确保为正。
    这样「+1」最多把权重放大到 1.5 倍，不会出现 0 或负数。
    """
    base = {**COST_WEIGHTS, **{k: float(v) for k, v in dict(base_weights or {}).items() if k in COST_WEIGHTS}}
    result = dict(base)
    delta: dict[str, float] = {}
    for key, value in adjustments.items():
        if key not in result:
            continue
        factor = 1.0 + _clamp(value) * float(max_delta)
        new_value = float(base[key]) * factor
        # 权重必须为正：极端情况下（base 很小）保底 0.01
        new_value = max(0.01, new_value)
        result[key] = new_value
        delta[key] = new_value - float(base[key])
    return result, delta


def replan_with_advice(
    chords: Sequence[ChordEvent],
    baseline: Sequence[VoicingChoice],
    *,
    parse: Callable[[Any], tuple[dict[str, float], list[str]]] | None = None,
    advisor_payload: Any = None,
    tuning: Sequence[int] | None = None,
    melody_notes: Sequence[MelodyNote] = (),
    key: Any = None,
    max_candidates: int = 8,
    weights: Mapping[str, float] | None = None,
) -> ReplanResult:
    """按 LLM 建议调整权重并**重跑 DP**，返回对比结果。

    ``advisor_payload`` 是 advisor 已解析好的结构化输出（含 ``cost_adjustments``）；
    传 ``None`` 表示没有可用建议 → 原样返回 baseline 并标 ``applied=False``。

    采用规则：**重解后的总成本更低才采纳**。LLM 的建议只是候选；重解更差就保留
    原方案，并在 ``warnings`` 里说明「建议未被采纳」。
    """
    base_weights = {
        **COST_WEIGHTS,
        **{k: float(v) for k, v in dict(weights or {}).items() if k in COST_WEIGHTS},
    }
    baseline_cost = plan_cost(list(baseline), weights=base_weights) if baseline else 0.0

    if advisor_payload is None:
        return ReplanResult(
            choices=list(baseline),
            cost=baseline_cost,
            weights=base_weights,
            applied=False,
            baseline_cost=baseline_cost,
            baseline_weights=base_weights,
            source="none",
            warnings=["没有 LLM 建议，保留 DP 原始结果"],
        )

    parser = parse or parse_adjustment
    adjustments, warnings = parser(advisor_payload)
    if not adjustments:
        return ReplanResult(
            choices=list(baseline),
            cost=baseline_cost,
            weights=base_weights,
            applied=False,
            baseline_cost=baseline_cost,
            baseline_weights=base_weights,
            warnings=[*warnings, "LLM 未给出有效权重调整，保留 DP 原始结果"],
        )

    new_weights, delta = apply_adjustment(adjustments, base_weights=base_weights)
    try:
        from aitsm.score.model import STANDARD_TUNING

        replanned = optimize_progression(
            chords,
            tuning=tuning or STANDARD_TUNING,
            melody_notes=melody_notes,
            key=key,
            max_candidates=max_candidates,
            weights=new_weights,
        )
    except Exception as exc:  # noqa: BLE001 - 重解失败不能拖垮编配
        return ReplanResult(
            choices=list(baseline),
            cost=baseline_cost,
            weights=base_weights,
            applied=False,
            requested=adjustments,
            delta=delta,
            baseline_cost=baseline_cost,
            baseline_weights=base_weights,
            warnings=[*warnings, f"按建议重解失败（{type(exc).__name__}: {exc}），保留原结果"],
        )

    new_cost = plan_cost(replanned, weights=new_weights) if replanned else baseline_cost
    # 成本可比性：两次用各自的权重算，属于「在该权重体系下的最优」比较。
    # 简单起见按新权重衡量两者：若新方案在新权重下更便宜，说明建议确实改进。
    baseline_under_new = plan_cost(list(baseline), weights=new_weights) if baseline else new_cost
    adopt = bool(replanned) and new_cost < baseline_under_new - 1e-9

    return ReplanResult(
        choices=list(replanned) if adopt else list(baseline),
        cost=new_cost if adopt else baseline_cost,
        weights=new_weights if adopt else base_weights,
        applied=adopt,
        requested=adjustments,
        delta=delta,
        baseline_cost=baseline_cost,
        baseline_weights=base_weights,
        warnings=warnings
        + ([] if adopt else ["按建议重解后成本未降低，已保留 DP 原始结果"]),
    )

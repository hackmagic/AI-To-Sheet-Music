"""编配层（v2 架构）：只吃符号谱，产出确定性的指法与节奏型决策。

模块
----
- :mod:`aitsm.arrangement.fingering_dp`：候选池 + 动态规划指法路径；
- :mod:`aitsm.arrangement.tart_rerank`：可选 TART 重排，离线走确定性微调；
- :mod:`aitsm.arrangement.strumming_rules`：按段落能量选节奏型；
- :mod:`aitsm.arrangement.llm_advisor`：可注入 transport 的大模型顾问，
  失败即规则兜底，绝不联网失败崩掉；
- :mod:`aitsm.arrangement.decisions`：汇总为 :class:`ArrangementPlan`。

边界：输入只允许 :class:`aitsm.contracts.SymbolicScore` 或其纯数据摘要；
本层不读音频、不 import ``aitsm.pipeline``。
"""

from __future__ import annotations

from .decisions import PLAN_SCHEMA, ArrangementPlan, arrange, build_plan, score_summary
from .fingering_dp import (
    COST_WEIGHTS,
    UNPLAYABLE_PENALTY,
    VoicingChoice,
    assign_path_costs,
    candidates_for,
    choice_from_voicing,
    local_cost,
    optimize_progression,
    plan_cost,
    transition_cost,
)
from .llm_advisor import (
    ADVICE_SCHEMA,
    LLM_URL_ENV,
    AdvisorResult,
    LLMAdvisor,
    advise,
    rule_advice,
    rule_review,
    rule_strum,
    rule_voicings,
)
from .strumming_rules import StrumDecision, recommend
from .tart_rerank import TART_ENV, active_engine, is_available, model_dir, rerank

__all__ = [
    "PLAN_SCHEMA",
    "ArrangementPlan",
    "arrange",
    "build_plan",
    "score_summary",
    "COST_WEIGHTS",
    "UNPLAYABLE_PENALTY",
    "VoicingChoice",
    "assign_path_costs",
    "candidates_for",
    "choice_from_voicing",
    "local_cost",
    "optimize_progression",
    "plan_cost",
    "transition_cost",
    "LLM_URL_ENV",
    "ADVICE_SCHEMA",
    "AdvisorResult",
    "LLMAdvisor",
    "advise",
    "rule_advice",
    "rule_review",
    "rule_strum",
    "rule_voicings",
    "StrumDecision",
    "recommend",
    "TART_ENV",
    "active_engine",
    "is_available",
    "model_dir",
    "rerank",
]

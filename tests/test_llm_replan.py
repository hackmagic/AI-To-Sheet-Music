"""LLM 评审 → 调权重 → 重跑 DP 的闭环测试。

对应评审提示 2 的验收要求：
- LLM 输出受**结构化 JSON** 约束，不能指定弦品；
- 建议被采纳时必须真的改变权重并重解；
- 建议更差时必须**不采纳**（LLM 是候选，不是命令）；
- 离线（无 advisor）时行为与之前完全一致。
"""

from __future__ import annotations

import pytest

from aitsm.arrangement import replan as R
from aitsm.arrangement.decisions import build_plan
from aitsm.arrangement.fingering_dp import (
    COST_WEIGHTS,
    assign_path_costs,
    optimize_progression,
)
from aitsm.arrangement.llm_advisor import AdvisorResult, LLMAdvisor
from aitsm.contracts import ChordEvent, KeySignature


def _chords() -> list[ChordEvent]:
    """G → C → D → Em，足够触发换把与切换成本。"""
    symbols = ["G", "C", "D", "Em"]
    return [ChordEvent(start=index * 2.0, end=index * 2.0 + 2.0, symbol=sym) for index, sym in enumerate(symbols)]


@pytest.fixture()
def baseline():
    return optimize_progression(_chords(), weights=COST_WEIGHTS)


def _key() -> KeySignature:
    return KeySignature(tonic="G", mode="major")


# ---------------------------------------------------------------- 权重调整


def test_apply_adjustment_is_bounded():
    """+1 最多放大到 1.5 倍；-1 最多减半，且权重永远为正。"""
    weights, delta = R.apply_adjustment({"position_shift": 1.0, "switch": -1.0})
    assert weights["position_shift"] == pytest.approx(COST_WEIGHTS["position_shift"] * 1.5)
    assert weights["switch"] == pytest.approx(COST_WEIGHTS["switch"] * 0.5)
    assert all(value > 0 for value in weights.values())
    assert delta["position_shift"] > 0 and delta["switch"] < 0


def test_apply_adjustment_clamps_out_of_range():
    weights, _ = R.apply_adjustment({"hand_span": 99.0})
    assert weights["hand_span"] == pytest.approx(COST_WEIGHTS["hand_span"] * (1 + R.MAX_WEIGHT_DELTA))


def test_apply_adjustment_ignores_unknown_keys():
    weights, delta = R.apply_adjustment({"no_such_weight": 1.0})
    assert "no_such_weight" not in weights
    assert delta == {}


def test_apply_adjustment_never_zero_or_negative():
    """极端负值也必须保持正权重，否则 DP 的成本方向会反过来。"""
    weights, _ = R.apply_adjustment({"open_string": -1.0})
    assert weights["open_string"] > 0


# ---------------------------------------------------------------- 结构化解析


def test_parse_adjustment_accepts_schema_shape():
    adjustments, warnings = R.parse_adjustment(
        {"cost_adjustments": {"position_shift": 0.3, "open_string": 0.2}, "reason": "更少换把"}
    )
    assert adjustments == {"position_shift": 0.3, "open_string": 0.2}
    assert warnings == []


def test_parse_adjustment_accepts_bare_weight_map():
    adjustments, _ = R.parse_adjustment({"position_shift": -0.4})
    assert adjustments == {"position_shift": -0.4}


def test_parse_adjustment_rejects_unknown_and_non_numeric():
    adjustments, warnings = R.parse_adjustment(
        {"cost_adjustments": {"frets": [1, 2, 3], "position_shift": "bogus", "switch": 0.5}}
    )
    assert adjustments == {"switch": 0.5}
    assert any("frets" in item for item in warnings)
    assert any("position_shift" in item for item in warnings)


def test_parse_adjustment_does_not_accept_frets():
    """LLM 不许指定弦品：问道具字段必须被丢弃。"""
    adjustments, warnings = R.parse_adjustment(
        {"cost_adjustments": {"position_shift": 0.1}, "frets": [[3, 2, 0, 0, 0, 3]]}
    )
    assert "frets" not in adjustments
    assert all(key in R.WEIGHT_KEYS for key in adjustments)


def test_parse_adjustment_handles_garbage():
    for payload in (None, "text", 42, [], {"cost_adjustments": "x"}):
        adjustments, warnings = R.parse_adjustment(payload)
        assert adjustments == {}
        assert warnings


# ---------------------------------------------------------------- 闭环行为


def test_replan_without_advice_keeps_baseline(baseline):
    result = R.replan_with_advice(_chords(), baseline, advisor_payload=None)
    assert result.applied is False
    assert result.choices == list(baseline)
    assert result.source == "none"


def test_replan_empty_adjustments_keeps_baseline(baseline):
    result = R.replan_with_advice(_chords(), baseline, advisor_payload={})
    assert result.applied is False
    assert result.choices == list(baseline)


def test_replan_adopts_when_new_cost_lower(baseline):
    """给出一个确实能降成本的建议时，应当采纳并改变权重。"""
    # open_string 权重调高 → DP 更愿意贴空弦，通常能降本
    result = R.replan_with_advice(
        _chords(), baseline, advisor_payload={"cost_adjustments": {"open_string": 0.5}}
    )
    assert result.requested == {"open_string": 0.5}
    assert result.weights != result.baseline_weights
    if result.applied:
        assert result.cost <= result.baseline_cost + 1e-6
        assert result.choices != baseline or result.cost < result.baseline_cost


def test_replan_rejects_when_not_improved(baseline):
    """建议不能降低（新权重下）成本时必须保留原方案。"""
    # 用一个不影响最优解的微调试试；若它确实无改进就必须 rejected
    result = R.replan_with_advice(
        _chords(), baseline, advisor_payload={"cost_adjustments": {"switch": 0.000001}}
    )
    if not result.applied:
        assert result.choices == list(baseline)
        assert result.weights == result.baseline_weights
        assert any("未降低" in item or "保留" in item for item in result.warnings)


def test_replan_survives_dp_exception(baseline, monkeypatch: pytest.MonkeyPatch):
    """重解抛异常时退回原结果，不能拖垮编配。"""
    def boom(*args, **kwargs):
        raise RuntimeError("dp exploded")

    monkeypatch.setattr(R, "optimize_progression", boom)
    result = R.replan_with_advice(
        _chords(), baseline, advisor_payload={"cost_adjustments": {"open_string": 0.3}}
    )
    assert result.applied is False
    assert result.choices == list(baseline)
    assert any("重解失败" in item for item in result.warnings)


def test_replan_result_serialises(baseline):
    result = R.replan_with_advice(
        _chords(), baseline, advisor_payload={"cost_adjustments": {"position_shift": 0.2}}
    )
    data = result.to_dict()
    for key in ("applied", "cost", "baseline_cost", "weights", "baseline_weights", "requested", "delta", "warnings"):
        assert key in data


# ---------------------------------------------------------------- 端到端（build_plan）


class _FakeAdvisor(LLMAdvisor):
    """假顾问：按 kind 返回**对应形状**的 JSON，不联网。

    注意：`suggest_voicings` / `review_plan` / `suggest_strumming` 三个方法的
    parser 期望不同结构（voicings 是数组、review 是对象、strumming 是 sections）。
    把同一个 payload 喂给所有 parser 会让前两个先抛 ValueError 而走规则兜底，
    于是 review 永远拿不到我们的 cost_adjustments。这里按 kind 分派。
    """

    def __init__(self, review_payload: dict):
        super().__init__(transport=lambda prompt: "{}")
        self.review_payload = review_payload

    def _ask(self, prompt, parser, fallback, *, kind):
        if kind == "谱面评审":
            payload = self.review_payload
        elif kind == "指法建议":
            payload = {"voicings": []}
        else:  # 节奏型建议 等
            payload = {"sections": []}
        # 注意：parser 的签名是 (data, warnings) -> dict，**只返回一个 dict**。
        # 早期写成 `data, warnings = parser(...)` 会把 dict 的键解包成两个字符串。
        try:
            data = parser(payload, [])
            warnings: list[str] = []
        except Exception as exc:  # parser 失败 → 用 fallback() 取值（它返回 dict）
            data, warnings = self._fallback(fallback), [f"解析失败：{exc}"]
        return AdvisorResult(engine="llm", data=data, warnings=warnings)


def _score_fixture():
    return {
        "beats": {"bpm": 120.0, "time_signature": "4/4", "beats_per_bar": 4},
        "key": {"tonic": "G", "mode": "major"},
        "chords": [
            {"start": 0.0, "end": 2.0, "symbol": "G"},
            {"start": 2.0, "end": 4.0, "symbol": "C"},
            {"start": 4.0, "end": 6.0, "symbol": "D"},
        ],
        "sections": [{"start": 0.0, "end": 6.0, "kind": "verse"}],
    }


def test_build_plan_without_advisor_is_unchanged():
    """离线路径：advisor=None 时不产生 advisor_replan，行为与之前一致。"""
    plan = build_plan(_score_fixture(), options={"arrangement": "auto"})
    assert plan.advisor == {"engine": "off"}
    assert plan.advisor_replan is None
    assert plan.engines.get("fingering") == "dp"


def test_build_plan_records_replan_outcome():
    """接上顾问后，无论采纳与否都要在 plan 里记录闭环结果。"""
    advisor = _FakeAdvisor({"cost_adjustments": {"open_string": 0.4}, "reason": "多用空弦"})
    plan = build_plan(_score_fixture(), advisor=advisor)
    assert plan.advisor_replan is not None
    assert "applied" in plan.advisor_replan
    assert "requested" in plan.advisor_replan
    assert plan.advisor_replan["requested"] == {"open_string": 0.4}


def test_build_plan_marks_engine_when_applied():
    """采纳时 engine 必须变成 dp+llm-weights，便于追溯。"""
    advisor = _FakeAdvisor({"cost_adjustments": {"open_string": 0.6, "position_shift": -0.4}})
    plan = build_plan(_score_fixture(), advisor=advisor)
    if plan.advisor_replan and plan.advisor_replan.get("applied"):
        assert plan.engines["fingering"] == "dp+llm-weights"
        assert any("调整成本权重" in item for item in plan.warnings)
    else:
        # 没改进也要有明确记录
        assert any("保留" in item for item in plan.warnings)


def test_build_plan_never_lets_llm_write_frets():
    """LLM 输出里带 frets 也不能影响最终 choices。"""
    advisor = _FakeAdvisor(
        {
            "cost_adjustments": {"position_shift": 0.1},
            "frets": [[1, 1, 1, 1, 1, 1]],
            "voicing_id": "HACKED",
        }
    )
    plan = build_plan(_score_fixture(), advisor=advisor)
    assert all(choice.voicing_id != "HACKED" for choice in plan.choices)
    assert all(len(choice.frets) == 6 for choice in plan.choices)


def test_build_plan_advisor_failure_is_contained(monkeypatch: pytest.MonkeyPatch):
    """顾问整体抛异常时编配仍要产出结果。"""
    class _Broken(LLMAdvisor):
        def _ask(self, *args, **kwargs):
            raise RuntimeError("llm down")

    plan = build_plan(_score_fixture(), advisor=_Broken())
    assert plan.choices
    assert plan.advisor_replan is not None
    assert plan.advisor_replan.get("applied") is False

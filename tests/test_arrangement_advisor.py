"""编配层大模型顾问的测试：注入假 transport，全部离线。"""

from __future__ import annotations

import json
import time

from aitsm.arrangement.decisions import build_plan
from aitsm.arrangement.fingering_dp import optimize_progression
from aitsm.arrangement.llm_advisor import ADVICE_SCHEMA, LLMAdvisor, advise, rule_review
from aitsm.contracts import BeatGrid, ChordEvent, KeySignature, Section, SymbolicScore


def _score() -> SymbolicScore:
    chords = [ChordEvent(0, 2, "G"), ChordEvent(2, 4, "C"), ChordEvent(4, 6, "D")]
    sections = [Section(0, 3, "intro", "Intro", 0.25), Section(3, 6, "chorus", "Chorus", 0.9)]
    return SymbolicScore(
        beats=BeatGrid(120, [0, 0.5, 1, 1.5], [0]),
        key=KeySignature("G"),
        chords=chords,
        sections=sections,
    )


def _valid_voicing_json(prompt: str) -> str:
    return '```json\n{"voicings": [{"symbol": "G", "voicing_id": "G_320003", "reason": "开放把位"}]}\n```'


def _bad_json(prompt: str) -> str:
    return "抱歉，我无法按要求返回 JSON。"


def _slow(prompt: str) -> str:
    time.sleep(0.8)
    return _valid_voicing_json(prompt)


def _boom(prompt: str) -> str:
    raise RuntimeError("模拟网络故障")


def _valid_advice_json(prompt: str) -> str:
    return json.dumps(
        {
            "section": "整体",
            "suggested_strumming": [
                {"start": 0.0, "kind": "intro", "pattern_id": "ballad8", "reason": "低能量"}
            ],
            "chord_substitutions": [
                {"original": "H13", "suggested": "G", "reason": "简化和弦"}
            ],
            "notes": "整体把位连贯",
        }
    )


def test_valid_json_uses_llm_engine():
    advisor = LLMAdvisor(transport=_valid_voicing_json, timeout=2.0)
    result = advisor.suggest_voicings(_score().chords)

    assert result.engine == "llm"
    assert result.advisory is True
    assert [item["voicing_id"] for item in result.data["voicings"]] == ["G_320003"]


def test_invalid_json_falls_back_to_rules_without_raising():
    advisor = LLMAdvisor(transport=_bad_json, timeout=2.0)
    result = advisor.suggest_voicings(_score().chords)

    assert result.engine == "rules"
    assert result.data["voicings"], "规则兜底必须给出可用候选"
    assert result.warnings
    assert result.error


def test_timeout_falls_back_quickly_to_rules():
    advisor = LLMAdvisor(transport=_slow, timeout=0.1)
    started = time.monotonic()
    result = advisor.suggest_voicings(_score().chords, )
    elapsed = time.monotonic() - started

    assert result.engine == "rules"
    assert result.data["voicings"]
    assert elapsed < 0.7, elapsed  # 不能被慢 transport 拖住
    assert "Timeout" in result.error or "超时" in " ".join(result.warnings)


def test_transport_exception_falls_back_to_rules():
    advisor = LLMAdvisor(transport=_boom, timeout=1.0)
    result = advisor.review_plan(optimize_progression(_score().chords))

    assert result.engine == "rules"
    assert 0.0 <= result.data["review"]["score"] <= 1.0


def test_llm_returning_unknown_voicing_id_is_rejected():
    advisor = LLMAdvisor(
        transport=lambda prompt: '{"voicings":[{"symbol":"G","voicing_id":"G_999999"}]}',
        timeout=1.0,
    )
    result = advisor.suggest_voicings([ChordEvent(0, 1, "G")])
    assert result.engine == "rules"
    assert any("未知指法" in warning for warning in result.warnings)


def test_valid_review_json_is_clamped():
    advisor = LLMAdvisor(
        transport=lambda prompt: '{"review":{"score":9,"issues":["a"],"suggestions":["b"]}}',
        timeout=1.0,
    )
    result = advisor.review_plan(optimize_progression(_score().chords))
    assert result.engine == "llm"
    assert result.data["review"]["score"] == 1.0


def test_noTransport_and_no_env_is_rules_without_network(monkeypatch):
    monkeypatch.delenv("AITSM_LLM_URL", raising=False)
    advisor = LLMAdvisor()
    assert advisor.configured is False

    started = time.monotonic()
    result = advisor.suggest_strumming(_score().sections)
    assert result.engine == "rules"
    assert result.data["sections"]
    assert time.monotonic() - started < 1.0


def test_empty_inputs_and_weird_choices_never_raise():
    advisor = LLMAdvisor(transport=_boom, timeout=0.5)
    assert advisor.suggest_voicings([]).engine == "rules"
    assert advisor.review_plan([]).engine == "rules"
    assert advisor.suggest_strumming([]).engine == "rules"


def test_advise_valid_json_passes_jsonschema_and_uses_llm():
    payload = advise(_score(), transport=_valid_advice_json)

    assert payload["engine"] == "llm"
    assert payload["advisory"] is True
    assert payload["notes"] == "整体把位连贯"
    assert payload["suggested_strumming"][0]["pattern_id"] == "ballad8"

    import jsonschema

    jsonschema.validate(instance=payload, schema=ADVICE_SCHEMA)


def test_advise_schema_violation_falls_back_to_rules():
    bad_schema = json.dumps({"section": 123, "suggested_strumming": "nope"})
    payload = advise(_score(), transport=lambda prompt: bad_schema)

    assert payload["engine"] == "rules"
    assert payload["suggested_strumming"], "规则兜底应给出节奏型建议"
    assert "engine" in payload and payload["warnings"]


def test_advise_pattern_outside_enum_falls_back():
    bad_enum = json.dumps(
        {
            "section": "整体",
            "suggested_strumming": [{"start": 0.0, "pattern_id": "not_a_pattern"}],
            "chord_substitutions": [],
            "notes": "",
        }
    )
    payload = advise(_score(), transport=lambda prompt: bad_enum)
    assert payload["engine"] == "rules"


def test_advise_without_transport_offline_is_rules(monkeypatch):
    monkeypatch.delenv("AITSM_LLM_URL", raising=False)
    payload = advise(
        {"chords": [{"symbol": "G"}, {"symbol": "G/C/G"}], "sections": [{"start": 0, "end": 4, "kind": "intro", "energy": 0.2}]}
    )
    assert payload["engine"] == "rules"
    assert payload["suggested_strumming"]
    assert any(item["original"] == "G/C/G" for item in payload["chord_substitutions"])


def test_advise_never_raises_on_bad_input():
    assert advise(None)["engine"] == "rules"
    assert advise(42)["engine"] == "rules"


def test_build_plan_accepts_options_dict():
    advisor = LLMAdvisor(transport=_bad_json, timeout=1.0)
    plan = build_plan(
        _score(),
        options={"style": "folk", "difficulty": "beginner", "advisor": advisor},
    )
    assert plan.meta["style"] == "folk"
    assert plan.meta["difficulty"] == "beginner"
    assert plan.advisor["engine"] == "rules"


def test_rule_review_is_deterministic():
    choices = optimize_progression(_score().chords)
    assert rule_review(choices) == rule_review(choices)


def test_advisor_does_not_override_plan_choices():
    score = _score()
    plain = build_plan(score)
    advised = build_plan(score, advisor=LLMAdvisor(transport=_bad_json, timeout=1.0))

    assert [c.voicing_id for c in advised.choices] == [c.voicing_id for c in plain.choices]
    assert advised.advisor["engine"] == "rules"
    assert advised.advisor["advisory"] is True

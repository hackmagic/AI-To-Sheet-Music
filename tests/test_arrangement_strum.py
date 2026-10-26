"""编配层节奏型规则测试：不同段落应得到不同 pattern。"""

from __future__ import annotations

from aitsm.arrangement.decisions import build_plan
from aitsm.arrangement.strumming_rules import recommend
from aitsm.contracts import BeatGrid, ChordEvent, KeySignature, Section, SymbolicScore
from aitsm.score.strum import get_pattern, pattern_ids


def _sections() -> list[Section]:
    return [
        Section(0, 4, "intro", "Intro", 0.25),
        Section(4, 12, "verse", "Verse", 0.45),
        Section(12, 20, "chorus", "Chorus", 0.9),
    ]


def _score() -> SymbolicScore:
    return SymbolicScore(
        beats=BeatGrid(120, [0, 0.5, 1, 1.5], [0]),
        key=KeySignature("G"),
        chords=[ChordEvent(0, 2, "G"), ChordEvent(2, 4, "C"), ChordEvent(4, 6, "D")],
        sections=_sections(),
    )


def test_chorus_and_intro_get_different_patterns():
    decisions = recommend(_sections())
    assert [item.section["kind"] for item in decisions] == ["intro", "verse", "chorus"]
    by_kind = {item.section["kind"]: item for item in decisions}

    assert by_kind["chorus"].pattern_id != by_kind["intro"].pattern_id
    # 副歌能量更高：扫弦型能量应不低于前奏
    chorus = get_pattern(by_kind["chorus"].pattern_id)
    intro = get_pattern(by_kind["intro"].pattern_id)
    assert chorus is not None and intro is not None
    assert float(chorus["energy"]) >= float(intro["energy"])


def test_patterns_are_valid_ids_with_confidence_and_reason():
    valid = set(pattern_ids())
    for item in recommend(_sections()):
        assert item.pattern_id in valid
        assert 0.35 <= item.confidence <= 0.95
        assert item.reason
        assert item.engine == "rules"
        assert item.to_dict()["pattern_id"] == item.pattern_id


def test_plain_dict_sections_match_dataclass_sections():
    from_objects = [item.to_dict() for item in recommend(_sections())]
    from_dicts = [item.to_dict() for item in recommend([section.to_dict() for section in _sections()])]
    assert from_objects == from_dicts


def test_recommend_is_deterministic_and_handles_empty():
    first = [item.to_dict() for item in recommend(_sections())]
    second = [item.to_dict() for item in recommend(_sections())]
    assert first == second
    assert recommend([]) == []


def test_difficulty_cap_is_respected():
    easy = recommend(_sections(), {"difficulty": "beginner"})
    for item in easy:
        pattern = get_pattern(item.pattern_id)
        assert pattern is not None
        assert int(pattern["difficulty"]) <= 3


def test_build_plan_attaches_strumming():
    plan = build_plan(_score())
    assert plan.engines["strumming"] == "rules"
    assert [item.section["kind"] for item in plan.strumming] == ["intro", "verse", "chorus"]
    assert len({item.pattern_id for item in plan.strumming}) >= 2

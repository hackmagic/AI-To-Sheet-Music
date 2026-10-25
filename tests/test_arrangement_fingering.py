"""编配层指法 DP 与 TART 兜底重排的测试（快速、离线）。"""

from __future__ import annotations

import importlib.util

from aitsm.arrangement.fingering_dp import (
    COST_WEIGHTS,
    UNPLAYABLE_PENALTY,
    assign_path_costs,
    candidates_for,
    choice_from_voicing,
    optimize_progression,
    plan_cost,
)
from aitsm.arrangement.tart_rerank import active_engine, is_available, rerank
from aitsm.contracts import BeatGrid, ChordEvent, KeySignature
from aitsm.score.voicings import validate_voicing


def _buttons(symbols: list[str]) -> list[ChordEvent]:
    return [ChordEvent(float(i) * 2, float(i) * 2 + 2, symbol) for i, symbol in enumerate(symbols)]


def _first_candidate_baseline(chords: list[ChordEvent]) -> float:
    """全选第一个候选的基准成本（不经过 DP）。"""
    picks = []
    for chord in chords:
        pool = candidates_for(chord)
        assert pool, f"{chord.symbol} 应有候选"
        picks.append(
            choice_from_voicing(
                chord,
                pool[0],
                candidates_considered=len(pool),
                melody_notes=[],
                weights=COST_WEIGHTS,
            )
        )
    return plan_cost(assign_path_costs(picks, weights=COST_WEIGHTS), weights=COST_WEIGHTS)


def test_dp_g_c_d_picks_valid_frets_and_beats_first_candidate_baseline():
    chords = _buttons(["G", "C", "D"])
    choices = optimize_progression(chords)

    assert [choice.symbol for choice in choices] == ["G", "C", "D"]
    for choice in choices:
        assert choice.frets, f"{choice.symbol} 不应为空指法"
        assert all(-1 <= fret <= 24 for fret in choice.frets), choice.frets
        assert validate_voicing(choice.frets, choice.symbol) == [], (choice.symbol, choice.frets)
        pool_ids = {item["id"] for item in candidates_for(choice.chord)}
        assert choice.voicing_id in pool_ids

    # 把位连续：相邻和弦的 base_fret 不跳，整体跨度小
    positions = [choice.base_fret for choice in choices]
    assert max(positions) - min(positions) <= 4, positions
    assert all(abs(b - a) <= 4 for a, b in zip(positions, positions[1:])), positions

    baseline = _first_candidate_baseline(chords)
    assert plan_cost(choices) <= baseline + 1e-9, (plan_cost(choices), baseline)


def test_cost_is_monotonic_when_an_unplayable_chord_is_appended():
    chords = _buttons(["G", "C"])
    base_cost = plan_cost(optimize_progression(chords))
    extended = optimize_progression(chords + [ChordEvent(4, 6, "H13")])

    assert len(extended) == 3
    assert extended[-1].frets == []
    assert plan_cost(extended) >= base_cost + UNPLAYABLE_PENALTY - 1e-9


def test_dp_empty_and_unknown_chords_do_not_crash():
    assert optimize_progression([]) == []

    choices = optimize_progression(_buttons(["G", "H13"]))
    assert len(choices) == 2
    assert choices[0].frets  # 已知和弦正常
    assert choices[1].frets == []  # 未知和弦只占位，不抛异常
    assert choices[1].local_cost >= 20.0


def test_transition_cost_prefers_staying_in_position():
    from dataclasses import replace

    from aitsm.arrangement.fingering_dp import transition_cost

    chords = _buttons(["C", "G"])
    pool = candidates_for(chords[1])
    open_pick = next(
        choice_from_voicing(chords[1], item, candidates_considered=len(pool), weights=COST_WEIGHTS)
        for item in pool
        if item["base_fret"] == 0
    )
    c_pick = choice_from_voicing(
        chords[0],
        next(item for item in candidates_for(chords[0]) if item["id"] == "C_x32010"),
        candidates_considered=1,
        weights=COST_WEIGHTS,
    )
    far_pick = replace(open_pick, base_fret=7)
    assert transition_cost(c_pick, far_pick) > transition_cost(c_pick, open_pick)


def test_rerank_never_worse_and_marks_engine():
    chords = _buttons(["G", "C", "D"])
    dp_choices = optimize_progression(chords)
    reranked = rerank(chords, dp_choices)

    assert len(reranked) == len(dp_choices)
    assert plan_cost(reranked) <= plan_cost(dp_choices) + 1e-9
    assert all(choice.engine == "tart-heuristic" for choice in reranked)
    for choice in reranked:
        assert validate_voicing(choice.frets, choice.symbol) == []


def test_tart_availability_is_offline_safe():
    if importlib.util.find_spec("transformers") is None:
        assert is_available() is False
        assert active_engine() == "tart-heuristic"


def test_rerank_handles_empty_and_mismatched_inputs():
    assert rerank([], []) == []
    chords = _buttons(["G", "C"])
    choices = optimize_progression(chords)
    # 长度不一致：原样返回副本，不猜
    shorter = rerank(chords, choices[:1])
    assert len(shorter) == 1 and shorter[0].voicing_id == choices[0].voicing_id

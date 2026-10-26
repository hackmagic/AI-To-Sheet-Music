"""Noise2Fret 五项可演奏性损失的移植测试。

这些公式来自 `RiccardoVib/Noise2Fret` 的 ``AuxiliaryLoss.py``（该仓库实测**只有
代码没有权重**，所以只移植数学公式）。测试用真实和弦形状验证语义方向正确。
"""

from __future__ import annotations

import pytest

from aitsm.arrangement import playability as P

# 常见开放和弦的六弦品位（低音 E → 高音 e）
C_MAJOR = [-1, 3, 2, 0, 1, 0]
G_MAJOR = [3, 2, 0, 0, 0, 3]
D_MAJOR = [-1, -1, 0, 2, 3, 2]
F_BARRE = [1, 3, 3, 2, 1, 1]
EM_OPEN = [0, 2, 2, 0, 0, 0]
IMPOSSIBLE = [1, -1, 9, 11, -1, 13]  # 跨度 12 品，几乎不可弹


# ---------------------------------------------------------------- 音级距离


def test_tonal_distance_identical_is_zero():
    assert P.jaccard_tonal_distance(C_MAJOR, C_MAJOR) == pytest.approx(0.0)


def test_tonal_distance_against_own_pitch_classes_is_zero():
    """指法本身的音级 vs 该和弦应有音级（C 大调 = C/E/G）应完全重合。"""
    assert P.pitch_classes_of(C_MAJOR) == {0, 4, 7}
    assert P.jaccard_tonal_distance(C_MAJOR, {0, 4, 7}) == pytest.approx(0.0)


def test_tonal_distance_different_chords_is_positive():
    distance = P.jaccard_tonal_distance(C_MAJOR, D_MAJOR)
    assert 0.0 < distance <= 1.0


def test_tonal_distance_disjoint_is_one():
    assert P.jaccard_tonal_distance({0}, {1}) == pytest.approx(1.0)


def test_tonal_distance_empty_sets_returns_zero():
    assert P.jaccard_tonal_distance([], []) == pytest.approx(0.0)


def test_pitch_classes_ignores_muted_strings():
    assert P.pitch_classes_of([-1, -1, -1, -1, -1, -1]) == set()
    # 只弹 6 弦 3 品 = G
    assert P.pitch_classes_of([3, -1, -1, -1, -1, -1]) == {7}


# ---------------------------------------------------------------- 把位距离


def test_fret_distance_identical_is_zero():
    assert P.fret_distance(C_MAJOR, C_MAJOR) == pytest.approx(0.0)


def test_fret_distance_ignores_muted_strings():
    """闷弦不参与比较：C 与 G 的 6 弦一个闷一个响，该弦被排除在外。"""
    left = [-1, 3, 2, 0, 1, 0]
    right = [3, 2, 0, 0, 0, 3]
    # 6 弦 left=-1 → 排除；其余 5 对：|3-2|,|2-0|,|0-0|,|1-0|,|0-3|
    # = 1 + 2 + 0 + 1 + 3 = 7，7 / 5 = 1.4
    assert P.fret_distance(left, right) == pytest.approx(1.4)


def test_fret_distance_no_overlap_is_zero():
    assert P.fret_distance([-1, -1, -1, -1, -1, -1], C_MAJOR) == pytest.approx(0.0)


# ---------------------------------------------------------------- 五度圈


def test_cof_distance_c_to_g_is_small():
    """C 与 G 在五度圈上相邻，距离应很小。"""
    assert P.cof_chord_distance({0}, {7}) == pytest.approx(1 / 6, abs=1e-6)


def test_cof_distance_c_to_fsharp_is_max():
    """C（圈位 0）与 F#（圈位 6）是对跖，归一化后应为最大值 1.0。"""
    assert P.cof_chord_distance({0}, {6}) == pytest.approx(1.0, abs=1e-6)


def test_cof_distance_identical_is_zero():
    assert P.cof_chord_distance({0, 4, 7}, {0, 4, 7}) == pytest.approx(0.0)


def test_cof_distance_tolerates_transposition():
    """转位/等音更宽容：同一和弦不同把位，五度圈距离应接近 0。"""
    assert P.cof_chord_distance(C_MAJOR, {0, 4, 7}) == pytest.approx(0.0)


# ---------------------------------------------------------------- 弦相似性


def test_string_similarity_identical_is_zero():
    assert P.string_activity_jaccard_loss(C_MAJOR, C_MAJOR) == pytest.approx(0.0)


def test_string_similarity_detects_different_string_sets():
    """C（5 弦弹 4 根）vs 只弹低音两弦 → 有差异。"""
    sparse = [3, -1, -1, -1, -1, -1]
    assert P.string_activity_jaccard_loss(C_MAJOR, sparse) > 0.5


def test_string_similarity_all_muted_is_zero():
    assert P.string_activity_jaccard_loss([-1] * 6, [-1] * 6) == pytest.approx(0.0)


def test_string_similarity_one_side_empty_is_one():
    assert P.string_activity_jaccard_loss(C_MAJOR, [-1] * 6) == pytest.approx(1.0)


# ---------------------------------------------------------------- 手跨


def test_hand_span_open_chord_is_zero():
    for chord in (C_MAJOR, G_MAJOR, EM_OPEN):
        assert P.hand_span_penalty(chord) == pytest.approx(0.0)


def test_hand_span_excludes_open_and_muted_strings():
    """空弦（0）与闷弦（-1）不算手跨：D 大调 [x,x,0,2,3,2] 跨度只有 1。"""
    assert P.hand_span_penalty(D_MAJOR) == pytest.approx(0.0)


def test_hand_span_barre_chord_zero():
    """F 横按 [1,3,3,2,1,1] 跨度 2，在 6 品阈值内。"""
    assert P.hand_span_penalty(F_BARRE) == pytest.approx(0.0)


def test_hand_span_impossible_chord_is_positive():
    penalty = P.hand_span_penalty(IMPOSSIBLE)
    assert penalty > 0.0


def test_hand_span_scales_with_excess():
    """跨度越大惩罚越大。"""
    five = [1, 2, 3, 4, 5, 6]  # 跨度 5
    eight = [1, 2, 3, 4, 5, 8]  # 跨度 7
    assert P.hand_span_penalty(five) == pytest.approx(0.0)
    assert P.hand_span_penalty(eight) > P.hand_span_penalty(five)


def test_hand_span_respects_custom_max_span():
    chord = [0, 0, 0, 0, 0, 0]  # 全空弦 → 无手跨
    assert P.hand_span_penalty(chord, max_span=2) == pytest.approx(0.0)
    # 全空弦不算手跨，即使 max_span 很小
    assert P.hand_span_penalty([1, 2, 3, -1, -1, -1], max_span=1) > 0.0


def test_hand_span_all_muted_is_zero():
    assert P.hand_span_penalty([-1] * 6) == pytest.approx(0.0)


# ---------------------------------------------------------------- 综合报告


def test_playability_report_covers_all_five():
    report = P.playability_report(
        F_BARRE, target_pcs={0, 4, 7}, previous=C_MAJOR
    )
    assert report["hand_span"] == pytest.approx(0.0)
    assert report["tonal_distance"] is not None
    assert report["cof_distance"] is not None
    assert report["fret_distance"] is not None
    assert report["string_similarity"] is not None
    assert isinstance(report["playability_cost"], float)


def test_playability_report_without_context_leaves_none():
    report = P.playability_report(C_MAJOR)
    assert report["tonal_distance"] is None
    assert report["fret_distance"] is None
    assert report["cof_distance"] is None
    assert report["string_similarity"] is None
    # 只用手跨一项也能给出综合分
    assert report["playability_cost"] == pytest.approx(report["hand_span"])


def test_playability_report_counts_sounding_strings():
    assert P.playability_report(C_MAJOR)["sounding_strings"] == 5
    assert P.playability_report([-1] * 6)["sounding_strings"] == 0


def test_playability_handles_none_values():
    """编配层可能拿到残缺指法（含 None），不能炸。"""
    report = P.playability_report([None, 3, 2, None, 1, 0])
    assert report["frets"] == [-1, 3, 2, -1, 1, 0]
    assert report["hand_span"] >= 0.0


def test_all_five_are_documented_in_module():
    for name in (
        "jaccard_tonal_distance",
        "fret_distance",
        "cof_chord_distance",
        "string_activity_jaccard_loss",
        "hand_span_penalty",
    ):
        assert callable(getattr(P, name))
        assert name in P.__all__

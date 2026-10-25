"""伴奏层拍号自洽性校验的测试。

实测背景：SheetSage2 在真实歌曲上把 **4/4 判成 2/4**（拍点间隔显示强拍在
每 2 个拍点处，但它报 ``denominator=4``）。若不校正，小节数会翻倍、和弦落位全错。
"""

from __future__ import annotations

import pytest

from aitsm.contracts import BeatGrid
from aitsm.pipeline import accompaniment_stage as A


def _grid(*, bpm=93.75, beats_per_bar=2, time_signature="2/4", beat_step=0.32, bar_beats=2, count=32):
    beats = [round(0.01 + index * beat_step, 2) for index in range(count)]
    downbeats = [round(0.01 + index * beat_step * bar_beats, 2) for index in range(count // bar_beats + 1)]
    grid = BeatGrid(
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=beats_per_bar,
        time_signature=time_signature,
    )
    return grid, {**grid.to_dict(), "bpm_display": bpm}


def test_declared_2_4_with_4_beat_strong_period_is_fixed():
    """真实误判场景：声明 2/4，但强拍每 4 个拍点出现一次 → 回退 4/4。"""
    grid, payload = _grid(bar_beats=4)
    fixed, note = A._sanity_check_meter(payload, grid)
    assert fixed["time_signature"] == "4/4"
    assert fixed["beats_per_bar"] == 4
    assert "拍号不可信" in note


def test_consistent_4_4_is_left_alone():
    """自洽的 4/4 不应被改动（不能无脑改写正确结果）。"""
    grid, payload = _grid(beats_per_bar=4, time_signature="4/4", bar_beats=4)
    fixed, note = A._sanity_check_meter(payload, grid)
    assert fixed["time_signature"] == "4/4"
    assert fixed["beats_per_bar"] == 4
    assert note == ""


def test_consistent_3_4_is_left_alone():
    grid, payload = _grid(beats_per_bar=3, time_signature="3/4", bar_beats=3, count=30)
    fixed, note = A._sanity_check_meter(payload, grid)
    assert fixed["time_signature"] == "3/4"
    assert note == ""


def test_uncommon_meter_is_flagged_but_preserved():
    """5/4 少见但**自洽**：只提示，不改写（不能把正确结果改错）。"""
    grid, payload = _grid(beats_per_bar=5, time_signature="5/4", bar_beats=5, count=40)
    fixed, note = A._sanity_check_meter(payload, grid)
    assert "不在常见集合" in note
    assert "已保留原值" in note
    assert fixed["time_signature"] == "5/4", "自洽的少见拍号必须保留"
    assert fixed["beats_per_bar"] == 5


def test_too_few_beats_returns_unchanged():
    """数据不足时不猜（少于 4 个拍点）。"""
    grid = BeatGrid(bpm=120, beats=[0.0, 0.5], downbeats=[0.0], beats_per_bar=4, time_signature="4/4")
    payload = {**grid.to_dict(), "bpm_display": 120}
    fixed, note = A._sanity_check_meter(payload, grid)
    assert note == ""
    assert fixed == payload


def test_too_few_downbeats_returns_unchanged():
    grid = BeatGrid(bpm=120, beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0], beats_per_bar=4, time_signature="4/4")
    payload = {**grid.to_dict(), "bpm_display": 120}
    _fixed, note = A._sanity_check_meter(payload, grid)
    assert note == ""


def test_implied_bars_is_clamped_to_sane_range():
    """推断出的每小节拍数必须落在 2..12，否则保守回退 4。"""
    grid, payload = _grid(beats_per_bar=2, time_signature="2/4", bar_beats=20, count=60)
    fixed, _note = A._sanity_check_meter(payload, grid)
    assert 2 <= fixed["beats_per_bar"] <= 12


def test_analysis_dicts_carries_meter_warning():
    """校验告警必须能经 analysis_dicts 传到上层（否则用户看不到）。"""
    from aitsm.contracts import AccompanimentResult, KeySignature

    grid, payload = _grid(bar_beats=4)
    result = AccompanimentResult(beats=grid, key=KeySignature(tonic="B", mode="minor"))
    data = A.analysis_dicts(result)
    assert data["beats"]["time_signature"] == "4/4"
    assert any("拍号" in item for item in data.get("warnings") or [])


def test_analysis_dicts_no_warning_for_consistent_meter():
    from aitsm.contracts import AccompanimentResult, KeySignature

    grid, _payload = _grid(beats_per_bar=4, time_signature="4/4", bar_beats=4)
    result = AccompanimentResult(beats=grid, key=KeySignature(tonic="C", mode="major"))
    data = A.analysis_dicts(result)
    assert data["beats"]["time_signature"] == "4/4"
    assert not (data.get("warnings") or [])


# ---------------------------------------------------------------- 段落 kind 规范化


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pre-chorus", "prechorus"),
        ("pre chorus", "prechorus"),
        ("Pre-Chorus", "prechorus"),
        ("chorus", "chorus"),
        ("verse", "verse"),
        ("breakdown", "bridge"),
        ("guitar solo", "solo"),
        ("weird-thing", "other"),
        ("", "other"),
        (None, "other"),
    ],
)
def test_section_kind_is_normalized_to_schema_enum(raw, expected):
    """SheetSage2 实测会输出 ``pre-chorus``（连字符），schema 只认 ``prechorus``。

    不规范化会让 analysis 校验失败（实测 3 处 kind 非法 → schema_invalid）。
    """
    assert A._normalize_section_kind(raw) == expected


def test_all_normalized_kinds_are_in_schema_enum():
    """规范化后的值必须全部落在 schema 枚举里，否则校验仍会失败。"""
    from aitsm.score.model import SECTION_KINDS

    samples = ["pre-chorus", "chorus", "verse", "breakdown", "guitar solo", "unknown", ""]
    for sample in samples:
        assert A._normalize_section_kind(sample) in SECTION_KINDS


def test_analysis_dicts_normalizes_real_pre_chorus():
    from aitsm.contracts import AccompanimentResult, KeySignature, Section

    grid, _payload = _grid(beats_per_bar=4, time_signature="4/4", bar_beats=4)
    result = AccompanimentResult(
        beats=grid,
        key=KeySignature(tonic="B", mode="minor"),
        sections=[
            Section(start=0.0, end=1.0, kind="pre-chorus"),
            Section(start=1.0, end=2.0, kind="chorus"),
        ],
    )
    data = A.analysis_dicts(result)
    assert [item["kind"] for item in data["sections"]] == ["prechorus", "chorus"]


# ---------------------------------------------------------------- 段落能量先验


def test_sections_get_energy_prior_by_kind():
    """SheetSage2 不给段落能量；若一律用默认 0.5，所有段落会落到同一个节奏型。

    实测：20 个段落全是 pop8。按 kind 给类型先验后，副歌能拿到 drive8。
    """
    from aitsm.contracts import AccompanimentResult, KeySignature, Section

    grid, _payload = _grid(beats_per_bar=4, time_signature="4/4", bar_beats=4)
    result = AccompanimentResult(
        beats=grid,
        key=KeySignature(tonic="B", mode="minor"),
        sections=[
            Section(start=0.0, end=1.0, kind="intro"),
            Section(start=1.0, end=2.0, kind="verse"),
            Section(start=2.0, end=3.0, kind="chorus"),
        ],
    )
    energies = {item["kind"]: item["energy"] for item in A.analysis_dicts(result)["sections"]}
    assert energies["intro"] < energies["verse"] < energies["chorus"]
    assert energies["chorus"] > 0.6, "副歌能量应明显高于主歌"

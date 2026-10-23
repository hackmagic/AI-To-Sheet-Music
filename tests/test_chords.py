"""和弦识别单测：模板打分 / Viterbi / 后处理 / 合成音频端到端。"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from aitsm.pipeline import chords as chords_module

np = pytest.importorskip("numpy")


def _write_triad_song(path: Path, *, sample_rate: int = 22050) -> Path:
    """前 4 秒 C 大三和弦，后 4 秒 G 大三和弦（含低音），模拟吉他分解和弦。"""
    seconds = 8.0
    frames = int(seconds * sample_rate)
    data = np.zeros(frames, dtype=np.float64)
    chords_plan = [
        {"start": 0.0, "end": 4.0, "freqs": [130.81, 261.63, 329.63, 392.00]},  # C
        {"start": 4.0, "end": 8.0, "freqs": [98.00, 196.00, 246.94, 293.66]},   # G
    ]
    for plan in chords_plan:
        start = int(plan["start"] * sample_rate)
        end = int(plan["end"] * sample_rate)
        local = np.arange(end - start) / sample_rate
        # 每 0.5 秒一次「扫弦」包络，让 chroma 有起落
        strum = 0.55 + 0.45 * np.abs(np.sin(2.0 * np.pi * 1.0 * local))
        tone = np.zeros_like(local)
        for index, frequency in enumerate(plan["freqs"]):
            weight = 1.0 if index == 0 else 0.55
            tone += weight * np.sin(2.0 * np.pi * frequency * local)
            tone += 0.25 * weight * np.sin(2.0 * np.pi * frequency * 2.0 * local)
        data[start:end] += tone * strum
    data *= 8000.0 / max(1e-9, float(np.max(np.abs(data))))
    pcm = np.clip(data, -32767, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def make_beats(bpm: float = 120.0, seconds: float = 8.0, beats_per_bar: int = 4) -> dict:
    interval = 60.0 / bpm
    times = [round(index * interval, 4) for index in range(int(seconds / interval) + 1)]
    return {
        "bpm": bpm,
        "bpm_display": bpm,
        "beats": times,
        "downbeats": times[::beats_per_bar],
        "beats_per_bar": beats_per_bar,
        "time_signature": "4/4",
        "confidence": 0.8,
    }


def _event(symbol: str, root_pc: int, quality: str, start: float, end: float, confidence: float = 0.6) -> dict:
    return {
        "start": start,
        "end": end,
        "symbol": symbol,
        "root_pc": root_pc,
        "quality": quality,
        "bass_pc": root_pc,
        "score": 0.8,
        "slot_confidence": confidence,
        "duration": end - start,
        "slot_count": 1,
    }


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #

def test_quality_template_has_chord_tones():
    template = chords_module.quality_template(0, "")
    assert template[0] > 0 and template[4] > 0 and template[7] > 0
    assert template[2] == 0 and template[3] == 0

    minor = chords_module.quality_template(9, "m")  # A minor
    assert minor[9] > 0 and minor[0] > 0 and minor[4] > 0


def test_chord_tone_mask_shape():
    mask = chords_module.chord_tone_mask(0, "7")
    assert mask.shape == (12,)
    assert bool(mask[0]) and bool(mask[4]) and bool(mask[7]) and bool(mask[10])


def test_score_slot_identifies_major_triad():
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = [1.0, 0.8, 0.8]
    top = chords_module.score_slot(chroma, None)[0]
    assert top["root_pc"] == 0
    assert top["quality"] == ""
    assert top["slot_confidence"] > 0.4


def test_score_slot_identifies_minor_triad():
    chroma = np.zeros(12)
    chroma[[9, 0, 4]] = [1.0, 0.8, 0.8]
    top = chords_module.score_slot(chroma, None)[0]
    assert top["root_pc"] == 9
    assert top["quality"] == "m"


def test_score_slot_uses_bass_for_inversion_root():
    # chroma 同时含 C 和 Am 的组成音时，低音决定根音判断
    chroma = np.zeros(12)
    chroma[[0, 4, 7, 9]] = [0.9, 0.8, 0.6, 0.5]
    bass = np.zeros(12)
    bass[9] = 1.0
    top = chords_module.score_slot(chroma, bass)[0]
    assert top["root_pc"] == 9


def test_transition_penalty_discourages_quality_flicker():
    """同根换质量要比重根换和弦更贵——真实歌里 C→G 常见，C→Cmaj7→Csus4 抖动是噪声。"""
    c_major = {"symbol": "C", "root_pc": 0, "quality": ""}
    c_major7 = {"symbol": "Cmaj7", "root_pc": 0, "quality": "maj7"}
    g_major = {"symbol": "G", "root_pc": 7, "quality": ""}
    assert chords_module.transition_penalty(c_major, c_major) == 0.0
    assert chords_module.transition_penalty(c_major, g_major) > 0.0
    assert chords_module.transition_penalty(c_major, c_major7) > chords_module.transition_penalty(c_major, g_major)


def test_viterbi_decode_suppresses_isolated_weak_outlier():
    c = {"symbol": "C", "root_pc": 0, "quality": "", "score": 0.90, "slot_confidence": 0.6}
    g = {"symbol": "G", "root_pc": 7, "quality": "", "score": 0.40, "slot_confidence": 0.3}
    c_weak = {**c, "score": 0.86}
    g_weak_win = {**g, "score": 0.90}  # 只比 C 强一点点
    emissions = [[c, g], [c_weak, g_weak_win], [c, g]]
    path = chords_module.viterbi_decode(emissions)
    # 单拍上微弱的 G 不敌两次换和弦的代价，应被平滑回 C
    assert [state["symbol"] for state in path] == ["C", "C", "C"]


def test_viterbi_decode_keeps_strong_real_change():
    c = {"symbol": "C", "root_pc": 0, "quality": "", "score": 0.90, "slot_confidence": 0.6}
    g = {"symbol": "G", "root_pc": 7, "quality": "", "score": 0.40, "slot_confidence": 0.3}
    # 证据差距远大于换和弦代价 → 真实的半小节和弦变化必须保留
    emissions = [[c, g], [{**c, "score": 0.30}, {**g, "score": 0.95}], [c, g]]
    path = chords_module.viterbi_decode(emissions)
    assert [state["symbol"] for state in path] == ["C", "G", "C"]


def test_apply_change_hysteresis_keeps_chord_when_gain_small():
    c = {"symbol": "C", "root_pc": 0, "quality": "", "score": 0.90, "slot_confidence": 0.6}
    g = {"symbol": "G", "root_pc": 7, "quality": "", "score": 0.40, "slot_confidence": 0.3}
    # 小节中途（1.3s 不在小节线上）G 只比 C 高 0.03，不足滞回阈值 0.04 → 保持 C
    emissions = [[c, g], [{**c, "score": 0.40}, {**g, "score": 0.43}]]
    states = chords_module.apply_change_hysteresis(
        [{**c}, {**g, "score": 0.43}], emissions, [0.0, 1.3], downbeats=[0.0, 2.6],
    )
    assert [state["symbol"] for state in states] == ["C", "C"]


def test_apply_change_hysteresis_allows_change_on_bar_line():
    c = {"symbol": "C", "root_pc": 0, "quality": "", "score": 0.90, "slot_confidence": 0.6}
    g = {"symbol": "G", "root_pc": 7, "quality": "", "score": 0.40, "slot_confidence": 0.3}
    # 2.6s 正是小节线 → 不受滞回约束，即使分差很小也换
    emissions = [[c, g], [{**c, "score": 0.41}, {**g, "score": 0.42}]]
    states = chords_module.apply_change_hysteresis(
        [{**c}, {**g, "score": 0.42}], emissions, [0.0, 2.6], downbeats=[0.0, 2.6],
    )
    assert [state["symbol"] for state in states] == ["C", "G"]


def test_apply_change_hysteresis_allows_large_gain_mid_bar():
    c = {"symbol": "C", "root_pc": 0, "quality": "", "score": 0.90, "slot_confidence": 0.6}
    g = {"symbol": "G", "root_pc": 7, "quality": "", "score": 0.40, "slot_confidence": 0.3}
    # 小节中途但证据很强（+0.20）→ 必须换
    emissions = [[c, g], [{**c, "score": 0.35}, {**g, "score": 0.55}]]
    states = chords_module.apply_change_hysteresis(
        [{**c}, {**g, "score": 0.55}], emissions, [0.0, 1.3], downbeats=[0.0, 2.6],
    )
    assert [state["symbol"] for state in states] == ["C", "G"]


def test_format_symbol_with_slash_bass():
    assert chords_module.format_symbol(0, "") == "C"
    assert chords_module.format_symbol(0, "maj7") == "Cmaj7"
    assert chords_module.format_symbol(0, "", 7) == "C/G"
    assert chords_module.format_symbol(9, "m", 4) == "Am/E"
    assert chords_module.format_symbol(0, "", 0) == "C"


def test_build_slot_times_covers_tail():
    beats = make_beats(bpm=120.0, seconds=1.0)
    slots = chords_module.build_slot_times(beats, 2.5)  # 尾巴比拍点长
    assert slots[0][0] == 0.0
    assert slots[-1][1] == pytest.approx(2.5, abs=0.01)
    assert all(end > start for start, end in slots)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(slots, slots[1:]))


def test_cleanup_events_absorbs_short_chord():
    events = [
        _event("C", 0, "", 0.0, 2.0),
        _event("G", 7, "", 2.0, 2.25),   # 过短
        _event("C", 0, "", 2.25, 4.0),
    ]
    cleaned = chords_module.cleanup_events(events, min_duration=0.5)
    assert [event["symbol"] for event in cleaned] == ["C"]
    assert cleaned[0]["start"] == 0.0
    assert cleaned[0]["end"] == pytest.approx(4.0, abs=0.01)


def test_cleanup_events_merges_adjacent_same_symbol():
    events = [
        _event("C", 0, "", 0.0, 2.0),
        _event("C", 0, "", 2.0, 4.0),
        _event("G", 7, "", 4.0, 6.0),
    ]
    cleaned = chords_module.cleanup_events(events, min_duration=0.5)
    assert [event["symbol"] for event in cleaned] == ["C", "G"]
    assert cleaned[0]["end"] == pytest.approx(4.0, abs=0.01)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #

def test_recognize_chords_on_synthetic_progression(tmp_path: Path):
    pytest.importorskip("librosa")
    source = _write_triad_song(tmp_path / "progression.wav")
    beats = make_beats(bpm=120.0, seconds=8.0)
    chords = chords_module.recognize_chords(
        source, beats, key={"tonic_pc": 0, "mode": "major", "tonic": "C"}, total_seconds=8.0,
    )

    assert chords, "应识别出和弦"
    first = chords[0]
    last = chords[-1]
    assert first["root_pc"] == 0 and first["quality"] == "", f"第一段应是 C，实际 {first['symbol']}"
    assert last["root_pc"] == 7 and last["quality"] == "", f"最后一段应是 G，实际 {last['symbol']}"

    # 事件必须连续覆盖、时间单调、置信度合法
    assert chords[0]["start"] == pytest.approx(0.0, abs=0.01)
    assert chords[-1]["end"] == pytest.approx(8.0, abs=0.15)
    for previous, current in zip(chords, chords[1:]):
        assert previous["end"] == pytest.approx(current["start"], abs=1e-3)
        assert previous["symbol"] != current["symbol"]
    for chord in chords:
        assert 0.0 <= chord["confidence"] <= 1.0
        assert 0 <= chord["root_pc"] <= 11
        assert 0 <= chord["bass_pc"] <= 11
        assert chord["start"] < chord["end"]
        assert chord["source"]


def test_recognize_chords_handles_degenerate_beats(tmp_path: Path):
    pytest.importorskip("librosa")
    source = _write_triad_song(tmp_path / "progression.wav")
    beats = {"bpm": 120.0, "beats": [], "downbeats": [], "beats_per_bar": 4}
    chords = chords_module.recognize_chords(source, beats, total_seconds=4.0)
    # 没有拍点也不能崩，至少要给出可用事件或空列表
    assert isinstance(chords, list)
    assert all(0.0 <= chord["start"] < 8.0 for chord in chords)


def test_diatonic_tables_are_semitone_indexed():
    """调内表按半音数索引：B 小调的 Gmaj7/Bm/D/A/F#m 都应判为调内，C 大三和弦不是。"""
    from aitsm.pipeline import keydet as keydet_module

    minor = keydet_module.MINOR_DIATONIC_QUALITIES
    assert "" in minor[8] or "maj7" in minor[8]     # VI 级：G（相对 B 是 8 个半音）
    assert "m" in minor[0]                          # i 级：Bm
    assert "" in minor[3]                           # III 级：D
    assert "" in minor[10]                          # VII 级：A
    assert "m" in minor[7]                          # v 级：F#m
    assert 1 not in minor and 4 not in minor        # C、E 不在 B 小调音阶内

    major = keydet_module.MAJOR_DIATONIC_QUALITIES
    assert "" in major[0] and "" in major[7]        # I / V
    assert "m" in major[9]                          # vi
    assert "dim" in major[11]                       # vii°


def test_diatonic_triad_quality_resolves_color_chords():
    """B 小调里 Asus2 应回退成 A 大三和弦，C#dim 保持减三和弦。"""
    bm = {"tonic_pc": 11, "mode": "minor"}
    assert chords_module.diatonic_triad_quality(9, bm) == ""     # A → A
    assert chords_module.diatonic_triad_quality(11, bm) == "m"   # B → Bm
    assert chords_module.diatonic_triad_quality(2, bm) == ""     # D 是 III 级（大三和弦）
    assert chords_module.diatonic_triad_quality(1, bm) is None   # C# 是 ii°，没有自然三和弦
    assert chords_module.diatonic_triad_quality(0, bm) is None   # C 不在 B 小调音阶内


def test_simplify_qualities_falls_back_to_diatonic_triad():
    """置信度不足的 sus4 必须被回退成调内三和弦。"""
    bm = {"tonic_pc": 11, "mode": "minor"}
    events = [
        {"start": 0.0, "end": 2.0, "symbol": "Asus4", "confidence": 0.45,
         "root_pc": 9, "bass_pc": 9, "quality": "sus4", "bass_strength": 0.1, "duration": 2.0},
        {"start": 2.0, "end": 4.0, "symbol": "Asus4", "confidence": 0.80,
         "root_pc": 9, "bass_pc": 9, "quality": "sus4", "bass_strength": 0.1, "duration": 2.0},
    ]
    simplified = chords_module.simplify_qualities(events, bm)
    assert simplified[0]["quality"] == ""
    assert simplified[0]["symbol"] == "A"
    assert simplified[0]["simplified_from"] == "sus4"
    # 高置信度的那条保留原样
    assert simplified[-1]["quality"] == "sus4"


# --------------------------------------------------------------------------- #
# 特征融合（中值 + 谱归一化）与重复一致性
# --------------------------------------------------------------------------- #


def _slot(chroma_columns: list[list[float]]) -> object:
    return np.asarray(chroma_columns, dtype=np.float64).T


def test_fusion_matrix_is_robust_to_attack_frames() -> None:
    """槽位里混进一两帧「宽带攻击」（鼓点）时，中值聚合不该被带偏。

    构造一段 C 大调三和弦的 chroma，其中第 2 帧是鼓点造成的全通道高能量；
    融合后的矩阵仍应在 C/E/G 上最高，而不是被那一帧拉成均匀分布。
    """
    c_major = [1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
    attack = [1.0] * 12
    columns = [c_major, attack, c_major, c_major]
    chroma = _slot(columns)
    frame_times = [(0, 4)]
    fused = chords_module._slot_matrix_fused(chroma, frame_times, np, fusion=True)
    column = fused[:, 0]
    assert column.shape == (12,)
    assert pytest.approx(float(column.sum()), abs=1e-9) == 1.0
    top = np.argsort(column)[::-1][:3]
    assert {int(index) for index in top} == {0, 4, 7}

    legacy = chords_module._slot_matrix_fused(chroma, frame_times, np, fusion=False, smooth=False)
    legacy_column = legacy[:, 0]
    # 两版都能排出正确的前三名，区别在「和弦音与非和弦音的差距」：
    # 均值聚合把攻击帧的能量平均到所有音级上，非和弦音被抬到 0.05 上下；
    # 中值 + 谱归一化融合后，非和弦音几乎归零，差距拉大三倍以上。
    fused_gap = float(column[0]) / max(1e-9, float(column[1]))
    legacy_gap = float(legacy_column[0]) / max(1e-9, float(legacy_column[1]))
    assert fused_gap > legacy_gap * 2.5


def test_fusion_weights_are_normalised_per_slot() -> None:
    columns = [[0.2] * 12, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    chroma = _slot(columns)
    fused = chords_module._slot_matrix_fused(chroma, [(0, 2)], np, fusion=True, smooth=False)
    assert pytest.approx(float(fused[:, 0].sum()), abs=1e-9) == 1.0
    # 单音级占优的槽位：融合后最大分量应该明显高于均匀分布的 1/12
    assert float(fused[:, 0].max()) > 0.25
    assert int(np.argmax(fused[:, 0])) == 0


def test_repetition_weights_ignore_adjacent_slots() -> None:
    """相邻槽位天然相似，必须排除，否则「永远不换和弦」会被奖励。"""
    rng = np.random.default_rng(7)
    matrix = rng.random((12, 40)) + 0.1
    weights = chords_module._repetition_weights(matrix, np)
    assert weights.shape == (40, 40)
    radius = chords_module._REPETITION_EXCLUDE
    for index in range(40):
        for offset in range(1, radius + 1):
            if index + offset < 40:
                assert weights[index, index + offset] == 0.0
        assert weights[index, index] == 0.0


def test_repetition_weights_empty_when_nothing_repeats() -> None:
    """每个槽位都和别的槽位不像时，不该产生任何权重。"""
    matrix = np.zeros((12, 12), dtype=np.float64)
    for index in range(12):
        matrix[index, index] = 1.0
    weights = chords_module._repetition_weights(matrix, np)
    assert float(weights.sum()) == 0.0


def test_repetition_prior_boosts_neighbour_majority() -> None:
    """邻居槽位大多解成 Am 时，当前槽位被加分的也是 Am。"""
    emissions = [
        [
            {"root_pc": 9, "quality": "m", "symbol": "Am", "score": 0.50},
            {"root_pc": 0, "quality": "", "symbol": "C", "score": 0.48},
        ]
        for _ in range(4)
    ]
    states = [
        {"root_pc": 9, "quality": "m", "symbol": "Am"},
        {"root_pc": 9, "quality": "m", "symbol": "Am"},
        {"root_pc": 9, "quality": "m", "symbol": "Am"},
        {"root_pc": 0, "quality": "", "symbol": "C"},
    ]
    weights = np.zeros((4, 4), dtype=np.float64)
    weights[3, 0] = weights[3, 1] = weights[3, 2] = 1.0
    boosted, touched = chords_module._apply_repetition_prior(emissions, weights, states, np)
    assert touched > 0
    am_entry = next(item for item in boosted[3] if item["symbol"] == "Am")
    assert float(am_entry["score"]) > 0.50
    # 前三个槽位没有邻居权重，排序与分数不应改变
    assert [item["symbol"] for item in boosted[0]] == ["Am", "C"]
    assert float(boosted[0][0]["score"]) == pytest.approx(0.50)


def test_recognize_chords_reports_stats(tmp_path: Path) -> None:
    """统计块要能被上层写进 manifest：融合开关、槽位数必须可见。"""
    path = _write_triad_song(tmp_path / "triad.wav")
    beats = {
        "bpm": 60.0, "bpm_display": 60.0, "beats_per_bar": 4, "time_signature": "4/4",
        "beats": [index * 1.0 for index in range(9)],
        "downbeats": [0.0, 4.0, 8.0],
        "confidence": 1.0,
    }
    stats: dict = {}
    events = chords_module.recognize_chords(
        path, beats, total_seconds=8.0, feature_fusion=True, repetition_pass=False, stats=stats
    )
    assert events
    assert stats["feature_fusion"] is True
    assert stats["enabled"] is False
    assert stats["slots"] >= 1
    assert all("source" in event for event in events)

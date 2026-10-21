"""结构切分单测：SSM/新颖度/聚类/标签/兜底，以及合成歌曲端到端。"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from aitsm.pipeline import structure as structure_module

np = pytest.importorskip("numpy")


def make_beats(bpm: float = 120.0, seconds: float = 32.0, beats_per_bar: int = 4) -> dict:
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


def write_sectioned_song(path: Path, *, sample_rate: int = 22050, seconds: float = 32.0) -> Path:
    """合成「安静前奏 → 主歌 → 明亮副歌 → 尾奏」，每段音色/能量明显不同。"""
    frames = int(seconds * sample_rate)
    data = np.zeros(frames, dtype=np.float64)
    plan = [
        (0.0, 8.0, 220.0, 0.20),   # 前奏：低能量
        (8.0, 16.0, 330.0, 0.35),  # 主歌
        (16.0, 26.0, 660.0, 0.85),  # 副歌：明亮、高能量
        (26.0, 32.0, 220.0, 0.18),  # 尾奏
    ]
    for start_sec, end_sec, frequency, amplitude in plan:
        start = int(start_sec * sample_rate)
        end = min(frames, int(end_sec * sample_rate))
        local = np.arange(end - start) / sample_rate
        signal = amplitude * np.sin(2.0 * np.pi * frequency * local)
        signal += 0.4 * amplitude * np.sin(2.0 * np.pi * frequency * 2.0 * local)
        signal += 0.2 * amplitude * np.sin(2.0 * np.pi * frequency * 3.01 * local)
        # 每 0.5 秒一次脉冲，制造频谱差异（音色证据）
        signal *= 0.7 + 0.3 * np.abs(np.sin(2.0 * np.pi * 2.0 * local))
        data[start:end] += signal
    data *= 8000.0 / max(1e-9, float(np.max(np.abs(data))))
    pcm = np.clip(data, -32767, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def _make_chords(beats: dict, symbols: list[str]) -> list[dict]:
    chord_names = {"C": 0, "G": 7, "Am": 9, "F": 5}
    times = beats["beats"]
    result = []
    for index in range(0, len(times) - 1, 4):
        start = times[index]
        end = times[min(len(times) - 1, index + 4)]
        symbol = symbols[(index // 4) % len(symbols)]
        result.append({
            "start": start,
            "end": end,
            "symbol": symbol,
            "confidence": 0.7,
            "source": "test",
            "root_pc": chord_names.get(symbol, 0),
            "bass_pc": chord_names.get(symbol, 0),
        })
    return result


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #

def test_build_bars_from_downbeats():
    beats = make_beats(bpm=120.0, seconds=8.0)
    bars = structure_module.build_bars(beats, 8.0)
    assert len(bars) >= 3
    assert bars[0][0] == pytest.approx(0.0, abs=0.01)
    assert bars[-1][1] == pytest.approx(8.0, abs=0.05)
    assert all(end > start for start, end in bars)


def test_build_bars_without_downbeats_uses_beats_per_bar():
    beats = {"beats": [index * 0.5 for index in range(17)], "downbeats": [], "beats_per_bar": 4}
    bars = structure_module.build_bars(beats, 8.0)
    assert len(bars) >= 3
    assert bars[0][1] - bars[0][0] == pytest.approx(2.0, abs=0.1)


def test_cosine_similarity_matrix_bounds():
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    matrix = structure_module.cosine_similarity_matrix(features)
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert matrix[0, 2] < 0.6
    assert np.allclose(np.diag(matrix), 1.0)


def test_zscore_rows_standardizes():
    matrix = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    standardized = structure_module.zscore_rows(matrix)
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-9)


def test_path_enhance_keeps_block_structure():
    block = np.ones((8, 8))
    block[:4, :4] = 0.9
    block[4:, 4:] = 0.9
    block[:4, 4:] = 0.1
    block[4:, :4] = 0.1
    enhanced = structure_module.path_enhance(block, width=2)
    assert enhanced[1, 1] > enhanced[1, 6]


def test_checkerboard_novelty_peaks_at_block_boundary():
    # 4 小节 A + 4 小节 B，B 与 A 完全不同
    size = 8
    features = np.zeros((size, 6))
    features[:4, 0] = 1.0
    features[4:, 1] = 1.0
    similarity = structure_module.cosine_similarity_matrix(features)
    novelty = structure_module.checkerboard_novelty(similarity, size=4)
    assert np.argmax(novelty) == 4


def test_pick_boundaries_finds_block_edges():
    novelty = np.zeros(16)
    novelty[8] = 1.0
    novelty[4] = 0.6
    bars = [(index * 2.0, index * 2.0 + 2.0) for index in range(16)]
    boundaries = structure_module.pick_boundaries(novelty, bars, min_bars=2)
    assert 8 in boundaries
    assert all(0 < value < 16 for value in boundaries)


def test_cluster_segments_groups_repeats():
    features = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.05],
        [0.0, 1.0, 0.02],
    ])
    labels = structure_module.cluster_segments(features, threshold=0.7)
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_assign_kinds_marks_chorus_and_intro():
    segments = [
        {"start": 0.0, "end": 8.0, "energy": 0.15, "confidence": 0.5},
        {"start": 8.0, "end": 24.0, "energy": 0.5, "confidence": 0.5},
        {"start": 24.0, "end": 44.0, "energy": 0.95, "confidence": 0.5},
        {"start": 44.0, "end": 60.0, "energy": 0.5, "confidence": 0.5},
        {"start": 60.0, "end": 80.0, "energy": 0.9, "confidence": 0.5},
    ]
    labels = [0, 1, 2, 1, 2]  # 前奏独一份；主歌、副歌各重复两次
    sections = structure_module.assign_kinds(segments, labels, total_seconds=80.0)
    kinds = [section["kind"] for section in sections]
    assert kinds[0] == "intro"
    assert kinds[2] == "chorus" and kinds[4] == "chorus"
    assert kinds[1] == "verse" and kinds[3] == "verse"
    # 重复段落要能指回第一次出现
    assert sections[4]["repeat_of"] == sections[2]["label"]
    assert sections[3]["repeat_of"] == sections[1]["label"]


def test_finalize_sections_covers_timeline():
    sections = [
        {"start": 0.5, "end": 9.0, "kind": "intro", "label": "前奏 1", "energy": 0.2,
         "repeat_of": None, "confidence": 0.4},
        {"start": 9.2, "end": 20.0, "kind": "verse", "label": "主歌 1", "energy": 0.5,
         "repeat_of": None, "confidence": 0.4},
        {"start": 20.0, "end": 20.5, "kind": "interlude", "label": "间奏 1", "energy": 0.4,
         "repeat_of": None, "confidence": 0.4},  # 太短，应被吸收
        {"start": 20.5, "end": 30.0, "kind": "chorus", "label": "副歌 1", "energy": 0.9,
         "repeat_of": None, "confidence": 0.4},
    ]
    finalized = structure_module.finalize_sections(sections, total_seconds=30.0)
    assert finalized[0]["start"] == 0.0
    assert finalized[-1]["end"] == pytest.approx(30.0, abs=0.01)
    for previous, current in zip(finalized, finalized[1:]):
        assert previous["end"] == pytest.approx(current["start"], abs=1e-6)
        assert current["end"] - current["start"] >= 2.0
    assert all(0.0 <= section["energy"] <= 1.0 for section in finalized)


def test_fallback_layout_is_valid():
    sections = structure_module.fallback_layout(200.0, make_beats(seconds=200.0))
    assert len(sections) >= 4
    assert sections[0]["start"] == 0.0
    assert sections[-1]["end"] == pytest.approx(200.0, abs=0.01)
    assert sections[0]["kind"] == "intro"
    assert sections[-1]["kind"] == "outro"
    for previous, current in zip(sections, sections[1:]):
        assert previous["end"] == pytest.approx(current["start"], abs=1e-6)
    assert all(section["confidence"] <= 0.3 for section in sections)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #

def test_analyze_structure_on_synthetic_song(tmp_path: Path):
    pytest.importorskip("librosa")
    source = write_sectioned_song(tmp_path / "song.wav", seconds=32.0)
    beats = make_beats(bpm=120.0, seconds=32.0)
    chords = _make_chords(beats, ["C", "G", "Am", "F"])
    sections = structure_module.analyze_structure(source, beats, chords, 32.0)

    assert sections, "必须返回段落"
    assert sections[0]["start"] == pytest.approx(0.0, abs=0.01)
    assert sections[-1]["end"] == pytest.approx(32.0, abs=0.05)
    valid_kinds = set(structure_module.KIND_LABELS)
    for previous, current in zip(sections, sections[1:]):
        assert previous["end"] == pytest.approx(current["start"], abs=1e-3)
    for section in sections:
        assert section["kind"] in valid_kinds
        assert 0.0 <= section["energy"] <= 1.0
        assert section["start"] < section["end"]
        assert section["label"]
        assert section["repeat_of"] is None or isinstance(section["repeat_of"], str)
        assert 0.0 <= section["confidence"] <= 1.0


def test_analyze_structure_too_short_uses_fallback(tmp_path: Path):
    pytest.importorskip("librosa")
    source = write_sectioned_song(tmp_path / "short.wav", seconds=2.0)
    sections = structure_module.analyze_structure(source, make_beats(seconds=2.0), [], 2.0)
    assert sections
    assert sections[0]["start"] == 0.0
    assert sections[-1]["end"] == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- #
# 整曲同一循环的诚实提示（run 层的辅助判断）
# --------------------------------------------------------------------------- #

def _loop_chords(symbols: list[str], bar_seconds: float) -> list[dict]:
    events = []
    for index, symbol in enumerate(symbols):
        events.append({
            "start": round(index * bar_seconds, 4),
            "end": round((index + 1) * bar_seconds, 4),
            "symbol": symbol,
        })
    return events


def test_chord_loop_period_detects_four_bar_loop():
    from aitsm.pipeline import run as run_module

    beats = make_beats(bpm=120.0, seconds=32.0, beats_per_bar=4)
    loop = ["C", "Am", "F", "G"] * 6
    chords = _loop_chords(loop, bar_seconds=2.0)
    assert run_module._chord_loop_period(chords, beats) == 4


def test_looks_uniform_structure_true_for_pure_loop_false_for_varied():
    from aitsm.pipeline import run as run_module

    beats = make_beats(bpm=120.0, seconds=32.0, beats_per_bar=4)
    sections = [
        {"start": 0.0, "end": 8.0, "kind": "verse", "energy": 0.5},
        {"start": 8.0, "end": 16.0, "kind": "chorus", "energy": 0.6},
        {"start": 16.0, "end": 24.0, "kind": "verse", "energy": 0.55},
    ]
    loop_chords = _loop_chords(["C", "Am", "F", "G"] * 6, bar_seconds=2.0)
    assert run_module._looks_uniform_structure(sections, loop_chords, beats=beats)

    # 段落和弦素材完全不同的歌：不应被判定为「整曲一致」
    varied = _loop_chords(
        ["C", "Am", "F", "G", "F", "G", "Em", "Am", "Dm", "G", "C", "Am", "Bb", "F", "Gm", "C"],
        bar_seconds=2.0,
    )
    varied = [dict(event) for event in varied]
    for event in varied:
        event["symbol"] = event["symbol"] + ("7" if int(event["start"]) >= 16 else "")
    assert not run_module._looks_uniform_structure(sections, varied, beats=beats)

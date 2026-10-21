"""F8：歌曲结构切分。

思路：以「小节」为最小单位（由 beats 的 downbeats 给出），
用三重证据构造自相似矩阵（SSM）：

- 音色：MFCC（均值 + 标准差）
- 和声：chroma 均值
- 能量/音质：RMS、频谱质心、谱宽、过零率

在 SSM 上做对角路径增强（抑制单点噪声）→ 棋盘核求新颖度曲线 → 取峰得边界；
再用「段落平均特征的层次聚类」判断哪些段落重复（repeat_of），
最后结合能量高低与位置给出 intro/verse/chorus/bridge 等标签。

对外接口（契约不可改）：``analyze_structure(source_wav, beats, chords, total_seconds)``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aitsm.pipeline.keydet import note_name

SAMPLE_RATE = 22050
HOP_LENGTH = 512

KIND_LABELS = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "预副歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "interlude": "间奏",
    "solo": "独奏",
    "outro": "尾奏",
    "other": "其他",
}

# 切分参数
_MIN_SECTION_SECONDS = 4.0
_MIN_SECTION_BARS = 4
_MAX_SECTION_BARS = 16
_MAX_SECTIONS = 12
_MAX_BOUNDARIES = 12
_REPEAT_SIMILARITY = 0.70  # 段落平均特征的余弦相似度阈值


def _require_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("结构分析需要 numpy 与 librosa。") from exc
    return librosa, np


def _require_np():
    """纯函数只需 numpy；单独拆出来，未装 librosa 时也能对内核做单测。"""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("该函数需要 numpy，请先安装核心依赖。") from exc
    return np


# --------------------------------------------------------------------------- #
# 纯函数内核
# --------------------------------------------------------------------------- #

def build_bars(beats: dict[str, Any], total_seconds: float) -> list[tuple[float, float]]:
    """由 downbeats 构造小节区间；downbeats 不足时按 beats_per_bar 现推。"""
    total = max(0.0, float(total_seconds))
    downbeats = [float(value) for value in (beats.get("downbeats") or []) if 0.0 <= float(value) < total]
    if len(downbeats) < 2:
        beat_list = [float(value) for value in (beats.get("beats") or []) if 0.0 <= float(value) < total]
        per_bar = int(beats.get("beats_per_bar") or 4)
        if len(beat_list) >= 2:
            downbeats = beat_list[:: max(1, per_bar)]
        else:
            # 完全没有拍点：退化成 2 秒一格
            downbeats = [index * 2.0 for index in range(int(total // 2.0) + 1)]
    if len(downbeats) < 2:
        return [(0.0, total)] if total > 0 else []
    bars: list[tuple[float, float]] = []
    for index, start in enumerate(downbeats):
        end = downbeats[index + 1] if index + 1 < len(downbeats) else total
        if end - start <= 0.05:
            continue
        bars.append((round(start, 4), round(min(end, total), 4)))
    if bars and bars[-1][1] < total - 0.15:
        bars.append((bars[-1][1], round(total, 4)))
    return bars


def frame_ranges(bars: list[tuple[float, float]], sr: int, hop_length: int) -> list[tuple[int, int]]:
    """小节时间 → 帧区间。"""
    ranges: list[tuple[int, int]] = []
    for start, end in bars:
        lo = int(round(start * sr / hop_length))
        hi = max(lo + 1, int(round(end * sr / hop_length)))
        ranges.append((lo, hi))
    return ranges


def aggregate_bars(feature: Any, ranges: list[tuple[int, int]], *, mode: str = "mean") -> Any:
    """按小节区间聚合特征矩阵（帧 × 维度）。"""
    np = _require_np()
    matrix = np.asarray(feature, dtype=np.float64)
    width = matrix.shape[0]
    rows = []
    for lo, hi in ranges:
        lo = max(0, min(int(lo), width - 1))
        hi = max(lo + 1, min(int(hi), width))
        block = matrix[lo:hi]
        if mode == "std":
            rows.append(block.std(axis=0) if block.size else np.zeros(matrix.shape[1]))
        else:
            rows.append(block.mean(axis=0) if block.size else np.zeros(matrix.shape[1]))
    return np.stack(rows) if rows else np.zeros((0, matrix.shape[1]))


def zscore_rows(matrix: Any) -> Any:
    """逐维度 z-score，避免不同特征量纲互相压倒。"""
    np = _require_np()
    values = np.asarray(matrix, dtype=np.float64)
    if values.size == 0:
        return values
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (values - mean) / std


def cosine_similarity_matrix(features: Any) -> Any:
    """归一化后的成对余弦相似度（映射到 0..1）。"""
    np = _require_np()
    values = np.asarray(features, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = values / norms
    similarity = normalized @ normalized.T
    return np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)


def combine_similarity(matrices: list[tuple[Any, float]]) -> Any:
    """加权融合多个 SSM。"""
    np = _require_np()
    total_weight = sum(weight for _, weight in matrices) or 1.0
    combined = None
    for matrix, weight in matrices:
        values = np.asarray(matrix, dtype=np.float64) * (weight / total_weight)
        combined = values if combined is None else combined + values
    if combined is None:
        return np.zeros((0, 0))
    return np.clip(combined, 0.0, 1.0)


def path_enhance(similarity: Any, width: int = 4) -> Any:
    """对角路径增强：沿 ±width 条对角线取中位数，突出成段重复而非单点相似。"""
    np = _require_np()
    values = np.asarray(similarity, dtype=np.float64)
    count = values.shape[0]
    if count <= 2 or width <= 0:
        return values
    stack = []
    for offset in range(-width, width + 1):
        shifted = np.zeros_like(values)
        if offset >= 0:
            if offset == 0:
                shifted = values
            elif count - offset > 0:
                shifted[: count - offset, : count - offset] = values[offset:, offset:]
        else:
            gap = -offset
            if count - gap > 0:
                shifted[gap:, gap:] = values[: count - gap, : count - gap]
        stack.append(shifted)
    return np.median(np.stack(stack), axis=0)


def checkerboard_novelty(similarity: Any, size: int = 8) -> Any:
    """棋盘核沿对角线卷积得到新颖度曲线。"""
    np = _require_np()
    values = np.asarray(similarity, dtype=np.float64)
    count = values.shape[0]
    if count < 4:
        return np.zeros(count)
    size = int(max(2, min(size, count // 2)))
    half = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for u in range(size):
        for v in range(size):
            sign = 1.0 if (u < half) == (v < half) else -1.0
            # 高斯锥形权重，弱化边缘
            weight = math.exp(-0.5 * (((u - half + 0.5) / (half + 0.5)) ** 2 + ((v - half + 0.5) / (half + 0.5)) ** 2))
            kernel[u, v] = sign * weight
    novelty = np.zeros(count, dtype=np.float64)
    for index in range(1, count - 1):
        start = index - half
        end = start + size
        if start < 0 or end > count:
            continue
        block = values[start:end, start:end]
        novelty[index] = float(np.sum(block * kernel))
    if float(np.max(novelty)) > 0:
        novelty = novelty / float(np.max(novelty))
    return novelty


def pick_boundaries(
    novelty: Any,
    bars: list[tuple[float, float]],
    *,
    min_bars: int = _MIN_SECTION_BARS,
    max_count: int = _MAX_BOUNDARIES,
    threshold_ratio: float = 0.28,
) -> list[int]:
    """从新颖度曲线取峰：先按阈值过滤，再按峰值降序贪心去重。"""
    np = _require_np()
    values = np.asarray(novelty, dtype=np.float64)
    count = len(values)
    if count < min_bars * 2:
        return []
    mean = float(values.mean())
    std = float(values.std())
    threshold = max(mean + threshold_ratio * std, float(values.max()) * 0.25)
    candidates = [index for index in range(min_bars, count - min_bars) if values[index] >= threshold]
    candidates.sort(key=lambda index: values[index], reverse=True)
    chosen: list[int] = []
    for candidate in candidates:
        if len(chosen) >= max_count:
            break
        if all(abs(candidate - existing) >= min_bars for existing in chosen):
            chosen.append(candidate)
    return sorted(chosen)


def enforce_section_constraints(
    boundaries: list[int],
    novelty: Any,
    bar_count: int,
    *,
    min_bars: int = 4,
    max_bars: int = 16,
    max_count: int = 12,
    similarity: Any | None = None,
) -> list[int]:
    """段落长度的硬约束：太短的合并、太长的二次切分、段数封顶。

    流行歌的一「段」通常是 4~16 小节（4/4 @92BPM 时 1 小节 ≈ 2.6s，即 10~41s）。
    自相似矩阵给出的边界常常过密（出现 5 秒的「主歌」）或过疏（出现 93 秒的
    「副歌」），这里用音乐上合理的长度区间把它压回来。
    """
    np = _require_np()
    values = np.asarray(novelty, dtype=np.float64)
    limit = max(1, int(bar_count))
    min_bars = max(1, int(min_bars))
    max_bars = max(min_bars + 1, int(max_bars))
    max_count = max(1, int(max_count))
    bounds = [0] + [int(item) for item in sorted(set(boundaries)) if 0 < int(item) < limit] + [limit]

    def sizes() -> list[int]:
        return [bounds[index + 1] - bounds[index] for index in range(len(bounds) - 1)]

    def novelty_at(index: int) -> float:
        if values.size == 0 or index >= values.size:
            return 0.0
        return float(values[index])

    # 1) 太短的段落：删掉它旁边更弱的那个边界，直到所有段落都够长
    while len(bounds) > 2:
        current = sizes()
        short = [index for index, length in enumerate(current) if length < min_bars]
        if not short:
            break
        victim = min(short, key=lambda index: current[index])
        if victim == 0:
            del bounds[1]
        elif victim == len(current) - 1:
            del bounds[-2]
        else:
            left_boundary, right_boundary = bounds[victim], bounds[victim + 1]
            if novelty_at(left_boundary) <= novelty_at(right_boundary):
                del bounds[victim]
            else:
                del bounds[victim + 1]

    # 2) 太长的段落：在段内找最强的新颖度峰做二次切分（最多切 6 次）
    peak = float(values.max()) if values.size else 0.0
    matrix = None
    if similarity is not None:
        matrix = np.asarray(similarity, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            matrix = None
    for _ in range(6):
        current = sizes()
        long_sections = [index for index, length in enumerate(current) if length > max_bars]
        if not long_sections:
            break
        victim = max(long_sections, key=lambda index: current[index])
        low, high = bounds[victim] + min_bars, bounds[victim + 1] - min_bars
        if high <= low:
            break
        window = values[low:high + 1]
        if window.size == 0:
            break
        candidate = int(np.argmax(window)) + low
        if novelty_at(candidate) >= max(peak * 0.15, 0.05):
            bounds = sorted(bounds + [candidate])
            continue
        # 全局阈值不够：副歌内部边界本来就比段落边界弱，改用「段内局部自相似矩阵」
        # 的新颖度突出程度（z 分数）判断，段内没有明显重复结构就不硬切。
        if matrix is None or matrix.shape[0] < 2 * min_bars + 2:
            break
        seg_start, seg_end = bounds[victim], bounds[victim + 1]
        if seg_end > matrix.shape[0]:
            break
        sub = matrix[seg_start:seg_end, seg_start:seg_end]
        size = max(2, min(4, sub.shape[0] // 4))
        try:
            local = np.asarray(checkerboard_novelty(sub, size=size), dtype=np.float64)
        except Exception:  # pragma: no cover
            break
        if local.size == 0:
            break
        usable_end = sub.shape[0] - min_bars - 1
        if usable_end <= min_bars - 1:
            break
        window_local = local[min_bars - 1:usable_end]
        if window_local.size == 0:
            break
        local_peak = float(np.max(window_local))
        background_mean = float(np.mean(local))
        background_std = float(np.std(local))
        if background_std <= 1e-9:
            break
        if (local_peak - background_mean) / background_std < 0.8:
            break
        split_at = seg_start + min_bars + int(np.argmax(window_local))
        bounds = sorted(bounds + [split_at])

    # 3) 段数封顶：优先丢掉最弱的内部边界
    while len(bounds) - 1 > max_count and len(bounds) > 2:
        inner = [(novelty_at(index), index) for index in bounds[1:-1]]
        if not inner:
            break
        weakest = min(inner)[1]
        bounds = [item for item in bounds if item != weakest]

    return [item for item in bounds if 0 < item < limit]


def segment_ranges(boundaries: list[int], count: int) -> list[tuple[int, int]]:
    """边界（小节索引）→ 段落的小节区间。"""
    edges = [0] + sorted(index for index in boundaries if 0 < index < count) + [count]
    return [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]


def cluster_segments(segment_features: Any, threshold: float = _REPEAT_SIMILARITY) -> list[int]:
    """单链层次聚类，返回每个段落的簇编号（用于判断重复段落）。"""
    np = _require_np()
    features = np.asarray(segment_features, dtype=np.float64)
    count = features.shape[0]
    if count == 0:
        return []
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = features / norms
    similarity = np.clip((normalized @ normalized.T + 1.0) / 2.0, 0.0, 1.0)
    labels = [-1] * count
    next_label = 0
    for index in range(count):
        if labels[index] >= 0:
            continue
        labels[index] = next_label
        queue = [index]
        while queue:
            current = queue.pop()
            for other in range(count):
                if labels[other] >= 0:
                    continue
                if similarity[current, other] >= threshold:
                    labels[other] = next_label
                    queue.append(other)
        next_label += 1
    return labels


def normalize_energy(values: Any, *, floor: float = 0.12) -> list[float]:
    """能量归一化到 floor..1（5%–95% 分位抗离群）。

    用极值归一化时，最安静的那一段永远显示 0.00，前端画图会误判成「静音」；
    这里改用分位数并保留一个下限，让段落能量保持相对关系但有可视高度。
    """
    np = _require_np()
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi - lo <= 1e-9:
        return [0.5 for _ in arr]
    floor = max(0.0, min(0.5, float(floor)))
    normalized = [(float(value) - lo) / (hi - lo) for value in arr]
    return [round(floor + (1.0 - floor) * max(0.0, min(1.0, value)), 4) for value in normalized]


def renumber_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按出现顺序重编「主歌 1/副歌 2」，并清理悬空的 repeat_of。"""
    counters: dict[str, int] = {}
    label_map: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for section in sections:
        kind = str(section.get("kind") or "other")
        counters[kind] = counters.get(kind, 0) + 1
        text = f"{KIND_LABELS.get(kind, '其他')} {counters[kind]}"
        old_label = str(section.get("label") or "")
        if old_label:
            label_map[old_label] = text
        item = dict(section)
        item["label"] = text
        repeat_of = item.get("repeat_of")
        item["repeat_of"] = label_map.get(str(repeat_of)) if repeat_of else None
        output.append(item)
    return output


_MERGE_ADJACENT_MAX_SECONDS = 10.0


def merge_adjacent_same_kind(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并「被切碎」的相邻同类型段落。

    只合并短碎片（<10s）或同一簇的相邻段：两个真实的副歌被切开时要保留编号，
    否则会出现「副歌 1 = 88 秒」这种下游无法排版的段落。
    """
    merged: list[dict[str, Any]] = []
    for section in sections:
        span = float(section["end"]) - float(section["start"])
        same_cluster = bool(
            merged and section.get("repeat_of") and merged[-1].get("repeat_of") == section.get("repeat_of")
        )
        if merged and merged[-1]["kind"] == section["kind"] and (
            span < _MERGE_ADJACENT_MAX_SECONDS or same_cluster
        ):
            previous = merged[-1]
            span_prev = float(previous["end"]) - float(previous["start"])
            span_cur = float(section["end"]) - float(section["start"])
            total_span = max(1e-6, span_prev + span_cur)
            previous["energy"] = round(
                (float(previous["energy"]) * span_prev + float(section["energy"]) * span_cur) / total_span, 3
            )
            previous["confidence"] = max(float(previous["confidence"]), float(section["confidence"]))
            previous["end"] = section["end"]
            continue
        merged.append(dict(section))
    return merged


# --------------------------------------------------------------------------- #
# 标签推断
# --------------------------------------------------------------------------- #

def assign_kinds(
    segments: list[dict[str, Any]],
    labels: list[int],
    *,
    total_seconds: float,
) -> list[dict[str, Any]]:
    """根据簇（重复关系）、能量与位置分配段落类型。"""
    if not segments:
        return []
    count = len(segments)
    cluster_members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(index)

    def cluster_energy(label: int) -> float:
        members = cluster_members[label]
        return sum(float(segments[index]["energy"]) for index in members) / len(members)

    def cluster_duration(label: int) -> float:
        return sum(float(segments[index]["end"]) - float(segments[index]["start"]) for index in cluster_members[label])

    # 副歌簇：能量最高、且重复过的簇优先（副歌通常最响且会再现）。
    # 只看「重复」会让只出现一次的真副歌被错标成桥段，所以重复只作为加成。
    repeated = [label for label, members in cluster_members.items() if len(members) >= 2]

    def chorus_score(label: int) -> float:
        bonus = 1.15 if len(cluster_members[label]) >= 2 else 1.0
        # 首段（前奏）不参与副歌竞争
        if cluster_members[label] == [0]:
            return -1.0
        return cluster_energy(label) * bonus

    chorus_label = max(cluster_members, key=chorus_score) if cluster_members else None
    if chorus_label is not None and chorus_score(chorus_label) < 0:
        chorus_label = None
    verse_label = None
    other_repeated = [label for label in repeated if label != chorus_label and cluster_members[label] != [0]]
    if other_repeated:
        verse_label = max(other_repeated, key=lambda label: len(cluster_members[label]))
    elif cluster_members:
        # 没有重复簇时，把能量第二高的簇当作主歌参照
        remaining = [label for label in cluster_members if label != chorus_label]
        if remaining:
            verse_label = max(remaining, key=cluster_energy)

    chorus_energy = cluster_energy(chorus_label) if chorus_label is not None else 0.0
    verse_energy = cluster_energy(verse_label) if verse_label is not None else 0.0

    def ranked_indices(values: list[float]) -> list[int]:
        """按能量降序的段落下标。"""
        return sorted(range(len(values)), key=lambda position: values[position], reverse=True)

    energies = [float(segment["energy"]) for segment in segments]
    # 能量分位阈值：定「响」和「安静」两条线，避免用绝对阈值（归一化后大多段落都挤在 0.8 以上）
    order = ranked_indices(energies)
    # 能量最高的前 45%（至少 3 段、至多 5 段）作为「副歌候选」：
    # 用排名而不是阈值，避免边界段落的能量恰好卡在阈值上换来换去。
    loud_count = max(2, min(6, int(round(len(energies) * 0.45))))
    loud_indices = set(order[:loud_count])
    quiet_indices = set(order[int(round(len(energies) * 0.70)):])
    high_energy = float(energies[order[0]]) if order else 1.0
    low_energy = float(energies[order[-1]]) if order else 0.0
    loud_floor = float(energies[order[min(len(order) - 1, loud_count - 1)]]) if order else 1.0
    quiet_ceiling = float(energies[order[min(len(order) - 1, max(0, int(round(len(energies) * 0.70)) - 1))]]) if order else 0.0

    kinds: list[str] = []
    for index, segment in enumerate(segments):
        label = labels[index]
        repeated_cluster = len(cluster_members[label]) >= 2
        energy = float(segment["energy"])
        duration = float(segment["end"]) - float(segment["start"])
        is_first = index == 0
        is_last = index == count - 1
        next_label = labels[index + 1] if index + 1 < count else None
        prev_label = labels[index - 1] if index > 0 else None

        # 1) 首段：安静 → 前奏
        if is_first:
            if label != chorus_label and energy < high_energy and duration < total_seconds * 0.35:
                kinds.append("intro")
            elif label == chorus_label:
                kinds.append("chorus")
            else:
                kinds.append("verse")
            continue

        # 2) 响且够长 → 副歌（副歌没被重复检测到时也要认出来）
        if energy >= loud_floor and duration >= 10.0 and label != chorus_label:
            kinds.append("chorus")
            continue
        if is_last and not repeated_cluster and (index in quiet_indices) and label != chorus_label:
            kinds.append("outro")
            continue

        # 3) 副歌 / 主歌簇（聚类给的强证据优先）
        if label == chorus_label:
            kinds.append("chorus")
            continue

        # 4) 安静的长段落 → 主歌；安静的短段落 → 间奏
        if index in quiet_indices or energy <= quiet_ceiling:
            kinds.append("verse" if duration >= 10.0 else "interlude")
            continue
        if label == verse_label:
            kinds.append("verse")
            continue
        if repeated_cluster:
            kinds.append("chorus" if energy >= max(chorus_energy * 0.8, verse_energy + 0.1) else "verse")
            continue

        # 5) 能量居中的唯一段落：按位置和时长归类
        if next_label is not None and next_label == chorus_label and duration <= max(24.0, total_seconds * 0.16):
            kinds.append("prechorus")
            continue
        if duration <= total_seconds * 0.08:
            kinds.append("interlude")
            continue
        if prev_label is not None and len(cluster_members.get(prev_label, [])) >= 2:
            kinds.append("bridge" if energy >= chorus_energy * 0.7 else "interlude")
            continue
        kinds.append("bridge")

    # 编号与 repeat_of
    counters: dict[str, int] = {}
    first_of_kind: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        kind = kinds[index]
        label_name = KIND_LABELS.get(kind, "其他")
        counters[kind] = counters.get(kind, 0) + 1
        text = f"{label_name} {counters[kind]}"
        repeat_of = None
        if labels[index] in cluster_members and len(cluster_members[labels[index]]) >= 2:
            key = f"{kind}:{labels[index]}"
            if key in first_of_kind:
                repeat_of = first_of_kind[key]
            else:
                first_of_kind[key] = text
        output.append({
            "start": segment["start"],
            "end": segment["end"],
            "kind": kind,
            "label": text,
            "energy": round(float(segment["energy"]), 3),
            "repeat_of": repeat_of,
            "confidence": segment["confidence"],
        })
    return output


def finalize_sections(
    sections: list[dict[str, Any]],
    *,
    total_seconds: float,
    snap_points: list[float] | None = None,
    min_seconds: float = _MIN_SECTION_SECONDS,
) -> list[dict[str, Any]]:
    """对齐边界、保证覆盖连续、合并过短段落。"""
    if not sections:
        return []
    ordered = sorted(sections, key=lambda item: float(item["start"]))
    # 边界吸附到最近的拍/小节线（±1s 内）
    if snap_points:
        points = sorted(float(value) for value in snap_points)
        for index in range(1, len(ordered)):
            boundary = float(ordered[index]["start"])
            nearest = min(points, key=lambda value: abs(value - boundary))
            if abs(nearest - boundary) <= 1.0:
                ordered[index]["start"] = round(nearest, 4)
                ordered[index - 1]["end"] = round(nearest, 4)
    cleaned: list[dict[str, Any]] = []
    for section in ordered:
        start = max(0.0, float(section["start"]))
        end = min(float(total_seconds), float(section["end"]))
        if end - start <= 0.05:
            continue
        cleaned.append({**section, "start": round(start, 4), "end": round(end, 4)})
    if not cleaned:
        return []
    # 过短段落并入前一段（第一段则并入后一段）
    merged: list[dict[str, Any]] = []
    for section in cleaned:
        if merged and section["end"] - section["start"] < min_seconds:
            previous = merged[-1]
            if previous["kind"] == section["kind"] and previous.get("repeat_of") is None:
                previous["end"] = section["end"]
                continue
            previous["end"] = section["end"]
            continue
        merged.append(section)
    # 收尾：强制首段从 0 开始、末段到 total_seconds
    merged[0]["start"] = 0.0
    merged[-1]["end"] = round(float(total_seconds), 4)
    for index in range(len(merged) - 1):
        merged[index]["end"] = merged[index + 1]["start"]
    result: list[dict[str, Any]] = []
    for section in merged:
        if section["end"] - section["start"] < 0.05:
            continue
        section["energy"] = round(max(0.0, min(1.0, float(section.get("energy", 0.5)))), 3)
        result.append(section)
    return result


def fallback_layout(total_seconds: float, beats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """证据不足时的通用曲式兜底（低置信度 + 明确标注）。"""
    total = max(1.0, float(total_seconds))
    bar = 2.0
    if beats:
        beat_list = [float(value) for value in (beats.get("beats") or [])]
        if len(beat_list) >= 4:
            intervals = sorted(b - a for a, b in zip(beat_list, beat_list[1:]))
            per_bar = int(beats.get("beats_per_bar") or 4)
            bar = intervals[len(intervals) // 2] * per_bar
    # 常见流行曲式占比：前奏 8%、主歌 20%、副歌 20%、主歌 16%、副歌 20%、尾奏 16%
    plan = [
        ("intro", 0.08), ("verse", 0.20), ("chorus", 0.20),
        ("verse", 0.16), ("chorus", 0.20), ("outro", 0.16),
    ]
    sections: list[dict[str, Any]] = []
    cursor = 0.0
    counters: dict[str, int] = {}
    first_of_kind: dict[str, str] = {}
    for index, (kind, ratio) in enumerate(plan):
        end = total if index == len(plan) - 1 else round(cursor + total * ratio, 4)
        counters[kind] = counters.get(kind, 0) + 1
        text = f"{KIND_LABELS[kind]} {counters[kind]}"
        repeat_of = None
        if kind in ("verse", "chorus"):
            if kind in first_of_kind:
                repeat_of = first_of_kind[kind]
            else:
                first_of_kind[kind] = text
        sections.append({
            "start": round(cursor, 4),
            "end": round(end, 4),
            "kind": kind,
            "label": text,
            "energy": 0.5,
            "repeat_of": repeat_of,
            "confidence": 0.15,
        })
        cursor = end
    return finalize_sections(sections, total_seconds=total, snap_points=None)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _chord_per_bar(chords: list[dict[str, Any]], bars: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """每个小节的主导和弦（按时长加权）。"""
    result: list[dict[str, Any]] = []
    for start, end in bars:
        weights: dict[str, float] = {}
        roots: dict[str, int] = {}
        for chord in chords or []:
            chord_start = max(start, float(chord.get("start") or 0.0))
            chord_end = min(end, float(chord.get("end") or 0.0))
            overlap = chord_end - chord_start
            if overlap <= 0:
                continue
            symbol = str(chord.get("symbol") or "")
            weights[symbol] = weights.get(symbol, 0.0) + overlap
            if chord.get("root_pc") is not None:
                roots[symbol] = int(chord["root_pc"])
        if weights:
            best = max(weights, key=weights.get)
            result.append({"symbol": best, "weight": weights[best], "root_pc": roots.get(best)})
        else:
            result.append({"symbol": "", "weight": 0.0, "root_pc": None})
    return result


def _is_minor_symbol(symbol: str, root_pc: int) -> bool:
    """判断和弦符号是否小和弦（Cm / Cm7 / Cdim / Caug 视为非大三）。"""
    body = symbol.split("/")[0].strip()
    for name in (note_name(int(root_pc)), note_name(int(root_pc), prefer_flat=True)):
        if body.startswith(name):
            return body[len(name):].startswith("m")
    return False


def analyze_structure(
    source_wav: str,
    beats: dict[str, Any],
    chords: list[dict[str, Any]],
    total_seconds: float,
) -> list[dict[str, Any]]:
    """给出歌曲段落列表（时间轴连续覆盖整首歌）。"""
    librosa, np = _require_libs()
    y, sr = librosa.load(str(source_wav), sr=SAMPLE_RATE, mono=True)
    duration = float(len(y)) / float(sr) if sr else 0.0
    total = float(total_seconds or duration) if (total_seconds or duration) else 0.0
    total = min(total, duration) if duration > 0 else total
    if total <= 1.0 or y.size == 0:
        return fallback_layout(max(total, 1.0), beats)

    bars = build_bars(beats, total)
    if len(bars) < 6:
        return fallback_layout(total, beats)

    ranges = frame_ranges(bars, sr, HOP_LENGTH)
    try:
        y_harm = librosa.effects.harmonic(y, margin=2.0)
    except Exception:  # pragma: no cover
        y_harm = y

    mfcc = librosa.feature.mfcc(y=y_harm, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)

    mfcc_mean = aggregate_bars(mfcc.T, ranges)
    mfcc_std = aggregate_bars(mfcc.T, ranges, mode="std")
    timbre = zscore_rows(np.concatenate([mfcc_mean, mfcc_std], axis=1))
    harmony = zscore_rows(aggregate_bars(chroma.T, ranges))
    energy_features = zscore_rows(np.concatenate([
        aggregate_bars(rms.T, ranges),
        aggregate_bars(centroid.T, ranges),
        aggregate_bars(bandwidth.T, ranges),
        aggregate_bars(zcr.T, ranges),
    ], axis=1))

    ssm_timbre = cosine_similarity_matrix(timbre)
    ssm_harmony = cosine_similarity_matrix(harmony)
    ssm_energy = cosine_similarity_matrix(energy_features)

    # 和弦重复证据：小节主导和弦组成 24 维特征（根音 + 质量粗分）
    bar_chords = _chord_per_bar(chords, bars)
    chord_vector = np.zeros((len(bars), 24), dtype=np.float64)
    for index, info in enumerate(bar_chords):
        if info["root_pc"] is None:
            continue
        root = int(info["root_pc"]) % 12
        is_minor = _is_minor_symbol(str(info["symbol"]), root)
        # 24 维 = 12 个大三和弦根音位 + 12 个小三和弦根音位
        chord_vector[index, root + (12 if is_minor else 0)] = 1.0
    ssm_chord = cosine_similarity_matrix(chord_vector)

    ssm = combine_similarity([
        (ssm_timbre, 0.42),
        (ssm_harmony, 0.24),
        (ssm_energy, 0.18),
        (ssm_chord, 0.16),
    ])
    enhanced = path_enhance(ssm, width=3)
    novelty = checkerboard_novelty(enhanced, size=max(4, min(10, len(bars) // 6)))
    boundaries = pick_boundaries(novelty, bars)
    # 段落长度约束：4~16 小节、最多 12 段（避免「5 秒主歌」或「93 秒副歌」）
    boundaries = enforce_section_constraints(
        boundaries, novelty, len(bars),
        min_bars=_MIN_SECTION_BARS,
        max_bars=_MAX_SECTION_BARS,
        max_count=_MAX_SECTIONS,
        similarity=enhanced,
    )

    # 段落特征与聚类
    ranges_seg = segment_ranges(boundaries, len(bars))
    segment_feats = []
    for start_bar, end_bar in ranges_seg:
        lo = max(0, start_bar)
        hi = max(lo + 1, min(len(bars), end_bar))
        segment_feats.append(np.concatenate([
            timbre[lo:hi].mean(axis=0),
            harmony[lo:hi].mean(axis=0),
            energy_features[lo:hi].mean(axis=0),
            chord_vector[lo:hi].mean(axis=0),
        ]))
    segment_matrix = zscore_rows(np.stack(segment_feats))
    labels = cluster_segments(segment_matrix, threshold=_REPEAT_SIMILARITY)
    energies = normalize_energy([float(np.mean(rms[0, max(0, ranges[start][0]):max(1, ranges[end - 1][1])]))
                                for start, end in ranges_seg])
    novelty_strength = []
    for start, _end in ranges_seg:
        index = min(len(novelty) - 1, max(0, start))
        novelty_strength.append(float(novelty[index]) if start > 0 else 1.0)

    segments: list[dict[str, Any]] = []
    for index, (start_bar, end_bar) in enumerate(ranges_seg):
        start = bars[start_bar][0]
        end = bars[min(len(bars) - 1, end_bar - 1)][1]
        confidence = 0.25 + 0.55 * float(novelty_strength[index])
        segments.append({
            "start": round(start, 4),
            "end": round(end, 4),
            "kind": "other",
            "label": "",
            "energy": energies[index] if index < len(energies) else 0.5,
            "repeat_of": None,
            "confidence": round(min(0.95, confidence), 3),
        })

    sections = assign_kinds(segments, labels, total_seconds=total)
    snap_points = [value for value in (beats.get("downbeats") or [])] + [
        float(chord.get("start") or 0.0) for chord in (chords or [])
    ]
    sections = finalize_sections(sections, total_seconds=total, snap_points=snap_points)

    # 退化兜底：只有 1 段，或两段且首段占比过大且是前奏
    if len(sections) <= 1 or (
        len(sections) == 2
        and sections[0]["kind"] == "intro"
        and (sections[0]["end"] - sections[0]["start"]) >= total * 0.65
    ):
        return renumber_sections(fallback_layout(total, beats))
    sections = merge_adjacent_same_kind(sections)
    sections = renumber_sections(sections)
    return sections

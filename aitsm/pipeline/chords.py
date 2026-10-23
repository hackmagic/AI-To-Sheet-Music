"""F6（后半）：和弦识别。

流水线：
1. 谐波分离后的 CQT chroma（36 bins/octave）+ 调音补偿；
2. 低音声部（优先 stems/bass.wav，其次 40~450Hz 带通）单独算 chroma，
   用于根音约束与斜杠和弦（转位）判断；
3. 以拍为槽位聚合 chroma，用「模板余弦 + 和弦音覆盖 + 溢出惩罚 + 低音根音
   奖励 + 调内先验 - 复杂度惩罚」给 132 个候选和弦打分；
4. Viterbi 解码（同和弦 0 代价、换和弦有代价）避免逐拍抖动；
5. 后处理：合并相邻同和弦、吸收短小噪声和弦、低音转位加斜杠低音。

对外接口（契约不可改）：``recognize_chords(source_wav, beats, *, key, total_seconds)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aitsm.pipeline.keydet import (
    MINOR_DIATONIC_QUALITIES,
    MAJOR_DIATONIC_QUALITIES,
    note_name,
    prefer_flat_key,
)

SAMPLE_RATE = 22050
HOP_LENGTH = 512

# 和弦质量模板：{后缀: [(相对根音的半音数, 权重)]}，权重沿用参考项目的人工调参。
QUALITY_TEMPLATES: dict[str, list[tuple[int, float]]] = {
    "": [(0, 1.35), (4, 1.0), (7, 0.95)],
    "m": [(0, 1.35), (3, 1.0), (7, 0.95)],
    "7": [(0, 1.25), (4, 0.95), (7, 0.9), (10, 0.7)],
    "maj7": [(0, 1.25), (4, 0.95), (7, 0.9), (11, 0.7)],
    "m7": [(0, 1.25), (3, 0.95), (7, 0.9), (10, 0.7)],
    "m7b5": [(0, 1.25), (3, 0.95), (6, 0.9), (10, 0.7)],
    "dim": [(0, 1.25), (3, 1.0), (6, 0.95)],
    "dim7": [(0, 1.25), (3, 0.95), (6, 0.9), (9, 0.7)],
    "aug": [(0, 1.25), (4, 1.0), (8, 0.95)],
    "sus2": [(0, 1.2), (2, 0.95), (7, 0.95)],
    "sus4": [(0, 1.2), (5, 0.95), (7, 0.95)],
    "6": [(0, 1.2), (4, 0.95), (7, 0.9), (9, 0.7)],
    "m6": [(0, 1.2), (3, 0.95), (7, 0.9), (9, 0.7)],
}

# 复杂度惩罚：证据相同的情况下先选三和弦，避免把经过音解释成七和弦。
COMPLEXITY_PENALTY = {
    "": 0.0, "m": 0.0,
    # 七和弦的惩罚要略重：四个音里只要有一个是邻近和弦渗过来的能量，
    # 三和弦就会被误判成 Xmaj7/Xm7（实测 waltz 的 Am 小节里 Fmaj7 与 Am 只差 0.004）。
    # 弹唱谱也更偏好简单三和弦。
    "7": 0.07, "maj7": 0.075, "m7": 0.075,
    "sus4": 0.045, "sus2": 0.05, "dim": 0.05,
    "aug": 0.06, "m7b5": 0.055, "6": 0.065, "m6": 0.07, "dim7": 0.08,
}

# 巴洛克式打分权重（与参考项目一致）
W_COSINE = 0.78
W_COVERAGE = 0.34
W_SPILL = 0.14
W_BASS_ROOT = 0.18

_TOP_K = 12
# 换和弦的基本代价。取值要小于「正确和弦与邻近调内和弦的典型分差」，
# 否则 Viterbi 会把 bar 线处的真实换和弦（如 Am→G，共同音为 0）抹平，
# 偏向「和声上更接近」的错误和弦。同根抖动另有更重的惩罚。
_CHORD_CHANGE_PENALTY = 0.05
# 同根换质量（C→Cmaj7→Csus4…）在真实流行歌里很少见，多半是 chroma 抖动；
# 惩罚必须大于「质量证据」的常见差值，否则会出现 sus 连用。
_SAME_ROOT_PENALTY = 0.10
_NON_DIATONIC_PENALTY = 0.06
_DIATONIC_ESCAPE_CONFIDENCE = 0.65
_QUALITY_CHANGE_PENALTY = 0.04


def _require_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("和弦识别需要 numpy 与 librosa。") from exc
    return librosa, np


def _require_np():
    """纯函数只需 numpy；单独拆出来，未装 librosa 时也能对内核做单测。"""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("该函数需要 numpy，请先安装核心依赖。") from exc
    return np


def _safe_tuning(librosa: Any, y: Any, sr: int, np: Any) -> float:
    """估计全局调音偏差（音分→半音）。

    短音频/纯音/静音时 librosa 会从「空频集」估 tuning（返回 0 并告警），
    这里静音处理告警并保证有限值，避免污染 chroma 频率轴。
    """
    import warnings as _warnings

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            value = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        return 0.0
    if not np.isfinite(value) or abs(value) > 0.5:
        return 0.0
    return value


def _diatonic_qualities(key: dict[str, Any] | None) -> dict[int, set[str]] | None:
    if not key:
        return None
    mode = str(key.get("mode") or "major")
    return MAJOR_DIATONIC_QUALITIES if mode == "major" else MINOR_DIATONIC_QUALITIES


_TRIAD_OF_QUALITY = {
    "": "",
    "maj7": "",
    "7": "",
    "6": "",
    "sus2": "",
    "sus4": "",
    "m": "m",
    "m7": "m",
    "m6": "m",
    "m7b5": "dim",
    "dim": "dim",
    "dim7": "dim",
    "aug": "aug",
}


def _is_diatonic(root_pc: int, quality: str, key: dict[str, Any] | None) -> bool:
    """该和弦是否属于调内自然音阶和弦（七和弦按其三和弦性质判定）。"""
    if not key:
        return True
    qualities = _diatonic_qualities(key)
    if not qualities:
        return True
    tonic = int(key.get("tonic_pc") or 0) % 12
    degree = (int(root_pc) - tonic) % 12
    allowed = qualities.get(degree)
    if not allowed:
        return False
    triad = _TRIAD_OF_QUALITY.get(quality)
    # 扩展和弦（maj7/7/m7/6/sus…）按三和弦性质判定，避免 "maj7".startswith("m") 把
    # 大七和弦当成小三和弦家族的一员。
    if triad is not None:
        return triad in allowed
    return quality in allowed


# 色彩和弦的采纳门槛：低于该置信度就回退成调内三和弦（避免 sus/增/减抖动）
_COLOR_THRESHOLDS = {
    "sus2": 0.60,
    "sus4": 0.58,
    "dim": 0.66,
    "aug": 0.72,
    "m7b5": 0.66,
    "6": 0.66,
    "m6": 0.70,
    "dim7": 0.74,
}
_SEVENTHS = {"7", "maj7", "m7"}
# 七和弦比三和弦多一个音，模板更容易「凑」出高分，门槛要更高才采纳
_SEVENTH_THRESHOLD = 0.68
# 斜杠低音需要低频能量的最低证据强度
_SLASH_MIN_STRENGTH = 0.35
# 斜杠和弦的取舍：真实流行歌里斜杠和弦远少于普通三和弦，因此
#   1) 候选打分阶段，若低音明确落在某个音上（占比 >= 0.30），
#      根音不是它的候选要额外扣分，让「根音 = 低音」的解释优先；
#   2) 只有普通三和弦候选明显更差时，才保留斜杠低音。
_SLASH_BASS_DOMINANCE = 0.30
_SLASH_ROOT_PENALTY = 0.16
_SLASH_PREFERENCE_MARGIN = 0.10
# 非小节线槽位换和弦的额外代价：小节线处换和弦是常规操作，
# 小节中途突然换和弦多半是 chroma 抖动/经过音。
# 注意：4/4 的槽位是半小节，所以每个小节都会有一个「小节中途」槽位，
# 正好是流行歌「每 2 拍换一次和弦」常发生的位置，惩罚不能给太大，
# 否则和声节奏会被压成「两小节一个和弦」（实测真歌事件数会从 ~55 掉到 ~40）。
_MID_BAR_CHANGE_PENALTY = 0.02
# 滞回（hysteresis）：小节中途换和弦时，新和弦的 emission 分必须比当前和弦
# 高出这个量才真的换过去；否则保持原和弦。作用是「真值两小节同一和弦」不会被
# 微弱证据打散（去除小节中途反复抖动），而明显更强的证据仍能换出来。
# 实测真歌：mid=0.02 + hyst=0.04 → 55 事件 / mean_conf 0.575 / A-B-A 抖动 12 次；
# 若把 hyst 提到 0.06 以上，事件数掉到 40 出头，和声节奏过粗。
_CHANGE_HYSTERESIS = 0.04
_HYSTERESIS_BAR_TOLERANCE = 0.08

# 特征融合权重：中值聚合（抗攻击瞬态）与谱归一化均值（抗响度偏差）。
# 中值路权重更高，因为它对「鼓点/起音把 chroma 摊宽」这个主要误差源更有效；
# 谱归一化路负责压住「贝斯一响低频音级通吃」的偏差。
_FUSION_WEIGHTS = (0.62, 0.38)
# 槽位两端各裁掉多少比例再聚合。想法是「和弦在槽位边界换，边界帧是两个和弦的混合」，
# 但实测裁剪反而更差（docs/09 §4：干净率 98.0% → 96.4%(0.15) / 95.2%(0.30)），
# 因为槽位变短后样本不足、中值不稳定。默认 0（不裁）。
_SLOT_TRIM = 0.0
# 重复一致性第二遍解码：槽位自相似度阈值、每个槽位最多参考几个邻居、
# 排除多近的相邻槽位（相邻天然相似，算进来等于奖励「别换和弦」）、
# 以及一致性证据的加分幅度（相对 emission 分数 0.3~0.6 的量级，必须很小）。
# ⚠️ 实测结论（docs/09-chord-tuning.md）：重复一致性这一遍在真歌上是**负收益**——
# 跨分轨一致率 55.2%→47.7%、调内三和弦干净率 93.9%→95.8% 但事件数乱跳，
# 所以默认关闭，保留实现供继续实验（recognize_chords(repetition_pass=True)
# 或环境变量 AITSM_CHORD_REPETITION=1）。
_REPETITION_SIMILARITY = 0.93
_REPETITION_MAX_NEIGHBORS = 24
_REPETITION_EXCLUDE = 3
_REPETITION_STRENGTH = 0.045


def diatonic_triad_quality(root_pc: int, key: dict[str, Any] | None) -> str | None:
    """返回该根音在当前调里的自然三和弦质量（"" 或 "m"），非调内返回 None。"""
    table = _diatonic_qualities(key)
    if not table:
        return None
    tonic = int(key.get("tonic_pc") or 0)
    degree = (int(root_pc) - tonic) % 12
    allowed = table.get(degree)
    if not allowed:
        return None
    if "" in allowed:
        return ""
    if "m" in allowed:
        return "m"
    return None


def simplify_qualities(
    events: list[dict[str, Any]],
    key: dict[str, Any] | None,
    *,
    prefer_flat: bool = False,
) -> list[dict[str, Any]]:
    """把证据不足的色彩和弦回退成调内三和弦，减少 sus/7th 抖动。"""
    if not events:
        return []
    output: list[dict[str, Any]] = []
    for event in events:
        quality = str(event.get("quality") or "")
        confidence = float(event.get("confidence") or 0.0)
        threshold = _COLOR_THRESHOLDS.get(quality)
        if threshold is None and quality in _SEVENTHS:
            threshold = _SEVENTH_THRESHOLD
        if threshold is not None and confidence < threshold:
            fallback = diatonic_triad_quality(int(event["root_pc"]), key)
            if fallback is not None and fallback != quality:
                replaced = dict(event)
                replaced["quality"] = fallback
                replaced["symbol"] = format_symbol(
                    int(replaced["root_pc"]), fallback,
                    int(replaced.get("bass_pc") or replaced["root_pc"]) if replaced.get("bass_pc") is not None else None,
                    prefer_flat=prefer_flat,
                )
                replaced["simplified_from"] = quality
                output.append(replaced)
                continue
        output.append(event)
    return _merge_adjacent_same_symbol(output, prefer_flat=prefer_flat)


def _best_candidate_with_root(candidates: list[dict[str, Any]], root_pc: int) -> dict[str, Any] | None:
    """在该槽位的候选里，取「指定音作根音」的最高分候选（用于斜杠和弦取舍）。"""
    best: dict[str, Any] | None = None
    for item in candidates:
        if int(item["root_pc"]) % 12 != int(root_pc) % 12:
            continue
        if best is None or float(item["score"]) > float(best["score"]):
            best = item
    return best


def drop_unsupported_slash(
    events: list[dict[str, Any]],
    *,
    prefer_flat: bool = False,
    min_strength: float = _SLASH_MIN_STRENGTH,
) -> list[dict[str, Any]]:
    """斜杠和弦必须有低频证据支持，否则去掉低音（避免 Asus2/B 这种噪声）。"""
    if not events:
        return []
    output: list[dict[str, Any]] = []
    for event in events:
        root_pc = int(event["root_pc"])
        bass_pc = int(event.get("bass_pc") if event.get("bass_pc") is not None else root_pc)
        strength = float(event.get("bass_strength") or 0.0)
        if bass_pc % 12 != root_pc % 12 and strength < min_strength:
            replaced = dict(event)
            replaced["bass_pc"] = root_pc % 12
            replaced["symbol"] = format_symbol(root_pc, str(event.get("quality") or ""), None, prefer_flat=prefer_flat)
            replaced["bass_dropped"] = True
            output.append(replaced)
            continue
        output.append(event)
    return _merge_adjacent_same_symbol(output, prefer_flat=prefer_flat)


def _merge_adjacent_same_symbol(
    events: list[dict[str, Any]], *, prefer_flat: bool = False,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for event in events:
        if merged and merged[-1]["symbol"] == event["symbol"]:
            previous = merged[-1]
            previous["end"] = event["end"]
            previous["duration"] = round(float(previous["end"]) - float(previous["start"]), 4)
            previous["confidence"] = round(
                max(float(previous.get("confidence") or 0.0), float(event.get("confidence") or 0.0)), 3
            )
            continue
        merged.append(dict(event))
    return merged


def enforce_boundaries(
    events: list[dict[str, Any]],
    total_seconds: float,
    *,
    prefer_flat: bool = False,
) -> list[dict[str, Any]]:
    """保证和弦事件从 0 覆盖到结尾、无缝隙无重叠。"""
    if not events:
        return []
    total = float(total_seconds)
    cleaned: list[dict[str, Any]] = []
    for event in events:
        start = max(0.0, float(event["start"]))
        end = min(total, float(event["end"]))
        if end - start <= 1e-6:
            continue
        item = dict(event)
        item["start"], item["end"] = start, end
        cleaned.append(item)
    if not cleaned:
        return []
    cleaned.sort(key=lambda item: item["start"])
    # 首尾对齐：第一段从 0 开始，最后一段到结尾
    cleaned[0]["start"] = 0.0
    cleaned[-1]["end"] = total
    for index in range(len(cleaned) - 1):
        cleaned[index]["end"] = cleaned[index + 1]["start"] = max(
            cleaned[index]["start"] + 0.01, min(cleaned[index]["end"], cleaned[index + 1]["start"])
        )
    output: list[dict[str, Any]] = []
    for event in cleaned:
        event["duration"] = round(float(event["end"]) - float(event["start"]), 4)
        if event["duration"] <= 0:
            continue
        event["start"] = round(float(event["start"]), 4)
        event["end"] = round(float(event["end"]), 4)
        # 尾部/头部的碎片（不足 1/4 拍）直接并进邻居，避免出现 0.02s 的和弦
        if output and event["duration"] < 0.2:
            output[-1]["end"] = event["end"]
            output[-1]["duration"] = round(float(output[-1]["end"]) - float(output[-1]["start"]), 4)
            continue
        output.append(event)
    return _merge_adjacent_same_symbol(output, prefer_flat=prefer_flat)


# --------------------------------------------------------------------------- #
# 模板打分（纯函数，只用 numpy）
# --------------------------------------------------------------------------- #

def quality_template(root_pc: int, quality: str) -> Any:
    """把和弦模板平移到某个根音，得到 12 维权重向量。"""
    np = _require_np()
    template = np.zeros(12, dtype=np.float64)
    tones = QUALITY_TEMPLATES.get(quality, QUALITY_TEMPLATES[""])
    for offset, weight in tones:
        template[(int(root_pc) + int(offset)) % 12] = float(weight)
    return template


def chord_tone_mask(root_pc: int, quality: str) -> Any:
    """和弦音掩码（哪些 pitch class 属于该和弦）。"""
    return quality_template(root_pc, quality) > 0


def score_slot(
    chroma: Any,
    bass_chroma: Any | None = None,
    key: dict[str, Any] | None = None,
    *,
    top_k: int = _TOP_K,
) -> list[dict[str, Any]]:
    """给一个时间槽位的 chroma 打分，返回得分最高的若干候选和弦。

    打分 = 0.78*余弦 + 0.34*和弦音覆盖 - 0.14*溢出 + 0.18*低音根音
           + 调内先验 - 复杂度惩罚
    """
    np = _require_np()
    values = np.clip(np.asarray(chroma, dtype=np.float64).reshape(-1)[:12], 0.0, None)
    norm = float(values.sum())
    normalized = values / norm if norm > 1e-9 else np.full(12, 1.0 / 12.0, dtype=np.float64)
    bass_norm = None
    if bass_chroma is not None:
        bass_values = np.clip(np.asarray(bass_chroma, dtype=np.float64).reshape(-1)[:12], 0.0, None)
        bass_total = float(bass_values.sum())
        if bass_total > 1e-9:
            bass_norm = bass_values / bass_total
    candidates: list[dict[str, Any]] = []
    bass_dominant_pc = None
    bass_dominant_strength = 0.0
    if bass_norm is not None:
        bass_dominant_pc = int(np.argmax(bass_norm))
        bass_dominant_strength = float(bass_norm[bass_dominant_pc])
    for root in range(12):
        for quality in QUALITY_TEMPLATES:
            template = quality_template(root, quality)
            template_norm = float(np.linalg.norm(template))
            cosine = (
                float(np.dot(normalized, template)) / (norm * template_norm + 1e-12)
                if norm > 0 else 0.0
            )
            mask = template > 0
            inside = float(normalized[mask].sum())
            coverage = inside
            spill = max(0.0, 1.0 - inside)
            score = W_COSINE * cosine + W_COVERAGE * coverage - W_SPILL * spill
            if bass_norm is not None:
                score += W_BASS_ROOT * float(bass_norm[root])
                # 低音明确指向别的音时，根音不是它的候选要扣分：
                # 真实低音支持的是「那个音作根音」的和弦，而不是转位斜杠和弦。
                if (bass_dominant_pc is not None
                        and bass_dominant_strength >= _SLASH_BASS_DOMINANCE
                        and root != bass_dominant_pc):
                    score -= _SLASH_ROOT_PENALTY * min(1.0, bass_dominant_strength / 0.45)
            # 调内先验：调外和弦必须靠强证据才能赢
            score += 0.025 if _is_diatonic(root, quality, key) else -_NON_DIATONIC_PENALTY
            score -= COMPLEXITY_PENALTY.get(quality, 0.03)
            candidates.append({
                "root_pc": root,
                "quality": quality,
                "symbol": f"{note_name(root)}{quality}",
                "score": float(score),
                "cosine": float(cosine),
                "coverage": float(coverage),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    top = candidates[: max(1, int(top_k))]
    # 槽位置信度：top1 与「符号不同的第一名」之间的差距
    margin = 0.0
    if candidates:
        first = candidates[0]
        for competitor in candidates[1:]:
            if competitor["symbol"] != first["symbol"]:
                margin = first["score"] - competitor["score"]
                break
    for item in top:
        item["slot_confidence"] = round(max(0.05, min(0.98, 0.45 + margin * 3.2)), 4)
    return top


def transition_penalty(previous: dict[str, Any], current: dict[str, Any]) -> float:
    """Viterbi 的换和弦代价：同根换质量最贵，共同音越多越便宜。"""
    np = _require_np()
    prev_root = int(previous["root_pc"])
    cur_root = int(current["root_pc"])
    if prev_root == cur_root:
        if str(previous.get("quality")) == str(current.get("quality")):
            return 0.0
        return _SAME_ROOT_PENALTY + _QUALITY_CHANGE_PENALTY
    shared = float(np.dot(
        chord_tone_mask(prev_root, str(previous.get("quality") or "")).astype(float),
        chord_tone_mask(cur_root, str(current.get("quality") or "")).astype(float),
    ))
    return max(0.02, _CHORD_CHANGE_PENALTY - 0.03 * shared)


def viterbi_decode(
    emissions: list[list[dict[str, Any]]],
    *,
    change_penalty: list[float] | None = None,
) -> list[dict[str, Any]]:
    """最小代价路径解码：代价 = -score + 换和弦代价。

    change_penalty 可选，为每个槽位给出「在这个槽位换和弦」的额外代价：
    小节线处换和弦是常规操作（代价 0），小节中途换和弦多半是 chroma 抖动，
    因此调用方可以给非小节线槽位加一点惩罚（见 recognize_chords）。
    """
    if not emissions:
        return []
    steps = len(emissions)
    scores = [-float(item["score"]) for item in emissions[0]]
    backpointers: list[list[int]] = []
    for step in range(1, steps):
        candidates = emissions[step]
        extra = float(change_penalty[step]) if change_penalty and step < len(change_penalty) else 0.0
        new_scores: list[float] = []
        pointers: list[int] = []
        for target in candidates:
            best_cost: float | None = None
            best_index = 0
            for index, previous in enumerate(emissions[step - 1]):
                change_cost = transition_penalty(previous, target)
                if change_cost > 0.0:
                    change_cost += extra
                cost = scores[index] + change_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = index
            new_scores.append(float(best_cost or 0.0) - float(target["score"]))
            pointers.append(best_index)
        scores = new_scores
        backpointers.append(pointers)
    cursor = min(range(len(scores)), key=lambda index: scores[index])
    path: list[dict[str, Any]] = [emissions[-1][cursor]]
    for step in range(steps - 1, 0, -1):
        cursor = backpointers[step - 1][cursor]
        path.append(emissions[step - 1][cursor])
    path.reverse()
    return path


def apply_change_hysteresis(
    states: list[dict[str, Any]],
    emissions: list[list[dict[str, Any]]],
    slot_starts: list[float],
    *,
    downbeats: list[float] | None = None,
    hysteresis: float = _CHANGE_HYSTERESIS,
    bar_tolerance: float = _HYSTERESIS_BAR_TOLERANCE,
) -> list[dict[str, Any]]:
    """和弦滞回：小节中途的换和弦必须有明显更强的证据，否则保持原和弦。

    与 Viterbi 的 change_penalty 互补——change_penalty 是「换和弦的固定门槛」，
    滞回是「针对当前正在持续的和弦，新和弦要额外赢多少分」。这样两小节同一和弦
    的真值不会被小节中途的微弱证据打散，而副歌每小节/每两拍换和弦的强证据仍能通过。

    参数
    ----
    states: Viterbi 解码出的逐槽位和弦候选（需含 ``symbol`` 与 ``score``）。
    emissions: 每个槽位的候选列表（按分数降序）。
    slot_starts: 每个槽位的起始时间（秒），用于判断是否在小节线上。
    downbeats: 小节线时间列表；为空时不做小节线豁免（全部按小节中途处理）。
    hysteresis: 新和弦相对当前和弦需要额外超出的分数。
    """
    if not states:
        return []
    bar_lines = sorted(float(value) for value in (downbeats or []))
    result: list[dict[str, Any]] = [dict(states[0])]
    held = str(result[0].get("symbol") or "")
    for index in range(1, len(states)):
        state = states[index]
        symbol = str(state.get("symbol") or "")
        start = float(slot_starts[index]) if index < len(slot_starts) else 0.0
        on_bar_line = any(abs(start - line) <= bar_tolerance for line in bar_lines)
        if symbol == held or on_bar_line or hysteresis <= 0.0:
            result.append(dict(state))
            held = symbol
            continue
        candidates = emissions[index] if index < len(emissions) else []
        held_candidate = next((item for item in candidates if str(item.get("symbol")) == held), None)
        if held_candidate is None:
            result.append(dict(state))
            held = symbol
            continue
        gain = float(state.get("score") or 0.0) - float(held_candidate.get("score") or 0.0)
        if gain >= hysteresis:
            result.append(dict(state))
            held = symbol
        else:
            # 证据不够强：维持上一个和弦
            result.append(dict(held_candidate))
    return result


# --------------------------------------------------------------------------- #
# 时间槽位与事件组装
# --------------------------------------------------------------------------- #

def slot_length_in_beats(beats_per_bar: int) -> int:
    """一个和弦槽位占几拍：4/4 用半小节（2 拍），其它拍号用一整小节。"""
    try:
        per_bar = int(beats_per_bar)
    except (TypeError, ValueError):
        per_bar = 4
    if per_bar <= 0:
        per_bar = 4
    if per_bar >= 4:
        return max(2, per_bar // 2)
    return per_bar


def build_slot_times(
    beats: dict[str, Any],
    total_seconds: float,
    *,
    slot_beats: int | None = None,
) -> list[tuple[float, float]]:
    """把拍点分组成和弦槽位时间区间（半小节粒度），并补齐到音频结尾。"""
    total = max(0.0, float(total_seconds or 0.0))
    if total <= 0.0:
        return []
    beat_list = sorted({
        float(value) for value in (beats.get("beats") or []) if float(value) >= 0.0
    })
    step = max(1, int(slot_beats)) if slot_beats else slot_length_in_beats(int(beats.get("beats_per_bar") or 4))
    if len(beat_list) >= 2:
        intervals = sorted(b - a for a, b in zip(beat_list, beat_list[1:]))
        beat_length = intervals[len(intervals) // 2]
    else:
        bpm = float(beats.get("bpm_display") or beats.get("bpm") or 0.0)
        beat_length = 60.0 / bpm if bpm > 0 else 0.5
    slot_length = max(0.12, float(beat_length) * step)
    downbeats = sorted(float(value) for value in (beats.get("downbeats") or []))
    anchor = downbeats[0] if downbeats else (beat_list[0] if beat_list else 0.0)
    anchor = max(0.0, min(float(anchor), total))
    starts: list[float] = []
    position = anchor
    while position < total - 1e-6:
        starts.append(round(position, 6))
        position += slot_length
    edges = sorted({0.0, round(total, 6), *[value for value in starts if 1e-6 < value < total - 1e-6]})
    slots = [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]
    return [(round(start, 4), round(end, 4)) for start, end in slots if end - start > 1e-6]


def format_symbol(
    root_pc: int,
    quality: str,
    bass_pc: int | None = None,
    *,
    prefer_flat: bool = False,
) -> str:
    """生成和弦符号；低音与根音不同（转位）时加斜杠低音。"""
    root = note_name(int(root_pc) % 12, prefer_flat=prefer_flat)
    symbol = f"{root}{quality}"
    if bass_pc is not None and int(bass_pc) % 12 != int(root_pc) % 12:
        symbol += f"/{note_name(int(bass_pc) % 12, prefer_flat=prefer_flat)}"
    return symbol


def merge_slots_to_events(
    states: list[dict[str, Any]],
    slot_times: list[tuple[float, float]],
    *,
    prefer_flat: bool = False,
    bass_pitch: list[int | None] | None = None,
    bass_strength: list[float] | None = None,
) -> list[dict[str, Any]]:
    """把逐槽位的 Viterbi 状态合并成和弦事件。"""
    events: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        if index >= len(slot_times):
            break
        start, end = slot_times[index]
        pitch = bass_pitch[index] if bass_pitch and index < len(bass_pitch) else None
        strength = float(bass_strength[index]) if bass_strength and index < len(bass_strength) else 0.0
        root_pc = int(state["root_pc"]) % 12
        quality = str(state.get("quality") or "")
        bass_value = int(pitch) % 12 if pitch is not None else root_pc
        symbol = format_symbol(root_pc, quality, bass_value, prefer_flat=prefer_flat)
        confidence = float(state.get("slot_confidence") or 0.3)
        if events and events[-1]["symbol"] == symbol:
            previous = events[-1]
            previous["end"] = end
            previous["duration"] = round(float(end) - float(previous["start"]), 4)
            count = int(previous.get("slot_count") or 1)
            previous["slot_confidence"] = (float(previous["slot_confidence"]) * count + confidence) / (count + 1)
            previous["slot_count"] = count + 1
            previous["bass_strength"] = max(float(previous.get("bass_strength") or 0.0), strength)
            previous["confidence"] = round(float(previous["slot_confidence"]), 3)
            continue
        events.append({
            "start": float(start),
            "end": float(end),
            "symbol": symbol,
            "confidence": round(confidence, 3),
            "slot_confidence": confidence,
            "score": float(state.get("score") or 0.0),
            "root_pc": root_pc,
            "bass_pc": bass_value,
            "quality": quality,
            "bass_strength": strength,
            "duration": round(float(end) - float(start), 4),
            "slot_count": 1,
            "source": "chroma-template+viterbi",
        })
    return events


def cleanup_events(
    events: list[dict[str, Any]],
    *,
    min_duration: float = 1.2,
    key: dict[str, Any] | None = None,
    prefer_flat: bool = False,
) -> list[dict[str, Any]]:
    """和弦事件清理：吸收被夹住的短和弦、丢弃极短低置信碎片、合并相邻同和弦。"""
    if not events:
        return []
    items = [dict(event) for event in events]
    items.sort(key=lambda item: float(item["start"]))

    def merge_same(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for event in sequence:
            if merged and merged[-1]["symbol"] == event["symbol"]:
                previous = merged[-1]
                previous["end"] = event["end"]
                previous["duration"] = round(float(previous["end"]) - float(previous["start"]), 4)
                previous["slot_confidence"] = max(
                    float(previous.get("slot_confidence") or 0.0),
                    float(event.get("slot_confidence") or 0.0),
                )
                previous["confidence"] = round(float(previous["slot_confidence"]), 3)
                previous["bass_strength"] = max(
                    float(previous.get("bass_strength") or 0.0),
                    float(event.get("bass_strength") or 0.0),
                )
                continue
            merged.append(event)
        return merged

    merged = merge_same(items)

    # 1) 前后同和弦夹住的短事件直接吸收（多半是经过音被误判）
    sandwiched: list[dict[str, Any]] = []
    for index, event in enumerate(merged):
        duration = float(event["end"]) - float(event["start"])
        if (
            0 < index < len(merged) - 1
            and duration < min_duration
            and merged[index - 1]["symbol"] == merged[index + 1]["symbol"]
        ):
            continue
        sandwiched.append(event)
    merged = merge_same(sandwiched)

    # 2) 极短且低置信的碎片并进更可信的邻居
    cleaned: list[dict[str, Any]] = []
    for event in merged:
        duration = float(event["end"]) - float(event["start"])
        slot_confidence = float(event.get("slot_confidence") or 0.0)
        if cleaned and len(merged) > 1 and duration < min_duration * 0.7 and slot_confidence < 0.22:
            neighbour = cleaned[-1]
            neighbour["end"] = event["end"]
            neighbour["duration"] = round(float(neighbour["end"]) - float(neighbour["start"]), 4)
            continue
        cleaned.append(event)
    merged = merge_same(cleaned)

    # 3) 重新生成符号（低音支持不足的斜杠在这里先按「去掉低音」处理）
    output: list[dict[str, Any]] = []
    for event in merged:
        root_pc = int(event["root_pc"]) % 12
        quality = str(event.get("quality") or "")
        bass_pc = int(event["bass_pc"] if event.get("bass_pc") is not None else root_pc) % 12
        if bass_pc != root_pc and float(event.get("bass_strength") or 0.0) < _SLASH_MIN_STRENGTH:
            bass_pc = root_pc
        event["bass_pc"] = bass_pc
        event["symbol"] = format_symbol(root_pc, quality, bass_pc, prefer_flat=prefer_flat)
        event["confidence"] = round(float(event.get("slot_confidence") or event.get("confidence") or 0.3), 3)
        output.append(event)
    return output


# --------------------------------------------------------------------------- #
# 音频载入辅助
# --------------------------------------------------------------------------- #

def _find_bass_wav(source_wav: str | Path) -> Path | None:
    """在分析目录里寻找 Demucs 输出的 bass stem。"""
    path = Path(source_wav)
    candidates = [
        path.parent / "stems" / "bass.wav",
        path.parent / "bass.wav",
        path.parent.parent / "stems" / "bass.wav",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _band_limit(y: Any, sr: int, low: float, high: float, librosa: Any, np: Any) -> Any:
    """用 STFT 掩码做粗带通（librosa 无现成带通，避免额外依赖 scipy）。"""
    spectrum = librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = ((freqs >= low) & (freqs <= high)).astype(np.float64)
    return librosa.istft(spectrum * mask[:, None], hop_length=HOP_LENGTH, length=len(y))


def _slot_aggregate(feature: Any, frames_start: int, frames_end: int, np: Any, *, mode: str = "mean") -> Any:
    lo = max(0, int(frames_start))
    hi = max(lo + 1, min(feature.shape[1], int(frames_end)))
    block = feature[:, lo:hi]
    if block.size == 0:
        return np.zeros(feature.shape[0], dtype=np.float64)
    if mode == "median":
        values = np.median(block, axis=1)
    elif mode == "max":
        values = block.max(axis=1)
    else:
        values = block.mean(axis=1)
    total = float(values.sum())
    if total <= 1e-9:
        return np.zeros(feature.shape[0], dtype=np.float64)
    return values / total


def _l1(vector: Any, np: Any) -> Any:
    values = np.clip(np.asarray(vector, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(values.sum())
    if total <= 1e-9:
        return np.full(values.shape, 1.0 / max(1, values.shape[0]), dtype=np.float64)
    return values / total


def _trim_frames(lo: int, hi: int, trim: float) -> tuple[int, int]:
    """把槽位帧区间的两端各裁掉 trim 比例（保持至少 1 帧）。"""
    if trim <= 0.0:
        return lo, hi
    span = hi - lo
    if span < 3:
        return lo, hi
    cut = int(round(span * min(0.45, float(trim))))
    return lo + cut, max(lo + cut + 1, hi - cut)


def _slot_matrix_fused(
    chroma: Any,
    frame_times: list[tuple[int, int]],
    np: Any,
    *,
    fusion: bool = True,
    smooth: bool = True,
    trim: float = _SLOT_TRIM,
) -> Any:
    """逐槽位聚合出 12×N 的和弦特征矩阵。

    两路特征：
    - **中值聚合**：对槽位内的帧取中位数。攻击瞬态（鼓点、拨弦起音）会把
      一两帧的 chroma 摊得很宽，均值会被这些离群帧带偏，中位数不会。
    - **谱归一化均值**：每帧先按自身能量归一化再取均值，压低「响的就是和弦」
      的偏差——贝斯/底鼓一响，整段的原始 chroma 会被低频音级主导。

    两路各自 L1 归一化后按 `_FUSION_WEIGHTS` 融合。`fusion=False` 时退回旧行为
    （纯 mean 聚合、不融合），用于 A/B 对比。
    """
    if not fusion:
        matrix = np.stack([
            _slot_aggregate(chroma, lo, hi, np, mode="mean") for lo, hi in frame_times
        ], axis=1)
        return _smooth_matrix(matrix, np) if smooth else matrix
    trimmed = [_trim_frames(lo, hi, trim) for lo, hi in frame_times]

    columns_median: list[Any] = []
    columns_whitened: list[Any] = []
    frame_norm = np.linalg.norm(np.asarray(chroma, dtype=np.float64), axis=0, keepdims=True)
    whitened = np.asarray(chroma, dtype=np.float64) / np.maximum(frame_norm, 1e-9)
    for lo, hi in trimmed:
        columns_median.append(_slot_aggregate(chroma, lo, hi, np, mode="median"))
        columns_whitened.append(_slot_aggregate(whitened, lo, hi, np, mode="mean"))
    median_matrix = np.stack(columns_median, axis=1)
    whitened_matrix = np.stack(columns_whitened, axis=1)
    weight_median, weight_whitened = _FUSION_WEIGHTS
    fused = np.zeros_like(median_matrix)
    for index in range(median_matrix.shape[1]):
        fused[:, index] = (
            weight_median * _l1(median_matrix[:, index], np)
            + weight_whitened * _l1(whitened_matrix[:, index], np)
        )
        fused[:, index] = _l1(fused[:, index], np)
    return _smooth_matrix(fused, np) if smooth else fused


def _repetition_weights(
    slot_matrix: Any,
    np: Any,
    *,
    threshold: float = _REPETITION_SIMILARITY,
    max_neighbors: int = _REPETITION_MAX_NEIGHBORS,
    exclude_radius: int = _REPETITION_EXCLUDE,
) -> Any:
    """算出「哪些槽位的伴奏长得像自己」的权重矩阵（N×N）。

    做法是槽位级自相似矩阵（余弦相似度），只保留相似度 ≥ threshold 的邻居，
    每个槽位最多留 max_neighbors 个；并排除 ±exclude_radius 内的相邻槽位——
    相邻槽位天然相似，把它们算进来等于奖励「永远别换和弦」，会把换和弦抹平。
    """
    matrix = np.asarray(slot_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        return np.zeros((matrix.shape[1] if matrix.ndim == 2 else 0,) * 2, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-9)
    similarity = np.clip(unit.T @ unit, -1.0, 1.0).astype(np.float64)
    slots = similarity.shape[0]
    weights = np.zeros_like(similarity)
    for index in range(slots):
        row = similarity[index].copy()
        low = max(0, index - int(exclude_radius))
        high = min(slots, index + int(exclude_radius) + 1)
        row[low:high] = 0.0
        candidates = np.where(row >= float(threshold))[0]
        if candidates.size == 0:
            continue
        if candidates.size > max_neighbors:
            order = np.argsort(row[candidates])[::-1][: int(max_neighbors)]
            candidates = candidates[np.sort(order)]
        # 相似度映射到 (0,1] 的权重：刚好过线的小、接近 1 的大
        weights[index, candidates] = (row[candidates] - float(threshold)) / max(1e-6, 1.0 - float(threshold))
    return weights


def _apply_repetition_prior(
    emissions: list[list[dict[str, Any]]],
    weights: Any,
    states: list[dict[str, Any]],
    np: Any,
    *,
    strength: float = _REPETITION_STRENGTH,
) -> tuple[list[list[dict[str, Any]]], int]:
    """用「相似槽位上一遍解出的是什么和弦」给候选加一点分。

    这是自训练式的平滑：一首歌里副歌唱三遍，其中两遍的 chroma 清楚、一遍被
    衰减或加花搞模糊。第一遍解码后，模糊那遍的邻居槽位已经给出了可信答案，
    第二遍就把这份一致性证据加回去。

    只在邻居相似度足够（权重 > 0）时才加，且幅度很小（`strength`），
    避免把整首歌压成同一个和弦。返回 (新打分, 被改动的槽位数)。
    """
    if weights.size == 0 or not states:
        return emissions, 0
    slots = min(weights.shape[0], len(emissions), len(states))
    if slots <= 0:
        return emissions, 0
    symbols = [str(state.get("symbol") or "") for state in states]
    adjusted: list[list[dict[str, Any]]] = []
    touched = 0
    for index in range(len(emissions)):
        candidates = emissions[index]
        if index >= slots:
            adjusted.append(candidates)
            continue
        row = weights[index]
        total = float(np.sum(row))
        if total <= 1e-9:
            adjusted.append(candidates)
            continue
        # 每个和弦符号在「相似槽位」里占的加权比例
        shares: dict[str, float] = {}
        for neighbor in range(slots):
            weight = float(row[neighbor])
            if weight <= 0.0:
                continue
            symbol = symbols[neighbor]
            if not symbol:
                continue
            shares[symbol] = shares.get(symbol, 0.0) + weight
        if not shares:
            adjusted.append(candidates)
            continue
        best_share = max(shares.values()) / total
        rescored: list[dict[str, Any]] = []
        changed = False
        for candidate in candidates:
            symbol = str(candidate.get("symbol") or "")
            share = shares.get(symbol, 0.0) / total
            bonus = strength * share
            item = dict(candidate)
            item["score"] = float(candidate.get("score") or 0.0) + bonus
            if symbol == symbols[index] and best_share >= 0.5:
                # 当前解码结果本来就是邻居里的大多数：额外小幅确认
                item["score"] += strength * 0.5 * best_share
            if abs(bonus) > 1e-9:
                changed = True
            rescored.append(item)
        rescored.sort(key=lambda entry: float(entry.get("score") or 0.0), reverse=True)
        adjusted.append(rescored)
        if changed:
            touched += 1
    return adjusted, touched


def _smooth_matrix(matrix: Any, np: Any) -> Any:
    """沿时间轴做 3 点轻度平滑（权重 0.2/0.6/0.2），抑制换和弦瞬间的混合。"""
    if matrix.shape[1] < 3:
        return matrix
    smoothed = matrix.copy()
    smoothed[:, 1:-1] = 0.2 * matrix[:, :-2] + 0.6 * matrix[:, 1:-1] + 0.2 * matrix[:, 2:]
    return smoothed


def _detect_bass_pitch(bass_slot: Any, root_pc: int, np: Any) -> tuple[int | None, float]:
    """判断该槽位的实际低音是否与根音不同（转位/斜杠和弦）。

    返回 (低音 pitch_class 或 None, 证据强度)。强度会一路带到事件里，
    后处理阶段用它决定「斜杠和弦到底有没有低频支持」。
    """
    if bass_slot is None:
        return None, 0.0
    total = float(np.sum(bass_slot))
    if total <= 1e-9:
        return None, 0.0
    normalized = bass_slot / total
    strongest = int(np.argmax(normalized))
    strength = float(normalized[strongest])
    if strongest == int(root_pc) % 12:
        return None, strength
    root_strength = float(normalized[int(root_pc) % 12])
    # 低音明显偏向非根音，且自身能量足够时，才认定转位
    if strength >= 0.30 and strength > root_strength * 1.40:
        return strongest, strength
    return None, strength * 0.5


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def recognize_chords(
    source_wav: Path | str,
    beats: dict[str, Any],
    *,
    key: dict[str, Any] | None = None,
    total_seconds: float | None = None,
    bass_wav: Path | str | None = None,
    feature_fusion: bool = True,
    repetition_pass: bool = False,
    slot_trim: float = _SLOT_TRIM,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按拍输出和弦事件列表。"""
    librosa, np = _require_libs()
    path = Path(source_wav)
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    duration = float(len(y)) / float(sr)
    if total_seconds is None:
        total_seconds = duration
    total_seconds = min(float(total_seconds), duration) if duration > 0 else float(total_seconds)

    slot_times = build_slot_times(beats, total_seconds)
    if not slot_times:
        return []

    # 1) 谐波分量 → 调音补偿 → CQT chroma
    try:
        y_harm = librosa.effects.harmonic(y, margin=3.0)
    except Exception:  # pragma: no cover
        y_harm = y
    tuning = 0.0
    try:
        tuning = _safe_tuning(librosa, y_harm, sr, np)
    except Exception:  # pragma: no cover
        tuning = 0.0
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=HOP_LENGTH, tuning=tuning,
        bins_per_octave=36, n_chroma=12,
    )

    # 2) 低音 chroma：优先 bass stem，否则带通
    bass_source = "mix-bandpass"
    bass_path = Path(bass_wav) if bass_wav else _find_bass_wav(path)
    bass_audio = None
    if bass_path is not None and bass_path.is_file():
        try:
            candidate, candidate_sr = librosa.load(str(bass_path), sr=SAMPLE_RATE, mono=True)
            if candidate_sr == sr and abs(len(candidate) / float(candidate_sr) - duration) < 1.0:
                bass_audio = candidate
                bass_source = "bass-stem"
        except Exception:  # pragma: no cover
            bass_audio = None
    if bass_audio is None:
        try:
            bass_audio = _band_limit(y_harm, sr, 40.0, 450.0, librosa, np)
        except Exception:  # pragma: no cover
            bass_audio = None
    bass_chroma = None
    if bass_audio is not None and len(bass_audio) > 0:
        try:
            bass_chroma = librosa.feature.chroma_cqt(
                y=bass_audio, sr=sr, hop_length=HOP_LENGTH, tuning=tuning,
                fmin=librosa.note_to_hz("C2"), n_octaves=3, bins_per_octave=36, n_chroma=12,
            )
        except Exception:  # pragma: no cover
            bass_chroma = None

    # 3) 逐槽聚合 + 平滑（中值 / 谱归一化两路融合，见 _slot_matrix_fused）
    frame_times = []
    for start, end in slot_times:
        frame_times.append((int(round(start * sr / HOP_LENGTH)), int(round(end * sr / HOP_LENGTH))))
    slot_matrix = _slot_matrix_fused(chroma, frame_times, np, fusion=feature_fusion, trim=slot_trim)
    bass_matrix = None
    if bass_chroma is not None:
        bass_matrix = np.stack([
            # 低音看「起音后一小段」，取最大值更能突出根音
            _slot_aggregate(
                bass_chroma, lo, lo + max(2, int((hi - lo) * 0.6)), np, mode="max",
            )
            for lo, hi in frame_times
        ], axis=1)

    # 4) 打分 + Viterbi
    emissions = [
        score_slot(slot_matrix[:, index], None if bass_matrix is None else bass_matrix[:, index], key)
        for index in range(slot_matrix.shape[1])
    ]
    downbeats = [float(value) for value in (beats.get("downbeats") or [])]
    change_penalty: list[float] = []
    for index, (start, _end) in enumerate(slot_times):
        if index == 0 or not downbeats:
            change_penalty.append(0.0)
            continue
        on_bar_line = any(abs(start - downbeat) <= 0.08 for downbeat in downbeats)
        change_penalty.append(0.0 if on_bar_line else _MID_BAR_CHANGE_PENALTY)
    states = viterbi_decode(emissions, change_penalty=change_penalty)
    if not states:
        return []

    # 4b) 重复一致性第二遍：相似槽位（副歌重复、主歌重复）互相背书。
    #     先把第一遍的解码结果当作「邻居的投票」，对候选重新加一点点分，再解一遍。
    #     这一步专门救「同一段伴奏被衰减/加花搞模糊」的槽位——它们的邻居槽位
    #     chroma 清楚，第一遍已经解对了，把这份一致性证据补回来。
    repetition_stats: dict[str, Any] = {"enabled": bool(repetition_pass), "touched_slots": 0}
    if repetition_pass and len(slot_times) >= 8:
        weights = _repetition_weights(slot_matrix, np)
        if float(np.sum(weights)) > 0.0:
            boosted, touched = _apply_repetition_prior(emissions, weights, states, np)
            repetition_stats["touched_slots"] = int(touched)
            if touched > 0:
                second = viterbi_decode(boosted, change_penalty=change_penalty)
                if second:
                    emissions = boosted
                    states = second
    if stats is not None:
        stats.update(repetition_stats)
        stats["feature_fusion"] = bool(feature_fusion)
        stats["slots"] = len(slot_times)
    # 滞回：小节中途换和弦必须赢过当前和弦 _CHANGE_HYSTERESIS 分
    states = apply_change_hysteresis(
        states,
        emissions,
        [start for start, _end in slot_times],
        downbeats=downbeats,
        hysteresis=_CHANGE_HYSTERESIS,
    )
    # 调外和弦必须有强证据（slot_confidence > 0.65）才保留，否则退回该槽位的调内最佳候选
    if key and _diatonic_qualities(key):
        diatonic_fallback: list[dict[str, Any] | None] = []
        for candidates in emissions:
            best_diatonic = next(
                (item for item in candidates if _is_diatonic(int(item["root_pc"]), str(item["quality"]), key)),
                None,
            )
            diatonic_fallback.append(best_diatonic)
        for index, state in enumerate(states):
            if _is_diatonic(int(state["root_pc"]), str(state["quality"]), key):
                continue
            if float(state.get("slot_confidence") or 0.0) >= _DIATONIC_ESCAPE_CONFIDENCE:
                continue
            replacement = diatonic_fallback[index] if index < len(diatonic_fallback) else None
            if replacement is not None:
                states[index] = dict(replacement)

    # 5) 低音转位判断（同时记录证据强度，供后处理决定要不要保留斜杠）
    #    斜杠和弦在流行歌里远少于普通三和弦：如果低音指向的音自己作根音
    #    也能得到几乎一样的分（差距 < _SLASH_PREFERENCE_MARGIN），就采信
    #    「根音 = 低音」的普通和弦（C/E → Em），而不是转位。
    bass_pitch: list[int | None] = []
    bass_strength: list[float] = []
    for index, state in enumerate(states):
        if bass_matrix is None or index >= bass_matrix.shape[1]:
            bass_pitch.append(None)
            bass_strength.append(0.0)
            continue
        root_pc = int(state["root_pc"])
        pitch, strength = _detect_bass_pitch(bass_matrix[:, index], root_pc, np)
        if pitch is not None and pitch != root_pc:
            alternative = _best_candidate_with_root(emissions[index], pitch)
            if (alternative is not None
                    and float(state.get("score") or 0.0) - float(alternative["score"])
                    < _SLASH_PREFERENCE_MARGIN):
                states[index] = dict(alternative)
                root_pc = int(alternative["root_pc"])
                pitch, strength = _detect_bass_pitch(bass_matrix[:, index], root_pc, np)
        bass_pitch.append(pitch)
        bass_strength.append(float(strength))

    # 6) 合并 + 清理
    median_beat = 0.5
    if len(slot_times) >= 2:
        intervals = sorted(end - start for start, end in slot_times)
        median_beat = intervals[len(intervals) // 2]
    prefer_flat = bool(key and prefer_flat_key(int(key.get("tonic_pc") or 0), str(key.get("mode") or "major")))
    events = merge_slots_to_events(
        states, slot_times, prefer_flat=prefer_flat,
        bass_pitch=bass_pitch, bass_strength=bass_strength,
    )
    events = cleanup_events(
        events,
        min_duration=max(0.28, median_beat * 0.85),
        key=key,
        prefer_flat=prefer_flat,
    )
    # 6b) 质量收敛：证据不足的 sus/dim/aug/七和弦回退到调内三和弦
    events = simplify_qualities(events, key, prefer_flat=prefer_flat)
    # 6c) 斜杠和弦需要低频证据；否则去低音
    events = drop_unsupported_slash(events, prefer_flat=prefer_flat)
    # 7) 收尾：对齐到 0..total、保证连续覆盖，再标注来源
    events = enforce_boundaries(events, total_seconds, prefer_flat=prefer_flat)
    bar_seconds, bar_starts = _bar_grid(beats, total_seconds)
    suffix = f"+repetition({repetition_stats['touched_slots']})" if repetition_stats.get("touched_slots") else ""
    for event in events:
        event["source"] = f"chroma-template+{bass_source}+viterbi{suffix}"
        event.pop("bass_strength", None)
        # 持续越久的和弦越可信（同一和弦占 4 小节显然比占半小节更确定）
        duration = float(event["end"]) - float(event["start"])
        bars = duration / bar_seconds if bar_seconds > 0 else 0.0
        bonus = min(0.18, 0.06 * bars)
        event["confidence"] = round(max(0.05, min(0.98, float(event["confidence"]) + bonus)), 3)
        event["bars"] = round(bars, 2)
        event["bar_aligned"] = bool(
            bar_starts and min(abs(float(event["start"]) - value) for value in bar_starts) <= max(0.18, bar_seconds * 0.08)
        )
    return events


def _bar_grid(beats: dict[str, Any], total_seconds: float) -> tuple[float, list[float]]:
    """小节长度与小节起点（用于和弦的 bar 对齐标记）。"""
    beat_list = [float(value) for value in (beats.get("beats") or [])]
    per_bar = int(beats.get("beats_per_bar") or 4)
    bar_seconds = 0.0
    if len(beat_list) >= 2:
        intervals = sorted(b - a for a, b in zip(beat_list, beat_list[1:]))
        bar_seconds = intervals[len(intervals) // 2] * per_bar
    if bar_seconds <= 0:
        bpm = float(beats.get("bpm_display") or beats.get("bpm") or 0.0)
        bar_seconds = (60.0 / bpm * per_bar) if bpm > 0 else 0.0
    downbeats = sorted(float(value) for value in (beats.get("downbeats") or []))
    if downbeats:
        starts = downbeats
    elif beat_list and per_bar > 0:
        starts = [beat_list[index] for index in range(0, len(beat_list), per_bar)]
    else:
        starts = []
    return bar_seconds, starts

"""F6（前半）：调性估计。

两路证据融合：
1. 音频侧：整曲 chroma 直方图与 Krumhansl 调性轮廓做余弦相关（24 个大小调轮转）；
2. 和弦侧：已识别和弦的调内功能得分（主/属/下属加权），可消除关系大小调歧义。

对外接口（契约不可改）：``detect_key(source_wav, chords=None) -> dict``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SAMPLE_RATE = 22050
HOP_LENGTH = 512

PITCH_CLASS_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASS_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl–Kessler 调性轮廓（原版数值）
KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# 五度圈上的调号数量（升号为 +，降号为 -）
_MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 5: -1, 10: -2, 3: -3, 8: -4}
_MINOR_FIFTHS = {9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: 6, 10: 7, 2: -1, 7: -2, 0: -3, 5: -4}

# 音阶内各音级允许的和弦质量（用于判断某个和弦是否「调内」）。
# 键是「相对主音的半音数」，不是音级序号——调用方一律用 (root_pc - tonic_pc) % 12 查表。
MAJOR_DIATONIC_QUALITIES: dict[int, set[str]] = {
    0: {"", "maj7", "6"},       # I
    2: {"m", "m7"},             # ii
    4: {"m", "m7"},             # iii
    5: {"", "maj7", "6"},       # IV
    7: {"", "7", "maj7"},       # V
    9: {"m", "m7"},             # vi
    11: {"dim", "m7b5"},        # vii°
}
MINOR_DIATONIC_QUALITIES: dict[int, set[str]] = {
    0: {"m", "m7"},             # i
    2: {"dim", "m7b5"},         # ii°
    3: {"", "maj7", "6"},       # III
    5: {"m", "m7"},             # iv
    7: {"m", "m7", "", "7"},    # v（自然小调）/ V（和声小调属和弦）
    8: {"", "maj7", "6"},       # VI
    10: {"", "7"},              # VII
}
# 各音级的功能权重：主功能 / 属功能 / 下属功能（同样按半音数索引）
_MAJOR_FUNCTION = {0: 1.0, 7: 0.6, 9: 0.5, 4: 0.25, 5: 0.28, 2: 0.2, 11: 0.15}
_MINOR_FUNCTION = {0: 1.0, 7: 0.6, 5: 0.5, 3: 0.25, 8: 0.28, 10: 0.2, 2: 0.15}


def _require_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("调性估计需要 numpy 与 librosa。") from exc
    return librosa, np


# --------------------------------------------------------------------------- #
# 纯函数内核
# --------------------------------------------------------------------------- #

def note_name(pitch_class: int, *, prefer_flat: bool = False) -> str:
    """音级 → 音名（默认升号，符合 schema 对 key.tonic 的要求）。"""
    index = int(pitch_class) % 12
    table = PITCH_CLASS_NAMES_FLAT if prefer_flat else PITCH_CLASS_NAMES_SHARP
    return table[index]


def prefer_flat_key(tonic_pc: int, mode: str) -> bool:
    """判断该调性用降号更自然（F、Bb、Eb、Ab、Db、Gb 及其关系小调）。"""
    fifths = (_MAJOR_FIFTHS if mode == "major" else _MINOR_FIFTHS).get(int(tonic_pc) % 12, 0)
    return fifths < 0


def key_signature_count(tonic_pc: int, mode: str) -> int:
    """调号数量：正数为升号数，负数为降号数。"""
    table = _MAJOR_FIFTHS if mode == "major" else _MINOR_FIFTHS
    return int(table.get(int(tonic_pc) % 12, 0))


def rotate_profile(profile: list[float], tonic_pc: int) -> list[float]:
    """把调性轮廓旋转到指定主音（profile[0] 对应主音）。"""
    size = len(profile)
    return [profile[(index - tonic_pc) % size] for index in range(size)]


def cosine_similarity(left: Any, right: Any) -> float:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("调性估计需要 numpy。") from exc
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_key_profiles(histogram: Any) -> list[dict[str, Any]]:
    """对 24 个大小调打分（Krumhansl 轮廓余弦相关）。"""
    values = list(float(v) for v in histogram)
    if len(values) != 12:
        raise ValueError(f"chroma 直方图长度应为 12，实际 {len(values)}")
    scores: list[dict[str, Any]] = []
    for mode, profile in (("major", KRUMHANSL_MAJOR), ("minor", KRUMHANSL_MINOR)):
        for tonic in range(12):
            scores.append({
                "tonic_pc": tonic,
                "mode": mode,
                "score": cosine_similarity(values, rotate_profile(profile, tonic)),
            })
    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores


def score_key_from_chords(chords: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """根据和弦事件（时长加权）给 24 个调打分：调内功能得分。"""
    if not chords:
        return []
    total = 0.0
    weighted: list[tuple[int, str, float]] = []
    for chord in chords:
        start = float(chord.get("start") or 0.0)
        end = float(chord.get("end") or start)
        duration = max(0.0, end - start)
        root = chord.get("root_pc")
        if duration <= 0 or root is None:
            continue
        quality = _quality_of(str(chord.get("symbol") or ""), root)
        weighted.append((int(root) % 12, quality, duration))
        total += duration
    if total <= 0:
        return []
    scores: list[dict[str, Any]] = []
    for mode in ("major", "minor"):
        table = MAJOR_DIATONIC_QUALITIES if mode == "major" else MINOR_DIATONIC_QUALITIES
        functions = _MAJOR_FUNCTION if mode == "major" else _MINOR_FUNCTION
        for tonic in range(12):
            score = 0.0
            for root, quality, duration in weighted:
                degree = (root - tonic) % 12
                weight = duration / total
                if degree in table and _quality_matches(quality, table[degree]):
                    score += weight * (1.0 + functions.get(degree, 0.0))
                else:
                    score -= weight * 0.6
            scores.append({"tonic_pc": tonic, "mode": mode, "score": score})
    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores


def _quality_of(symbol: str, root_pc: int) -> str:
    """从和弦符号里剥出质量后缀（如 C#m7/G# → m7）。"""
    body = symbol.split("/")[0].strip()
    for prefix in (PITCH_CLASS_NAMES_SHARP[int(root_pc) % 12], PITCH_CLASS_NAMES_FLAT[int(root_pc) % 12]):
        if body.startswith(prefix):
            return body[len(prefix):]
    return body[1:] if body else ""


def _quality_matches(quality: str, allowed: set[str]) -> bool:
    if quality in allowed:
        return True
    # 允许省略：maj 和弦标成 ""，属七标成 "7"，增和弦等视为调外。
    return quality in {"", "m", "7", "maj7", "m7", "6", "dim", "m7b5"} and any(
        quality.startswith(candidate) or candidate.startswith(quality) for candidate in allowed if candidate
    )


def normalize_scores(scores: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    """把一路候选分线性归一化到 0..1。

    音频路（Krumhansl 余弦）在 24 个调上几乎是饱和的（实测 top4 差距 <0.02），
    和弦路量纲完全不同（可达 ±1.5）。直接按权重相加会让音频路变成噪声，
    所以两路先各自 min-max 归一化，融合结果才有「相对可信度」的含义。
    """
    if not scores:
        return {}
    values = [float(item["score"]) for item in scores]
    low, high = min(values), max(values)
    span = high - low
    output: dict[tuple[int, str], float] = {}
    for item in scores:
        key = (int(item["tonic_pc"]), str(item["mode"]))
        output[key] = 0.5 if span <= 1e-9 else (float(item["score"]) - low) / span
    return output


def combine_key_scores(
    audio_scores: list[dict[str, Any]],
    chord_scores: list[dict[str, Any]] | None,
    *,
    audio_weight: float = 0.58,
) -> list[dict[str, Any]]:
    """融合音频与和弦两路打分（各自归一化后按权重相加）。

    返回项包含归一化后的 `score`，以及未归一化的 `audio_raw` / `chord_raw`，
    方便排查「为什么选了这个调」。
    """
    audio_map = normalize_scores(audio_scores)
    chord_map = normalize_scores(chord_scores or [])
    raw_audio = {(int(i["tonic_pc"]), str(i["mode"])): float(i["score"]) for i in audio_scores}
    raw_chord = {(int(i["tonic_pc"]), str(i["mode"])): float(i["score"]) for i in (chord_scores or [])}
    weight = float(audio_weight) if chord_map else 1.0
    if chord_map:
        weight = min(1.0, max(0.0, weight))
    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for key, audio_norm in audio_map.items():
        item = {
            "tonic_pc": key[0],
            "mode": key[1],
            "score": audio_norm * weight,
            "audio_raw": round(raw_audio.get(key, 0.0), 6),
            "chord_raw": round(raw_chord.get(key, 0.0), 6),
        }
        if chord_map:
            item["chord_score"] = round(float(chord_map.get(key, 0.0)), 6)
        merged[key] = item
    if chord_map:
        chord_weight = 1.0 - weight
        for key, chord_norm in chord_map.items():
            item = merged.get(key)
            if item is None:
                item = {
                    "tonic_pc": key[0], "mode": key[1], "score": 0.0,
                    "audio_raw": round(raw_audio.get(key, 0.0), 6),
                    "chord_raw": round(raw_chord.get(key, 0.0), 6),
                }
                merged[key] = item
            item["chord_score"] = round(float(chord_norm), 6)
            item["score"] = float(item["score"]) + chord_norm * chord_weight
    ranked = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
    for item in ranked:
        item["score"] = round(float(item["score"]), 6)
    return ranked


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def detect_key(source_wav: str, chords: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """估计调性（主音 + 大小调）。

    返回 {"tonic", "mode", "confidence", "num_sharps_flats",
          "name", "alternatives", "warnings"}
    """
    librosa, np = _require_libs()
    warnings: list[str] = []
    y, sr = librosa.load(str(source_wav), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        warnings.append("音频为空，调性无法估计，默认 C 大调。")
        return _fallback_key(warnings)

    try:
        y_harm = librosa.effects.harmonic(y, margin=2.0)
    except Exception:  # pragma: no cover
        y_harm = y
        warnings.append("谐波分离失败，直接用原始波形估计调性。")

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    # 用帧能量加权，静音段不参与统计
    frame_energy = np.sqrt(np.mean(chroma ** 2, axis=0))
    weights = np.clip(frame_energy, 0.0, None)
    if float(weights.sum()) <= 1e-9:
        weights = np.ones_like(weights)
    histogram = np.sum(chroma * weights, axis=1)
    if float(histogram.sum()) <= 1e-9:
        warnings.append("chroma 能量过低，调性主要依赖和弦证据。")
        histogram = np.ones(12, dtype=np.float64)

    audio_scores = score_key_profiles(histogram)
    chord_scores = score_key_from_chords(chords or [])
    if not chord_scores:
        warnings.append("未提供和弦事件，调性仅由音频 chroma 推断。")
    ranked = combine_key_scores(audio_scores, chord_scores)

    best = ranked[0]
    relative_major_swap = False
    # 关系大小调歧义：吉他弹唱谱行业惯例按大调标注（调号相同、音阶相同、和弦相同，
    # 但「D 大调 + capo 0」对业余用户比「B 小调」友好得多）。delta < 0.05 时改选大调。
    if best["mode"] == "minor":
        relative_major = (int(best["tonic_pc"]) + 3) % 12
        for candidate in ranked[1:]:
            if (int(candidate["tonic_pc"]), candidate["mode"]) != (relative_major, "major"):
                continue
            if float(best["score"]) - float(candidate["score"]) < 0.05:
                warnings.append(
                    f"关系大小调歧义（{note_name(int(best['tonic_pc']))} 小调 vs "
                    f"{note_name(relative_major)} 大调，delta="
                    f"{float(best['score']) - float(candidate['score']):.3f}），按行业惯例按大调标注。"
                )
                best = candidate
                relative_major_swap = True
            break
    # 置信度只看「和关系调之外的候选」的差距：关系大小调本来就分不开（调号/音阶/和弦全同），
    # 把它们算进 margin 会让用户看到莫名其妙的低置信度。
    pair_pcs = {
        int(best["tonic_pc"]) % 12,
        (int(best["tonic_pc"]) + (3 if str(best["mode"]) == "minor" else 9)) % 12,
    }
    others = [
        candidate for candidate in ranked
        if (int(candidate["tonic_pc"]) % 12, str(candidate["mode"])) != (int(best["tonic_pc"]) % 12, str(best["mode"]))
        and int(candidate["tonic_pc"]) % 12 not in pair_pcs
    ]
    second = others[0] if others else {"score": float(best["score"]) - 0.2, "tonic_pc": -1, "mode": "major"}
    margin = max(0.0, float(best["score"]) - float(second["score"]))
    confidence = max(0.05, min(0.98, 0.45 + margin * 2.2))
    # 两路证据一致时加一点置信度
    if chord_scores and (chord_scores[0]["tonic_pc"], chord_scores[0]["mode"]) == (best["tonic_pc"], best["mode"]):
        confidence = min(0.98, confidence + 0.06)


    tonic_pc = int(best["tonic_pc"])
    mode = str(best["mode"])
    flat = prefer_flat_key(tonic_pc, mode)
    return {
        "tonic": note_name(tonic_pc, prefer_flat=False),
        "mode": mode,
        "confidence": round(float(confidence), 3),
        "score": round(float(best["score"]), 4),
        "num_sharps_flats": key_signature_count(tonic_pc, mode),
        # 为 True 时 alternatives 里可能有一条 score 更高的关系大小调（按大调惯例未采用）
        "relative_major_swap": bool(relative_major_swap),
        "tonic_pc": tonic_pc,
        "name": f"{note_name(tonic_pc, prefer_flat=flat)} {'大调' if mode == 'major' else '小调'}",
        "display_name": f"{note_name(tonic_pc, prefer_flat=flat)}{'' if mode == 'major' else 'm'}",
        "prefer_flat": flat,
        "alternatives": [
            {
                "tonic": note_name(item["tonic_pc"]),
                "tonic_pc": int(item["tonic_pc"]),
                "mode": item["mode"],
                "score": round(float(item["score"]), 4),
                "delta": round(float(best["score"]) - float(item["score"]), 4),
            }
            for item in [entry for entry in ranked if entry is not best][:4]
        ],
        "warnings": warnings,
    }


def _fallback_key(warnings: list[str]) -> dict[str, Any]:
    return {
        "tonic": "C",
        "mode": "major",
        "confidence": 0.05,
        "score": 0.0,
        "num_sharps_flats": 0,
        "tonic_pc": 0,
        "name": "C 大调",
        "display_name": "C",
        "prefer_flat": False,
        "alternatives": [],
        "warnings": warnings,
    }

"""F5：节拍、速度与拍号。

策略（纯上下文，方便对照代码）：
1. 先做谐波/打击分离，用打击分量算 onset 包络，节拍更稳；
2. 用 onset 自相关 + 对数速度先验挑选「音乐速度」，解决 beat tracker
   常见的半速/倍速锁相问题（八分音符锁相 → 显示 BPM 翻倍）；
3. 用选定的显示速度重新约束 beat_track，得到与 BPM 自洽的拍点；
4. 用低频重音（底鼓）+ 和声变化两路证据检测小节相位，给出拍号与 downbeats。

对外接口（契约不可改）：``analyze_beats(source_wav) -> dict``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# 分析统一规格，与 aitsm.pipeline.ingest 保持一致。
SAMPLE_RATE = 22050
HOP_LENGTH = 512  # ≈23.2ms

# 音乐速度先验：以 100BPM 为中心的宽高斯（对数域），避免极端速度被选中。
_TEMPO_PRIOR_CENTER = 100.0
_TEMPO_PRIOR_SIGMA_OCTAVES = 0.85
# 候选速度范围（显示用 BPM）。
_MIN_BPM = 45.0
_MAX_BPM = 210.0


def _require_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "节拍分析需要 numpy 与 librosa，请先安装核心依赖（numpy scipy librosa soundfile）。"
        ) from exc
    return librosa, np


def _require_np():
    """纯函数只需 numpy；单独拆出来，未装 librosa 时也能对内核做单测。"""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("该函数需要 numpy，请先安装核心依赖。") from exc
    return np


# --------------------------------------------------------------------------- #
# 纯函数内核（输入数组，可在单测里直接喂合成数据）
# --------------------------------------------------------------------------- #

def autocorrelate(x: Any, max_size: int) -> Any:
    """onset 包络自相关，返回长度 max_size、已按 ac[0] 归一化的数组。"""
    np = _require_np()
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return np.zeros(1, dtype=np.float64)
    values = values - values.mean()
    norm = float(np.sqrt(np.sum(values * values)))
    if norm <= 1e-9:
        return np.zeros(max_size, dtype=np.float64)
    size = 1
    while size < values.size * 2:
        size *= 2
    spectrum = np.fft.rfft(values, n=size)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:max_size]
    return ac / (abs(float(ac[0])) + 1e-12)


def tempo_prior(bpm: float) -> float:
    """对数域高斯先验：60~160BPM 最自然，越极端得分越低。"""
    if bpm <= 0:
        return 0.0
    octaves = math.log2(float(bpm) / _TEMPO_PRIOR_CENTER)
    return math.exp(-0.5 * (octaves / _TEMPO_PRIOR_SIGMA_OCTAVES) ** 2)


def score_tempo_candidates(ac: Any, sr: int, hop_length: int, candidates: list[float]) -> list[dict[str, Any]]:
    """用自相关 + 先验给速度候选打分。

    自相关在「真实脉冲周期」及其倍数处都会出现峰，因此把候选的 1/2、1、2
    倍周期一起纳入（谐波求和），让半速/倍速候选的强弱自然区分开。
    """
    np = _require_np()
    ac = np.asarray(ac, dtype=np.float64)
    frames_per_second = sr / float(hop_length)
    scored: list[dict[str, Any]] = []
    for bpm in candidates:
        if bpm <= 0:
            continue
        lag = 60.0 * frames_per_second / float(bpm)
        if lag < 1.5 or lag >= len(ac) - 1:
            continue
        # 取 lag 邻域最大值，容忍 ±3% 的节拍漂移。
        lo = max(1, int(round(lag * 0.97)))
        hi = min(len(ac) - 1, int(round(lag * 1.03)) + 1)
        peak = float(np.max(ac[lo:hi])) if hi > lo else 0.0
        harmonic = 0.0
        for multiplier, weight in ((2.0, 0.5), (0.5, 0.35), (4.0, 0.2)):
            sub_lag = 60.0 * frames_per_second / (bpm * multiplier)
            if 1.5 <= sub_lag < len(ac) - 1:
                sub_lo = max(1, int(round(sub_lag * 0.96)))
                sub_hi = min(len(ac) - 1, int(round(sub_lag * 1.04)) + 1)
                if sub_hi > sub_lo:
                    harmonic += weight * float(np.max(ac[sub_lo:sub_hi]))
        score = (0.68 * peak + 0.32 * harmonic) * tempo_prior(bpm)
        scored.append({"bpm": round(float(bpm), 3), "score": round(score, 6),
                       "autocorr_peak": round(peak, 6)})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def choose_display_tempo(raw_bpm: float, ac: Any, sr: int, hop_length: int) -> dict[str, Any]:
    """在 raw_bpm 的半速/倍速八度里挑一个最像「人听的拍子」的显示速度。"""
    if raw_bpm <= 0:
        return {"bpm": 120.0, "normalization": None, "candidates": []}
    candidates: list[float] = []
    for factor in (0.5, 1.0, 2.0):
        value = raw_bpm * factor
        if _MIN_BPM <= value <= _MAX_BPM:
            candidates.append(round(value, 3))
    if not candidates:
        candidates = [min(max(raw_bpm, _MIN_BPM), _MAX_BPM)]
    scored = score_tempo_candidates(ac, sr, hop_length, candidates)
    if not scored:
        return {"bpm": round(candidates[0], 3), "normalization": None, "candidates": []}
    best = scored[0]
    # 用比例判定八度修正：候选速度是四舍五入到 3 位小数的，直接比较绝对差会
    # 把「同一个速度」误判成 half_time/double_time。
    normalization = None
    if raw_bpm > 0:
        ratio = float(best["bpm"]) / float(raw_bpm)
        if 0.45 <= ratio <= 0.55:
            normalization = "half_time"
        elif 1.8 <= ratio <= 2.2:
            normalization = "double_time"
    return {"bpm": float(best["bpm"]), "normalization": normalization, "candidates": scored}


def fit_beat_grid(beat_times: list[float]) -> dict[str, Any]:
    """对拍点做等距网格最小二乘拟合，返回拟合速度/周期/起点与最大残差。

    拍的原始估计受帧长量化影响（hop 512 @22050Hz ≈ 23ms），逐拍间隔
    逐次取中位数会把这个量化误差变成系统性速度误差（例：真值 138BPM
    可能测成 136BPM），4 分钟下来累积偏移可达数秒。用整串拍点做直线
    拟合可以把这个误差平均掉。
    """
    np = _require_np()
    times = np.asarray(beat_times, dtype=np.float64)
    if times.size < 4:
        return {"bpm": 0.0, "period": 0.0, "offset": 0.0,
                "max_residual": float("inf"), "times": list(beat_times)}
    index = np.arange(times.size, dtype=np.float64)
    design = np.vstack([np.ones_like(index), index]).T
    solution, *_ = np.linalg.lstsq(design, times, rcond=None)
    offset, period = float(solution[0]), float(solution[1])
    if period <= 1e-6:
        return {"bpm": 0.0, "period": 0.0, "offset": 0.0,
                "max_residual": float("inf"), "times": list(beat_times)}
    snapped = offset + period * index
    residual = float(np.max(np.abs(times - snapped)))
    return {
        "bpm": 60.0 / period,
        "period": period,
        "offset": offset,
        "max_residual": residual,
        "times": [round(float(value), 4) for value in snapped],
    }


def beat_regularity(beat_times: list[float]) -> float:
    """拍点间隔一致性：1 表示完全等距，0 表示杂乱。"""
    if len(beat_times) < 3:
        return 0.0
    intervals = [b - a for a, b in zip(beat_times, beat_times[1:]) if b > a]
    if not intervals:
        return 0.0
    median = sorted(intervals)[len(intervals) // 2]
    if median <= 0:
        return 0.0
    deviation = sum(abs(value - median) for value in intervals) / len(intervals)
    return float(max(0.0, min(1.0, 1.0 - (deviation / median) * 1.6)))


# 拍号先验：4/4 最常见，3/4 次之，2/4 再次；6/8 不给先验（必须靠证据赢）。
_METER_PRIORITY = {4: 1.0, 3: 0.85, 2: 0.45, 6: 0.0}
# 6/8 的八分音符拍通常很快；如果拍间隔超过这个值（<150 拍/分），
# 把它解释成「每拍八分」的 6/8 就不合理，更可能是 3/4 的四分音符拍。
_SIX_EIGHT_MAX_BEAT_INTERVAL = 0.40
_SIX_EIGHT_SLOW_PENALTY = 0.12


def _zscore(values: Any) -> Any:
    """标准化到零均值单位方差；全常量输入返回全零（表示「无证据」）。"""
    np = _require_np()
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    spread = float(arr.std())
    if spread <= 1e-9:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / spread


def detect_meter(
    low_accents: list[float],
    change_accents: list[float],
    *,
    prefer: tuple[int, ...] = (4, 3, 2),
    prior_bonus: float = 0.05,
    beat_interval: float | None = None,
) -> dict[str, Any]:
    """根据逐拍证据判定每小节拍数与 downbeat 相位。

    low_accents: 每拍的低频重音强度（底鼓/贝斯起音）
    change_accents: 每拍的和声变化强度（和弦切换多发生在小节头）
    beat_interval: 拍点间隔中位数（秒），用于区分 3/4 与 6/8

    打分方式：把两路证据各自标准化成 z 分数，再看「候选小节头位置上的
    平均 z」——重音与和声变化越集中在该位置，分越高。这样 6 拍候选
    不会仅因为「只是 3 拍候选的抽样」而占便宜（两者的 z 相同），
    6/8 必须靠额外的次强拍证据与更快的八分音符拍才能赢。
    """
    np = _require_np()
    count = len(low_accents)
    if count < 4:
        return {"beats_per_bar": 4, "time_signature": "4/4", "phase": 0,
                "confidence": 0.1, "candidates": []}
    low_z = _zscore(low_accents)
    change_z = _zscore(change_accents)
    # 没有和声变化证据（纯打击乐/单和弦循环）时只看重音。
    has_change = float(np.std(np.asarray(change_accents, dtype=np.float64))) > 1e-6
    weight_low = 0.45 if has_change else 0.80
    weight_change = 0.55 if has_change else 0.0
    scored: list[dict[str, Any]] = []
    for bpb in (2, 3, 4, 6):
        if count < bpb * 2:
            continue
        best_phase, best_score = 0, -1e9
        for phase in range(bpb):
            positions = np.arange(phase, count, bpb)
            if positions.size < 2:
                continue
            score = (weight_low * float(np.mean(low_z[positions]))
                     + weight_change * float(np.mean(change_z[positions])))
            if score > best_score:
                best_phase, best_score = phase, score
        bonus = prior_bonus * _METER_PRIORITY.get(bpb, 0.0) if bpb in prefer else 0.0
        penalty = 0.0
        if (bpb == 6 and beat_interval is not None
                and beat_interval >= _SIX_EIGHT_MAX_BEAT_INTERVAL):
            penalty = _SIX_EIGHT_SLOW_PENALTY
        scored.append({
            "beats_per_bar": bpb,
            "phase": best_phase,
            "score": round(best_score + bonus - penalty, 6),
            "raw_score": round(best_score, 6),
        })
    if not scored:
        return {"beats_per_bar": 4, "time_signature": "4/4", "phase": 0,
                "confidence": 0.1, "candidates": []}
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    demoted_from = None
    # 3/4 与 6/8 的差别本质是「拍单位」：6/8 的八分音符拍必须够快
    # （复合拍子的记谱拍是附点四分音符，约 50~120BPM）。如果检测到的拍
    # 间隔 >= 0.40s，把它当成八分音符会得到 <50BPM 的附点四分，音乐上
    # 不合理；此时按 3/4（同样 3 个八分/小节，但拍是四分音符）解释。
    if (int(best["beats_per_bar"]) == 6 and beat_interval is not None
            and beat_interval >= _SIX_EIGHT_MAX_BEAT_INTERVAL):
        alternatives = [item for item in scored if int(item["beats_per_bar"]) != 6]
        if alternatives:
            preferred = min(
                alternatives,
                key=lambda item: (abs(int(item["beats_per_bar"]) - 3),
                                   -float(item["score"])),
            )
            demoted_from = 6
            best = preferred
            second_score = float(scored[1]["score"]) if scored[1] is not best else float(scored[0]["score"])
            margin = max(0.0, float(best["score"]) - second_score)
        else:
            margin = max(0.0, float(best["score"]) - float(scored[1]["score"])) if len(scored) > 1 else 0.15
    else:
        second = scored[1]["score"] if len(scored) > 1 else best["score"] - 0.15
        margin = max(0.0, float(best["score"]) - float(second))
    signature = {2: "2/4", 3: "3/4", 4: "4/4", 6: "6/8"}[int(best["beats_per_bar"])]
    result: dict[str, Any] = {
        "beats_per_bar": int(best["beats_per_bar"]),
        "time_signature": signature,
        "phase": int(best["phase"]),
        "confidence": round(max(0.1, min(0.95, 0.35 + margin * 1.5)), 3),
        "candidates": scored[:3],
    }
    if demoted_from is not None:
        result["demoted_from"] = demoted_from
        result["note"] = (
            f"6/8 候选得分更高，但拍间隔 {beat_interval:.2f}s 作为八分音符太慢"
            f"（附点四分 <50BPM），已按 3/4 解释。"
        )
    return result


def _normalize01(values: Any) -> list[float]:
    np = _require_np()
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-9:
        return [0.0 for _ in arr]
    return [float((value - lo) / (hi - lo)) for value in arr]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def analyze_beats(source_wav: Path | str) -> dict[str, Any]:
    """估计 BPM、拍点、小节线与拍号。

    返回 {"bpm", "bpm_display", "confidence", "beats", "downbeats",
          "beats_per_bar", "time_signature", "beat_strength",
          "first_downbeat_sec", "pickup_beats", "raw_bpm", "warnings"}
    """
    librosa, np = _require_libs()
    path = str(source_wav)
    warnings: list[str] = []

    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    duration = float(len(y)) / float(sr)
    if duration < 1.5:
        warnings.append(f"音频仅 {duration:.2f}s，节拍与拍号不可靠。")
        return _degenerate_result(duration, warnings)

    # 打击分量更贴近鼓点，谐波分量用于和声变化；一次 HPSS 同时拿到两份，
    # 比先 percussive 再 harmonic 省掉一遍 STFT + 中值滤波（4 分钟歌约省 10s）。
    try:
        y_harm, y_perc = librosa.effects.hpss(y, margin=(2.0, 2.0), kernel_size=17)
    except Exception:  # pragma: no cover - 极端短音频可能异常
        y_harm, y_perc = y, y
        warnings.append("谐波/打击分离失败，使用原始波形估计节拍。")

    onset_env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    if float(np.max(onset_env, initial=0.0)) <= 1e-8:
        warnings.append("未检测到明显起音（可能是静音或纯持续音），速度不可靠。")
        return _degenerate_result(duration, warnings, old_result=True)

    # 1) 自相关 + 先验给出显示速度（修半速/倍速）
    ac = autocorrelate(onset_env, max_size=int(60.0 / 45.0 * sr / HOP_LENGTH) + 4)
    # 粗速度只需要一个八度内的参考值：用 tempo(ac) 聚合，比跑一次完整 beat_track
    # （动态规划，60s 音频约 1.5s）便宜一个数量级。
    try:
        rough_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        rough_bpm = float(np.atleast_1d(rough_tempo)[0])
    except Exception:  # pragma: no cover - 老版本 librosa 没有 feature.tempo
        rough_tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, trim=False,
        )
        rough_bpm = float(np.atleast_1d(rough_tempo)[0]) if np.size(rough_tempo) else 0.0
    display = choose_display_tempo(rough_bpm, ac, sr, HOP_LENGTH)
    if display.get("normalization"):
        warnings.append(
            f"速度经过八度修正（{display['normalization']}）：{rough_bpm:.1f}BPM → "
            f"{float(display['bpm']):.1f}BPM。"
        )

    # 2) 用显示速度约束重新跟踪拍点，保证 beats 与 bpm 自洽
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        start_bpm=float(display["bpm"]), tightness=200, trim=False, units="frames",
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    if beat_frames.size < 2:
        # 退一步：不约束再试一次
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, trim=False, units="frames",
        )
        beat_frames = np.asarray(beat_frames, dtype=int)
    if beat_frames.size < 2:
        warnings.append("拍点跟踪只返回不足 2 个拍点，节拍结果不可靠。")
        return _degenerate_result(duration, warnings, old_result=True)

    beat_times = [round(float(value), 4) for value in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)]
    # 过滤落在音轨之外的拍点
    beat_times = [t for t in beat_times if 0.0 <= t <= duration]
    if len(beat_times) < 2:
        warnings.append("有效拍点不足，节拍结果不可靠。")
        return _degenerate_result(duration, warnings, old_result=True)

    intervals = [b - a for a, b in zip(beat_times, beat_times[1:]) if b > a]
    median_interval = sorted(intervals)[len(intervals) // 2] if intervals else 0.0
    measured_bpm = 60.0 / median_interval if median_interval > 0 else float(display["bpm"])
    measured_bpm = float(min(320.0, max(20.0, measured_bpm)))

    # 2b) 等距网格拟合：修正帧量化带来的系统性速度误差，减少整曲累积偏移。
    #     残差很小 → 拍点本来就规整，直接把 times 换成正弦网格；
    #     残差较大（有 rubato/现场演奏）→ 只改速度，不动拍点。
    grid = fit_beat_grid(beat_times)
    grid_info: dict[str, Any] = {
        "fitted_bpm": round(float(grid["bpm"]), 2) if grid["period"] > 0 else None,
        "max_residual": round(float(grid["max_residual"]), 4) if grid["period"] > 0 else None,
        "snapped": False,
    }
    if grid["period"] > 0 and grid["bpm"] > 0:
        snap_limit = max(0.06, 0.14 * median_interval) if median_interval > 0 else 0.06
        if float(grid["max_residual"]) <= snap_limit:
            measured_bpm = float(min(320.0, max(20.0, 60.0 / float(grid["period"]))))
            # 保留拍点个数（重音列表按下标对齐），只把越界值夹回音轨内。
            beat_times = [round(min(duration, max(0.0, float(value))), 4) for value in grid["times"]]
            grid_info["snapped"] = True
        elif abs(float(grid["bpm"]) - measured_bpm) / max(measured_bpm, 1e-9) < 0.08:
            measured_bpm = float(min(320.0, max(20.0, float(grid["bpm"]))))
    if not beat_times:
        warnings.append("有效拍点不足，节拍结果不可靠。")
        return _degenerate_result(duration, warnings, old_result=True)

    # 3) 逐拍证据：低频重音 + 和声变化
    # 低音重音只关心 30~200Hz：必须收窄 mel 通道数，否则窄低频带会退化成空滤波器
    # （librosa 1.0 会打印 "Empty filters detected"，且那些通道全是 0）。
    low_env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, hop_length=HOP_LENGTH,
        fmin=30, fmax=200, n_mels=8, n_fft=2048, aggregate=np.max,
    )
    beat_strength = _sample_envelope(onset_env, beat_frames, sr, HOP_LENGTH, duration)
    low_accents = _sample_envelope(low_env, beat_frames, sr, HOP_LENGTH, duration)
    change_accents = _harmonic_change_per_beat(y_harm, sr, beat_times, librosa, np)

    meter = detect_meter(low_accents, change_accents, beat_interval=median_interval)
    if meter.get("note"):
        warnings.append(str(meter["note"]))
    bpb = int(meter["beats_per_bar"])
    phase = int(meter["phase"])
    downbeats = beat_times[phase::bpb] if len(beat_times) > phase else []

    # 4) 置信度：拍点规整度 + 脉冲清晰度 + 起音覆盖
    regularity = beat_regularity(beat_times)
    lag = max(1, int(round(60.0 * sr / (measured_bpm * HOP_LENGTH))))
    clarity = float(ac[lag]) if 0 < lag < len(ac) else 0.0
    coverage = float(np.mean(np.asarray(beat_strength) > 0.25 * max(float(np.max(beat_strength, initial=0.0)), 1e-9))) if beat_strength else 0.0
    confidence = max(0.0, min(0.98, 0.18 + 0.42 * regularity + 0.26 * max(0.0, clarity) + 0.14 * coverage))
    if len(beat_times) < 8:
        confidence *= 0.6
        warnings.append("拍点数量过少，速度置信度已下调。")

    return {
        "bpm": round(measured_bpm, 2),
        "bpm_display": round(float(display["bpm"]), 2),
        "confidence": round(float(confidence), 3),
        "beats": beat_times,
        "downbeats": [round(float(value), 4) for value in downbeats],
        "beats_per_bar": bpb,
        "time_signature": meter["time_signature"],
        "beat_strength": [round(float(value), 4) for value in beat_strength],
        "first_downbeat_sec": round(float(downbeats[0]), 4) if len(downbeats) else round(beat_times[0], 4),
        "pickup_beats": int(phase),
        "raw_bpm": round(rough_bpm, 2),
        "beat_grid": grid_info,
        "normalization": display.get("normalization"),
        "meter_candidates": meter.get("candidates", []),
        "hop_length": HOP_LENGTH,
        "sample_rate": sr,
        "warnings": warnings,
    }


def _sample_envelope(envelope: Any, beat_frames: Any, sr: int, hop_length: int, duration: float) -> list[float]:
    """在拍点邻域取包络峰值（±40ms），并把结果归一化到 0..1。"""
    librosa, np = _require_libs()
    if len(beat_frames) == 0:
        return []
    window = max(1, int(round(0.04 * sr / hop_length)))
    values: list[float] = []
    for frame in beat_frames:
        lo = max(0, int(frame) - window)
        hi = min(len(envelope), int(frame) + window + 1)
        values.append(float(np.max(envelope[lo:hi])) if hi > lo else 0.0)
    return _normalize01(values)


def _harmonic_change_per_beat(y_harm: Any, sr: int, beat_times: list[float], librosa: Any, np: Any) -> list[float]:
    """逐拍和声变化量（相邻拍 chroma 的余弦距离），用于找小节头。

    y_harm 应当是已经做过 HPSS 的谐波分量（调用方共享，避免重复分解）。
    """
    if len(beat_times) < 2:
        return [0.0 for _ in beat_times]
    try:
        # 这里只用 chroma 看「和声是否变化」，不需要绝对音高，传 tuning=0 避免
        # librosa 从稀疏频谱估 tuning 时空频集告警（短音频/纯音合成常见）。
        chroma = librosa.feature.chroma_cqt(
            y=y_harm, sr=sr, hop_length=HOP_LENGTH, tuning=0.0,
        )
    except Exception:  # pragma: no cover
        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    frames = librosa.time_to_frames(np.asarray(beat_times, dtype=float), sr=sr, hop_length=HOP_LENGTH)
    vectors: list[Any] = []
    for index, frame in enumerate(frames):
        nxt = int(frames[index + 1]) if index + 1 < len(frames) else int(frame) + 1
        lo = max(0, int(frame))
        hi = max(lo + 1, min(chroma.shape[1], nxt))
        vectors.append(chroma[:, lo:hi].mean(axis=1))
    changes: list[float] = [0.0]
    for left, right in zip(vectors, vectors[1:]):
        left_n = left / (np.linalg.norm(left) + 1e-9)
        right_n = right / (np.linalg.norm(right) + 1e-9)
        changes.append(float(max(0.0, 1.0 - float(np.dot(left_n, right_n)))))
    return changes


def _degenerate_result(duration: float, warnings: list[str], *, old_result: bool = False) -> dict[str, Any]:
    """退化结果：给一个合法的默认速度网格，但如实标低置信度。"""
    bpm = 120.0
    interval = 60.0 / bpm
    beats = [round(index * interval, 4) for index in range(max(1, int(duration / interval) + 1))]
    beats = [t for t in beats if t <= duration + 1e-6]
    downbeats = beats[::4]
    return {
        "bpm": bpm,
        "bpm_display": bpm,
        "confidence": 0.05,
        "beats": beats,
        "downbeats": [round(float(value), 4) for value in downbeats],
        "beats_per_bar": 4,
        "time_signature": "4/4",
        "beat_strength": [0.0 for _ in beats],
        "first_downbeat_sec": round(float(beats[0]), 4) if beats else 0.0,
        "pickup_beats": 0,
        "raw_bpm": 0.0,
        "normalization": "degenerate_grid",
        "meter_candidates": [],
        "hop_length": HOP_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "warnings": warnings + ["已退化为 120BPM 均匀网格，仅用于让下游流程可继续，不代表真实节拍。"],
    }

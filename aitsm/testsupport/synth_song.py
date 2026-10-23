"""生成带「已知正确答案」的合成测试歌曲，用于客观评估扒谱准确率。

为什么需要它
------------
真实歌曲没有标准答案，只能"听着像不像"。合成歌曲的调性、速度、拍号、和弦、
旋律全部是我们自己写下去的，因此可以逐个字段比对，量化每个模块的准确率：

    BPM 误差、调性是否命中、和弦命中率（按时间加权）、拍号是否命中、段落是否命中

这也让回归测试有了意义：改动算法后能立刻看出是变好还是变坏。

合成的局限（写在验收文档里，不能假装是真歌）
--------------------------------------------
- 音色是简单振荡器，没有真实乐器的泛音与动态
- 没有真实的混响、压缩、母带处理
- 人声是用带颤音的正弦近似，不是真实歌唱，因此 **不用于评估歌词/旋律模块**

用法::

    python -m aitsm.testsupport.synth_song --out work/synth/song-a.wav --profile waltz
    python -m aitsm.testsupport.synth_song --list
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 44100

# 音名 -> pitch class
NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# 和弦性质 -> 相对根音的半音
CHORD_TONES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "sus4": [0, 5, 7],
}

# 大调音阶级数 -> 半音
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]


def _midi(name: str, octave: int) -> int:
    return 12 * (octave + 1) + NOTE_PC[name]


def _osc(freq: float, t: np.ndarray, kind: str = "saw") -> np.ndarray:
    """基础振荡器。saw 的谐波更丰富，接近真实乐器，chroma 更容易识别。"""
    phase = 2.0 * math.pi * freq * t
    if kind == "sine":
        return np.sin(phase)
    if kind == "saw":
        # 用前 8 次谐波累加，避免理想锯齿的混叠
        out = np.zeros_like(t)
        for h in range(1, 9):
            out += np.sin(phase * h) / h
        return out * 0.6
    if kind == "square":
        out = np.zeros_like(t)
        for h in range(1, 10, 2):
            out += np.sin(phase * h) / h
        return out * 0.7
    raise ValueError(kind)


def _env(length: int, sr: int = SAMPLE_RATE, attack: float = 0.01,
         release: float = 0.12) -> np.ndarray:
    """简单的 ADSR 包络，避免爆音并让起音清晰（利于 onset 检测）。"""
    env = np.ones(length, dtype=np.float64)
    a = min(int(attack * sr), length // 2)
    r = min(int(release * sr), length // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] *= np.linspace(1.0, 0.0, r)
    # 轻微衰减，给真实感
    env *= np.linspace(1.0, 0.72, length)
    return env


def _place(buffer: np.ndarray, start: int, signal: np.ndarray) -> None:
    """把 signal 叠加到 buffer 的 start 位置，越界自动截断。"""
    if start >= len(buffer) or len(signal) == 0:
        return
    if start < 0:
        signal = signal[-start:]
        start = 0
    end = min(len(buffer), start + len(signal))
    buffer[start:end] += signal[: end - start]


@dataclass
class Profile:
    """一首合成歌曲的完整配方（同时是它的「正确答案」）。"""

    name: str
    title: str
    style: str
    bpm: float
    time_signature: tuple[int, int]
    key_tonic: str
    key_mode: str
    # 每个和弦先用 (根音名, 性质)，每小节一个
    progression: list[tuple[str, str]]
    # 段落划分：(段落类型, 重复次数)，决定小节数
    structure: list[tuple[str, int]]
    # 是否有人声旋律
    with_vocals: bool = True
    # 把最后一段（通常是重复出现的副歌）做旧：低通 + 衰减。
    # 用途是造一个「同一个人重复段落里有一遍信号变差」的用例，
    # 检查「重复一致性」这类跨段证据能不能把它救回来。
    degrade_last_section: bool = False
    # 做旧方式：lowpass 只压高频；thin_fifth 再把和弦垫的三音去掉（只剩根音+五度），
    # 让大小三和弦在频谱上几乎无法区分——这时候只有「同一段副歌前面唱过什么」能定案。
    degrade_mode: str = "lowpass"
    # 配器风格：决定各声部音量
    mix: dict[str, float] = field(default_factory=lambda: {
        "chords": 0.34, "bass": 0.30, "melody": 0.24, "drums": 0.16,
    })

    @property
    def beats_per_bar(self) -> int:
        return self.time_signature[0]

    @property
    def bar_seconds(self) -> float:
        return 60.0 / self.bpm * self.beats_per_bar

    def total_bars(self) -> int:
        return sum(count for _, count in self.structure) * len(self.progression)


PROFILES: dict[str, Profile] = {
    # 1) 慢速三拍子民谣：测 3/4 拍号、宽松速度、开放和弦
    "waltz": Profile(
        name="waltz",
        title="合成测试 · 慢三拍民谣",
        style="folk",
        bpm=96.0,
        time_signature=(3, 4),
        key_tonic="C",
        key_mode="major",
        progression=[("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")],
        structure=[("intro", 1), ("verse", 2), ("chorus", 2)],
        with_vocals=True,
        mix={"chords": 0.30, "bass": 0.32, "melody": 0.26, "drums": 0.12},
    ),
    # 2) 快速四拍摇滚：测 4/4、强鼓点、力量和弦、别名调（E 大调）
    "rock": Profile(
        name="rock",
        title="合成测试 · 快速摇滚",
        style="rock",
        bpm=138.0,
        time_signature=(4, 4),
        key_tonic="E",
        key_mode="major",
        progression=[("E", "maj"), ("B", "maj"), ("C#", "min"), ("A", "maj")],
        structure=[("intro", 1), ("verse", 2), ("chorus", 3)],
        with_vocals=True,
        mix={"chords": 0.36, "bass": 0.30, "melody": 0.22, "drums": 0.22},
    ),
    # 3) 无vocal 中速流行：测降级路径（没有人声时不能崩、不能编造歌词）
    "instrumental": Profile(
        name="instrumental",
        title="合成测试 · 纯伴奏流行",
        style="pop",
        bpm=104.0,
        time_signature=(4, 4),
        key_tonic="G",
        key_mode="major",
        progression=[("G", "maj"), ("D", "maj"), ("E", "min"), ("C", "maj")],
        structure=[("intro", 1), ("verse", 2), ("chorus", 2)],
        with_vocals=False,
        mix={"chords": 0.38, "bass": 0.32, "melody": 0.24, "drums": 0.16},
    ),
    # 4) 编配更满的流行：测「鼓/贝斯很吵、和弦垫被压」时的稳健性，
    #    并且把最后一遍副歌做旧（低通+衰减），专门考跨重复段的一致性证据。
    "busy": Profile(
        name="busy",
        title="合成测试 · 满编配流行（末段做旧）",
        style="pop",
        bpm=122.0,
        time_signature=(4, 4),
        key_tonic="G",
        key_mode="major",
        progression=[("G", "maj"), ("E", "min"), ("C", "maj"), ("D", "maj")],
        structure=[("intro", 1), ("verse", 2), ("chorus", 2), ("verse", 1), ("chorus", 1)],
        with_vocals=True,
        mix={"chords": 0.24, "bass": 0.34, "melody": 0.22, "drums": 0.30},
        degrade_last_section=True,
        degrade_mode="noise_floor",
    ),
}


def _chord_track(profile: Profile, total_bars: int, sr: int) -> np.ndarray:
    """和弦垫：每小节一个和弦，按拍做分解（arpeggio），利于 chroma 识别。"""
    bar_len = int(profile.bar_seconds * sr)
    buffer = np.zeros(bar_len * total_bars)
    beat_len = profile.bar_seconds / profile.beats_per_bar
    per_section = len(profile.progression)
    thin_from = total_bars - per_section if profile.degrade_last_section and profile.degrade_mode == "thin_fifth" else total_bars
    for bar in range(total_bars):
        root_name, quality = profile.progression[bar % len(profile.progression)]
        root = _midi(root_name, 4)
        tones = [root + step for step in CHORD_TONES[quality]]
        if bar >= thin_from:
            # 只剩根音+五度：三音缺失，maj/min 在 chroma 上无法区分
            tones = [offset for offset in tones if (offset - root) % 12 in (0, 7)]
        # 低八度补一个根音，让和弦更有厚度
        tones = [root - 12] + tones
        bar_start = bar * bar_len
        # 分解：每个音一个八分音符（3/4 拍时按拍走）
        step = beat_len if profile.beats_per_bar == 3 else beat_len / 2
        for i in range(int(profile.bar_seconds / step)):
            note = tones[i % len(tones)]
            freq = 440.0 * 2 ** ((note - 69) / 12)
            n = int(step * sr * 0.92)
            t = np.arange(n) / sr
            sig = _osc(freq, t, "saw") * _env(n)
            _place(buffer, bar_start + int(i * step * sr), sig * profile.mix["chords"])
    return buffer


def _bass_track(profile: Profile, total_bars: int, sr: int) -> np.ndarray:
    """贝斯：每拍根音，给和弦识别提供低音证据（斜杠和弦判定的关键）。"""
    bar_len = int(profile.bar_seconds * sr)
    buffer = np.zeros(bar_len * total_bars)
    beat_len = profile.bar_seconds / profile.beats_per_bar
    for bar in range(total_bars):
        root_name, _ = profile.progression[bar % len(profile.progression)]
        root = _midi(root_name, 2)
        freq = 440.0 * 2 ** ((root - 69) / 12)
        for beat in range(profile.beats_per_bar):
            n = int(beat_len * sr * 0.88)
            t = np.arange(n) / sr
            sig = _osc(freq, t, "sine") * _env(n, attack=0.005, release=0.08)
            _place(buffer, bar * bar_len + int(beat * beat_len * sr), sig * profile.mix["bass"])
    return buffer


def _melody_track(profile: Profile, total_bars: int, sr: int) -> np.ndarray:
    """人声旋律：只弹（合成）不用来评估歌词，但用于验证旋律识别管线。"""
    if not profile.with_vocals:
        return np.zeros(int(profile.bar_seconds * sr) * total_bars)
    bar_len = int(profile.bar_seconds * sr)
    buffer = np.zeros(bar_len * total_bars)
    beat_len = profile.bar_seconds / profile.beats_per_bar
    tonic_pc = NOTE_PC[profile.key_tonic]
    # 旋律走音阶级数，音符落在人声音域（C4-C5 之间）
    degrees = [4, 4, 2, 0, 2, 4, 4, 2]  # 简单动机，可重复
    for bar in range(total_bars):
        root_name, _ = profile.progression[bar % len(profile.progression)]
        root_pc = NOTE_PC[root_name]
        for beat in range(profile.beats_per_bar):
            deg = degrees[(bar * profile.beats_per_bar + beat) % len(degrees)]
            semis = (root_pc - tonic_pc + MAJOR_STEPS[deg % 7]) % 12
            note = _midi(profile.key_tonic, 4) + semis
            # 落在 C4..C5
            while note > _midi("C", 5):
                note -= 12
            while note < _midi("C", 4):
                note += 12
            freq = 440.0 * 2 ** ((note - 69) / 12)
            n = int(beat_len * sr * 0.9)
            t = np.arange(n) / sr
            # 加一点颤音，模拟歌唱的连续音高变化
            vib = 1.0 + 0.004 * np.sin(2 * math.pi * 5.2 * t)
            sig = _osc(freq * vib, t, "sine") * _env(n, attack=0.03, release=0.1)
            _place(buffer, bar * bar_len + int(beat * beat_len * sr), sig * profile.mix["melody"])
    return buffer


def _drum_track(profile: Profile, total_bars: int, sr: int) -> np.ndarray:
    """鼓：底鼓/军鼓/踩镲，给节拍检测提供清晰的瞬态。"""
    bar_len = int(profile.bar_seconds * sr)
    buffer = np.zeros(bar_len * total_bars)
    beat_len = profile.bar_seconds / profile.beats_per_bar
    rng = np.random.default_rng(20261021)  # 固定种子，保证可复现
    amount = profile.mix["drums"]

    def kick() -> np.ndarray:
        n = int(0.12 * sr)
        t = np.arange(n) / sr
        freq = 110.0 * np.exp(-t * 26) + 45.0
        return np.sin(2 * math.pi * freq * t) * np.exp(-t * 18)

    def snare() -> np.ndarray:
        n = int(0.10 * sr)
        t = np.arange(n) / sr
        noise = rng.normal(0, 1, n) * np.exp(-t * 32)
        tone = np.sin(2 * math.pi * 190 * t) * np.exp(-t * 26)
        return 0.7 * noise + 0.5 * tone

    def hat() -> np.ndarray:
        n = int(0.045 * sr)
        t = np.arange(n) / sr
        return rng.normal(0, 1, n) * np.exp(-t * 90) * 0.45

    for bar in range(total_bars):
        bar_start = bar * bar_len
        for beat in range(profile.beats_per_bar):
            pos = bar_start + int(beat * beat_len * sr)
            # 底鼓在强拍，军鼓在三拍（4/4 的第 3 拍），其它拍加军鼓变体
            if beat == 0:
                _place(buffer, pos, kick() * amount)
            elif profile.beats_per_bar == 4 and beat == 2:
                _place(buffer, pos, snare() * amount * 0.9)
            elif profile.beats_per_bar == 3 and beat == 1:
                _place(buffer, pos, snare() * amount * 0.55)
            # 八分踩镲
            _place(buffer, pos, hat() * amount)
            _place(buffer, pos + int(beat_len * sr / 2), hat() * amount * 0.7)
    return buffer


def synthesize(profile: Profile, sr: int = SAMPLE_RATE) -> np.ndarray:
    """按配方渲染出立体声（双声道相同）的混音。"""
    total_bars = profile.total_bars()
    length = int(profile.bar_seconds * sr) * total_bars
    mix = np.zeros(length)
    for track in (_chord_track, _bass_track, _melody_track, _drum_track):
        mix += track(profile, total_bars, sr)
    # 归一化到 -1..1，留 3dB 余量
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = mix / peak * 0.71
    if profile.degrade_last_section:
        per_section = len(profile.progression)
        bar_len = int(profile.bar_seconds * sr)
        start = max(0, (total_bars - per_section) * bar_len)
        if profile.degrade_mode == "noise_floor":
            mix = degrade_segment(mix, start, length, sr, mode="noise_floor", gain=0.22, noise=0.05)
        else:
            mix = degrade_segment(mix, start, length, sr, mode=profile.degrade_mode)
    return mix


def degrade_segment(
    mix: np.ndarray,
    start: int,
    end: int,
    sr: int = SAMPLE_RATE,
    *,
    mode: str = "lowpass",
    gain: float = 0.62,
    noise: float = 0.0,
) -> np.ndarray:
    """把一段音频做旧，模拟「同一段副歌第二遍录得更闷/更轻/更脏」。

    mode="lowpass"     一阶低通 + 衰减（闷）
    mode="noise_floor" 大幅衰减 + 加宽带噪声（远、糊），信噪比低到单看这一段很难定和弦
    mode="thin_fifth"  低通 + 只保留根音五度（由和弦轨处理），大小三和弦无法区分
    """
    out = mix.copy()
    start = max(0, int(start))
    end = min(len(out), int(end))
    if end - start < 2:
        return out
    alpha = 0.10  # 截止频率约 400Hz，高频（镲片、泛音）基本被抹掉
    segment = out[start:end]
    filtered = np.empty_like(segment)
    state = 0.0
    for index in range(segment.shape[0]):
        state += alpha * (segment[index] - state)
        filtered[index] = state
    processed = filtered * float(gain)
    if mode == "noise_floor" and noise > 0:
        # 固定种子：同一个 profile 每次渲染出同一段音频，评估才可复现
        rng = np.random.default_rng(20240607)
        processed = processed + rng.normal(0.0, float(noise), size=processed.shape)
    out[start:end] = processed
    return out


def ground_truth(profile: Profile) -> dict[str, Any]:
    """输出该合成歌曲的「正确答案」，供验收脚本比对。"""
    total_bars = profile.total_bars()
    bar_seconds = profile.bar_seconds
    sections: list[dict[str, Any]] = []
    bar_cursor = 0
    per_section_bars = len(profile.progression)
    for kind, count in profile.structure:
        for _ in range(count):
            sections.append({
                "kind": kind,
                "start_bar": bar_cursor + 1,
                "end_bar": bar_cursor + per_section_bars,
                "start_sec": round(bar_cursor * bar_seconds, 4),
                "end_sec": round((bar_cursor + per_section_bars) * bar_seconds, 4),
            })
            bar_cursor += per_section_bars
    degraded_from_bar = total_bars - len(profile.progression) + 1 if profile.degrade_last_section else None
    return {
        "schema": "aitsm.groundtruth/v1",
        "degraded_from_bar": degraded_from_bar,
        "profile": profile.name,
        "title": profile.title,
        "style": profile.style,
        "bpm": profile.bpm,
        "time_signature": list(profile.time_signature),
        "key": {"tonic": profile.key_tonic, "mode": profile.key_mode},
        "duration": round(total_bars * bar_seconds, 4),
        "total_bars": total_bars,
        "chords": [
            {
                "bar": bar + 1,
                "start_sec": round(bar * bar_seconds, 4),
                "end_sec": round((bar + 1) * bar_seconds, 4),
                "root": root,
                "quality": quality,
            }
            for bar in range(total_bars)
            for root, quality in [profile.progression[bar % len(profile.progression)]]
        ],
        "sections": sections,
        "with_vocals": profile.with_vocals,
        "limits": [
            "合成音频使用简单振荡器，无真实乐器泛音与动态处理",
            "人声为带颤音的正弦近似，不能用于评估歌词识别准确率",
        ],
    }


def write_wav(path: Path, mix: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """写 16bit 立体声 PCM WAV（不依赖 soundfile，便于离线复现）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(mix, -1.0, 1.0)
    ints = (data * 32767).astype("<i2")
    stereo = np.repeat(ints[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(stereo.tobytes())


def build(profile_name: str, out_dir: Path) -> dict[str, Any]:
    """生成一首合成歌曲：WAV + ground-truth.json，返回元数据。"""
    profile = PROFILES[profile_name]
    mix = synthesize(profile)
    wav_path = out_dir / f"{profile.name}.wav"
    write_wav(wav_path, mix)
    truth = ground_truth(profile)
    truth["audio"] = str(wav_path)
    (out_dir / f"{profile.name}.groundtruth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "profile": profile_name,
        "wav": str(wav_path),
        "groundtruth": str(out_dir / f"{profile.name}.groundtruth.json"),
        "duration": truth["duration"],
        "bpm": truth["bpm"],
        "key": truth["key"],
        "bars": truth["total_bars"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成带 ground truth 的合成测试歌曲")
    parser.add_argument("--out", default="work/synth", help="输出目录")
    parser.add_argument("--profile", action="append", default=None,
                        help="要生成的 profile（可重复），默认全部")
    parser.add_argument("--list", action="store_true", help="列出可用 profile")
    args = parser.parse_args()

    if args.list:
        for name, profile in PROFILES.items():
            print(f"{name:14s} {profile.bpm:5.1f}BPM {profile.time_signature[0]}/"
                  f"{profile.time_signature[1]} {profile.key_tonic} {profile.key_mode} "
                  f"bars={profile.total_bars()} vocals={profile.with_vocals}  {profile.title}")
        return 0

    names = args.profile or list(PROFILES)
    out_dir = Path(args.out)
    results = [build(name, out_dir) for name in names]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

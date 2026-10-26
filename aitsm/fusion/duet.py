"""DuET 适配器：把 ``fusion.quantize`` 的旋律量化请求接到真实模型。

接口契约（由 :func:`aitsm.fusion.quantize._quantize_with_duet` 决定，不是猜的）::

    quantize_melody(
        notes=[{"start","end","midi","confidence","lyric"}, ...],
        beats={"bpm","beats","downbeats","beats_per_bar","time_signature",...},
        key={"tonic","mode","confidence","fifths"},
        work_dir=str | None,
        device="auto" | "cpu" | "cuda",
        duet_dir=str,
    ) -> [{"start","end","midi","confidence","lyric"}, ...]   # 绝对秒

实现路径：

1. 把音符写成 **performance MIDI**（真实演奏时间，不量化）；
2. 调 DuET（``DuET.from_pretrained(dir, local_files_only=True).transcribe(mid, out)``）
   得到量化后的 ``score.musicxml`` + ``notes.json``；
3. 优先读 ``notes.json``（DuET 的原始预测，含 bar/beat/时长 tick），按
   tempo + tick 换算回绝对秒；读不到时解析 MusicXML 的 measure/beat 位置。

为什么走 ``notes.json`` 而不是 MusicXML：MusicXML 里小节/拍位是**记谱**信息，
乘回拍长才能得到绝对时间，精度受 divisions 限制；``notes.json`` 直接给 tick 网格，
换算一步到位。

能力边界（必须如实告知）：DuET 是**钢琴独奏** performance→score 模型，
输入是 MIDI，输出是记谱节奏——对吉他弹唱谱它只贡献「节奏量化」这一层价值，
不提供 TAB 或指法。任何失败都抛异常，由 ``quantize`` 降级到 HMM。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "VENDOR_ENV",
    "DUET_DIR_ENV",
    "vendor_src",
    "is_available",
    "probe",
    "quantize_melody",
    "describe",
]

#: DuET 源码根（内含 ``src/duet``）；默认指向仓库内 ``vendor/DuET``
VENDOR_ENV = "AITSM_DUET_VENDOR"
#: DuET 权重目录（``model.safetensors`` + ``config.json``）
DUET_DIR_ENV = "AITSM_DUET_DIR"

_DEFAULT_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "DuET"
_TICKS_PER_BEAT = 480



# --------------------------------------------------------------------------- #
# 可用性
# --------------------------------------------------------------------------- #


def vendor_src() -> Path | None:
    """DuET 源码里的 ``src`` 目录；不存在返回 ``None``。"""
    raw = str(os.environ.get(VENDOR_ENV) or "").strip()
    root = Path(raw).expanduser() if raw else _DEFAULT_VENDOR
    candidate = root / "src"
    return candidate if (candidate / "duet" / "__init__.py").is_file() else None


def _ensure_importable() -> bool:
    src = vendor_src()
    if src is None:
        return False
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)
    try:
        importlib.import_module("duet")
    except Exception:  # pragma: no cover - 依赖损坏
        return False
    return True


def _duet_dir() -> Path | None:
    raw = str(os.environ.get(DUET_DIR_ENV) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def probe(device: str | None = None) -> dict[str, Any]:
    """不发网络请求地检查 DuET 是否可跑。"""
    duet_dir = _duet_dir()
    checks: dict[str, Any] = {
        "torch": importlib.util.find_spec("torch") is not None,
        "vendor_src": str(vendor_src() or ""),
        "duet_dir": str(duet_dir or ""),
        "weights": bool(duet_dir and any(duet_dir.glob("*.safetensors"))),
        "device": str(device or "auto"),
    }
    reason = ""
    if not checks["torch"]:
        reason = "未安装 torch"
    elif not checks["vendor_src"]:
        reason = f"缺少 DuET 源码：{VENDOR_ENV} 或 vendor/DuET 不可用"
    elif duet_dir is None:
        reason = f"未配置 {DUET_DIR_ENV}（DuET 权重目录）"
    elif not checks["weights"]:
        reason = f"{duet_dir} 里没有 *.safetensors 权重"
    checks["available"] = not reason
    checks["reason"] = reason
    return checks


def is_available(device: str | None = None) -> bool:
    return bool(probe(device)["available"])


def describe() -> str:
    info = probe()
    return f"DuET 可用（{info['duet_dir']}）" if info["available"] else f"DuET 不可用：{info['reason']}"


def _resolve_device(device: str | None) -> str | None:
    requested = str(device or "auto").strip().lower()
    if requested in ("", "auto"):
        return None  # 让 DuET 自己选（有 CUDA 就用）
    return requested


# --------------------------------------------------------------------------- #
# performance MIDI 读写
# --------------------------------------------------------------------------- #


def write_performance_midi(
    notes: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    bpm: float,
) -> Path:
    """把绝对秒的音符写成 performance MIDI（保留真实演奏时间，不量化）。

    DuET 只吃 performance MIDI；这里不做任何网格吸附，量化交给模型。
    """
    import mido  # 惰性：只有真跑 DuET 才需要

    tempo = int(round(60_000_000 / max(1.0, float(bpm or 120.0))))
    mid = mido.MidiFile(ticks_per_beat=_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    seconds_per_tick = (60.0 / max(1.0, float(bpm or 120.0))) / _TICKS_PER_BEAT
    events: list[tuple[int, str, int, int]] = []
    for note in notes:
        pitch = int(note.get("midi", 60))
        if not 0 <= pitch <= 127:
            continue
        start = max(0.0, float(note.get("start", 0.0)))
        end = max(start, float(note.get("end", start)))
        on_tick = int(round(start / seconds_per_tick))
        off_tick = max(on_tick + 1, int(round(end / seconds_per_tick)))
        # 同拍多个音时用序数稳定排序，保证 on/off 不交叉
        events.append((on_tick, "on", pitch, 0))
        events.append((off_tick, "off", pitch, 1))
    # 同一 tick 上先 off 后 on，避免同音高重叠被 DuET 当成连音
    events.sort(key=lambda item: (item[0], item[3], item[2]))

    cursor = 0
    for tick, kind, pitch, _order in events:
        delta = max(0, tick - cursor)
        cursor = tick
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(target))
    return target


def _read_notes_json(score_dir: Path) -> list[dict[str, Any]] | None:
    """读 DuET 的 ``notes.json``（原始预测，含 tick 网格）。"""
    candidate = score_dir / "notes.json"
    if not candidate.is_file():
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        for key in ("notes", "predictions", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    return [item for item in data if isinstance(item, Mapping)] if isinstance(data, list) else None


def _tick_to_seconds(ticks: float, bpm: float) -> float:
    return float(ticks) / _TICKS_PER_BEAT * (60.0 / max(1.0, float(bpm or 120.0)))


def _notes_from_json(
    raw: Sequence[Mapping[str, Any]], *, bpm: float
) -> list[dict[str, Any]] | None:
    """把 DuET 的 ``notes.json`` 换算成绝对秒。

    DuET 的每个音符是 ``{"pitch": int, "bar": int, "row": [...11 个 class ID...]}``，
    ``row`` 必须用官方 :class:`duet.codec.Codec` 解码：它的
    ``attributes()`` 给出 ``onset_tick`` / ``duration_tick``（480 tick = 1 四分音符）。
    自己拆 class ID 会跟 vocabulary 版本耦合，因此一律走官方解码器。
    """
    from duet.codec import Codec  # type: ignore[import-not-found]

    rows: list[list[int]] = []
    pitches: list[int] = []
    for item in raw:
        pitch = item.get("pitch", item.get("midi"))
        row = item.get("row")
        if pitch is None or not isinstance(row, (list, tuple)) or len(row) != 11:
            return None
        rows.append([int(value) for value in row])
        pitches.append(int(pitch))
    if not rows:
        return None

    codec = Codec()
    decoded = codec.attributes(rows)
    pairs = sorted(
        zip(pitches, decoded["onset_tick"], decoded["duration_tick"]),
        key=lambda item: (int(item[1]), int(item[0])),
    )
    out: list[dict[str, Any]] = []
    for pitch, onset_tick, duration_tick in pairs:
        start = _tick_to_seconds(int(onset_tick), bpm)
        out.append(
            {
                "start": start,
                "end": start + _tick_to_seconds(max(1, int(duration_tick)), bpm),
                "midi": int(pitch),
                "confidence": 1.0,
                "lyric": "",
            }
        )
    return out or None


def quantize_melody(
    notes: Sequence[Mapping[str, Any]] | None = None,
    *,
    beats: Mapping[str, Any] | None = None,
    key: Mapping[str, Any] | None = None,
    work_dir: str | Path | None = None,
    device: str = "auto",
    duet_dir: str | None = None,
) -> list[dict[str, Any]]:
    """用 DuET 量化旋律，返回绝对秒的音符列表。

    接口与 :func:`aitsm.fusion.quantize._quantize_with_duet` 的调用一致。
    任何失败都抛 ``RuntimeError``（调用方会降级 HMM），绝不返回伪造结果。
    """
    checks = probe(device)
    if not checks["available"]:
        raise RuntimeError(checks["reason"])
    if not _ensure_importable():  # pragma: no cover
        raise RuntimeError("DuET 源码不可导入")

    beat_info = dict(beats or {})
    bpm = float(beat_info.get("bpm") or 120.0)
    payload = [dict(item) for item in (notes or []) if isinstance(item, Mapping)]
    if not payload:
        raise RuntimeError("没有旋律音可供量化")

    root = Path(work_dir) if work_dir else Path(duet_dir or "work") / "duet-run"
    root.mkdir(parents=True, exist_ok=True)
    stamp = f"duet-{int(time.time() * 1000)}"
    midi_path = write_performance_midi(payload, root / f"{stamp}.mid", bpm=bpm)

    from duet.pipeline import DuET  # type: ignore[import-not-found]

    target_dir = root / f"{stamp}-score"
    if target_dir.exists():  # DuET 要求不存在的输出目录
        for child in target_dir.iterdir():
            child.unlink()
        target_dir.rmdir()

    model = DuET.from_pretrained(
        str(duet_dir or checks["duet_dir"]),
        device=_resolve_device(device),
        local_files_only=True,
    )
    report = model.transcribe(str(midi_path), str(target_dir))

    raw_notes = _read_notes_json(target_dir)
    converted = _notes_from_json(raw_notes, bpm=bpm) if raw_notes else None
    if converted is None:
        raise RuntimeError("DuET 输出里没有可换算的 notes.json（无法回读绝对时间）")

    public_report = report if isinstance(report, Mapping) else {}
    coverage = public_report.get("input_coverage")
    if isinstance(coverage, (int, float)) and float(coverage) < 0.99:
        raise RuntimeError(f"DuET 输入覆盖不完整（input_coverage={coverage}）")

    # ---- 只借用「节奏」，音高与歌词仍用我们自己更可靠的结果 ----
    #
    # DuET 是钢琴模型：它会把输入当成多声部，同起点/相邻的音会被重排（实测把
    # 末尾的高音排到了开头）。对单声部人声旋律，音高与歌词来自 VocalParse/pyin
    # 比 DuET 更可信；DuET 真正有价值的是**量化后的节奏网格**。所以这里按起点
    # 顺序把量化后的 (start, duration) 逐个移植到原始音符上，音高/歌词/置信度
    # 全部保留原值。
    #
    # 数量不一致时不做任何猜测：直接失败，让调用方降级到 HMM。
    if len(converted) != len(payload):
        raise RuntimeError(
            f"DuET 输出音符数 {len(converted)} 与输入 {len(payload)} 不一致，"
            "无法按单声部移植节奏"
        )

    timings = sorted(
        ((float(item["start"]), max(1e-4, float(item["end"]) - float(item["start"]))) for item in converted),
        key=lambda item: item[0],
    )
    ordered = sorted(
        range(len(payload)),
        key=lambda index: (float(payload[index].get("start", 0.0)), int(payload[index].get("midi", 60))),
    )
    out: list[dict[str, Any]] = []
    for slot, source_index in enumerate(ordered):
        original = payload[source_index]
        start, duration = timings[slot]
        out.append(
            {
                "start": round(start, 6),
                "end": round(start + duration, 6),
                "midi": int(original.get("midi", 60)),
                "confidence": float(original.get("confidence", 1.0)),
                "lyric": str(original.get("lyric") or ""),
            }
        )
    out.sort(key=lambda item: item["start"])
    return out


if __name__ == "__main__":  # pragma: no cover - 手动排查用
    print(json.dumps(probe(), ensure_ascii=False, indent=2))

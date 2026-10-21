"""F1/F2：媒体输入与标准化。

职责：把任意音/视频容器（mp3/wav/flac/m4a/ogg/aac/mp4/mov/mkv/webm…）
解码成后续分析需要的三份资产：

- ``analysis.wav``：22.05kHz 单声道，节拍/调性/和弦/结构分析统一入口
- ``mix_44k.wav``：44.1kHz 双声道，Demucs 分轨的模型输入（模型要求 44.1k/2ch）
- ``playback.mp3`` / ``playback.m4a``：浏览器同步播放音源

时间轴一律为秒，从 0 开始（片段输入的时间轴相对于片段起点）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aitsm import media

ANALYSIS_WAV_NAME = "analysis.wav"
MIX_WAV_NAME = "mix_44k.wav"
PLAYBACK_STEM = "playback"


def _as_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def ingest(
    source: Path | str,
    work_dir: Path | str,
    *,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> dict[str, Any]:
    """把输入媒体标准化成分析/分轨/播放三份资产。

    参数
    ----
    source：原始音频或视频文件（只读，不会被修改）
    work_dir：产物目录，不存在会自动创建
    max_seconds：只处理 N 秒（与 start_seconds 搭配得到片段）
    start_seconds：从第几秒开始（截取片段调试用）

    返回
    ----
    {"source_wav", "playback_wav" / "playback_mp3", "mix_wav", "duration",
     "sample_rate", "channels", "sha256", "original_name", "size_bytes",
     "probe", "warnings", "tools"}
    """
    started = time.monotonic()
    source_path = _as_path(source)
    work_path = _as_path(work_dir)
    warnings: list[str] = []

    if not source_path.exists():
        raise FileNotFoundError(f"输入媒体不存在：{source_path}")
    if not source_path.is_file():
        raise media.MediaError(f"输入不是普通文件：{source_path}")
    if source_path.stat().st_size <= 0:
        raise media.MediaError(f"输入文件为空：{source_path}")
    if not media.is_supported_media(source_path):
        # 后缀不认识也继续尝试：ffprobe 能识别就按音频处理，减少无谓失败。
        warnings.append(f"未识别的扩展名 {source_path.suffix or '(无)'}，仍尝试用 ffmpeg 解码。")

    work_path.mkdir(parents=True, exist_ok=True)

    probe = media.probe_media(source_path)
    if not probe["has_audio"]:
        raise media.MediaError(f"输入不含音频流：{source_path.name}")

    source_duration = float(probe["duration"] or 0.0)
    if source_duration <= 0:
        raise media.MediaError(f"无法探测音频时长：{source_path.name}")

    start = max(0.0, float(start_seconds or 0.0))
    if start >= source_duration:
        raise media.MediaError(f"起点 {start:.2f}s 超出音频时长 {source_duration:.2f}s。")
    available = source_duration - start
    duration = available
    if max_seconds is not None and float(max_seconds) > 0:
        duration = min(available, float(max_seconds))
    trim_duration = duration if duration < available - 0.005 else None
    trim_start = start if start > 0 else None

    # 1) 分析音频：22.05kHz 单声道
    analysis_info = media.decode_to_wav(
        source_path,
        work_path / ANALYSIS_WAV_NAME,
        sample_rate=media.ANALYSIS_SAMPLE_RATE,
        channels=media.ANALYSIS_CHANNELS,
        start_seconds=trim_start,
        duration_seconds=trim_duration,
    )

    # 2) 分轨输入：44.1kHz 双声道（时长与 analysis.wav 对齐，保证 stems 时间轴一致）
    mix_info = media.decode_to_wav(
        source_path,
        work_path / MIX_WAV_NAME,
        sample_rate=media.SEPARATION_SAMPLE_RATE,
        channels=media.SEPARATION_CHANNELS,
        start_seconds=trim_start,
        duration_seconds=trim_duration,
    )

    # 3) 播放音源：mp3，失败退 m4a
    playback_info = media.encode_playback(
        source_path,
        work_path / f"{PLAYBACK_STEM}.mp3",
        start_seconds=trim_start,
        duration_seconds=trim_duration,
    )

    if probe["has_video"]:
        warnings.append("输入含视频流，已自动抽取音轨。")
    if probe["sample_rate"] and probe["sample_rate"] != media.ANALYSIS_SAMPLE_RATE:
        warnings.append(
            f"原采样率 {probe['sample_rate']}Hz，分析音频已重采样至 {media.ANALYSIS_SAMPLE_RATE}Hz。"
        )
    if trim_start is not None or (trim_duration is not None and trim_duration < available - 0.05):
        warnings.append(
            f"仅处理 {start:.2f}s~{start + duration:.2f}s 片段（原始时长 {source_duration:.2f}s）。"
        )

    result: dict[str, Any] = {
        "source_wav": analysis_info["path"],
        "playback_wav": playback_info["path"],   # 契约别名：实际是 mp3/m4a 压缩音源
        "playback_mp3": playback_info["path"] if playback_info["codec"] == "libmp3lame" else None,
        "mix_wav": mix_info["path"],
        "duration": round(duration, 6),
        "source_duration": round(source_duration, 6),
        "start_seconds": round(start, 6),
        "sample_rate": media.ANALYSIS_SAMPLE_RATE,
        "channels": media.ANALYSIS_CHANNELS,
        "sha256": media.sha256_file(source_path),
        "original_name": source_path.name,
        "size_bytes": media.file_size(source_path),
        "playback_codec": playback_info["codec"],
        "probe": {
            "container": probe["container"],
            "codec": probe["codec"],
            "sample_rate": probe["sample_rate"],
            "channels": probe["channels"],
            "bit_rate": probe["bit_rate"],
            "has_video": probe["has_video"],
        },
        "warnings": warnings,
        "tools": {
            "ffmpeg": media.ffmpeg_version(),
            "ffmpeg_exe": media.resolve_ffmpeg_exe(),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return result

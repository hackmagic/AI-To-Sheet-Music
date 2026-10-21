"""ffmpeg / ffprobe 封装。

只依赖标准库，方便在没有 numpy/librosa 的环境里也能做媒体探测与转码。
上层 `aitsm.pipeline.ingest` 负责把探测结果整理成分析用的音频资产。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

# 分析用音频统一规格：单声道 22.05kHz，够 librosa 做 chroma/beat，体积小。
ANALYSIS_SAMPLE_RATE = 22050
ANALYSIS_CHANNELS = 1
# Demucs 模型要求 44.1kHz 双声道，所以另外留一份分轨输入。
SEPARATION_SAMPLE_RATE = 44100
SEPARATION_CHANNELS = 2
# 浏览器播放音源：mp3 兼容性最好，失败再退 m4a(aac)。
PLAYBACK_BITRATE = "192k"
PLAYBACK_CODECS = ("libmp3lame", "aac")

_SUPPORTED_SUFFIXES = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".ape",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".ts", ".m4v",
}


class MediaError(RuntimeError):
    """媒体探测 / 转码失败。"""


def _works(exe: str, args: tuple[str, ...] = ("-version",)) -> bool:
    """确认可执行文件真的能跑起来（WinGet shim / 破损 PATH 都可能指向不可用目标）。"""
    try:
        completed = subprocess.run(
            [exe, *args],
            capture_output=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _candidate_paths(binary_name: str) -> list[str]:
    """在 WinGet / chocolatey / Program Files 等常见位置找 ffmpeg(.exe)。"""
    names = [binary_name]
    if os.name == "nt" and not binary_name.lower().endswith(".exe"):
        names.append(f"{binary_name}.exe")
    roots: list[Path] = []
    local_appdata = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local_appdata:
        roots.append(Path(local_appdata) / "Microsoft" / "WinGet" / "Links")
        packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            roots.extend(sorted(packages.glob("*FFmpeg*/**/bin"), reverse=True))
    programdata = (os.environ.get("ProgramData") or "").strip()
    if programdata:
        roots.append(Path(programdata) / "chocolatey" / "bin")
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = (os.environ.get(env_name) or "").strip()
        if base:
            roots.append(Path(base) / "ffmpeg" / "bin")
    found: list[str] = []
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                found.append(str(candidate))
    return found


def _resolve_binary(binary_name: str, env_names: tuple[str, ...]) -> str:
    for env_name in env_names:
        candidate = (os.environ.get(env_name) or "").strip().strip('"')
        if candidate and _works(candidate):
            return candidate
    found = shutil.which(binary_name)
    if found and _works(found):
        return found
    for candidate in _candidate_paths(binary_name):
        if _works(candidate):
            return candidate
    # 全部探测失败时返回裸名字，让 subprocess 抛 FileNotFoundError 并给出明确错误。
    return found or binary_name


@lru_cache(maxsize=1)
def resolve_ffmpeg_exe() -> str:
    """定位 ffmpeg：环境变量 > PATH > WinGet/chocolatey/Program Files 常见目录。"""
    return _resolve_binary("ffmpeg", ("AITSM_FFMPEG", "FFMPEG_EXE"))


@lru_cache(maxsize=1)
def resolve_ffprobe_exe() -> str:
    """定位 ffprobe：优先取 ffmpeg 同目录的 ffprobe，再退 PATH。"""
    for env_name in ("AITSM_FFPROBE", "FFPROBE_EXE"):
        candidate = (os.environ.get(env_name) or "").strip().strip('"')
        if candidate and _works(candidate):
            return candidate
    ffmpeg_path = Path(resolve_ffmpeg_exe())
    if ffmpeg_path.name.lower().startswith("ffmpeg"):
        for replacement in ("ffprobe", "FFPROBE"):
            sibling = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", replacement, 1))
            if sibling.is_file() and _works(str(sibling)):
                return str(sibling)
    return _resolve_binary("ffprobe", ("AITSM_FFPROBE", "FFPROBE_EXE"))


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """取 ffmpeg 版本号写进 manifest；探测失败时返回 "unknown"。"""
    try:
        result = subprocess.run(
            [resolve_ffmpeg_exe(), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
    return first_line.strip() or "unknown"


def _run(argv: list[str], *, timeout: float = 900.0, action: str = "ffmpeg") -> subprocess.CompletedProcess[str]:
    """执行命令，失败时把 stderr 尾部带进异常，方便定位真实原因。"""
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到 {argv[0]}，请确认 ffmpeg 已安装并在 PATH 中。") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{action} 超时（>{timeout:.0f}s）：{' '.join(argv[:4])} …") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-6:]
        detail = " | ".join(line.strip() for line in tail if line.strip()) or "无 stderr 输出"
        raise MediaError(f"{action} 失败（exit={completed.returncode}）：{detail}")
    return completed


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 sha256（大文件不整读进内存）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path | str) -> int:
    return int(Path(path).stat().st_size)


def is_supported_media(path: Path | str) -> bool:
    return Path(path).suffix.lower() in _SUPPORTED_SUFFIXES


def probe_media(path: Path | str) -> dict[str, Any]:
    """用 ffprobe 探测时长/采样率/声道/音视频流。

    返回 {"duration", "sample_rate", "channels", "codec", "container",
          "bit_rate", "has_audio", "has_video", "streams"}
    时长探测不到时 duration=0.0，由调用方决定是否报错。
    """
    source = Path(path)
    if not source.exists():
        raise MediaError(f"输入文件不存在：{source}")
    argv = [
        # 注意：ffprobe 9 用 -show_entries 取 disposition 会丢字段，这里直接 -show_streams。
        resolve_ffprobe_exe(),
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(source),
    ]
    completed = _run(argv, timeout=120.0, action="ffprobe")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe 输出无法解析：{exc}") from exc

    streams = payload.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    # 封面图（flac/mp3 内嵌 attached_pic）也是 video 流，但不能当成视频容器处理。
    video_streams = [
        s for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    fmt = payload.get("format") or {}

    duration = 0.0
    for raw in (fmt.get("duration"), *(s.get("duration") for s in audio_streams)):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            duration = value
            break

    primary = audio_streams[0] if audio_streams else {}
    try:
        sample_rate = int(primary.get("sample_rate") or 0)
    except (TypeError, ValueError):
        sample_rate = 0
    try:
        channels = int(primary.get("channels") or 0)
    except (TypeError, ValueError):
        channels = 0
    try:
        bit_rate = int(float(fmt.get("bit_rate") or 0))
    except (TypeError, ValueError):
        bit_rate = 0

    return {
        "path": str(source),
        "duration": round(duration, 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "codec": str(primary.get("codec_name") or ""),
        "container": str(fmt.get("format_name") or ""),
        "bit_rate": bit_rate,
        "has_audio": bool(audio_streams),
        "has_video": bool(video_streams),
        "streams": streams,
    }


def decode_to_wav(
    source: Path | str,
    out_path: Path | str,
    *,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    channels: int = ANALYSIS_CHANNELS,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    subtype: str = "s16",
    timeout: float = 900.0,
) -> dict[str, Any]:
    """把任意音/视频容器解码成指定规格的 PCM WAV（视频自动丢弃视频流）。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = [resolve_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    # -ss 放在 -i 之前是输入侧快速定位，长文件更省时间。
    if start_seconds is not None and start_seconds > 0:
        argv += ["-ss", f"{float(start_seconds):.6f}"]
    argv += ["-i", str(source)]
    if duration_seconds is not None and duration_seconds > 0:
        argv += ["-t", f"{float(duration_seconds):.6f}"]
    codec = {"s16": "pcm_s16le", "f32": "pcm_f32le"}.get(subtype, "pcm_s16le")
    argv += ["-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", str(channels), "-ar", str(sample_rate),
             "-c:a", codec, "-f", "wav", str(target)]
    _run(argv, timeout=timeout, action="ffmpeg 解码")
    return {
        "path": str(target),
        "sample_rate": sample_rate,
        "channels": channels,
        "subtype": subtype,
        "size_bytes": file_size(target),
    }


def encode_playback(
    source: Path | str,
    out_path: Path | str,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    bitrate: str = PLAYBACK_BITRATE,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """生成浏览器播放音源：优先 mp3，编码器缺失时退 m4a(aac)。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: MediaError | None = None
    for codec, suffix in (("libmp3lame", ".mp3"), ("aac", ".m4a")):
        candidate = target if target.suffix.lower() == suffix else target.with_suffix(suffix)
        argv = [resolve_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if start_seconds is not None and start_seconds > 0:
            argv += ["-ss", f"{float(start_seconds):.6f}"]
        argv += ["-i", str(source)]
        if duration_seconds is not None and duration_seconds > 0:
            argv += ["-t", f"{float(duration_seconds):.6f}"]
        argv += ["-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "2", "-ar", "44100",
                 "-c:a", codec, "-b:a", bitrate, str(candidate)]
        try:
            _run(argv, timeout=timeout, action=f"ffmpeg 编码({codec})")
        except MediaError as exc:
            last_error = exc
            continue
        return {
            "path": str(candidate),
            "codec": codec,
            "bit_rate": bitrate,
            "size_bytes": file_size(candidate),
        }
    raise last_error or MediaError("生成播放音源失败：没有可用的音频编码器。")


def slice_media(
    source: Path | str,
    out_path: Path | str,
    *,
    start_seconds: float,
    duration_seconds: float | None = None,
    sample_rate: int = SEPARATION_SAMPLE_RATE,
    channels: int = SEPARATION_CHANNELS,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """截取片段（用于 --start/--duration 调试），保留较高保真度以便后续分轨。"""
    start = max(0.0, float(start_seconds or 0.0))
    return decode_to_wav(
        source,
        out_path,
        sample_rate=sample_rate,
        channels=channels,
        start_seconds=start,
        duration_seconds=duration_seconds,
        subtype="s16",
        timeout=timeout,
    )


def ffmpeg_capabilities() -> dict[str, bool]:
    """检查关键编码器是否可用（写 manifest，避免事后猜）。"""
    caps: dict[str, bool] = {}
    try:
        result = subprocess.run(
            [resolve_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        text = result.stdout or ""
    except (OSError, subprocess.SubprocessError):
        text = ""
    for codec in ("libmp3lame", "aac", "pcm_s16le", "pcm_f32le"):
        caps[codec] = codec in text
    return caps

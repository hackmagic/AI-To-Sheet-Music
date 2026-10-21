"""ingest / media 的单测：只用标准库合成音频，不依赖 numpy。

用 ``wave`` 直接写 16bit PCM，避免在依赖没装齐时整文件被跳过。
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path

import pytest

from aitsm import media
from aitsm.pipeline import ingest as ingest_module


def write_tone_wav(
    path: Path,
    *,
    seconds: float = 3.0,
    sample_rate: int = 44100,
    channels: int = 2,
    frequency: float = 440.0,
) -> Path:
    """写一段带左右声相差的测试音（模拟真实立体声歌曲）。"""
    frames = int(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            position = index / sample_rate
            base = 0.5 * math.sin(2.0 * math.pi * frequency * position)
            for channel in range(channels):
                # 左右声道幅度不同，模拟「中置人声 + 两侧伴奏」
                amplitude = base * (1.0 if channel == 0 else 0.55)
                value = int(max(-1.0, min(1.0, amplitude)) * 20000)
                payload += struct.pack("<h", value)
        handle.writeframes(bytes(payload))
    return path


# --------------------------------------------------------------------------- #
# media 层
# --------------------------------------------------------------------------- #

def test_ffmpeg_resolvable():
    exe = media.resolve_ffmpeg_exe()
    assert exe, "ffmpeg 可执行文件应能定位"
    assert media.ffmpeg_version() != ""


def test_probe_media_reads_wav(tmp_path: Path):
    source = write_tone_wav(tmp_path / "tone.wav", seconds=2.0, sample_rate=44100, channels=2)
    probe = media.probe_media(source)
    assert probe["has_audio"] is True
    assert probe["has_video"] is False
    assert probe["sample_rate"] == 44100
    assert probe["channels"] == 2
    assert probe["duration"] == pytest.approx(2.0, abs=0.05)


def test_probe_media_missing_file(tmp_path: Path):
    with pytest.raises(media.MediaError):
        media.probe_media(tmp_path / "nope.wav")


def test_sha256_matches_hashlib(tmp_path: Path):
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.5)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert media.sha256_file(source) == expected


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

def test_ingest_produces_analysis_assets(tmp_path: Path):
    source = write_tone_wav(tmp_path / "song.wav", seconds=3.0)
    work = tmp_path / "work"
    result = ingest_module.ingest(source, work)

    analysis_wav = Path(result["source_wav"])
    assert analysis_wav.is_file()
    assert result["sample_rate"] == media.ANALYSIS_SAMPLE_RATE
    assert result["channels"] == media.ANALYSIS_CHANNELS
    assert result["duration"] == pytest.approx(3.0, abs=0.1)
    assert len(result["sha256"]) == 64
    assert result["original_name"] == "song.wav"

    # 播放音源
    assert Path(result["playback_wav"]).is_file()
    assert result["playback_codec"] in ("libmp3lame", "aac")
    # Demucs 输入：44.1k 双声道
    assert Path(result["mix_wav"]).is_file()
    mix_probe = media.probe_media(result["mix_wav"])
    assert mix_probe["sample_rate"] == media.SEPARATION_SAMPLE_RATE
    assert mix_probe["channels"] == media.SEPARATION_CHANNELS
    assert result["elapsed_seconds"] > 0


def test_ingest_max_seconds_trims_all_outputs(tmp_path: Path):
    source = write_tone_wav(tmp_path / "song.wav", seconds=4.0)
    result = ingest_module.ingest(source, tmp_path / "work", max_seconds=1.5)
    assert result["duration"] == pytest.approx(1.5, abs=0.1)
    assert result["source_duration"] == pytest.approx(4.0, abs=0.1)
    assert media.probe_media(result["source_wav"])["duration"] == pytest.approx(1.5, abs=0.1)
    assert media.probe_media(result["playback_wav"])["duration"] == pytest.approx(1.5, abs=0.2)
    assert any("片段" in message for message in result["warnings"])


def test_ingest_start_seconds_offsets(tmp_path: Path):
    source = write_tone_wav(tmp_path / "song.wav", seconds=4.0)
    result = ingest_module.ingest(source, tmp_path / "work", max_seconds=1.0, start_seconds=2.0)
    assert result["start_seconds"] == pytest.approx(2.0, abs=0.01)
    assert result["duration"] == pytest.approx(1.0, abs=0.1)
    assert media.probe_media(result["source_wav"])["duration"] == pytest.approx(1.0, abs=0.1)


def test_ingest_rejects_missing_file(tmp_path: Path):
    """输入不存在时应抛出带路径的 FileNotFoundError，而不是等 ffmpeg 子进程失败。"""
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError) as excinfo:
        ingest_module.ingest(missing, tmp_path / "work")
    assert "missing.wav" in str(excinfo.value)


def test_ingest_rejects_unsupported_suffix(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("not audio", encoding="utf-8")
    with pytest.raises(media.MediaError):
        ingest_module.ingest(source, tmp_path / "work")


def test_ingest_start_beyond_duration(tmp_path: Path):
    source = write_tone_wav(tmp_path / "song.wav", seconds=2.0)
    with pytest.raises(media.MediaError):
        ingest_module.ingest(source, tmp_path / "work", start_seconds=10.0)

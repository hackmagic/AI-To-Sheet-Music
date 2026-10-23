"""F4：人声/伴奏分离（Demucs htdemucs，失败则降级）。

契约要求 stems 转成 22050Hz 单声道 WAV 供后续分析；同时额外保留一份
44.1kHz 双声道的 HQ 人声（melody/lyrics 队友做音高跟踪时更准）。

降级策略（任一步失败都不会让流水线中断，只写 warnings）：
- Demucs 不可用（未安装/权重下载失败/显存不足且 CPU 也失败）→ 中置声道 + HPSS 粗分离；
- CUDA OOM → 先缩小 segment 再试一次 GPU，仍失败则转 CPU 重试一次；
- 模型只加载一次（模块级缓存），重复调用不重复下载。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from aitsm import media

STEM_NAMES = ("vocals", "drums", "bass", "other")
SAMPLE_RATE = 22050
HQ_SAMPLE_RATE = 44100

_SEED = 17


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


# 分轨质量旋钮（``--quality`` 预设会写这些环境变量）：
#   AITSM_DEMUCS_MODEL   htdemucs（默认）| htdemucs_ft（微调版，人声更干净，约 4 倍耗时）
#   AITSM_DEMUCS_SHIFTS  随机位移次数，越大越稳越慢
#   AITSM_DEMUCS_OVERLAP 分段重叠率，越大边界越平滑越慢
_SHIFTS_DEFAULT = 1
_OVERLAP_DEFAULT = 0.5


def demucs_quality() -> tuple[str, int, float]:
    """当前分轨质量设置：(模型名, shifts, overlap)。

    每次调用现读环境变量，这样 ``--quality`` 可以在同一进程里按任务调整，
    而不用重新 import 模块。
    """
    model = (os.environ.get("AITSM_DEMUCS_MODEL") or "htdemucs").strip() or "htdemucs"
    shifts = max(1, _env_int("AITSM_DEMUCS_SHIFTS", _SHIFTS_DEFAULT))
    overlap = min(0.95, max(0.0, _env_float("AITSM_DEMUCS_OVERLAP", _OVERLAP_DEFAULT)))
    return model, shifts, overlap
_SEGMENT = 7.0
_CPU_SEGMENT = 4.0

_DEVICE_CHOICES = ("auto", "cuda", "cpu")

_MODEL_CACHE: dict[str, Any] = {}


def _require_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("分轨后处理需要 numpy 与 librosa。") from exc
    return librosa, np


def _demucs_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("demucs", "torch", "torchaudio"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover
            versions[name] = "missing"
    return versions


def _pick_device(requested: str, torch: Any) -> str:
    requested = (requested or "auto").lower()
    if requested not in _DEVICE_CHOICES:
        requested = "auto"
    if requested == "cpu":
        return "cpu"
    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        available = False
    if requested == "cuda":
        return "cuda" if available else "cpu"
    return "cuda" if available else "cpu"


def _prepare_mix(source_wav: Path, work_dir: Path) -> tuple[Path, dict[str, Any]]:
    """找到/生成 Demucs 需要的 44.1kHz 双声道输入。"""
    info: dict[str, Any] = {"reused": False}
    candidates = [
        source_wav.parent / "mix_44k.wav",
        work_dir / "mix_44k.wav",
    ]
    for candidate in candidates:
        if candidate.is_file():
            probe = media.probe_media(candidate)
            if probe["sample_rate"] == HQ_SAMPLE_RATE and probe["channels"] >= 2:
                info.update({"path": str(candidate), "reused": True})
                return candidate, info
    target = work_dir / "mix_44k.wav"
    media.decode_to_wav(
        source_wav, target, sample_rate=HQ_SAMPLE_RATE, channels=2,
        subtype="s16",
    )
    info.update({"path": str(target), "reused": False})
    return target, info


def _load_demucs_model(name: str) -> tuple[Any, dict[str, str]]:
    """加载并缓存 Demucs 模型（权重走 dl.fbaipublicfiles.com，已在要求范围内）。"""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name], _demucs_versions()
    from demucs.pretrained import get_model  # type: ignore[import-not-found]

    model = get_model(name)
    model.eval()
    _MODEL_CACHE[name] = model
    return model, _demucs_versions()


def _run_demucs(
    mix_path: Path,
    out_hq: Path,
    out_analysis: Path,
    *,
    device: str,
    warnings: list[str],
) -> dict[str, Any]:
    """执行 Demucs 分轨并写出 HQ / 分析两种 stem。失败会抛异常由上层降级。"""
    import numpy as np
    import soundfile as sf  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from demucs.apply import apply_model  # type: ignore[import-not-found]

    librosa, _ = _require_libs()
    model_name, shifts, overlap = demucs_quality()
    model, versions = _load_demucs_model(model_name)
    chosen_device = _pick_device(device, torch)
    model.to(chosen_device)

    data, sample_rate = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if sample_rate != HQ_SAMPLE_RATE:
        raise RuntimeError(f"分轨输入采样率应为 {HQ_SAMPLE_RATE}Hz，实际 {sample_rate}Hz。")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
        warnings.append("分轨输入是单声道，已复制为双声道（中置提取无意义，音质会略降）。")
    wav = torch.from_numpy(np.ascontiguousarray(data.T)).float()
    reference = wav.mean(0, keepdim=True)
    scale = float(reference.std()) or 1.0
    normalized = (wav - reference.mean()) / scale

    attempts: list[dict[str, Any]] = [
        {"device": chosen_device, "segment": _SEGMENT},
        {"device": chosen_device, "segment": _CPU_SEGMENT},
        {"device": "cpu", "segment": _CPU_SEGMENT},
    ]
    last_error: Exception | None = None
    sources = None
    used = attempts[0]
    for attempt in attempts:
        try:
            if attempt["device"] == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            segment = attempt["segment"]
            sources = apply_model(
                model,
                normalized[None],
                device=attempt["device"],
                shifts=shifts,
                split=True,
                overlap=overlap,
                segment=segment,
                progress=False,
                num_workers=0,
            )[0]
            used = attempt
            break
        except TypeError:
            # 旧版 demucs 的 apply_model 没有 segment 参数
            try:
                sources = apply_model(
                    model, normalized[None], device=attempt["device"],
                    shifts=shifts, split=True, overlap=overlap,
                    progress=False, num_workers=0,
                )[0]
                used = attempt
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
        except Exception as exc:  # 含 CUDA OOM / 显存碎片
            last_error = exc
            warnings.append(f"Demucs 在 {attempt['device']}/segment={attempt['segment']} 上失败：{exc}")
            if attempt["device"] == "cpu" and attempt is attempts[-1]:
                break

    if sources is None:
        raise RuntimeError(f"Demucs 分轨失败：{last_error}") from last_error

    sources = sources * scale + reference.mean()
    names = list(getattr(model, "sources", STEM_NAMES))
    order = {name: index for index, name in enumerate(names)}

    out_hq.mkdir(parents=True, exist_ok=True)
    out_analysis.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for stem in STEM_NAMES:
        if stem not in order:
            continue
        tensor = sources[order[stem]].cpu().numpy()
        # HQ：44.1kHz 双声道 float32（避免峰值削顶，保留动态）
        hq_path = out_hq / f"{stem}.wav"
        sf.write(str(hq_path), tensor.T, sample_rate, subtype="FLOAT")
        # 分析：22050Hz 单声道
        mono = tensor.mean(axis=0)
        mono_sr = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
        analysis_path = out_analysis / f"{stem}.wav"
        sf.write(str(analysis_path), mono_sr.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        written[stem] = str(analysis_path)

    return {
        "engine": "demucs",
        "model": model_name,
        "device": used["device"],
        "segment": used["segment"],
        "stems": written,
        "hq_stems": {stem: str(out_hq / f"{stem}.wav") for stem in written},
        "versions": versions,
        "warnings": warnings,
    }


def _fallback_separate(
    mix_path: Path,
    out_analysis: Path,
    *,
    warnings: list[str],
) -> dict[str, Any]:
    """降级分离：中置声道（人声居中假设）+ HPSS + 低通（贝斯）。"""
    import numpy as np
    import soundfile as sf  # type: ignore[import-not-found]

    librosa, _ = _require_libs()
    data, sample_rate = sf.read(str(mix_path), dtype="float32", always_2d=True)
    if sample_rate != HQ_SAMPLE_RATE:
        data = librosa.resample(data.T, orig_sr=sample_rate, target_sr=HQ_SAMPLE_RATE).T
        sample_rate = HQ_SAMPLE_RATE
    mono44 = data.mean(axis=1)
    if data.shape[1] >= 2:
        mid = 0.5 * (data[:, 0] + data[:, 1])
        side = 0.5 * (data[:, 0] - data[:, 1])
    else:
        mid = mono44
        side = np.zeros_like(mono44)

    harmonic = librosa.effects.harmonic(mid, margin=2.0)
    percussive = librosa.effects.percussive(mid, margin=2.0)
    # 贝斯：谐波里的低频
    spectrum = librosa.stft(harmonic, n_fft=2048, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    low_mask = (freqs <= 250.0).astype(np.float32)[:, None]
    bass = librosa.istft(spectrum * low_mask, hop_length=512, length=len(harmonic))
    vocals = harmonic
    other = mono44 - vocals - percussive

    out_analysis.mkdir(parents=True, exist_ok=True)
    tracks = {"vocals": vocals, "drums": percussive, "bass": bass, "other": other}
    written: dict[str, str] = {}
    for stem, signal in tracks.items():
        signal = np.asarray(signal, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
        path = out_analysis / f"{stem}.wav"
        sf.write(str(path), signal.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        written[stem] = str(path)

    warnings.append(
        "Demucs 不可用，已降级为中置声道 + HPSS 粗分离：人声轨含伴奏残差、鼓/贝斯轨不精确，"
        "和弦与歌词结果的可信度会下降。"
    )
    return {
        "engine": "fallback",
        "model": "center-hpss",
        "device": "cpu",
        "segment": None,
        "stems": written,
        "hq_stems": {},
        "versions": _demucs_versions(),
        "warnings": warnings,
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def separate(source_wav: Path | str, work_dir: Path | str, *, device: str = "auto") -> dict[str, Any]:
    """分离出 vocals/drums/bass/other 四轨。

    返回 {"engine", "stems", "warnings", "device", "model", "elapsed_seconds", ...}
    """
    started = time.monotonic()
    source = Path(source_wav)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_analysis = work / "stems"
    out_hq = work / "stems_hq"
    manifest_path = out_analysis / "manifest.json"
    warnings: list[str] = []

    if not source.is_file():
        raise FileNotFoundError(f"分轨输入不存在：{source}")

    mix_path, mix_info = _prepare_mix(source, work)
    input_sha = media.sha256_file(mix_path)
    # 质量旋钮每次调用现读环境变量（--quality 预设通过环境变量生效），
    # 而且必须同时进缓存键：换了模型/位移次数就不能复用旧分轨。
    model_name, shifts, overlap = demucs_quality()
    cache_key = {
        "input_sha256": input_sha,
        "input_frames": media.probe_media(mix_path)["duration"],
        "device": (device or "auto").lower(),
        "model": model_name,
        "shifts": shifts,
        "overlap": overlap,
        "segment": _SEGMENT,
    }

    # 缓存复用：输入哈希 + 参数完全一致且四轨都在，才复用
    cached = _read_manifest(manifest_path)
    if cached and cached.get("cache_key") == cache_key:
        stems = cached.get("stems") or {}
        if all(Path(stems.get(stem, "")).is_file() for stem in STEM_NAMES):
            return {
                "engine": cached.get("engine", "demucs"),
                "model": cached.get("model", "htdemucs"),
                "device": cached.get("device", "cpu"),
                "stems": stems,
                "hq_stems": cached.get("hq_stems") or {},
                # 复用缓存时也要带上质量参数，否则上层不知道这份 stem 是哪个档位产的
                "shifts": (cached.get("cache_key") or {}).get("shifts"),
                "overlap": (cached.get("cache_key") or {}).get("overlap"),
                "warnings": ["复用已有分轨结果（输入哈希与参数一致）。"],
                "versions": cached.get("versions") or _demucs_versions(),
                "cached": True,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    try:
        result = _run_demucs(mix_path, out_hq, out_analysis, device=device, warnings=warnings)
    except Exception as exc:  # 任何失败都降级，不中断流水线
        warnings.append(f"Demucs 不可用（{type(exc).__name__}: {exc}），转降级分离。")
        result = _fallback_separate(mix_path, out_analysis, warnings=warnings)

    result.update({
        "cached": False,
        "mix_wav": str(mix_path),
        "mix_reused": mix_info.get("reused", False),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        # 质量旋钮随结果上报，manifest 里才能解释「为什么这份比那份慢/准」
        "model": result.get("model") or model_name,
        "shifts": shifts,
        "overlap": overlap,
    })
    try:
        manifest_path.write_text(json.dumps({
            "cache_key": cache_key,
            "engine": result["engine"],
            "model": result.get("model") or model_name,
            "shifts": shifts,
            "overlap": overlap,
            "device": result.get("device"),
            "stems": result["stems"],
            "hq_stems": result.get("hq_stems") or {},
            "versions": result.get("versions") or {},
            "warnings": result.get("warnings") or [],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        result.setdefault("warnings", []).append(f"分轨 manifest 写入失败：{exc}")
    return result

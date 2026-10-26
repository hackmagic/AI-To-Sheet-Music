"""模块 0：人声/伴奏分离层。

架构首选 **htdemucs_ft**（MUSDB18-HQ vocals SDR 9.19 dB）。本层只做两件事：

1. 决定用哪个后端：ONNX（``demucs_onnx`` + 权重目录，部署时避免拉满 PyTorch）
   > torch Demucs（本机已装 demucs 4.1.0）> 中置声道 + HPSS 降级（由
   :mod:`aitsm.pipeline.separate` 提供，绝不静默）。
2. 把结果收敛成 :class:`aitsm.contracts.SeparationResult`，把「实际用了什么」
   写进 provenance。

**不做**：重采样、缓存、写 manifest —— 那是 :mod:`aitsm.pipeline.separate` 的
职责，本层只做适配，不复制算法。

环境变量：

- ``AITSM_DEMUCS_MODEL``：默认 ``htdemucs_ft``；``htdemucs`` 更快但人声更脏。
- ``AITSM_DEMUCS_ONNX_DIR``：ONNX 权重目录（内含 ``htdemucs_ft.onnx`` 等）。
- ``AITSM_DEMUCS_SHIFTS`` / ``AITSM_DEMUCS_OVERLAP``：质量旋钮，透传给 demucs。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

from aitsm.contracts import SeparationResult

__all__ = [
    "REQUESTED_MODEL",
    "DEFAULT_MODEL",
    "available_backends",
    "onnx_available",
    "separate",
]

#: 架构首选模型（微调版 htdemucs，人声最干净，约 4 倍耗时）
REQUESTED_MODEL = "htdemucs_ft"
#: 本机默认模型：与既有 ``--quality`` 预设保持一致，可用环境变量覆盖
DEFAULT_MODEL = "htdemucs"

#: demucs 官方四轨名（顺序即模型输出顺序）
STEM_NAMES = ("vocals", "drums", "bass", "other")


def _onnx_dir() -> Path | None:
    raw = (os.environ.get("AITSM_DEMUCS_ONNX_DIR") or "").strip()
    return Path(raw) if raw else None


def onnx_available() -> bool:
    """ONNX 后端是否真的可用。

    满足任一条件即可：

    * 官方 ``bag_infer.py`` 在权重目录里（``onnxruntime`` + numpy + soundfile
      就够了，不需要任何 Demucs/torch 包）；
    * 有 ``demucs_onnx`` 包且目录里至少一个 ``*.onnx``。
    """
    directory = _onnx_dir()
    if directory is None or not directory.is_dir():
        return False
    if not any(directory.glob("*.onnx")):
        return False
    # 路径 1：仓库自带的官方推理脚本
    if (directory / "bag_infer.py").is_file():
        try:
            import importlib.util

            for module_name in ("onnxruntime", "soundfile", "numpy"):
                if importlib.util.find_spec(module_name) is None:
                    return False
            return True
        except (ImportError, ValueError):  # pragma: no cover
            pass
    # 路径 2：demucs_onnx 包
    try:
        import importlib.util

        return importlib.util.find_spec("demucs_onnx") is not None
    except (ImportError, ValueError):  # pragma: no cover - 依赖损坏
        return False


def available_backends() -> dict[str, bool]:
    """各后端可用性快照，供 doctor / manifest 打印。"""
    demucs_ok = False
    try:
        import importlib.util

        demucs_ok = importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):  # pragma: no cover
        demucs_ok = False
    return {"onnx": onnx_available(), "demucs": demucs_ok}


def _bag_infer_module(directory: Path) -> Any | None:
    """导入 htdemucs_ft ONNX 仓库自带的 ``bag_infer``（官方脚本，无需 torch）。

    ``StemSplitio/htdemucs-ft-onnx`` 的用法不是 ``demucs_onnx`` 包，而是仓库里的
    ``bag_infer.py``：它只依赖 numpy + onnxruntime + soundfile，跑四轨 specialist
    并按 htdemucs_ft 的 one-hot bag 权重聚合。
    """
    script = directory / "bag_infer.py"
    if not script.is_file():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("aitsm_htdemucs_ft_bag", script)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # pragma: no cover - 依赖缺失/脚本损坏
        return None


def _prepare_onnx_input(
    source_wav: Path, work_dir: Path, warnings: list[str] | None = None
) -> tuple[Path, str] | None:
    """ONNX 模型要求 44.1kHz；输入是 22.05kHz 分析音轨时先升采样。

    升采样结果按 ``work_dir/onnx_input_44k.wav`` 复用，避免每次重算。
    失败原因写进 ``warnings``（不静默），返回 ``None`` 让调用方降级。
    """
    sink = warnings if warnings is not None else []
    try:
        import soundfile as sf

        info = sf.info(str(source_wav))
        if int(info.samplerate) == 44100:
            return source_wav, ""
    except Exception as exc:
        sink.append(f"ONNX 输入探测失败（{type(exc).__name__}: {exc}），尝试直接转换。")

    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "onnx_input_44k.wav"
    if target.is_file():
        return target, ""
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        data, sample_rate = sf.read(str(source_wav), dtype="float32", always_2d=True)
        channels = data.T
        if int(sample_rate) != 44100:
            channels = librosa.resample(
                channels, orig_sr=int(sample_rate), target_sr=44100, res_type="soxr_hq"
            )
        channels = np.asarray(channels, dtype=np.float32)
        # bag_infer 明确要求双声道；单声道复制成两轨
        if channels.shape[0] == 1:
            channels = np.repeat(channels, 2, axis=0)
        sf.write(str(target), channels.T, 44100, subtype="FLOAT")
        return target, f"ONNX 输入已转换 {sample_rate} Hz → 44100 Hz（双声道）。"
    except Exception as exc:
        sink.append(f"ONNX 输入转换失败（{type(exc).__name__}: {exc}）。")
        return None


def _separate_onnx(
    source_wav: Path,
    work_dir: Path,
    *,
    model_name: str,
    device: str,
) -> SeparationResult | None:
    """ONNX 推理适配器（两条路径，都只读本地）。

    1. **官方 bag_infer**（首选）：``StemSplitio/htdemucs-ft-onnx`` 自带
       ``bag_infer.py``，直接调 ``separate_all(wav)`` 拿四个 numpy stem；
    2. ``demucs_onnx`` 包（兜底）：按 ``load_model(dir)`` + ``separate(...)`` 的最小约定。

    任何不符合约定的情况都返回 ``None``，由调用方降级到 torch Demucs——
    不猜、不对着未知 API 硬写。
    """
    directory = _onnx_dir()
    if directory is None or not onnx_available():
        return None

    started = time.monotonic()
    out_dir = work_dir / "stems_onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 路径 1：官方 bag_infer ----
    bag = _bag_infer_module(directory)
    if bag is not None and callable(getattr(bag, "separate_all", None)):
        onnx_warnings: list[str] = []
        prepared = _prepare_onnx_input(source_wav, work_dir, onnx_warnings)
        onnx_input = prepared[0] if prepared else None
        if onnx_input is not None:
            try:
                import numpy as np
                import soundfile as sf

                stems_raw = bag.separate_all(str(onnx_input))
                written: dict[str, str | None] = {}
                for name, array in dict(stems_raw).items():
                    data = np.asarray(array, dtype=np.float32)
                    # bag_infer 返回 (channels, samples)；soundfile 要 (samples, channels)
                    samples = data.T if data.ndim == 2 else data.reshape(-1, 1)
                    path = out_dir / f"{name}.wav"
                    sf.write(str(path), samples, 44100, subtype="FLOAT")
                    written[str(name)] = str(path)
                if "vocals" in written:
                    notes = list(onnx_warnings)
                    if prepared and prepared[1]:
                        notes.append(prepared[1])
                    return SeparationResult(
                        engine="demucs-onnx",
                        model="htdemucs_ft",
                        device="cpu",
                        stems=written,
                        cached=False,
                        elapsed_seconds=time.monotonic() - started,
                        warnings=notes,
                        detail={"backend": "bag_infer", "sample_rate": 44100},
                    )
            except Exception as exc:  # pragma: no cover - 推理失败降级
                onnx_warnings.append(f"bag_infer 推理失败（{type(exc).__name__}: {exc}）。")
                warnings = onnx_warnings

    # ---- 路径 2：demucs_onnx 包 ----
    try:
        import demucs_onnx  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - 依赖存在但导入失败
        return None
    load_model = getattr(demucs_onnx, "load_model", None)
    apply_model = getattr(demucs_onnx, "separate", None)
    if not callable(load_model) or not callable(apply_model):
        return None
    try:
        model = load_model(str(directory))
        raw = apply_model(model, str(source_wav), str(out_dir), sample_rate=44100)
    except Exception:  # pragma: no cover - 推理失败一律降级
        return None
    if not isinstance(raw, dict):
        return None
    stems = {str(key): str(value) for key, value in raw.items() if value}
    if "vocals" not in stems:
        return None
    return SeparationResult(
        engine="demucs-onnx",
        model=model_name,
        device=device,
        stems=stems,
        cached=False,
        elapsed_seconds=time.monotonic() - started,
        warnings=["ONNX 后端：未做 HQ 44.1kHz 双声道 stem 落盘，音量归一化由调用方复核。"],
        detail={"backend": "demucs_onnx"},
    )


def _configured_model(explicit: str | None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    raw = (os.environ.get("AITSM_DEMUCS_MODEL") or "").strip()
    return raw or DEFAULT_MODEL


def _engine_name(raw_engine: str | None) -> str:
    engine = str(raw_engine or "").strip().lower()
    if engine in ("demucs", "htdemucs", "demucs-onnx"):
        return engine
    if engine in ("fallback", "center-hpss", "none", ""):
        return "demucs-fallback"
    return engine or "unknown"


def _normalize_stems(stems: Iterable[Any] | dict[str, Any]) -> dict[str, str | None]:
    data = dict(stems or {}) if isinstance(stems, dict) else {}
    normalized: dict[str, str | None] = {}
    for name in STEM_NAMES:
        value = data.get(name)
        normalized[name] = str(value) if value else None
    # 保留后端可能多给的轨道（如 6-stem 模型），但不要让它们覆盖四轨
    for name, value in data.items():
        if name not in normalized:
            normalized[str(name)] = str(value) if value else None
    return normalized


def separate(
    source_wav: str | Path,
    work_dir: str | Path,
    *,
    device: str = "auto",
    model: str | None = None,
    prefer_onnx: bool = True,
) -> SeparationResult:
    """分离出 vocals / drums / bass / other 四轨。

    参数
    ----
    source_wav:
        分析音轨（通常是 ``analysis.wav``，44.1kHz 混音由内部再生成）。
    work_dir:
        产物目录，stem 落在 ``work/stems/`` 与 ``work/stems_hq/``。
    device:
        ``auto`` / ``cuda`` / ``cpu``，透传给 Demucs。
    model:
        覆盖模型名；默认读 ``AITSM_DEMUCS_MODEL``，再退回 :data:`DEFAULT_MODEL`。
    prefer_onnx:
        有可用 ONNX 权重时才走 ONNX，否则直接 torch Demucs。

    返回
    ----
    :class:`aitsm.contracts.SeparationResult`
        ``engine`` 一定是实际生效的实现（``demucs-onnx`` / ``demucs`` /
        ``demucs-fallback``），``requested`` 固定为 :data:`REQUESTED_MODEL`。
    """
    started = time.monotonic()
    source = Path(source_wav)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"分离输入不存在：{source}")

    model_name = _configured_model(model)

    if prefer_onnx:
        result = _separate_onnx(source, work, model_name=model_name, device=device)
        if result is not None:
            return result

    # torch Demucs 走既有实现：它已经处理了缓存、降级、HQ/分析双份 stem。
    from aitsm.pipeline import separate as torch_separate

    previous = os.environ.get("AITSM_DEMUCS_MODEL")
    os.environ["AITSM_DEMUCS_MODEL"] = model_name
    try:
        raw = torch_separate.separate(source, work, device=device)
    finally:
        if previous is None:
            os.environ.pop("AITSM_DEMUCS_MODEL", None)
        else:
            os.environ["AITSM_DEMUCS_MODEL"] = previous

    warnings = [str(item) for item in (raw.get("warnings") or []) if str(item).strip()]
    engine = _engine_name(raw.get("engine"))
    if engine == "demucs" and str(raw.get("model") or "") != REQUESTED_MODEL:
        warnings.append(
            f"架构首选 {REQUESTED_MODEL}，本机实际使用 {raw.get('model') or model_name}；"
            "人声残留会略多，旋律与歌词的置信度已相应下调。"
        )
    detail = {
        key: raw.get(key)
        for key in ("hq_stems", "stems_hq", "shifts", "overlap", "model", "quality")
        if raw.get(key)
    }
    return SeparationResult(
        engine=engine,
        model=str(raw.get("model") or model_name),
        device=str(raw.get("device") or device),
        stems=_normalize_stems(raw.get("stems") or {}),
        warnings=warnings,
        cached=bool(raw.get("cached", False)),
        elapsed_seconds=time.monotonic() - started,
        detail=detail,
    )

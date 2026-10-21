#!/usr/bin/env python
"""DSH Skill 环境自检：确认能否产出高质量结果。

输出 JSON（``--json``）供 agent 直接读：
- ``ready``：能否运行；
- ``capabilities``：分轨 / 歌词 / 旋律三项能力的可用性；
- ``degraded``：会降低质量的缺失组件；
- ``paths``：项目根、解释器、默认输出目录（含可写性）；
- ``advice``：一句话结论 + 下一步。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _locate  # noqa: E402

REQUIRED = {
    "numpy": "数值计算",
    "scipy": "信号处理",
    "librosa": "节拍/音高/和弦分析",
    "soundfile": "音频读写",
    "jsonschema": "产物校验",
}
OPTIONAL = {
    "torch": "人声/伴奏分轨（缺失则旋律与和弦质量显著下降）",
    "demucs": "分轨模型（缺失则无法分轨）",
    "faster_whisper": "歌词识别（缺失则无歌词）",
}
VIEWER = {
    "fastapi": "Web 查看器",
    "uvicorn": "Web 查看器",
}
MIN_PYTHON = (3, 11)


def probe(module: str) -> bool:
    """当前解释器能否 import 该模块（仅在拿不到目标解释器时使用）。"""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def probe_version(module: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(module.replace("_", "-"))
    except Exception:  # noqa: BLE001 - 版本号只是锦上添花
        return None


PROBE_SCRIPT = """
import importlib.util, json, sys
names = sys.argv[1:]
out = {}
for name in names:
    try:
        out[name] = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        out[name] = False
print(json.dumps(out))
"""


def probe_in(python: str, root: Path, modules: list[str]) -> dict[str, bool] | None:
    """在**目标解释器**里探测模块 —— 自检必须描述实际执行分析的那个环境。

    当前解释器可能是系统的 python（能写文件但没有 torch），用它探测会给出误导结论。
    """
    if not python or not os.path.isfile(python):
        return None
    try:
        proc = subprocess.run(
            [python, "-c", PROBE_SCRIPT, *modules],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed((proc.stdout or "").strip().splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            return {str(k): bool(v) for k, v in data.items()}
    return None


def model_cache(root: Path) -> dict:
    """模型权重位置（只检查是否存在，不触发下载）。"""
    home = root / "work" / "cache"
    hub = home / "hf" / "hub"
    whisper = sorted(p.name for p in hub.glob("models--Systran--faster-whisper-*")) if hub.is_dir() else []
    demucs_dirs = home / "torch" / "hub" / "checkpoints"
    demucs = sorted(p.name for p in demucs_dirs.glob("*.th")) if demucs_dirs.is_dir() else []
    hf_demucs = sorted(p.name for p in hub.glob("models--*--HTDemucs*")) if hub.is_dir() else []
    return {
        "cache_dir": str(home),
        "cache_exists": home.is_dir(),
        "whisper": whisper,
        "demucs": demucs or hf_demucs,
        "ready": bool(whisper) and bool(demucs or hf_demucs),
    }


def main() -> int:
    _locate.force_utf8()

    parser = argparse.ArgumentParser(description="环境自检")
    parser.add_argument("--json", action="store_true", help="只输出 JSON（agent 用）")
    parser.add_argument("--verbose", action="store_true", help="文本模式下列出全部细节")
    args = parser.parse_args()

    result: dict = {"ok": True}

    # ---- 路径：项目根 / 解释器 / 输出目录
    locate_error = None
    located: dict = {}
    try:
        located = _locate.resolve(__file__)
    except _locate.LocateError as exc:
        locate_error = {"error": str(exc), "hint": exc.hint, "tried": exc.tried}

    paths = {
        "skill_dir": str(_locate.script_root(__file__)),
        "aitsm_root": located.get("aitsm_root"),
        "aitsm_root_source": located.get("aitsm_root_source"),
        "python": located.get("python"),
        "python_source": located.get("python_source"),
        "default_out_dir": located.get("default_out_dir"),
        "out_candidates": located.get("out_candidates"),
    }
    if locate_error:
        result["ok"] = False
        result["locate_error"] = locate_error

    root = Path(located["aitsm_root"]) if located.get("aitsm_root") else None

    # ---- 依赖探测：优先在 aitsm 实际使用的解释器里探测
    all_modules = list(REQUIRED) + list(OPTIONAL) + list(VIEWER)
    target_probe = None
    probe_source = "当前解释器"
    if located.get("python") and root:
        target_probe = probe_in(located["python"], root, all_modules)
        if target_probe is not None:
            probe_source = f"目标解释器（{located.get('python_source')}）"

    def available(module: str) -> bool:
        if target_probe is not None:
            return bool(target_probe.get(module))
        return probe(module)

    required = {name: {"ok": available(name), "why": why} for name, why in REQUIRED.items()}
    optional = {name: {"ok": available(name), "why": why} for name, why in OPTIONAL.items()}
    viewer = {name: {"ok": available(name), "why": why} for name, why in VIEWER.items()}

    missing_required = [name for name, info in required.items() if not info["ok"]]
    degraded = [name for name, info in optional.items() if not info["ok"]]

    ffmpeg = shutil.which("ffmpeg") or ""
    ffprobe = shutil.which("ffprobe") or ""
    if not ffmpeg:
        missing_required.append("ffmpeg")
    if not ffprobe:
        missing_required.append("ffprobe")

    python_ok = sys.version_info[:2] >= MIN_PYTHON
    if not python_ok:
        missing_required.append("python>=3.11")

    # ---- aitsm 所用的解释器是否真的能 import（比 find_spec 更硬的证据）
    target_python_ok = None
    if located.get("python") and root:
        target_python_ok = _locate.probe_python(located["python"], root)

    models = model_cache(root) if root else {"cache_dir": None, "ready": False,
                                             "whisper": [], "demucs": [], "cache_exists": False}

    can_separate = available("torch") and available("demucs")
    can_transcribe = available("faster_whisper")

    ready = not missing_required and locate_error is None and target_python_ok is not False

    # ---- 结论
    capabilities = {
        "separation": {
            "ok": can_separate,
            "impact": "缺失则旋律不可信（会锁到贝斯）、和弦质量下降",
        },
        "lyrics": {
            "ok": can_transcribe,
            "impact": "缺失则没有歌词，只有旋律与和弦",
        },
        "melody": {
            "ok": can_separate,
            "impact": "依赖人声分轨",
        },
        "viewer": {
            "ok": all(info["ok"] for info in viewer.values()),
            "impact": "缺失只影响 serve.py 网页查看，不影响出谱",
        },
    }

    if locate_error:
        advice = "项目本体未找到 —— 按 locate_error.hint 修复后重跑。"
    elif not ready:
        advice = "缺少必需组件，无法运行：" + ", ".join(missing_required)
    elif degraded:
        advice = ("可以运行，但质量会下降（缺 " + ", ".join(degraded)
                  + "）。交付时**必须**告知用户具体影响。")
    elif models["cache_exists"] and not models["whisper"]:
        advice = "环境完整，但 whisper 权重未缓存 —— 首次识别歌词会联网下载约 460MB。"
    else:
        advice = "环境完整，可产出完整弹唱谱。"

    result.update({
        "ready": ready,
        "capabilities": capabilities,
        "can_separate": can_separate,
        "can_transcribe_lyrics": can_transcribe,
        "missing_required": sorted(set(missing_required)),
        "degraded": degraded,
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ok": python_ok,
            "minimum": "3.11",
            "executable": sys.executable,
            "target_usable": target_python_ok,
        },
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "models": models,
        "paths": paths,
        "details": {
            "required": required,
            "optional": optional,
            "viewer": viewer,
            "probe_source": probe_source,
            "versions": {
                name: probe_version(name)
                for name in ("numpy", "scipy", "librosa", "soundfile",
                             "torch", "demucs", "faster-whisper", "jsonschema")
            },
        },
        "advice": advice,
    })

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ready else 1

    # ---- 人类可读
    print(f"就绪：{'是' if ready else '否'}")
    print(f"项目根：{paths['aitsm_root'] or '未找到'}"
          + (f"（{paths['aitsm_root_source']}）" if paths.get("aitsm_root_source") else ""))
    print(f"解释器：{paths['python'] or '未找到'}")
    print(f"默认输出目录：{paths['default_out_dir'] or '无可写目录'}")
    print(f"ffmpeg：{ffmpeg or '未找到'}")
    print("\n必需：")
    for name, info in required.items():
        print(f"  {name:16s} {'OK' if info['ok'] else '缺失':4s}  {info['why']}")
    print("\n可选（影响质量）：")
    for name, info in optional.items():
        print(f"  {name:16s} {'OK' if info['ok'] else '缺失':4s}  {info['why']}")
    print("\n能力：")
    for name, info in capabilities.items():
        print(f"  {name:12s} {'可用' if info['ok'] else '不可用':4s}  {info['impact']}")
    print(f"\n模型缓存：{models['cache_dir'] or '—'}")
    if models["whisper"]:
        print(f"  whisper: {', '.join(models['whisper'])}")
    if models["demucs"]:
        print(f"  demucs:  {', '.join(models['demucs'])}")
    if locate_error:
        print(f"\n路径错误：{locate_error['error']}")
        print(f"修复：\n{locate_error['hint']}")
    print(f"\n建议：{advice}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

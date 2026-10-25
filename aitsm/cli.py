"""命令行入口。

    python -m aitsm.cli serve                    启动 Web 服务
    python -m aitsm.cli generate <audio> --out d  直接生成弹唱谱
    python -m aitsm.cli bench <dir> --out d       多曲目转录基准（对比每层引擎）
    python -m aitsm.cli demo --out d              用内置示例跑一遍前端数据
    python -m aitsm.cli voicings                  生成 web/voicings.json
    python -m aitsm.cli doctor                    环境自检

``generate`` 是 DSH 插件将来调用的主要入口，因此它只依赖文件系统与 stdout，
不依赖任何网络与前端状态。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .config import REPO_ROOT, SETTINGS, WEB_DIR, check_optional_engines, resolve_binary


#: ``--quality`` 预设：只拧「更慢但更准」的旋钮，不改算法结构。
#: standard = 默认（htdemucs + medium 歌词）；high = 微调分轨 + 更稳的分段；
#: max = 微调分轨 + 多次位移 + large-v3 歌词（CPU 上非常慢）。
QUALITY_PRESETS: dict[str, dict[str, str]] = {
    "standard": {},
    "high": {
        "AITSM_DEMUCS_MODEL": "htdemucs_ft",
        "AITSM_DEMUCS_SHIFTS": "2",
        "AITSM_DEMUCS_OVERLAP": "0.75",
    },
    "max": {
        "AITSM_DEMUCS_MODEL": "htdemucs_ft",
        "AITSM_DEMUCS_SHIFTS": "4",
        "AITSM_DEMUCS_OVERLAP": "0.9",
        "AITSM_WHISPER_MODEL": "large-v3",
    },
}


def apply_quality_preset(name: str) -> dict[str, str]:
    """把 ``--quality`` 预设写进环境变量，返回实际生效的项。

    已经显式设置过的环境变量不覆盖——这样用户可以用
    ``AITSM_WHISPER_MODEL=medium`` 手动调和预设里更激进的值。
    """
    applied: dict[str, str] = {}
    for key, value in QUALITY_PRESETS.get(name, {}).items():
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host or SETTINGS.host
    port = args.port or SETTINGS.port
    print(f"AI-To-Sheet-Music {__version__}  http://{host}:{port}")
    uvicorn.run("aitsm.api:app", host=host, port=port, reload=args.reload, log_level="info")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .service import run_job

    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        print(f"找不到音频文件：{source}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve()
    options = {
        "key": args.key,
        "capo": args.capo,
        "difficulty": args.difficulty,
        "style": args.style,
        "transpose": args.transpose,
        "include_tab": not args.no_tab,
        "start_sec": args.start,
        "duration_sec": args.duration,
        "device": args.device,
        "language": args.language,
        "whisper_model": args.whisper_model,
        "quality": getattr(args, "quality", None) or "standard",
        "no_separation": args.no_separation,
        "no_lyrics": args.no_lyrics,
        "no_melody": args.no_melody,
    }
    options = {k: v for k, v in options.items() if v is not None}

    # --quality 预设通过环境变量生效（分轨在调用时读取，歌词模型在下游读取）；
    # 显式 --whisper-model 优先级最高。
    quality = str(options.get("quality") or "standard")
    applied = apply_quality_preset(quality)
    if args.whisper_model:
        os.environ["AITSM_WHISPER_MODEL"] = args.whisper_model
        applied["AITSM_WHISPER_MODEL"] = args.whisper_model
    if quality != "standard":
        detail = ", ".join(f"{k.replace('AITSM_', '')}={v}" for k, v in sorted(applied.items()))
        print(f"质量档：{quality}（{detail or '无额外设置'}）")

    def progress(stage: str, percent: float, message: str) -> None:
        bar = "#" * int(percent / 4)
        print(f"\r[{stage:<12}] {bar:<25} {percent:5.1f}% {message[:40]:<40}", end="", flush=True)

    started = time.time()
    result = run_job(source_path=source, out_dir=out_dir, options=options, progress=progress)
    print()
    print(f"完成，用时 {time.time() - started:.1f}s")
    print(f"  谱面: {result['score_path']}")
    print(f"  分析: {result['analysis_path']}")
    for fmt, path in result["exports"].items():
        print(f"  导出: {fmt:<9} {path}")
    warnings = result.get("warnings") or []
    if warnings:
        print(f"  告警 {len(warnings)} 条：")
        for warning in warnings[:10]:
            print(f"    - [{warning.get('level')}] {warning.get('code')}: {warning.get('message')}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """多曲目基准：同一套参数跑一批音频，汇总每层引擎与质量分。"""
    from .benchmark import expand_sources, run_benchmark

    sources = [Path(item).expanduser() for item in args.sources]
    out_dir = Path(args.out).expanduser().resolve()
    paths = expand_sources(sources)
    if not paths:
        print(f"没有找到音频文件：{', '.join(str(item) for item in sources)}", file=sys.stderr)
        return 2

    options = {
        "start_sec": args.start,
        "duration_sec": args.duration,
        "device": args.device,
        "language": args.language,
        "whisper_model": args.whisper_model,
        "no_lyrics": args.no_lyrics or None,
    }
    options = {key: value for key, value in options.items() if value is not None}
    if args.whisper_model:
        os.environ["AITSM_WHISPER_MODEL"] = args.whisper_model

    print(f"基准：{len(paths)} 首 -> {out_dir}")

    def progress(stage: str, percent: float, message: str) -> None:
        bar = "#" * int(percent / 4)
        print(f"\r[{stage:<6}] {bar:<25} {percent:5.1f}% {message[:44]:<44}", end="", flush=True)

    report = run_benchmark(sources, out_dir=out_dir, options=options, progress=progress)
    print()
    summary = report.get("summary") or {}
    mean = summary.get("mean_overall")
    mean_text = f"{float(mean) * 100:.1f}" if isinstance(mean, (int, float)) else "—"
    print(
        f"完成：成功 {summary.get('succeeded', 0)}/{summary.get('count', 0)}"
        f"，失败 {summary.get('failed', 0)}，平均总分 {mean_text}"
    )
    degraded = summary.get("degraded_counts") or {}
    if degraded:
        print("降级层：" + "，".join(f"{key}×{value}" for key, value in sorted(degraded.items())))
    outputs = report.get("outputs") or {}
    for name, path in outputs.items():
        print(f"  {name:<9} {path}")
    for item in report.get("items") or []:
        if not item.get("ok"):
            print(f"  失败：{item.get('name')} — {item.get('error')}")
    return 0 if not summary.get("failed") else 1


def cmd_demo(args: argparse.Namespace) -> int:
    """把示例谱面复制到指定目录，便于前端/打印自测。"""
    import shutil

    source = REPO_ROOT / "samples" / "demo-score.json"
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "score.json"
    shutil.copyfile(source, target)
    print(f"示例谱面已写入 {target}")
    return 0


def cmd_voicings(args: argparse.Namespace) -> int:
    from .score.voicings import build_voicing_db

    db = build_voicing_db()
    path = WEB_DIR / "voicings.json"
    total = sum(len(items) for items in db.get("standard", {}).values())
    print(f"指法库已生成：{path}（{len(db.get('standard', {}))} 个和弦，{total} 个把位）")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    engines = check_optional_engines()
    print(f"AI-To-Sheet-Music {__version__}")
    print(f"Python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"工作目录    {SETTINGS.work_root}")
    print(f"ffmpeg      {resolve_binary('ffmpeg', SETTINGS.ffmpeg) or '未找到（将无法解析音频）'}")
    print(f"ffprobe     {resolve_binary('ffprobe', SETTINGS.ffprobe) or '未找到'}")
    for name, ok in engines.items():
        print(f"{name:<12}{'可用' if ok else '缺失（相关阶段将降级）'}")
    schemas = [p.name for p in (REPO_ROOT / "schemas").glob("*.json")]
    print(f"schemas     {', '.join(sorted(schemas)) or '缺失'}")
    _print_v2_backends()
    _print_deployment()
    return 0


def _print_v2_backends() -> None:
    """v2 四层专用模型的就绪情况（只看本地，不联网）。"""
    print("— v2 专用模型 —")
    try:
        from .transcribe import qwen_backend

        info = qwen_backend.probe()
        detail = info["reason"] or f"qwen-asr {info['version']}（pin {info['pinned_version']}）"
        if info["available"] and not info["version_matches_pin"]:
            detail += "｜版本与实测 pin 不同，注意 API 可能变化"
        mark = "可用" if info["available"] else "降级"
        print(f"{'人声 VocalParse':<16}{mark}｜{detail}")
    except Exception as exc:  # noqa: BLE001
        print(f"{'人声 VocalParse':<16}降级｜{exc}")

    for label, module_name in (
        ("伴奏 SheetSage2", "aitsm.transcribe.accompaniment"),
        ("量化 DuET", "aitsm.fusion.quantize"),
        ("分离 htdemucs_ft", "aitsm.separation.engine"),
    ):
        try:
            module = __import__(module_name, fromlist=["is_available", "onnx_available"])
            if module_name.endswith("separation.engine"):
                ok = bool(module.onnx_available())
                detail = "ONNX 权重已就绪" if ok else "未配置 AITSM_DEMUCS_ONNX_DIR 或无权重"
            else:
                ok = bool(module.is_available())
                detail = "" if ok else "权重目录/依赖缺失"
            mark = "可用" if ok else "降级"
            print(f"{label:<16}{mark}" + (f"｜{detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<16}降级｜{exc}")
    print("→ 四层全可用时 degraded_layers 为空；任一降级都会写进 symbolic.provenance")


def _print_deployment() -> None:
    """部署约束检查：多 worker + 大模型会 OOM。"""
    from .deployment import check_deployment, model_memory_report

    report = model_memory_report()
    print("— 部署 —")
    print(
        f"模型常驻内存 约 {report['total_gb']} GB"
        f"（含预留 {report['with_reserve_gb']} GB / 建议上限 {report['ceiling_gb']} GB）"
    )
    result = check_deployment()
    prefix = {"ok": "OK", "info": "·", "warn": "!", "error": "X"}.get(result["level"], "·")
    print(f"[{prefix}] {result['message']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aitsm", description="音频 -> 网页弹唱吉他谱")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 Web 服务")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    gen = sub.add_parser("generate", help="从音频生成弹唱谱")
    gen.add_argument("audio", help="音频或视频文件")
    gen.add_argument("--out", required=True, help="输出目录")
    gen.add_argument("--key", default=None, help="指定调，如 C / G / Am")
    gen.add_argument("--capo", type=int, default=None, help="指定变调夹品位 0-9")
    gen.add_argument("--difficulty", default=None, choices=["beginner", "elementary", "intermediate", "advanced"])
    gen.add_argument("--style", default=None, help="pop / folk / rock ...")
    gen.add_argument("--transpose", type=int, default=None, help="整体移调半音数")
    gen.add_argument("--no-tab", action="store_true")
    gen.add_argument("--start", type=float, default=None, help="只分析起始秒（调试）")
    gen.add_argument("--duration", type=float, default=None, help="只分析多少秒（调试）")
    gen.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    gen.add_argument("--language", default=None, help="歌词语言：auto/zh/en/ja …，默认 auto")
    gen.add_argument(
        "--whisper-model",
        default=None,
        help="faster-whisper 模型：tiny/base/small/medium/large-v3（默认 small）",
    )
    gen.add_argument("--no-separation", action="store_true")
    gen.add_argument("--no-lyrics", action="store_true")
    gen.add_argument("--no-melody", action="store_true")
    gen.add_argument(
        "--quality",
        default="standard",
        choices=["standard", "high", "max"],
        help="质量档：standard 默认；high 用微调分轨（slow）；max 再加多次位移与 large-v3 歌词（very slow，CPU 上可能 30 分钟以上）",
    )
    gen.set_defaults(func=cmd_generate)

    bench = sub.add_parser("bench", help="多曲目转录基准")
    bench.add_argument("sources", nargs="+", help="音频文件或目录（目录只收一层）")
    bench.add_argument("--out", default="work/bench", help="输出目录（每首一个子目录）")
    bench.add_argument("--start", type=float, default=None, help="只分析起始秒（调试，全批共用）")
    bench.add_argument("--duration", type=float, default=None, help="只分析多少秒（调试，全批共用）")
    bench.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    bench.add_argument("--language", default=None, help="歌词语言：auto/zh/en/ja …")
    bench.add_argument("--whisper-model", default=None, help="faster-whisper 模型")
    bench.add_argument("--no-lyrics", action="store_true")
    bench.set_defaults(func=cmd_bench)

    demo = sub.add_parser("demo", help="导出示例谱面")
    demo.add_argument("--out", default="work/demo")
    demo.set_defaults(func=cmd_demo)

    voicings = sub.add_parser("voicings", help="生成和弦指法库")
    voicings.set_defaults(func=cmd_voicings)

    doctor = sub.add_parser("doctor", help="环境自检")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def _force_utf8_stdio() -> None:
    """Windows 控制台默认是 GBK，中文进度与告警会乱码或直接抛 UnicodeEncodeError。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - 极少数终端不支持
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

"""命令行入口。

    python -m aitsm.cli serve                    启动 Web 服务
    python -m aitsm.cli generate <audio> --out d  直接生成弹唱谱
    python -m aitsm.cli demo --out d              用内置示例跑一遍前端数据
    python -m aitsm.cli voicings                  生成 web/voicings.json
    python -m aitsm.cli doctor                    环境自检

``generate`` 是 DSH 插件将来调用的主要入口，因此它只依赖文件系统与 stdout，
不依赖任何网络与前端状态。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import REPO_ROOT, SETTINGS, WEB_DIR, check_optional_engines, resolve_binary


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
        "no_separation": args.no_separation,
        "no_lyrics": args.no_lyrics,
        "no_melody": args.no_melody,
    }
    options = {k: v for k, v in options.items() if v is not None}

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
    return 0


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
    gen.set_defaults(func=cmd_generate)

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

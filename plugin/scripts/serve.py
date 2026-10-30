#!/usr/bin/env python
"""启动本地网页乐谱查看器（skill 的可选可视化）。

    python scripts/serve.py --score work/out/score.json

只做两件事：把 ``<repo>/web`` 静态目录与 ``samples/`` 一起提供出去，然后打开浏览器。
项目根由 ``_locate.py`` 解析，因此 skill 装到任意目录都能用。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _locate  # noqa: E402


def main() -> int:
    _locate.force_utf8()

    parser = argparse.ArgumentParser(description="弹唱谱查看器")
    parser.add_argument("--score", default=None, help="要预览的 score.json（挂到 /samples/preview.json）")
    parser.add_argument("--port", type=int, default=7874)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--json", action="store_true", help="以 JSON 报告地址（供 agent 读）")
    args = parser.parse_args()

    try:
        located = _locate.find_root(__file__)
        root, source = located
    except _locate.LocateError as exc:
        payload = {"ok": False, "error": str(exc), "hint": exc.hint, "tried": exc.tried}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json
              else f"{exc}\n\n{exc.hint}", file=sys.stdout if args.json else sys.stderr)
        return 2

    web_dir = root / "web"
    samples_dir = root / "samples"
    if not web_dir.is_dir():
        payload = {"ok": False, "error": f"找不到 web 目录：{web_dir}",
                   "hint": "仓库不完整，重新克隆或检查 AITSM_ROOT 是否指到了正确位置。"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["error"])
        return 2

    preview_file: Path | None = None
    if args.score:
        source_score = Path(args.score).expanduser()
        if not source_score.is_absolute():
            source_score = (Path.cwd() / source_score)
        if not source_score.is_file():
            payload = {"ok": False, "error": f"找不到谱面：{source_score}"}
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["error"])
            return 2
        # 优先放在 score 同目录（可写时），避免污染仓库 samples/
        target = source_score.parent / "preview.json"
        try:
            shutil.copyfile(source_score, target)
            preview_file = target
        except OSError:
            # 兜底**绝不**写进仓库的 samples/：那里有被 git 跟踪的 preview.json，
            # 覆盖它会让工作区出现莫名其妙的脏 diff（和单测直接写 web/ 是同一类坑）。
            try:
                fallback_dir = Path(tempfile.mkdtemp(prefix="aitsm-preview-"))
                target = fallback_dir / "preview.json"
                shutil.copyfile(source_score, target)
                preview_file = target
            except OSError as exc:
                payload = {"ok": False, "error": f"无法准备预览文件：{exc}",
                           "hint": "把 score.json 复制到可写目录后再试。"}
                print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json
                      else payload["error"])
                return 2

    class Handler(SimpleHTTPRequestHandler):
        """``/samples/*`` 映射到谱面目录，``/preview.json`` 指向本次谱面，其余走 ``web/``。"""

        def translate_path(self, path: str) -> str:  # noqa: D102
            clean = path.split("?", 1)[0].split("#", 1)[0]
            if clean.startswith("/samples/"):
                name = clean[len("/samples/"):]
                if preview_file is not None and preview_file.is_file():
                    if name in ("preview.json", "demo-score.json") or name.endswith(".json"):
                        # 优先用本次生成的谱面
                        if name == "preview.json":
                            return str(preview_file)
                return str(samples_dir / name)
            if clean in ("/preview.json", "/score.json"):
                if preview_file is not None:
                    return str(preview_file)
                return str(samples_dir / "preview.json")
            if clean.startswith("/voicings.json"):
                return str(web_dir / "voicings.json")
            return str(web_dir / clean.lstrip("/") or web_dir / "index.html")

        def log_message(self, fmt: str, *fmt_args) -> None:  # 静默默认日志
            return

        def end_headers(self) -> None:  # 开发用：禁用缓存
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        payload = {"ok": False, "error": f"端口 {args.port} 不可用：{exc}",
                   "hint": "换一个端口：--port 7875"}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else payload["error"])
        return 2

    url = f"http://127.0.0.1:{args.port}/"
    if args.json:
        print(json.dumps({
            "ok": True,
            "url": url,
            "preview": str(preview_file) if preview_file else None,
            "aitsm_root": str(root),
            "aitsm_root_source": source,
            "hint": "这是常驻进程；agent 应把它放到后台运行，然后只把 url 告诉用户。",
        }, ensure_ascii=False, indent=2))
    else:
        print(f"弹唱谱查看器：{url}")
        print(f"项目根：{root}（{source}）")
        if preview_file:
            print(f"预览谱面：{args.score} -> {preview_file.name}")
            print("打开后点「加载示例谱」或直接访问 /preview.json")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not args.json:
            print("\n已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

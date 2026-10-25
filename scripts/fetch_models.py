"""下载 v2 专用模型权重（国内走 hf-mirror.com 镜像）。

    .venv/Scripts/python.exe scripts/fetch_models.py --list
    .venv/Scripts/python.exe scripts/fetch_models.py --only duet sheetsage2
    .venv/Scripts/python.exe scripts/fetch_models.py --all

设计要点：

* **幂等**：已经完整下载的仓库跳过（检查关键权重文件是否在）；
* **镜像可换**：``--endpoint`` 或环境变量 ``AITSM_HF_ENDPOINT``；
* **Xet 关闭**：HF 的 Xet 传输在部分 Windows 环境会报 ``I/O error 5``，
  统一走普通 HTTP；
* **不假装成功**：每个仓库下载后校验关键文件存在，失败会明确报错并给出退出码。

体积参考（2026-10）：

| 仓库 | 关键文件 | 体积 |
| --- | --- | --- |
| Wayne42/DuET | model.safetensors | 0.43 GB |
| m-a-p/SheetSage2 | model.safetensors | 0.23 GB |
| pymaster/VocalParse | model.safetensors | 4.1 GB |
| StemSplitio/htdemucs-ft-onnx | htdemucs_ft_*.onnx | 0.05 GB/轨 |
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://hf-mirror.com"

#: repo -> (本地目录名, 关键文件列表)
REPOS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Wayne42/DuET": ("DuET", ("model.safetensors", "config.json")),
    "m-a-p/SheetSage2": ("SheetSage2", ("model.safetensors", "config.json")),
    "pymaster/VocalParse": ("VocalParse", ("model.safetensors", "config.json")),
    "StemSplitio/htdemucs-ft-onnx": ("htdemucs_ft_onnx", ("htdemucs_ft_vocals.onnx",)),
}

#: 别名 -> 仓库 id，方便 --only duet
ALIASES = {
    "duet": "Wayne42/DuET",
    "sheetsage2": "m-a-p/SheetSage2",
    "vocalparse": "pymaster/VocalParse",
    "demucs": "StemSplitio/htdemucs-ft-onnx",
}

#: 下载完把环境变量指向这些目录（打印出来让用户 copy）
ENV_FOR = {
    "DuET": "AITSM_DUET_DIR",
    "SheetSage2": "AITSM_SHEETSAGE2_DIR",
    "VocalParse": "AITSM_VOCALPARSE_DIR",
    "htdemucs_ft_onnx": "AITSM_DEMUCS_ONNX_DIR",
}


def _hf_cli() -> str | None:
    for candidate in (ROOT / ".venv" / "Scripts" / "hf.exe", ROOT / ".venv" / "bin" / "hf"):
        if candidate.is_file():
            return str(candidate)
    found = shutil_which("hf")
    return found


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    return None


def is_complete(target: Path, required: tuple[str, ...]) -> bool:
    return all((target / name).is_file() for name in required)


def size_hint(repo: str) -> str:
    return {
        "Wayne42/DuET": "0.43 GB",
        "m-a-p/SheetSage2": "0.23 GB",
        "pymaster/VocalParse": "4.1 GB",
        "StemSplitio/htdemucs-ft-onnx": "约 0.2 GB（四轨 ONNX）",
    }.get(repo, "?")


def download(
    repo: str,
    *,
    models_dir: Path,
    endpoint: str,
    cli: str,
    force: bool = False,
) -> dict[str, object]:
    folder_name, required = REPOS[repo]
    target = models_dir / folder_name
    if not force and is_complete(target, required):
        return {"repo": repo, "status": "skipped", "dir": str(target), "seconds": 0.0}

    env = dict(os.environ)
    env["HF_ENDPOINT"] = endpoint
    # Xet 传输在部分 Windows 环境会报 I/O error 5，强制走普通 HTTP
    env["HF_HUB_DISABLE_XET"] = "1"
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

    started = time.time()
    target.mkdir(parents=True, exist_ok=True)
    command = [cli, "download", repo, "--local-dir", str(target)]
    try:
        completed = subprocess.run(command, env=env, cwd=str(ROOT), check=False)
        code = completed.returncode
    except FileNotFoundError as exc:
        return {"repo": repo, "status": "error", "error": f"找不到 hf 命令：{exc}", "seconds": 0.0}

    elapsed = round(time.time() - started, 1)
    if code != 0:
        return {"repo": repo, "status": "failed", "error": f"退出码 {code}", "seconds": elapsed}
    if not is_complete(target, required):
        missing = [name for name in required if not (target / name).is_file()]
        return {
            "repo": repo,
            "status": "incomplete",
            "error": f"缺少关键文件：{', '.join(missing)}",
            "dir": str(target),
            "seconds": elapsed,
        }
    return {"repo": repo, "status": "ok", "dir": str(target), "seconds": elapsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_models",
        description="下载 v2 专用模型权重（默认走 hf-mirror.com）",
    )
    parser.add_argument("--list", action="store_true", help="只列出仓库与体积，不下载")
    parser.add_argument("--all", action="store_true", help="下载全部仓库")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="只下载指定仓库（别名 duet/sheetsage2/vocalparse/demucs 或完整 id）",
    )
    parser.add_argument("--models-dir", default=str(ROOT / "models"), help="权重落盘根目录")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AITSM_HF_ENDPOINT") or DEFAULT_ENDPOINT,
        help=f"HF 镜像端点（默认 {DEFAULT_ENDPOINT}）",
    )
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models_dir = Path(args.models_dir).expanduser()

    if args.list or not (args.all or args.only):
        print(f"镜像端点：{args.endpoint}")
        print(f"落盘目录：{models_dir}\n")
        for repo, (folder, required) in REPOS.items():
            state = "已就绪" if is_complete(models_dir / folder, required) else "缺失"
            print(f"  {repo:34s} -> models/{folder:20s} {size_hint(repo):>22s}  [{state}]")
        if not (args.all or args.only):
            print("\n用 --all 下载全部，或 --only duet sheetsage2 只下部分。")
        return 0

    resolved: list[str] = []
    if args.all:
        resolved = list(REPOS)
    else:
        for item in args.only or []:
            repo = ALIASES.get(str(item).lower(), item)
            if repo not in REPOS:
                print(f"未知仓库：{item}（可用别名：{', '.join(ALIASES)}）", file=sys.stderr)
                return 2
            resolved.append(repo)

    cli = _hf_cli()
    if cli is None:
        print(
            "找不到 hf 命令。请先安装 huggingface_hub：\n"
            "  .venv/Scripts/python.exe -m pip install --index-url https://pypi.org/simple/ huggingface_hub",
            file=sys.stderr,
        )
        return 2

    results = []
    for repo in resolved:
        print(f"\n== {repo}（{size_hint(repo)}）")
        result = download(repo, models_dir=models_dir, endpoint=args.endpoint, cli=cli, force=args.force)
        results.append(result)
        status = result["status"]
        if status == "skipped":
            print("   已就绪，跳过")
        elif status == "ok":
            print(f"   完成（{result['seconds']}s）-> {result['dir']}")
        else:
            print(f"   失败：{result.get('error')}", file=sys.stderr)

    ready = [item for item in results if item["status"] in ("ok", "skipped")]
    print(f"\n成功 {len(ready)}/{len(results)}")
    if ready:
        print("\n设置环境变量即可启用对应层：")
        for item in ready:
            folder = Path(str(item["dir"])).name
            env_name = ENV_FOR.get(folder)
            if env_name:
                print(f"  export {env_name}={item['dir']}")
    return 0 if len(ready) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""定位 aitsm 项目本体与可用的 Python 解释器。

DSH skill 会被复制到 ``~/.agents/skills/audio-to-chord-sheet/``
（DSH 0.2.x 的用户级扫描根之一），
此时 ``Path(__file__).resolve().parents[2]`` 指向 ``~/.agents/``，
里面没有 ``aitsm`` 包 —— 所以不能靠脚本自身位置推断仓库。

解析顺序（第一个通过的胜出）：

1. 环境变量 ``AITSM_ROOT``；
2. 配置文件 ``aitsm-root.json``（安装时写入，记录绝对的仓库路径与解释器）；
3. 开发布局：``<repo>/plugin/scripts/_locate.py`` → ``parents[2]``；
4. 便携布局：``<skill>/vendor/``（把 aitsm 一起打包时）；
5. 从 skill 目录向上逐级找同名 ``aitsm/`` 包；
6. pip 安装位置：用候选解释器执行 ``import aitsm`` 反查包路径。

每一步都会校验 ``<root>/aitsm/__init__.py`` 存在，避免误认。
全部失败时抛 ``LocateError``，消息里列出试过的路径与修复命令（供 agent 直接转述）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONFIG_NAME = "aitsm-root.json"
PACKAGE_MARKER = Path("aitsm") / "__init__.py"

# 子进程探针超时（秒）：仅 import，正常 <2s，冷启动留足余量
PROBE_TIMEOUT = 120

_probe_cache: dict[tuple[str, str], bool] = {}


class LocateError(RuntimeError):
    """找不到项目本体或可用解释器。``hint`` 供 agent 原样转述给用户。"""

    def __init__(self, message: str, hint: str = "", tried: list[dict] | None = None) -> None:
        super().__init__(message)
        self.hint = hint
        self.tried = tried or []


# ------------------------------------------------------------------ 基础工具


def _dedupe(paths: list[Path | None]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        if path is None:
            continue
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, ValueError):
            continue
        key = str(resolved).lower() if os.name == "nt" else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def is_project_root(path: Path | str) -> bool:
    """真正的项目根：包目录存在。"""
    try:
        return (Path(path) / PACKAGE_MARKER).is_file()
    except OSError:
        return False


def script_root(script_file: str | Path) -> Path:
    """skill 根目录（scripts/ 的上一级）。"""
    return Path(script_file).resolve().parent.parent


def read_config(script_file: str | Path) -> dict:
    """读安装时写入的 aitsm-root.json（没有就返回空 dict）。"""
    base = script_root(script_file)
    for candidate in (base / CONFIG_NAME, base / "scripts" / CONFIG_NAME):
        try:
            if candidate.is_file():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["_source_file"] = str(candidate)
                    return data
        except (OSError, ValueError):
            continue
    return {}


# ------------------------------------------------------------------ 解释器


def probe_python(python: str, root: Path) -> bool:
    """该解释器能否在 ``root`` 下 import aitsm（带缓存）。"""
    key = (str(root).lower(), os.path.normcase(python))
    if key in _probe_cache:
        return _probe_cache[key]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [python, "-c", "import aitsm"],
            cwd=str(root), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT,
        )
        ok = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    _probe_cache[key] = ok
    return ok


def python_candidates(root: Path, script_file: str | Path) -> list[tuple[str, str]]:
    """候选解释器列表：``[(可执行路径, 来源说明)]``，按优先级排序。"""
    config = read_config(script_file)
    candidates: list[tuple[str, str]] = []

    env_python = os.environ.get("AITSM_PYTHON")
    if env_python:
        candidates.append((env_python, "环境变量 AITSM_PYTHON"))
    if config.get("python"):
        candidates.append((str(config["python"]), f"配置文件 {CONFIG_NAME}"))
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python", "venv/Scripts/python.exe"):
        candidate = root / relative
        if candidate.is_file():
            candidates.append((str(candidate), f"项目 venv（{relative}）"))
    candidates.append((sys.executable, "当前解释器"))
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append((found, f"PATH 上的 {name}"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for path, why in candidates:
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, why))
    return unique


def find_python(root: Path, script_file: str | Path) -> tuple[str, str]:
    """返回 ``(解释器路径, 来源)``；找不到能 import aitsm 的则抛 LocateError。"""
    tried: list[dict] = []
    for path, why in python_candidates(root, script_file):
        if not os.path.isfile(path):
            tried.append({"python": path, "why": why, "result": "文件不存在"})
            continue
        if probe_python(path, root):
            return path, why
        tried.append({"python": path, "why": why, "result": "无法 import aitsm"})
    raise LocateError(
        "找不到能 import aitsm 的 Python 解释器",
        hint=(
            "项目需要先建好 venv 并安装依赖：\n"
            f'  "{root / ".venv" / "Scripts" / "python.exe"}" -m pip install -r requirements.txt\n'
            "或用 AITSM_PYTHON 指定解释器。"
        ),
        tried=tried,
    )


# ------------------------------------------------------------------ 项目根


def candidate_roots(script_file: str | Path) -> list[tuple[Path, str]]:
    """候选项目根本列表：``[(路径, 来源说明)]``。"""
    base = script_root(script_file)
    config = read_config(script_file)
    candidates: list[tuple[Path, str]] = []

    env_root = os.environ.get("AITSM_ROOT")
    if env_root:
        candidates.append((Path(env_root), "环境变量 AITSM_ROOT"))
    if config.get("aitsm_root"):
        candidates.append((Path(str(config["aitsm_root"])), f"配置文件 {CONFIG_NAME}"))
    # 开发布局：<repo>/plugin/scripts/_locate.py
    resolved = Path(script_file).resolve()
    if len(resolved.parents) > 2:
        candidates.append((resolved.parents[2], "开发布局（脚本上两级）"))
    # 便携布局：<skill>/vendor/aitsm/...
    candidates.append((base / "vendor", "skill 内置 vendor/"))
    # 从 skill 目录逐级向上找 aitsm/ 包（安装到 skills 目录时通常无用，但代价低）
    for parent in list(base.parents)[:6]:
        candidates.append((parent, f"向上查找 {parent.name or parent}"))
    return [(path, why) for path, why in candidates]


def find_root(script_file: str | Path) -> tuple[Path, str]:
    """返回 ``(项目根, 来源)``；失败抛 LocateError（含已试路径与修复建议）。"""
    tried: list[dict] = []
    for path, why in candidate_roots(script_file):
        if is_project_root(path):
            return path, why
        tried.append({"path": str(path), "why": why, "result": "没有 aitsm/__init__.py"})

    # 最后手段：候选解释器里可能 pip 安装过 aitsm
    base = script_root(script_file)
    for python, why in python_candidates(Path.cwd(), script_file)[:4]:
        if not os.path.isfile(python):
            continue
        try:
            proc = subprocess.run(
                [python, "-c", "import aitsm, pathlib; print(pathlib.Path(aitsm.__file__).parent.parent)"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        found = (proc.stdout or "").strip().splitlines()
        if proc.returncode == 0 and found and is_project_root(found[-1]):
            return Path(found[-1]).resolve(), f"pip 安装位置（{why}）"
        tried.append({"path": found[-1] if found else python, "why": f"pip 探测（{why}）", "result": "不可用"})

    raise LocateError(
        "找不到 AI-To-Sheet-Music 项目本体（aitsm 包）",
        hint=(
            "skill 脚本需要指向仓库根目录。任选一种修复：\n"
            "1) 设置环境变量：AITSM_ROOT=<仓库根>\n"
            f"2) 写配置文件：{base / CONFIG_NAME}"
            ' 内容 {"aitsm_root": "<仓库根>", "python": "<venv python>"}\n'
            "3) 重新执行安装脚本（install.py）自动写入上述配置"
        ),
        tried=tried,
    )


# ------------------------------------------------------------------ 输出目录


def _writable(path: Path) -> tuple[bool, str]:
    """实际写一个探针文件来确认可写（Windows 的权限/完整性标签差异只能实测）。"""
    probe = path / f".aitsm-write-probe-{os.getpid()}"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def _system_temp() -> Path | None:
    """系统临时目录；低完整性环境下 tempfile 可能找不到可用目录（会抛异常）。"""
    try:
        return Path(tempfile.gettempdir())
    except (OSError, FileNotFoundError, AssertionError, ValueError):
        for env_name in ("TEMP", "TMP", "TMPDIR"):
            value = os.environ.get(env_name)
            if value:
                candidate = Path(value)
                try:
                    if candidate.is_dir():
                        return candidate
                except OSError:
                    continue
        return None


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def writable_candidates(root: Path, script_file: str | Path) -> list[tuple[Path, str]]:
    """默认输出目录候选，按优先级排序。

    关键：**仓库内的 plugin/ 不能成为产物目录**。开发布局下 skill 目录是
    ``<repo>/plugin``，若把 work/ 放在那里，跑一次生成就在仓库里留下运行产物。
    因此 dev 布局把输出指向 ``<repo>/work/skill-out``（已被 .gitignore 覆盖）；
    只有 skill 被装到仓库外（如 ~/.agents/skills/）时才用自带的 work/。
    """
    base = script_root(script_file)
    out: list[tuple[Path, str]] = []

    env_out = os.environ.get("AITSM_OUT_DIR")
    if env_out:
        out.append((Path(env_out), "环境变量 AITSM_OUT_DIR"))

    skill_in_repo = _is_inside(base, root)
    if skill_in_repo:
        out.append((root / "work" / "skill-out", "项目 work/skill-out（仓库内不写 plugin/）"))
        out.append((base / "work" / "out", "skill 目录 work/out"))
    else:
        out.append((base / "work" / "out", "skill 目录 work/out"))
        out.append((root / "work" / "skill-out", "项目 work/skill-out"))

    system_temp = _system_temp()
    if system_temp is not None:
        out.append((system_temp / "aitsm-out", "系统临时目录"))
    return out


def ensure_dir(path: Path) -> tuple[bool, str]:
    """确保目录存在且可写。"""
    return _writable(Path(path))


def default_out_dir(root: Path, script_file: str | Path) -> tuple[Path | None, list[dict]]:
    """挑第一个可写的默认输出目录；返回 ``(目录或 None, 探测明细)``。"""
    tried: list[dict] = []
    for candidate, why in writable_candidates(root, script_file):
        ok, error = _writable(candidate)
        tried.append({"path": str(candidate), "why": why, "writable": ok, "error": error})
        if ok:
            return candidate.resolve(), tried
    return None, tried


# ------------------------------------------------------------------ 便捷入口


def resolve(script_file: str | Path) -> dict:
    """一次性解析 root / python / 输出目录，供三个脚本共用。"""
    root, root_source = find_root(script_file)
    python, python_source = find_python(root, script_file)
    out_dir, out_tried = default_out_dir(root, script_file)
    return {
        "aitsm_root": str(root),
        "aitsm_root_source": root_source,
        "python": python,
        "python_source": python_source,
        "default_out_dir": str(out_dir) if out_dir else None,
        "out_candidates": out_tried,
    }


def child_env(root: Path) -> dict[str, str]:
    """给 aitsm 子进程准备环境：保证能 import、UTF-8 输出、缓存落在项目内。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("AITSM_WORK_DIR", str(root / "work"))
    return env


def force_utf8() -> None:
    """Windows 控制台默认 GBK，agent 读 stdout 会拿到坏编码。"""
    for stream_name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, stream_name, None), "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

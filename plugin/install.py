#!/usr/bin/env python
"""把 plugin/ 安装成 DSH skill，并写入定位配置。

用法（在仓库根执行）：:

    python plugin/install.py                       # 安装到默认位置
    python plugin/install.py --dest <skills-dir>   # 指定 skills 根目录
    python plugin/install.py --dry-run             # 只看会做什么

它做四件事：

1. **自动挑一个能写的解释器** —— 某些 Windows 机器上项目 ``.venv`` 的解释器带
   「低完整性」标记，往 ``%USERPROFILE%`` 写文件会被系统拒绝（连 mkdir 都不行）。
   本脚本先用当前解释器实测写入，失败就自动换一个（PATH 上的 python、
   py launcher、系统安装目录），并把该解释器的绝对路径回显出来。
2. 复制 SKILL.md / scripts / references / agents 到 ``<skills>/audio-to-chord-sheet/``。
3. 写 ``aitsm-root.json``，记录仓库根与 venv 解释器的绝对路径
   —— 这是 skill 被复制走之后仍能找到 ``aitsm`` 包的关键。
4. 跑一次 ``doctor.py --json`` 确认新装的文件可用。

不修改任何 DSH 配置，也不重启服务：DSH 会自动扫描该 skills 目录
（本机实测：复制完成后当前会话的可用技能里即出现 audio-to-chord-sheet）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_DIR.parent
SKILL_NAME = "audio-to-chord-sheet"
CONFIG_NAME = "aitsm-root.json"
# 只复制运行必需的内容，避免把运行产物带进用户目录
INCLUDE = ("SKILL.md", "scripts", "references", "agents")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "work", ".pytest_cache")

WRITE_PROBE = (
    "import sys,pathlib;"
    "p=pathlib.Path(sys.argv[1]);"
    "d=p/'probe';d.mkdir(parents=True,exist_ok=True);"
    "(d/'f').write_text('ok');(d/'f').unlink();d.rmdir();print('WRITABLE')"
)


def force_utf8() -> None:
    """Windows 控制台默认 GBK：中文 + 替换字符会直接抛 UnicodeEncodeError。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def default_dest() -> Path:
    """DSH 读取用户级 skill 的默认位置（可用环境变量覆盖）。"""
    override = os.environ.get("DSH_SKILLS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "opencode" / "skills"


def venv_python() -> str | None:
    """项目 venv 的解释器 —— 这是**运行**时该用的（有 torch/demucs/whisper）。"""
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = REPO_ROOT / relative
        if candidate.is_file():
            return str(candidate)
    return None


# ---------------------------------------------------------------- 解释器选择


def _scratch_dir() -> Path:
    """一个确定可写的临时目录（低完整性环境下 tempfile 可能整个不可用）。"""
    try:
        return Path(tempfile.gettempdir())
    except (OSError, FileNotFoundError, AssertionError, ValueError):
        pass
    for env_name in ("TEMP", "TMP", "TMPDIR"):
        value = os.environ.get(env_name)
        if value and Path(value).is_dir():
            return Path(value)
    return REPO_ROOT


def probe_writable(python: str, directory: Path) -> tuple[bool, str]:
    """跑一个子进程，实测该解释器能否在 directory 下 mkdir + 写文件。"""
    try:
        proc = subprocess.run(
            [python, "-c", WRITE_PROBE, str(directory)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180, cwd=str(_scratch_dir()),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0 and "WRITABLE" in (proc.stdout or ""):
        return True, ""
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (detail[-1] if detail else f"exit {proc.returncode}")


def interpreter_candidates() -> list[tuple[str, str]]:
    """可能用来执行安装的解释器：``[(路径, 来源)]``，当前解释器排第一。"""
    candidates: list[tuple[str, str]] = [(sys.executable, "当前解释器")]

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append((found, f"PATH 上的 {name}"))
    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append((py_launcher, "py launcher"))

    # 常见系统安装位置（覆盖 conda / 官方安装器 / Windows Store）
    home = Path.home()
    patterns = [
        "AppData/Local/Programs/Python/Python3*/python.exe",
        "AppData/Local/Microsoft/WindowsApps/python3*.exe",
        "anaconda3/python.exe",
        "miniconda3/python.exe",
    ]
    roots = [Path("C:/"), Path("D:/"), home]
    for root in roots:
        for pattern in patterns[:2] if root == home else patterns[2:]:
            try:
                for match in sorted(root.glob(pattern)):
                    candidates.append((str(match), f"常见安装位置 {match}"))
            except OSError:
                continue
        try:
            for match in sorted(root.glob("ProgramData/anaconda3/python.exe")):
                candidates.append((str(match), f"常见安装位置 {match}"))
        except OSError:
            pass
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        value = os.environ.get(env_name)
        if value:
            for relative in ("python.exe", "bin/python"):
                candidate = Path(value) / relative
                if candidate.is_file():
                    candidates.append((str(candidate), f"{env_name} 环境"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for path, why in candidates:
        if not path:
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, why))
    return unique


def choose_installer_python(probe_target: Path) -> tuple[str | None, str, list[dict]]:
    """挑一个能写 ``probe_target`` 的解释器。

    返回 ``(解释器或 None, 来源说明, 探测明细)``。
    """
    probe_dir = probe_target.parent if probe_target.name else probe_target
    # 目标目录可能还不存在，先探测它的父目录（能不能在里面建目录）
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent

    tried: list[dict] = []
    for path, why in interpreter_candidates():
        if not os.path.isfile(path):
            continue
        ok, error = probe_writable(path, probe_dir)
        tried.append({"python": path, "why": why, "writable": ok, "error": error})
        if ok:
            return path, why, tried
    return None, "", tried


def run_install_with(python: str) -> tuple[bool, str]:
    """用指定解释器重跑本脚本（带标记避免无限递归）。"""
    env = dict(os.environ)
    env["AITSM_INSTALL_CHILD"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [python, str(Path(__file__).resolve()), *sys.argv[1:]],
            env=env, cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return proc.returncode == 0, f"exit {proc.returncode}"


# ---------------------------------------------------------------- 可写性 / 完整性


def can_write(python: str | None, path: Path) -> tuple[bool, str]:
    """用**运行解释器**实测能否写 path（生成时的默认输出目录）。"""
    runner = python or sys.executable
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return probe_writable(runner, path)


def label_low(path: Path) -> tuple[bool, str]:
    """把目录标成低完整性，让带低完整性标签的解释器也能写入（Windows only）。"""
    if os.name != "nt":
        return False, "非 Windows"
    icacls = shutil.which("icacls")
    if not icacls:
        return False, "找不到 icacls"
    try:
        proc = subprocess.run(
            [icacls, str(path), "/setintegritylevel", "(OI)(CI)Low"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return proc.returncode == 0, (proc.stdout or proc.stderr or "").strip()


def copy_plugin(target: Path) -> tuple[bool, str]:
    """复制插件文件；逐项容错，返回首个失败原因。"""
    target.mkdir(parents=True, exist_ok=True)
    for name in INCLUDE:
        source = PLUGIN_DIR / name
        if not source.exists():
            continue
        destination = target / name
        try:
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                shutil.copytree(source, destination, ignore=IGNORE)
            else:
                shutil.copy2(source, destination)
        except OSError as exc:
            return False, f"复制 {name} 失败：{type(exc).__name__}: {exc}"
    return True, ""


def write_config(target: Path, python: str | None) -> tuple[bool, str]:
    config = {"aitsm_root": str(REPO_ROOT)}
    if python:
        config["python"] = python
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    try:
        (target / CONFIG_NAME).write_text(text, encoding="utf-8")
        return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def self_check(python: str | None, target: Path) -> int:
    doctor = target / "scripts" / "doctor.py"
    if not doctor.is_file():
        print("[4/4] 跳过自检（未找到 doctor.py）")
        return 0
    runner = python or sys.executable
    print(f"[4/4] 自检：{runner}")
    try:
        proc = subprocess.run(
            [runner, str(doctor), "--json"],
            cwd=str(target), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        print(f"[!] 无法运行自检：{exc}")
        return 1
    try:
        report = json.loads(proc.stdout)
    except ValueError:
        print("[!] 自检输出无法解析：")
        print(f"    stdout: {(proc.stdout or '')[-1200:]}")
        print(f"    stderr: {(proc.stderr or '')[-1200:]}")
        return 1

    ready = report.get("ready")
    print(f"    就绪：{'是' if ready else '否'}")
    if report.get("locate_error"):
        print(f"    路径错误：{report['locate_error']['error']}")
        hint = str(report["locate_error"].get("hint", ""))
        print("    修复：" + hint.replace("\n", "\n          "))
    if report.get("degraded"):
        print(f"    降级组件：{', '.join(report['degraded'])}")
        print(f"    影响：{report.get('advice')}")
    paths = report.get("paths") or {}
    print(f"    项目根：{paths.get('aitsm_root')}（{paths.get('aitsm_root_source')}）")
    print(f"    输出目录：{paths.get('default_out_dir')}")
    if ready:
        print(f"    建议：{report.get('advice')}")
    return 0 if ready else 1


# ---------------------------------------------------------------- main


def main() -> int:
    force_utf8()

    parser = argparse.ArgumentParser(description="安装 audio-to-chord-sheet skill")
    parser.add_argument("--dest", default=None, help="skills 根目录（默认 ~/.config/opencode/skills）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true")
    args = parser.parse_args()

    skills_root = Path(args.dest).expanduser().resolve() if args.dest else default_dest().resolve()
    target = skills_root / SKILL_NAME
    runner_python = venv_python()

    if not (PLUGIN_DIR / "SKILL.md").is_file():
        print(f"[x] 找不到 {PLUGIN_DIR / 'SKILL.md'}")
        print("    请在仓库内的 plugin/ 目录运行本脚本。")
        return 2

    print(f"仓库根      : {REPO_ROOT}")
    print(f"运行解释器  : {runner_python or '（未找到 .venv，将用当前解释器）'}")
    print(f"安装目标    : {target}")

    if args.dry_run:
        print("\n[dry-run] 将复制：", ", ".join(INCLUDE))
        print("[dry-run] 将写入：", target / CONFIG_NAME)
        return 0

    # ---- 0) 挑一个能写的解释器来执行安装
    is_child = os.environ.get("AITSM_INSTALL_CHILD") == "1"
    if not is_child:
        installer, source, tried = choose_installer_python(target)
        if installer is None:
            print(f"\n[x] 当前环境没有任何解释器能写入 {skills_root}")
            print("    探测结果：")
            for item in tried:
                print(f"      - {item['python']}  [{item['why']}]")
                if item["error"]:
                    print(f"          {item['error']}")
            print("\n    解决方式（任选）：")
            print(f"      1) 用管理员身份的终端重新执行：python \"{Path(__file__).resolve()}\"")
            print(f"      2) 装到有写权限的位置：python \"{Path(__file__).resolve()}\" --dest <可写目录>")
            print("      3) 手动把 plugin/ 下的 SKILL.md、scripts/、references/、agents/")
            print(f"         复制到 {target}")
            return 2
        if os.path.normcase(os.path.abspath(installer)) != os.path.normcase(os.path.abspath(sys.executable)):
            print(f"\n当前解释器无法写入 {skills_root}（完整性/权限限制）。")
            print(f"自动改用：{installer}  [{source}]")
            print(f"等价命令：\"{installer}\" \"{Path(__file__).resolve()}\" {' '.join(sys.argv[1:])}\n")
            ok, message = run_install_with(installer)
            if not ok:
                print(f"[x] 用 {installer} 重试仍失败：{message}")
                return 2
            return 0
        print(f"本次用当前解释器执行安装（已实测可写）")

    # ---- 1) 复制
    ok, error = copy_plugin(target)
    if not ok:
        print(f"[x] {error}")
        print(f"    目标目录：{target}")
        return 2
    print("[1/4] 已复制 SKILL.md / scripts / references / agents")

    # ---- 2) 定位配置
    ok, error = write_config(target, runner_python)
    if ok:
        print(f"[2/4] 已写入 {CONFIG_NAME}")
    else:
        print(f"[!] 写 {CONFIG_NAME} 失败：{error}")
        print(f"    可改用环境变量：AITSM_ROOT={REPO_ROOT}")

    # ---- 3) work/ 可写（生成时的默认输出目录，运行解释器必须能写）
    work_dir = target / "work"
    work_ok, work_error = can_write(runner_python, work_dir)
    if work_ok:
        print("[3/4] work/ 可写")
    else:
        labelled, message = label_low(work_dir)
        if labelled:
            work_ok, work_error = can_write(runner_python, work_dir)
            print(f"[3/4] work/ 原本不可写，已标记低完整性：{'已修复' if work_ok else '仍失败'}")
        else:
            first_line = (work_error or "?").strip().splitlines()[-1]
            print(f"[3/4] work/ 不可写（{first_line}）")
            print(f"      自动修正失败：{message}")
            print(f"      生成时请指定可写目录：--out {REPO_ROOT / 'work' / 'skill-out'}")
        # 成功或失败都不阻塞安装：doctor 会报告可写候选

    # 确保不把空的 work/ 留下来当作仓库产物
    try:
        if work_dir.is_dir() and not any(work_dir.iterdir()):
            work_dir.rmdir()
    except OSError:
        pass

    # ---- 4) 自检
    if args.skip_doctor:
        print("[4/4] 跳过自检")
        return 0
    return self_check(runner_python, target)


if __name__ == "__main__":
    raise SystemExit(main())

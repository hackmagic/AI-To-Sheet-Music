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
2. 复制 SKILL.md / scripts / references / agents 到 ``<skills>/audio-to-chord-sheet/``，
   并逐字节校验复制后的 SKILL.md 与仓库一致（防止又出现「仓库改了、副本还是旧的」）。
3. 写 ``aitsm-root.json``，记录仓库根与 venv 解释器的绝对路径
   —— 这是 skill 被复制走之后仍能找到 ``aitsm`` 包的关键。
4. 跑一次 ``doctor.py --json`` 确认新装的文件可用。

默认目标是 **DSH 0.2.x 真正扫描的用户级目录** ``~/.agents/skills``
（另三个根是 ``<projectRoot>/.dsh/skills``、``<projectRoot>/.agents/skills``、``~/.dsh/skills``；
详见 ``docs/03-dsh-plugin.md`` 的「DSH 0.2.x 技能发现契约与版本」）。
旧版手册里的 ``~/.config/opencode/skills`` **已不在扫描列表里**，脚本检测到那边的遗留副本会给出迁移提示。

不修改任何 DSH 配置，也不重启服务：DSH 会按需扫描上面这些目录
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
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_DIR.parent
SKILL_NAME = "audio-to-chord-sheet"
CONFIG_NAME = "aitsm-root.json"
# 只复制运行必需的内容，避免把运行产物带进用户目录
INCLUDE = ("SKILL.md", "scripts", "references", "agents")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "work", ".pytest_cache")

#: DSH 0.2.x skill-filesystem provider 扫描的用户级根（按优先级）
AGENTS_HOME = Path.home() / ".agents"
DSH_HOME = Path.home() / ".dsh"
#: 旧版（opencode 时代）的安装位置。DSH 0.2.x 已不读，仅用于迁移提示。
LEGACY_DEST = Path.home() / ".config" / "opencode" / "skills"

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
    """DSH 0.2.x 读取用户级 skill 的默认位置（可用环境变量覆盖）。

    依据：``@deepseek-ai/dsh-skill-filesystem`` 的 provider 只扫四个根
    —— ``<projectRoot>/.dsh/skills``、``<projectRoot>/.agents/skills``、
    ``~/.dsh/skills``、``~/.agents/skills``。用户级安装走 ``~/.agents/skills``。
    旧版的 ``~/.config/opencode/skills`` 在 0.2.x 里**根本不会被扫到**。
    """
    override = os.environ.get("DSH_SKILLS_DIR")
    if override:
        return Path(override).expanduser()
    return AGENTS_HOME / "skills"


def legacy_skill_dir() -> Path:
    """旧版 opencode 位置的同名副本（存在则说明需要提示迁移）。"""
    return LEGACY_DEST / SKILL_NAME


def warn_legacy_copy(target: Path) -> None:
    """旧位置还有副本就明说 —— 不静默，因为「改了没生效」几乎都是这个坑。"""
    stale = legacy_skill_dir()
    if not stale.is_dir():
        return
    try:
        same_target = os.path.normcase(os.path.abspath(stale)) == os.path.normcase(
            os.path.abspath(target)
        )
    except OSError:
        same_target = False
    if same_target:
        return
    print("")
    print(f"[!] 发现旧版遗留副本：{stale}")
    print("    DSH 0.2.x 的 skill-filesystem 只扫描 ~/.agents/skills、~/.dsh/skills")
    print("    以及项目内 .agents/skills、.dsh/skills，**不会读** ~/.config/opencode/skills。")
    print("    也就是说这份旧副本不会被任何会话加载，改了它也不生效。")
    print("    留着的风险：别的 agent 工具（opencode / claude code 等）仍会扫这个目录，")
    print("    于是会出现两个同名、内容不同版本的技能，谁生效取决于工具。")
    print("    本脚本默认**不删**它（里面可能有你跑出来的 work/ 产物）。")
    print("    确认新副本可用后，可以用下面任一方式处理：")
    print("      - 让脚本备份并移除：加 --remove-legacy（先改名为 .bak-<时间戳>，可回滚）")
    if os.name == "nt":
        print(f"      - 手动删：rmdir /s /q {stale}")
    else:
        print(f"      - 手动删：rm -rf {stale}")
    print("    本脚本不会自动删除它——里面可能有你跑出来的 work/ 产物。")


def backup_legacy_copy() -> tuple[bool, str]:
    """把旧副本**改名**成 ``.bak-<时间戳>``，而不是 rmtree。

    为什么先改名：那是用户的目录，可能含 ``work/`` 里跑了几十分钟的产物。
    直接删是不可逆的；改名后用户随时可以翻回去找东西，也可以自己删。
    改名失败（跨卷/占用/权限）就不动它并报错，绝不退化成删除。
    """
    stale = legacy_skill_dir()
    if not stale.is_dir():
        return True, "旧副本不存在，无需处理"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = stale.with_name(f"{stale.name}.bak-{stamp}")
    # 极端情况下同名备份已存在（同秒重跑），追加序号而不是覆盖
    index = 1
    while backup.exists():
        backup = stale.with_name(f"{stale.name}.bak-{stamp}-{index}")
        index += 1
    try:
        stale.rename(backup)
    except OSError as exc:
        return False, f"备份旧副本失败（未删除任何东西）：{type(exc).__name__}: {exc}"
    return True, f"旧副本已备份为 {backup}（确认无用后可自行删除）"


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
    """跑一个子进程，实测该解释器能否在 directory 下 mkdir + 写文件。

    **注意**：这条探测只在父进程本身不是 Low 完整性时可信。Windows 上子进程会
    **继承父进程的完整性级别**，所以从一个 Low 进程里派生出来的任何解释器都还是 Low，
    探测会把「其实能写的 Medium 解释器」也判成写不了。要判断能不能装，
    先看 :func:`in_process_writable`（自己动手写，不派生）。
    """
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


def in_process_writable(directory: Path) -> tuple[bool, str]:
    """**当前进程**自己试着建目录 + 写文件，不派生任何子进程。

    这是最可信的「能不能装」判据：就是本进程的实际权限，没有继承问题。
    """
    probe = Path(directory) / ".aitsm-install-probe"
    try:
        probe.mkdir(parents=True, exist_ok=True)
        target = probe / "f.txt"
        target.write_text("ok", encoding="utf-8")
        target.unlink()
        probe.rmdir()
        return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def integrity_label(executable: str) -> str:
    """查某个 exe 的强制完整性标签（``Low`` / ``Medium`` / …），查不到返回空串。

    Windows 会把标签写进文件的安全描述符：``Mandatory Label\\Low Mandatory Level``。
    带 Low 标签的解释器**即使从普通（Medium）终端启动**，进程也会跑在 Low 下，
    于是写不了 ``%USERPROFILE%``。这是本机 ``.venv`` 的真实状态。
    """
    if os.name != "nt":
        return ""
    icacls = shutil.which("icacls")
    if not icacls:
        return ""
    try:
        proc = subprocess.run(
            [icacls, str(executable)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for token in ("Low", "Medium", "High", "System"):
        if f"Mandatory Label\\{token} Mandatory Level" in text:
            return token
    return ""


def current_is_low() -> bool:
    """当前进程是否跑在 Low 完整性下（据自身 exe 的标签判断，够用且无副作用）。"""
    return integrity_label(sys.executable) == "Low"


def install_hint(installer: str) -> str:
    """给「必须换个终端重跑」的提示：直接可复制的命令。"""
    return f'"{installer}" "{Path(__file__).resolve()}"'


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
    """挑一个能写 ``probe_target`` 的解释器（**仅当当前进程不是 Low 时才有意义**）。

    返回 ``(解释器或 None, 来源说明, 探测明细)``。

    为什么不总能靠它：Windows 子进程**继承父进程完整性级别**。从 Low 进程派生出来的
    解释器一律仍是 Low，哪怕它自己的 exe 是 Medium —— 所以「换解释器重跑」在 Low 父进程下
    是**结构性无效**的，本函数探测结果会全部是 False。调用方必须先看
    :func:`current_is_low`，别把「全都写不了」的错误结论甩给用户。
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


def non_low_candidates() -> list[tuple[str, str]]:
    """exe 本身**不带 Low 标签**的解释器 —— 从普通终端跑它就不会掉到 Low。"""
    out: list[tuple[str, str]] = []
    for path, why in interpreter_candidates():
        if not os.path.isfile(path):
            continue
        # py launcher 只是个转发器，标签要看它真正的目标，这里跳过
        if os.path.normcase(os.path.basename(path)) == os.path.normcase("py.exe"):
            continue
        if integrity_label(path) != "Low":
            out.append((path, why))
    return out


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


def verify_skill_md(target: Path) -> tuple[bool, str]:
    """校验副本里的 SKILL.md 与仓库逐字节一致。

    为什么值得单独校验：skill 目录是**复制品**，曾经真实出现过「仓库版已加 v2 内容、
    装好的副本还是 2026-10-23 的旧版」——agent 读的是旧说明，于是漏讲
    symbolic.json/provenance、把降级链路说成用了专用模型。复制成功不代表内容对。
    """
    source = PLUGIN_DIR / "SKILL.md"
    copied = target / "SKILL.md"
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        return False, f"读仓库 SKILL.md 失败：{type(exc).__name__}: {exc}"
    try:
        copied_bytes = copied.read_bytes()
    except OSError as exc:
        return False, f"读副本 SKILL.md 失败：{type(exc).__name__}: {exc}"
    if source_bytes != copied_bytes:
        return False, (
            f"SKILL.md 副本与仓库不一致（仓库 {len(source_bytes)} 字节 / "
            f"副本 {len(copied_bytes)} 字节）；重跑本脚本或检查目标目录是否只读"
        )
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


def finish_legacy_migration() -> int:
    """``--remove-legacy`` 的执行体：备份（改名）旧副本，不直接删除。"""
    print("")
    print("[5/5] 迁移旧副本")
    ok, message = backup_legacy_copy()
    print(f"    {message}")
    if not ok:
        print("    旧副本保持原样，未做任何删除。可手动处理或检查占用后重试。")
        return 1
    return 0


def main() -> int:
    force_utf8()

    parser = argparse.ArgumentParser(description="安装 audio-to-chord-sheet skill")
    parser.add_argument(
        "--dest", default=None,
        help="skills 根目录（默认 ~/.agents/skills；也可用 DSH_SKILLS_DIR 覆盖）",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true")
    parser.add_argument(
        "--remove-legacy", action="store_true",
        help="把 ~/.config/opencode/skills 的旧副本改名备份为 .bak-<时间戳>（不直接删除）",
    )
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
    warn_legacy_copy(target)

    if args.dry_run:
        print("\n[dry-run] 将复制：", ", ".join(INCLUDE))
        print("[dry-run] 将写入：", target / CONFIG_NAME)
        print("[dry-run] 将校验：副本 SKILL.md 与仓库逐字节一致")
        return 0

    # ---- 0) 能不能写：先用「本进程自己写」判，不靠派生子进程
    #      Windows 子进程继承父进程完整性级别，所以从 Low 的 .venv 里派生任何解释器
    #      都还是 Low，探测会一律失败——那是假信号，不是「机器上没一个能写的」。
    is_child = os.environ.get("AITSM_INSTALL_CHILD") == "1"
    can_write_here, write_error = in_process_writable(skills_root)

    if not can_write_here:
        # 当前进程写不了：可能是文件系统权限，也可能是 Low 完整性继承。
        low = current_is_low()
        alternatives = non_low_candidates()
        if low:
            print(f"\n[x] 当前进程跑在 **Low 完整性**下，写不了 {skills_root}")
            print(f"    （{write_error}）")
            print("    根因不是权限配置，而是「Low 进程派生的一切都是 Low」——")
            print("    换解释器重跑也没用，因此本脚本没有尝试「自动换一个解释器重试」。")
            if alternatives:
                print("\n    这台机器上能用的 Medium 解释器（在**普通终端**里直接跑它即可）：")
                for path, why in alternatives:
                    print(f"      {install_hint(path)}")
                    print(f"        [{why}]")
            else:
                print("\n    没找到不带 Low 标签的解释器。改用管理员/普通终端里的系统 python：")
                print(f'      python "{Path(__file__).resolve()}"')
            print("\n    其余出路（任选）：")
            print(f"      1) 装到有写权限的位置（项目盘不受 Low 限制）：")
            print(f'         python "{Path(__file__).resolve()}" --dest {REPO_ROOT / ".agents"}'
                  f"{os.sep}skills")
            print("      2) 手动把 plugin/ 下的 SKILL.md、scripts/、references/、agents/")
            print(f"         复制到 {target}（资源管理器复制不受完整性限制）")
            return 2

        # 非 Low 却写不了：才值得试「换一个能写的解释器」
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
            print("本次用当前解释器执行安装（已实测可写）")
    else:
        print(f"写入权限    : 已实测可写（{skills_root}）")

    # ---- 1) 复制
    ok, error = copy_plugin(target)
    if not ok:
        print(f"[x] {error}")
        print(f"    目标目录：{target}")
        return 2
    print("[1/4] 已复制 SKILL.md / scripts / references / agents")

    # ---- 1b) 副本内容校验（防「仓库改了、副本没变」）
    ok, error = verify_skill_md(target)
    if ok:
        print("[1b] SKILL.md 副本与仓库一致")
    else:
        print(f"[!] {error}")
        return 2

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
        if args.remove_legacy:
            return finish_legacy_migration()
        return 0
    code = self_check(runner_python, target)
    if args.remove_legacy:
        migration = finish_legacy_migration()
        if migration != 0:
            return migration
    return code


if __name__ == "__main__":
    raise SystemExit(main())

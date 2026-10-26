"""集中配置：路径、模型选择、运行环境和可选引擎开关。

所有路径都可用环境变量覆盖，便于 DSH 插件把工作目录指向用户自己的位置。
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

SCHEMA_DIR = REPO_ROOT / "schemas"
WEB_DIR = REPO_ROOT / "web"
SAMPLES_DIR = REPO_ROOT / "samples"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _temp_dir_is_writable() -> bool:
    """当前 ``tempfile`` 选中的临时目录是否真的可写。

    只探测、不创建残留文件。沙箱环境里 Python 会逐个试
    ``%TEMP%`` / ``%WINDOWS%\\Temp`` / ``c:\\temp``，全部不可写时**回退到当前工作目录**
    —— 那正是项目根目录被堆满 ``pip-build-tracker-*`` / ``pip-install-*`` 的原因。
    """
    import tempfile

    try:
        directory = Path(tempfile.gettempdir())
    except Exception:  # pragma: no cover
        return False
    probe = directory / ".aitsm-write-probe"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
    except OSError:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        # 回退到 cwd 时 gettempdir() 就是工作目录本身，判为不可用
        return directory.resolve() != Path.cwd().resolve()
    except OSError:  # pragma: no cover
        return True


def _redirect_temp_dir(root: Path) -> Path | None:
    """把 ``TEMP`` / ``TMP`` / ``TMPDIR`` 重定向到工作区缓存目录。

    为什么要做：Python 的 ``tempfile`` 在所有候选系统临时目录都不可写时，
    **会静默回退到当前工作目录**。pip 于是把 ``pip-build-tracker-*`` /
    ``pip-install-*`` / ``pip-ephem-wheel-cache-*`` 直接建在项目根，
    一次装依赖能留下上百个目录（本机实测 306 个 / 165 MB）。

    只在该回退**真的发生**时才重定向，避免覆盖用户正常的系统临时目录。
    """
    if _temp_dir_is_writable():
        return None
    target = root / "tmp"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - 只读工作区
        return None
    for env_name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[env_name] = str(target)
    # tempfile 会缓存首次解析结果，必须清掉才会用新环境变量
    try:
        import tempfile

        tempfile.tempdir = None
    except Exception:  # pragma: no cover
        pass
    return target


def prepare_cache_dirs(work_root: Path | None = None) -> Path:
    """把第三方库的缓存目录重定向到工作区。

    沙箱与部分企业环境下 ``%USERPROFILE%\\.cache`` 不可写，torch / huggingface /
    demucs 会因此直接抛 ``PermissionError``。这些库都在 import 时读取环境变量，
    所以必须在导入它们之前调用本函数（``aitsm/__init__.py`` 已保证这一点）。
    """
    root = (work_root or _env_path("AITSM_WORK_DIR", REPO_ROOT / "work")) / "cache"
    mapping = {
        "TORCH_HOME": root / "torch",
        "HF_HOME": root / "hf",
        "HUGGINGFACE_HUB_CACHE": root / "hf" / "hub",
        "XDG_CACHE_HOME": root / "xdg",
        "MPLCONFIGDIR": root / "matplotlib",
        "NUMBA_CACHE_DIR": root / "numba",
    }
    for env_name, path in mapping.items():
        if name_set := os.environ.get(env_name):
            # 用户显式指定时尊重用户设置
            if name_set:
                continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - 只读工作区时放弃重定向
            continue
        os.environ[env_name] = str(path)
    # Windows 上 huggingface_hub 无法创建符号链接，用复制模式并静音警告
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Xet 存储后端在本机会返回 401（CAS 重建鉴权失败），必须回退到普通 HTTP 下载
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # 系统临时目录不可写时（沙箱常见），Python 会回退到工作目录 →
    # pip 的 pip-build-tracker-* / pip-install-* 会堆在项目根
    _redirect_temp_dir(root)
    return root



@dataclass(slots=True)
class Settings:
    """一次进程运行使用的配置快照。"""

    work_root: Path = field(default_factory=lambda: _env_path("AITSM_WORK_DIR", REPO_ROOT / "work"))
    host: str = field(default_factory=lambda: os.environ.get("AITSM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("AITSM_PORT", 7863))

    # 上传限制
    max_upload_mb: int = field(default_factory=lambda: _env_int("AITSM_MAX_UPLOAD_MB", 120))
    max_duration_sec: float = field(default_factory=lambda: _env_float("AITSM_MAX_DURATION_SEC", 600.0))

    # 分析引擎
    separate_device: str = field(default_factory=lambda: os.environ.get("AITSM_DEVICE", "auto"))
    demucs_model: str = field(default_factory=lambda: os.environ.get("AITSM_DEMUCS_MODEL", "htdemucs"))
    enable_separation: bool = field(default_factory=lambda: _env_bool("AITSM_ENABLE_SEPARATION", True))
    whisper_model: str = field(default_factory=lambda: os.environ.get("AITSM_WHISPER_MODEL", "medium"))
    enable_lyrics: bool = field(default_factory=lambda: _env_bool("AITSM_ENABLE_LYRICS", True))
    enable_melody: bool = field(default_factory=lambda: _env_bool("AITSM_ENABLE_MELODY", True))
    lyrics_language: str = field(default_factory=lambda: os.environ.get("AITSM_LANGUAGE", "zh"))

    # 外部可执行文件
    ffmpeg: str = field(default_factory=lambda: os.environ.get("AITSM_FFMPEG", ""))
    ffprobe: str = field(default_factory=lambda: os.environ.get("AITSM_FFPROBE", ""))

    # 任务并发（分析阶段单任务串行，避免显存争抢）
    max_workers: int = field(default_factory=lambda: _env_int("AITSM_MAX_WORKERS", 1))
    keep_jobs: int = field(default_factory=lambda: _env_int("AITSM_KEEP_JOBS", 20))

    def job_dir(self, job_id: str) -> Path:
        return self.work_root / "jobs" / job_id

    def ensure_dirs(self) -> None:
        for path in (self.work_root, self.work_root / "jobs"):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def resolve_binary(name: str, override: str = "") -> str:
    """定位 ffmpeg/ffprobe 等外部可执行文件。

    顺序：显式覆盖 -> PATH -> Windows WinGet 链接目录 -> 常见安装位置。
    找不到时返回空字符串，由调用方决定如何降级。
    """
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(override)
        if found:
            return found
    found = shutil.which(name)
    if found:
        return found
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        guesses = [
            local / "Microsoft" / "WinGet" / "Links" / f"{name}.exe",
            local / "Microsoft" / "WinGet" / "Packages",
        ]
        for guess in guesses:
            if guess.is_file():
                return str(guess)
            if guess.is_dir():
                for match in guess.glob(f"**/bin/{name}.exe"):
                    return str(match)
    return ""


def check_optional_engines() -> dict[str, bool]:
    """探测可选重依赖是否可导入，供 API / CLI 在启动时给出明确提示。"""
    import importlib.util

    def has(*names: str) -> bool:
        return any(importlib.util.find_spec(name) is not None for name in names)

    return {
        "torch": has("torch"),
        "demucs": has("demucs"),
        # 本项目用 faster-whisper（CTranslate2），openai-whisper 在 py312/Windows 装不上
        "whisper": has("faster_whisper", "whisper"),
        "librosa": has("librosa"),
        "soundfile": has("soundfile"),
        # 中文歌词繁简统一用；缺失时歌词原样输出（繁体保留）
        "zhconv": has("zhconv"),
    }

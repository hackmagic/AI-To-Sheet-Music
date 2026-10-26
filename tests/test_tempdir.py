"""临时目录重定向的测试。

**为什么需要这个**：沙箱环境里 Python 的 ``tempfile`` 会在所有候选系统临时目录
都不可写时**静默回退到当前工作目录**。pip 于是把 ``pip-build-tracker-*`` /
``pip-install-*`` / ``pip-ephem-wheel-cache-*`` 直接建在项目根——
本机实测一次装依赖留下 306 个目录 / 165 MB。

``aitsm.config.prepare_cache_dirs()`` 现在会检测该回退并重定向到
``<work>/cache/tmp``（已被 gitignore 的 work/ 覆盖）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from aitsm.config import _redirect_temp_dir, _temp_dir_is_writable


@pytest.fixture()
def clean_temp_env(monkeypatch: pytest.MonkeyPatch):
    """保存/恢复 TEMP 系列变量与 tempfile 缓存，避免污染整个测试会话。"""
    keys = ("TEMP", "TMP", "TMPDIR")
    saved = {key: os.environ.get(key) for key in keys}
    saved_tempdir = tempfile.tempdir
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = saved_tempdir


# ---------------------------------------------------------------- 可写性探测


def test_writable_probe_returns_bool():
    assert isinstance(_temp_dir_is_writable(), bool)


def test_probe_does_not_leave_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env):
    """探测不能在工作目录里留残留文件。"""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = {item.name for item in tmp_path.iterdir()}
    _temp_dir_is_writable()
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_cwd_counts_as_not_writable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env):
    """tempfile 回退到 cwd 时（gettempdir() == cwd），必须判为不可用。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert _temp_dir_is_writable() is False


# ---------------------------------------------------------------- 重定向


def test_redirect_skips_when_system_temp_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env
):
    """系统临时目录正常可写时**不要**动它——不能覆盖用户环境。"""
    usable = tmp_path / "system-temp"
    usable.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(usable))
    before = os.environ.get("TEMP")
    assert _redirect_temp_dir(tmp_path / "cache") is None
    assert os.environ.get("TEMP") == before


def test_redirect_moves_temp_into_work_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env
):
    """探测到不可写时，TEMP/TMP/TMPDIR 全部指向 <work>/cache/tmp。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # 模拟回退到 cwd
    cacheroot = tmp_path / "cache"

    target = _redirect_temp_dir(cacheroot)

    assert target == cacheroot / "tmp"
    assert target.is_dir()
    for key in ("TEMP", "TMP", "TMPDIR"):
        assert Path(os.environ[key]) == target


def test_redirect_resets_tempfile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env
):
    """tempfile 缓存了旧路径，必须清掉才会用新环境变量。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _redirect_temp_dir(tmp_path / "cache")
    assert tempfile.tempdir is None


def test_after_redirect_mkdtemp_lands_in_work_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_temp_env
):
    """端到端：重定向之后 mkdtemp 必须落在 work/cache/tmp 里，而不是项目根。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cacheroot = tmp_path / "cache"
    _redirect_temp_dir(cacheroot)

    created = Path(tempfile.mkdtemp(prefix="pip-install-"))
    try:
        assert created.parent == cacheroot / "tmp"
        assert created.parent != tmp_path
    finally:
        created.rmdir()


def test_gitignore_covers_work_cache():
    """重定向目标必须在 gitignore 覆盖范围内，否则等于换个地方污染仓库。"""
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in ignored.splitlines()}
    assert "work/" in lines, "work/ 必须被忽略，临时目录才会落在忽略范围里"

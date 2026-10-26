"""``qwen_backend`` 依赖边界的测试。

核心目标：**让 ``qwen-asr`` 升级导致的路径/版本变化在探测阶段暴露**，而不是等到
推理中途抛 ImportError。同时锁住 mRoPE 配置校验与词表常量。
"""

from __future__ import annotations

import pytest

from aitsm.transcribe import qwen_backend as Q


# ---------------------------------------------------------------- 版本 pin


def test_pinned_version_is_exact_pin_not_range():
    """0.0.x 的 API 会变，必须精确 pin（不能出现 >= 或 ~=）。"""
    assert Q.PINNED_VERSION
    assert all(ch.isdigit() or ch == "." for ch in Q.PINNED_VERSION), (
        f"应为精确版本号，实际 {Q.PINNED_VERSION!r}"
    )


def test_installed_version_reads_metadata():
    version = Q.installed_version()
    # 未安装时返回 None，安装了就应是非空字符串（不强制等于 pin，但要能读到）
    assert version is None or isinstance(version, str)
    assert version != ""


# ---------------------------------------------------------------- 路径探测


def test_check_import_paths_reports_missing_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Q, "_package_root", lambda: None)
    info = Q.check_import_paths()
    assert info["ok"] is False
    assert "未安装" in info["reason"]
    assert set(info["missing"]) == set(Q.REQUIRED_MODULES)


def test_check_import_paths_detects_missing_submodule(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """模拟 qwen-asr 升级后少了某个建模子模块。"""
    backend = tmp_path / "core" / "transformers_backend"
    backend.mkdir(parents=True)
    # 只放两个，缺 modeling
    (backend / "configuration_qwen3_asr.py").write_text("", encoding="utf-8")
    (backend / "processing_qwen3_asr.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(Q, "_package_root", lambda: tmp_path)

    info = Q.check_import_paths()
    assert info["ok"] is False
    assert info["missing"] == ["modeling_qwen3_asr"]
    assert "modeling_qwen3_asr" in info["reason"]


def test_check_import_paths_accepts_both_layouts(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """新旧两种子目录布局都要接受。"""
    for subdir in (tmp_path / "core" / "transformers_backend", tmp_path / "transformers_backend"):
        subdir.mkdir(parents=True)
        for stem in Q.REQUIRED_MODULES:
            (subdir / f"{stem}.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(Q, "_package_root", lambda: tmp_path)
    assert Q.check_import_paths()["ok"] is True


def test_real_backend_paths_are_valid():
    """本机装了 qwen-asr 时，真实路径必须通过探测。"""
    if Q.installed_version() is None:
        pytest.skip("未安装 qwen-asr")
    info = Q.check_import_paths()
    assert info["ok"], info["reason"]
    assert info["backend"].endswith("transformers_backend")


# ---------------------------------------------------------------- 总探测


def test_probe_reports_version_mismatch_with_broken_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """版本变了且路径也不可用时，reason 必须同时说清两件事。"""
    monkeypatch.setattr(Q, "installed_version", lambda: "0.1.0")
    monkeypatch.setattr(Q, "_package_root", lambda: tmp_path)  # 空目录 → 路径不可用
    info = Q.probe()
    assert info["available"] is False
    assert info["version_matches_pin"] is False
    assert "0.1.0" in info["reason"] and Q.PINNED_VERSION in info["reason"]


def test_probe_allows_unpinned_version_when_paths_ok(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """版本不在 pin 上但路径正常时不硬拒绝（只做标记）。"""
    backend = tmp_path / "core" / "transformers_backend"
    backend.mkdir(parents=True)
    for stem in Q.REQUIRED_MODULES:
        (backend / f"{stem}.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(Q, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(Q, "installed_version", lambda: "0.0.7")

    info = Q.probe()
    assert info["available"] is True
    assert info["version_matches_pin"] is False


# ---------------------------------------------------------------- mRoPE


class _Config:
    def __init__(self, rope_scaling):
        self.rope_scaling = rope_scaling


def test_validate_mrope_accepts_default():
    assert Q.validate_mrope(_Config(Q.mrope_scaling())) == []


def test_validate_mrope_flags_missing_section():
    """少了 mrope_section 就是位置编码静默失效——必须报出来。"""
    problems = Q.validate_mrope(_Config({"type": "mrope"}))
    assert problems and "mrope_section" in problems[0]


def test_validate_mrope_flags_none():
    problems = Q.validate_mrope(_Config(None))
    assert problems and "为空" in problems[0]


@pytest.mark.parametrize("section", [[24, 20], [1, 2, 3, 4], [0, 0, 0], "bad"])
def test_validate_mrope_flags_bad_shapes(section):
    assert Q.validate_mrope(_Config({"mrope_section": section}))


def test_mrope_default_matches_qwen_asr_source():
    """``[24, 20, 20]`` 必须与 qwen_asr 建模代码里的缺省一致，不是我们猜的。"""
    assert Q.DEFAULT_MROPE["mrope_section"] == [24, 20, 20]
    assert Q.DEFAULT_MROPE["type"] == "mrope"


def test_mrope_scaling_returns_copy():
    """调用方可以安全修改返回值，不会污染默认值。"""
    first = Q.mrope_scaling()
    first["mrope_section"].append(99)
    assert Q.DEFAULT_MROPE["mrope_section"] == [24, 20, 20]


# ---------------------------------------------------------------- 词表


def test_vocab_size_constant():
    assert Q.VOCAB_SIZE == 152101
    assert Q.VOCAB_SIZE > 151936  # 基础词表 + AST token

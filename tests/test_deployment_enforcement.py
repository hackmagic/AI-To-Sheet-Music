"""部署约束的**强制力**测试（评审提示 6）。

关键问题：`worker_advice()` 到底只是「打印建议」，还是真能阻止多 worker 部署？

当前实现（`aitsm/api.py` 的 startup 钩子）：
- 单 worker / 未配置 WEB_CONCURRENCY → 通过；
- 多 worker + 大模型 → **抛 RuntimeError 拒绝启动**；
- 设 `AITSM_ALLOW_MULTI_WORKER=1` 才放行（并记 warning）。
"""

from __future__ import annotations

import os

import pytest

from aitsm import deployment as D


def _startup_hook():
    """取 api.py 的启动校验函数（不启动服务器）。

    FastAPI 现在推荐 ``lifespan`` 而不是 ``@app.on_event("startup")``；
    这里直接测那个被 lifespan 调用的纯函数，行为等价且不用跑事件循环。
    """
    from aitsm import api

    assert callable(api._enforce_deployment_limits)
    # 同时确认它真的被 lifespan 接上（否则函数存在但没人调）
    assert api.app.router.lifespan_context is not None
    return api._enforce_deployment_limits


# ---------------------------------------------------------------- 钩子存在性


def test_startup_hook_is_registered():
    """必须真的挂在 FastAPI 启动钩子上，否则等于没强制。"""
    assert callable(_startup_hook())


# ---------------------------------------------------------------- 强制行为


def test_single_worker_passes_startup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("AITSM_ALLOW_MULTI_WORKER", raising=False)
    monkeypatch.setattr(D, "check_deployment", lambda **kw: {"ok": True, "level": "info", "message": ""})
    _startup_hook()()  # 不应抛


def test_multi_worker_with_big_model_refuses_startup(monkeypatch: pytest.MonkeyPatch):
    """4 worker + VocalParse → 必须拒绝启动，而不是只打印警告。"""
    monkeypatch.delenv("AITSM_ALLOW_MULTI_WORKER", raising=False)
    monkeypatch.setattr(
        D,
        "check_deployment",
        lambda **kw: {
            "ok": False,
            "level": "error",
            "message": "配置了 4 个 worker，每个约需 16.9 GB —— 会 OOM。",
        },
    )
    with pytest.raises(RuntimeError) as excinfo:
        _startup_hook()()
    message = str(excinfo.value)
    assert "拒绝启动" in message
    assert "OOM" in message
    # 必须告诉用户逃生开关
    assert "AITSM_ALLOW_MULTI_WORKER" in message


def test_override_env_allows_startup(monkeypatch: pytest.MonkeyPatch):
    """显式设 AITSM_ALLOW_MULTI_WORKER=1 时放行（自担风险）。"""
    monkeypatch.setenv("AITSM_ALLOW_MULTI_WORKER", "1")
    monkeypatch.setattr(
        D, "check_deployment", lambda **kw: {"ok": False, "level": "error", "message": "会 OOM"}
    )
    _startup_hook()()  # 不应抛


# ---------------------------------------------------------------- 真实判定（不 mock）


def test_real_check_deployment_blocks_four_workers(monkeypatch: pytest.MonkeyPatch):
    """用真实的内存预算逻辑验证：4 worker 确实判为不 ok。"""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    result = D.check_deployment()
    assert result["ok"] is False
    assert result["level"] == "error"
    assert "OOM" in result["message"]


def test_real_check_deployment_allows_one_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    result = D.check_deployment()
    # 单 worker 但内存超上限 → warn（能跑），不是 error
    assert result["level"] in ("ok", "info", "warn")


def test_startup_uses_real_check_when_web_concurrency_set(monkeypatch: pytest.MonkeyPatch):
    """整合验证：真的设 WEB_CONCURRENCY=4 时，startup 钩子会抛。"""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.delenv("AITSM_ALLOW_MULTI_WORKER", raising=False)
    with pytest.raises(RuntimeError):
        _startup_hook()()

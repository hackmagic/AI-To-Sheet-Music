"""``/api/demo`` 与演示音频挂载的测试。

约定：交付产物 ``artifacts/score.json`` 里**不写**播放地址（产物不该绑定本地目录），
播放地址只在 HTTP 响应里补，且只有 ``work/jobs/demo/playback.mp3`` 真的存在时才补。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aitsm import api


def _make_demo_audio(tmp_path: Path) -> Path:
    jobs = tmp_path / "jobs" / "demo"
    jobs.mkdir(parents=True)
    playback = jobs / "playback.mp3"
    playback.write_bytes(b"ID3fake")
    return playback


def test_attach_sets_audio_url_when_playback_exists(tmp_path, monkeypatch):
    _make_demo_audio(tmp_path)
    monkeypatch.setattr(api.SETTINGS, "work_root", tmp_path)

    score = {"meta": {"title": "x"}}
    api._attach_demo_audio(score)

    assert score["meta"]["source"]["audio_url"] == "/jobs/demo/playback.mp3"


def test_attach_noop_without_playback(tmp_path, monkeypatch):
    monkeypatch.setattr(api.SETTINGS, "work_root", tmp_path)

    score = {"meta": {"title": "x"}}
    api._attach_demo_audio(score)

    assert "source" not in score["meta"]


def test_attach_keeps_existing_audio_url(tmp_path, monkeypatch):
    _make_demo_audio(tmp_path)
    monkeypatch.setattr(api.SETTINGS, "work_root", tmp_path)

    score = {"meta": {"source": {"audio_url": "/elsewhere.mp3"}}}
    api._attach_demo_audio(score)

    assert score["meta"]["source"]["audio_url"] == "/elsewhere.mp3"


def test_attach_creates_source_dict_when_meta_lacks_it(tmp_path, monkeypatch):
    _make_demo_audio(tmp_path)
    monkeypatch.setattr(api.SETTINGS, "work_root", tmp_path)

    score = {"meta": {}}
    api._attach_demo_audio(score)

    assert isinstance(score["meta"]["source"], dict)
    assert score["meta"]["source"]["audio_url"].endswith("playback.mp3")


def test_attach_ignores_non_dict_meta(tmp_path, monkeypatch):
    _make_demo_audio(tmp_path)
    monkeypatch.setattr(api.SETTINGS, "work_root", tmp_path)

    score = {"meta": "broken"}
    api._attach_demo_audio(score)  # 不应抛出

    assert score["meta"] == "broken"


@pytest.fixture()
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    return fastapi_testclient.TestClient(api.app)


def test_demo_endpoint_returns_score(client):
    res = client.get("/api/demo")
    assert res.status_code == 200
    body = res.json()
    assert "bars" in body and "meta" in body


def test_demo_sample_flag_returns_handwritten_sample(client):
    res = client.get("/api/demo?sample=1")
    assert res.status_code == 200
    body = res.json()
    assert len(body["bars"]) <= 20  # 手写样例 12 小节，真曲 88 小节


def test_demo_endpoint_does_not_pollute_delivery_file():
    """只有响应里带播放地址，产物文件本身不应被写脏。"""
    delivery = api.REPO_ROOT / "artifacts" / "score.json"
    if not delivery.is_file():
        pytest.skip("没有交付产物")
    import json

    data = json.loads(delivery.read_text(encoding="utf-8"))
    source = (data.get("meta") or {}).get("source") or {}
    assert not source.get("audio_url")

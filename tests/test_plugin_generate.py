"""plugin/scripts/generate.py 的回归测试。

**为什么有这个文件**：这个脚本在仓库外被复制走（安装到
``~/.agents/skills/audio-to-chord-sheet/scripts/``，DSH 0.2.x 用户级扫描根）后再运行，
``aitsm/`` 的单元测试完全覆盖不到它。曾经真实踩过一次：

``aitsm/service.py`` 把 ``manifest["stages"]["analysis"]`` 从「service 写的摘要 dict」
改成「pipeline 写的阶段记录 list」之后，``generate.py`` 里
``((manifest.get("stages") or {}).get("analysis") or {}).get("engines")``
在端到端跑完整首歌、全部产物都写完之后才崩在
``AttributeError: 'list' object has no attribute 'get'``
—— 用户拿到的是 ``{"ok": false}``，而 ``aitsm`` 的 479 个测试全绿。

所以这里锁三件事：
1. ``_manifest_engines`` 对新旧两种 manifest 结构都要能读；
2. ``stages.analysis`` 是 list 时，``collect_needs_review`` / ``build_report`` 不得抛异常；
3. 两个入口在「什么引擎信息都没有」的退化 manifest 上也要返回结果。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "plugin" / "scripts"


def _load_generate():
    """按路径加载 plugin/scripts/generate.py（不能当普通包导入）。"""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / "generate.py"
    spec = importlib.util.spec_from_file_location("plugin_generate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generate()


# --------------------------------------------------------------- 构造测试数据


def _stage_records() -> list[dict]:
    """pipeline 写的阶段记录（aitsm/pipeline/run.py 的真实形状）。"""
    return [
        {"name": "ingest", "elapsed_seconds": 1.2, "warnings": []},
        {"name": "separate", "elapsed_seconds": 147.4, "warnings": []},
        {"name": "beats", "elapsed_seconds": 3.1, "warnings": []},
        {"name": "chords", "elapsed_seconds": 8.8, "warnings": []},
    ]


def _tools() -> dict:
    return {
        "python": "3.12.8",
        "numpy": "2.5.4",
        "torch": "2.14.1+cpu",
        "demucs": "4.1.0",
        "ffmpeg": "ffmpeg version 9.0.1",
        "lyrics": {"available": False, "engine": "none", "line_count": 0},
        "melody": {"available": True, "engine": "pyin", "note_count": 37},
    }


def _manifest_with_list_stages() -> dict:
    """service 合并之后的 manifest：stages.analysis 是 list。"""
    return {
        "schema": "aitsm.manifest/v1",
        "generator": "aitsm 0.1.0",
        "options": {"no_lyrics": True},
        "tools": _tools(),
        "stages": {
            "analysis": _stage_records(),
            "analysis_summary": {"duration_sec": 37.5, "warnings": []},
            "arrange": {"elapsed_seconds": 0.4, "warnings": []},
            "export": {"elapsed_seconds": 1.1, "warnings": [], "outputs": {}},
        },
    }


def _manifest_old_service_summary() -> dict:
    """老版 service 写的 manifest：stages.analysis 是摘要 dict。"""
    return {
        "schema": "aitsm.manifest/v1",
        "stages": {
            "analysis": {
                "duration_sec": 37.5,
                "engines": {
                    "lyrics": {"available": True, "engine": "faster-whisper", "line_count": 18},
                    "melody": {"available": True, "engine": "pyin", "note_count": 120},
                    "torch": "2.14.1+cpu",
                },
            }
        },
    }


def _score() -> dict:
    """最小可用 score：vocal 有歌词、guitar 有 hits，避免报告层崩在空数据上。"""
    return {
        "schema": "aitsm.score/v1",
        "meta": {
            "title": "回归测试曲",
            "key": {"tonic": "C", "mode": "major", "fifths": 0},
            "tempo": 96.0,
            "time_signature": [4, 4],
            "capo": 0,
            "difficulty": "elementary",
        },
        "quality": {"overall": 0.8, "warnings": [], "stages": {"chords": 0.8}},
        "sections": [
            {"id": "s1", "label": "主歌", "kind": "verse", "start_bar": 1, "end_bar": 4,
             "start_sec": 0.0, "end_sec": 10.0, "repeat_of": None},
        ],
        "bars": [
            {
                "index": i,
                "start_sec": (i - 1) * 2.5,
                "end_sec": i * 2.5,
                "time_signature": [4, 4],
                "chords": [{"beat": 0, "duration": 4.0, "symbol": "C"}],
                "guitar": {"pattern_id": "pop8", "hits": []},
                "vocal": [{"beat": 0.0, "duration": 1.0, "midi": 60, "lyric": "你"}],
            }
            for i in range(1, 5)
        ],
    }


def _analysis() -> dict:
    return {
        "schema": "aitsm.analysis/v1",
        "meta": {"duration_sec": 10.0},
        "beats": {"bpm": 96.0, "beats": [], "time_signature": [4, 4]},
        "key": {"tonic": "C", "mode": "major", "confidence": 0.8},
        "chords": [{"start_sec": 0.0, "end_sec": 10.0, "symbol": "C", "confidence": 0.9}],
        "stems": {"engine": "demucs", "device": "cpu", "cached": True},
        "tools": _tools(),
    }


# ------------------------------------------------------- 1) _manifest_engines


def test_engines_from_tools_when_stages_analysis_is_list():
    """回归：service 合并后 stages.analysis 变成 list，必须退到 manifest.tools。"""
    engines = gen._manifest_engines(_manifest_with_list_stages(), _analysis())
    assert engines == _tools()
    assert engines["torch"] == "2.14.1+cpu"


def test_engines_from_old_service_summary_dict():
    engines = gen._manifest_engines(_manifest_old_service_summary(), None)
    assert engines["lyrics"]["engine"] == "faster-whisper"
    assert engines["lyrics"]["line_count"] == 18


def test_engines_from_new_service_analysis_summary():
    manifest = {
        "stages": {
            "analysis": _stage_records(),
            "analysis_summary": {"engines": {"lyrics": {"available": True, "engine": "faster-whisper"}}},
        }
    }
    engines = gen._manifest_engines(manifest, None)
    assert engines["lyrics"]["engine"] == "faster-whisper"


def test_engines_falls_back_to_analysis_tools():
    manifest = {"stages": {"analysis": _stage_records()}}
    engines = gen._manifest_engines(manifest, _analysis())
    assert engines["melody"]["note_count"] == 37


def test_engines_returns_empty_dict_when_nothing_available():
    """任何字段都没有时必须返回空 dict 而不是 None / 抛异常。"""
    assert gen._manifest_engines({}, None) == {}
    assert gen._manifest_engines(None, None) == {}
    assert gen._manifest_engines({"stages": []}, None) == {}
    assert gen._manifest_engines({"stages": {"analysis": _stage_records()}}, {}) == {}


# ------------------------------------- 2) 端到端不崩（这正是线上炸的那一处）


@pytest.mark.parametrize(
    "manifest_factory",
    [
        _manifest_with_list_stages,
        _manifest_old_service_summary,
        lambda: {"stages": {"analysis": _stage_records()}},
        lambda: {},
    ],
    ids=["service-merged", "old-service-summary", "list-only", "empty"],
)
def test_collect_needs_review_never_raises(manifest_factory):
    """全部产物都写完之后才崩，是最贵的失败；这里锁住不抛异常。"""
    items = gen.collect_needs_review(_score(), manifest_factory(), _analysis())
    assert isinstance(items, list)
    for item in items:
        # 每一项都必须是可交付的结构，缺字段会让 agent 无法转述位置
        assert set(item) >= {"severity", "code", "what", "action"}
        assert item["severity"] in {"high", "medium", "low"}
        assert item["code"] and item["what"] and item["action"]


@pytest.mark.parametrize(
    "manifest_factory",
    [
        _manifest_with_list_stages,
        _manifest_old_service_summary,
        lambda: {},
    ],
    ids=["service-merged", "old-service-summary", "empty"],
)
def test_build_report_never_raises(tmp_path, manifest_factory):
    manifest = manifest_factory()
    needs_review = gen.collect_needs_review(_score(), manifest, _analysis())
    text = gen.build_report(_score(), manifest, _analysis(), tmp_path, needs_review)
    assert isinstance(text, str) and len(text) > 200
    assert "引擎" in text or "降级" in text


# ------------------------------- 3) 关键内容确实写进了报告（不是空报告）


def test_report_mentions_engines_when_tools_present(tmp_path):
    """tools 里的引擎信息要真的出现在 report.md 的引擎小节里。"""
    manifest = _manifest_with_list_stages()
    needs_review = gen.collect_needs_review(_score(), manifest, _analysis())
    text = gen.build_report(_score(), manifest, _analysis(), tmp_path, needs_review)
    assert "2.14.1+cpu" in text
    assert "demucs" in text


def test_lyrics_none_is_flagged_high(tmp_path):
    """lyrics.available=false 时必须给 high 级别告警（红线：不隐藏降级）。"""
    manifest = _manifest_with_list_stages()
    items = gen.collect_needs_review(_score(), manifest, _analysis())
    lyrics = [i for i in items if i["code"] == "lyrics.none"]
    assert lyrics, f"应当有 lyrics.none 告警，实际告警码={[i['code'] for i in items]}"
    assert lyrics[0]["severity"] == "high"


def test_lyrics_present_does_not_flag_none():
    manifest = _manifest_with_list_stages()
    manifest["tools"] = dict(_tools())
    manifest["tools"]["lyrics"] = {"available": True, "engine": "faster-whisper", "line_count": 12}
    items = gen.collect_needs_review(_score(), manifest, _analysis())
    assert not [i for i in items if i["code"] == "lyrics.none"]


def test_report_json_serializable(tmp_path):
    """generate.py 的 stdout 必须是可 json.dumps 的对象。"""
    manifest = _manifest_with_list_stages()
    needs_review = gen.collect_needs_review(_score(), manifest, _analysis())
    payload = {
        "ok": True,
        "needs_review": needs_review,
        "needs_review_counts": {},
        "sections": _score()["sections"],
        "report_path": str(tmp_path / "report.md"),
    }
    json.dumps(payload, ensure_ascii=False)

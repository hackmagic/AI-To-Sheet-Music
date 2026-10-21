"""service 层的集成边界测试：选项归一 + manifest 合并。

这两块都是「静默失效」型 bug 的高发区：
- 选项名不匹配时不会报错，只是被忽略（``--no-lyrics`` 曾因此仍跑满 ASR）；
- manifest 同路径两次写盘时后写的会覆盖先写的，可复现性字段悄悄丢失。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aitsm.service import _merge_manifests, _normalize_options  # noqa: E402


# ---------------------------------------------------------------- 选项归一


def test_no_separation_maps_to_pipeline_keys():
    opts = _normalize_options({"no_separation": True})
    assert opts["separate"] is False
    assert opts["no_separate"] is True
    assert "no_separation" not in opts


def test_no_lyrics_actually_skips_asr():
    """pipeline 只认 skip_lyrics；只设 enable_lyrics 会让 ASR 白跑几分钟。"""
    opts = _normalize_options({"no_lyrics": True})
    assert opts["skip_lyrics"] is True
    assert opts["enable_lyrics"] is False


def test_explicit_skip_lyrics_is_preserved():
    opts = _normalize_options({"skip_lyrics": True})
    assert opts["skip_lyrics"] is True


def test_normalize_options_is_pure():
    original = {"no_lyrics": True}
    _normalize_options(original)
    assert original == {"no_lyrics": True}


# ---------------------------------------------------------------- manifest 合并


def _pipeline_manifest() -> dict:
    return {
        "schema": "aitsm.manifest/v1",
        "created_at": "2026-01-01T00:00:00+08:00",
        "input": {"sha256": "abc123", "size_bytes": 4096, "duration": 229.03},
        "options": {"separate": True},
        "stages": [{"name": "ingest", "elapsed_seconds": 1.2, "ok": True}],
        "stage_timings": {"ingest": 1.2},
        "degraded": [{"stage": "lyrics", "level": "warn", "code": "lyrics.low_conf"}],
        "outputs": {
            "analysis": "work/jobs/x/analysis.json",
            "analysis_sha256": "def456",
            "playback_audio": "work/jobs/x/playback.mp3",
            "stems": {"vocals": "work/jobs/x/stems/vocals.wav"},
        },
        "tools": {"demucs": "4.1.0"},
        "quality": {"overall": 0.829},
        "schema_validation": {"valid": True, "errors": []},
        "total_elapsed_seconds": 70.1,
    }


def _integration_manifest() -> dict:
    return {
        "schema": "aitsm.manifest/v1",
        "generator": {"name": "AI-To-Sheet-Music", "version": "0.1.0"},
        "options": {"separate": True, "skip_lyrics": False},
        "started_at": 1770000000.0,
        "stages": {
            "arrange": {"duration_sec": 0.4},
            "export": {"outputs": {"musicxml": "a.musicxml"}, "errors": {}},
        },
        "warnings": [{"level": "warn", "code": "structure.coarse", "message": "段落偏长", "stage": "analyzing"}],
        "duration_sec": 75.5,
        "outputs": {"analysis": "work/jobs/x/analysis.json", "score": "work/jobs/x/score.json"},
    }


def test_merge_keeps_reproducibility_fields():
    merged = _merge_manifests(_pipeline_manifest(), _integration_manifest())
    # 这些字段以前会被 service 覆盖掉
    assert merged["input"]["sha256"] == "abc123"
    assert merged["tools"] == {"demucs": "4.1.0"}
    assert merged["quality"]["overall"] == 0.829
    assert merged["schema_validation"]["valid"] is True
    assert merged["stage_timings"] == {"ingest": 1.2}
    assert merged["outputs"]["stems"]["vocals"].endswith("vocals.wav")
    assert merged["degraded"][0]["code"] == "lyrics.low_conf"


def test_merge_keeps_integration_fields():
    merged = _merge_manifests(_pipeline_manifest(), _integration_manifest())
    assert merged["generator"]["version"] == "0.1.0"
    assert merged["duration_sec"] == 75.5
    assert merged["outputs"]["score"].endswith("score.json")
    assert merged["outputs"]["musicxml"] == "a.musicxml"


def test_merge_stages_preserves_analysis_list():
    merged = _merge_manifests(_pipeline_manifest(), _integration_manifest())
    assert [s["name"] for s in merged["stages"]["analysis"]] == ["ingest"]
    assert merged["stages"]["arrange"]["duration_sec"] == 0.4
    assert merged["stages"]["export"]["outputs"]["musicxml"] == "a.musicxml"
    # service 层自己的摘要不能覆盖分析阶段列表
    assert "analysis_summary" not in merged["stages"] or isinstance(
        merged["stages"].get("analysis"), list
    )


def test_merge_dedupes_warnings():
    base = _pipeline_manifest()
    base["warnings"] = [{"level": "warn", "code": "c", "message": "m", "stage": "s"}]
    integration = _integration_manifest()
    integration["warnings"] = [{"level": "warn", "code": "c", "message": "m", "stage": "s"}]
    merged = _merge_manifests(base, integration)
    assert len(merged["warnings"]) == 1


def test_merge_without_base_manifest_still_works():
    merged = _merge_manifests(None, _integration_manifest())
    assert merged["schema"] == "aitsm.manifest/v1"
    assert merged["stages"]["analysis"] == []
    assert merged["outputs"]["score"].endswith("score.json")
    # 没有 base 时用合并后的告警推出 degraded
    assert [d["code"] for d in merged["degraded"]] == ["structure.coarse"]


def test_merge_result_is_json_serialisable():
    merged = _merge_manifests(_pipeline_manifest(), _integration_manifest())
    json.dumps(merged, ensure_ascii=False)


def test_merge_does_not_mutate_inputs():
    base = _pipeline_manifest()
    integration = _integration_manifest()
    base_copy = json.loads(json.dumps(base))
    integration_copy = json.loads(json.dumps(integration))
    _merge_manifests(base, integration)
    assert base == base_copy
    assert integration == integration_copy

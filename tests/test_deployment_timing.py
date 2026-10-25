"""部署约束与耗时分布的测试。

对应评审风险 2（多 worker OOM）与建议点 10（耗时分布）。
"""

from __future__ import annotations

import pytest

from aitsm import deployment as D
from aitsm.pipeline.run import _timing_breakdown


# ---------------------------------------------------------------- 内存模型


def test_model_memory_all_enabled_is_above_single_worker_ceiling():
    """四层全开时必然超过「建议单 worker 上限」——这正是要提示的前提。"""
    report = D.model_memory_report()
    assert report["total_gb"] > 10
    assert set(report["models"]) == set(D.MODEL_MEMORY_GB)


def test_model_memory_respects_disabled_models():
    report = D.model_memory_report({"vocalparse": False, "sheetsage2": True, "duet": True})
    assert "vocalparse" not in report["models"]
    assert report["total_gb"] < D.MODEL_MEMORY_GB["vocalparse"]


def test_vocalparse_dominates_memory():
    """VocalParse 单独就占了大头，这条决定了「拆独立服务」的策略。"""
    assert D.MODEL_MEMORY_GB["vocalparse"] > 4 * D.MODEL_MEMORY_GB["sheetsage2"]


# ---------------------------------------------------------------- worker 建议


def test_default_worker_count_is_safe():
    """未配置 worker 时，uvicorn 默认单进程 → 安全。"""
    advice = D.worker_advice({"vocalparse": True}, workers=None)
    assert advice["recommended_workers"] == 1
    assert advice["level"] == "info"


def test_single_worker_with_all_models_is_warn_not_error():
    """单 worker 但总内存超上限：警告（能跑，但要留神），不是错误。"""
    advice = D.worker_advice(workers=1)
    assert advice["level"] == "warn"
    assert advice["report"]["fits_single_worker"] is False


def test_multi_worker_with_big_model_is_error():
    """`--workers 4` + VocalParse = 必然 OOM，必须报 error。"""
    advice = D.worker_advice({"vocalparse": True, "sheetsage2": False, "duet": False}, workers=4)
    assert advice["level"] == "error"
    assert advice["projected_total_gb"] > 30
    assert "独立推理服务" in advice["message"]


def test_multi_worker_with_small_models_is_ok():
    """只跑轻量模型时，多 worker 是允许的。"""
    advice = D.worker_advice({"vocalparse": False, "sheetsage2": False, "duet": True}, workers=3)
    assert advice["level"] == "ok"
    assert advice["recommended_workers"] == 3


def test_check_deployment_maps_level_to_ok():
    assert D.check_deployment({"vocalparse": True}, workers=4)["ok"] is False
    assert D.check_deployment({"duet": True}, workers=2)["ok"] is True


def test_runtime_flags_reads_web_concurrency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    assert D.runtime_flags()["workers"] == 4


def test_runtime_flags_tolerates_bad_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "not-a-number")
    assert D.runtime_flags()["workers"] is None


# ---------------------------------------------------------------- 耗时分布


def _records():
    return [
        {"name": "ingest", "elapsed_seconds": 2.0},
        {"name": "separate", "elapsed_seconds": 58.0},
        {"name": "sheetsage2", "elapsed_seconds": 45.0},
        {"name": "lyrics", "elapsed_seconds": 150.0},
        {"name": "melody", "elapsed_seconds": 60.0},
        {"name": "symbolic", "elapsed_seconds": 5.0},
        {"name": "assemble", "elapsed_seconds": 1.0},
    ]


def test_timing_breakdown_groups_stages_by_layer():
    data = _timing_breakdown(_records(), 321.0)
    assert data["by_layer_seconds"]["vocal"] == 210.0  # lyrics + melody
    assert data["by_layer_seconds"]["accompaniment"] == 45.0
    assert data["by_layer_seconds"]["separation"] == 58.0


def test_timing_breakdown_sorts_by_cost():
    data = _timing_breakdown(_records(), 321.0)
    layers = list(data["by_layer_seconds"])
    assert layers[0] == "vocal", "耗时最多的一层应排在最前"
    assert data["slowest_layer"] == "vocal"


def test_timing_breakdown_reports_share_and_hint():
    data = _timing_breakdown(_records(), 321.0)
    assert 0.5 < data["by_layer_share"]["vocal"] < 0.7
    assert "vocal" in data["gpu_hint"]
    assert "%" in data["gpu_hint"]


def test_timing_breakdown_handles_empty_records():
    data = _timing_breakdown([], 10.0)
    assert data["by_layer_seconds"] == {}
    assert data["slowest_layer"] == ""
    assert data["gpu_hint"]


def test_timing_breakdown_ignores_bad_records():
    data = _timing_breakdown(
        [{"name": "", "elapsed_seconds": 5}, {"elapsed_seconds": 3}, {"name": "ingest", "elapsed_seconds": None}],
        10.0,
    )
    assert data["by_layer_seconds"] == {"ingest": 0.0}


def test_timing_breakdown_unknown_stage_goes_to_other():
    data = _timing_breakdown([{"name": "mystery", "elapsed_seconds": 7.0}], 10.0)
    assert data["by_layer_seconds"] == {"other": 7.0}

"""service 层的渲染 payload 集成：符号契约 -> payload.json。

转录层（analysis/symbolic）与渲染层（payload）之间只允许有一条数据通路，
这里把这条通路钉死：符号契约优先读盘、缺失时重建、ABC 兜底不抛异常、
payload 十路数据齐全（MusicXML/ABC/timing/指法图）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aitsm.contracts import SymbolicScore  # noqa: E402
from aitsm.service import _abc_fallback, _symbolic_payload, build_render_payload  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _analysis() -> dict:
    return json.loads((ARTIFACTS / "analysis.json").read_text(encoding="utf-8"))


def _score() -> dict:
    return json.loads((ARTIFACTS / "score.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 符号契约取值


def test_symbolic_falls_back_to_analysis_when_no_file(tmp_path):
    symbolic = _symbolic_payload(_analysis(), tmp_path)
    assert symbolic is not None
    assert symbolic["schema"] == "aitsm.symbolic/v1"
    assert len(symbolic["melody"]) > 0 and len(symbolic["chords"]) > 0


def test_symbolic_prefers_written_contract(tmp_path):
    written = SymbolicScore.from_analysis(_analysis()).to_dict()
    written["meta"]["marker"] = "from-disk"
    (tmp_path / "symbolic.json").write_text(
        json.dumps(written, ensure_ascii=False), encoding="utf-8"
    )
    loaded = _symbolic_payload(_analysis(), tmp_path)
    assert loaded is not None
    assert loaded["meta"]["marker"] == "from-disk"


def test_symbolic_reads_path_recorded_in_analysis(tmp_path):
    symbolic = SymbolicScore.from_analysis(_analysis()).to_dict()
    nested = tmp_path / "sub"
    nested.mkdir()
    target = nested / "symbolic.json"
    target.write_text(json.dumps(symbolic, ensure_ascii=False), encoding="utf-8")
    analysis = _analysis()
    analysis["symbolic"] = {"available": True, "paths": {"json": str(target)}}
    loaded = _symbolic_payload(analysis, tmp_path)
    assert loaded is not None
    assert loaded["schema"] == "aitsm.symbolic/v1"


def test_symbolic_handles_corrupt_file(tmp_path):
    (tmp_path / "symbolic.json").write_text("{not json", encoding="utf-8")
    symbolic = _symbolic_payload(_analysis(), tmp_path)
    assert symbolic is not None and symbolic["schema"] == "aitsm.symbolic/v1"


# ---------------------------------------------------------------- ABC 兜底


def test_abc_fallback_from_score():
    abc = _abc_fallback(None, _score())
    assert abc is not None and abc.startswith("X:")


def test_abc_fallback_never_raises_on_garbage():
    assert _abc_fallback(None, {"bars": "not-a-list"}) is None
    assert _abc_fallback(None, {}) is None


# ---------------------------------------------------------------- payload 集成


def test_payload_has_all_render_channels(tmp_path):
    payload = build_render_payload(_analysis(), _score(), tmp_path)

    assert payload["schema"].startswith("aitsm.payload")
    assert "<score-partwise" in payload["musicxml"] or "<score-timewise" in payload["musicxml"]
    assert payload["abc"] and payload["abc"].startswith("X:")
    assert payload["timing"]["bars"], "timing.bars 不能为空，播放同步依赖它"
    assert payload["timing"]["notes"], "timing.notes 不能为空"
    assert payload["chord_diagrams"]["count"] > 0
    for key in ("title", "meta", "sections", "stats", "audio_url"):
        assert key in payload


def test_payload_serialises_to_json(tmp_path):
    payload = build_render_payload(_analysis(), _score(), tmp_path)
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == payload["schema"]


# ---------------------------------------------------------------- timing 独立产物


def test_timing_matches_render_layer_contract(tmp_path):
    """payload.timing 必须与 render.timing 的输出一致：它是单独落盘的 timing.json。"""
    from aitsm.render.timing import build_timing

    payload = build_render_payload(_analysis(), _score(), tmp_path)
    timing = build_timing(_score())
    assert payload["timing"] == json.loads(json.dumps(timing))
    assert len(timing["bars"]) == len(_score()["bars"])
    # 单调递增，播放器按顺序推进
    starts = [bar["start"] for bar in timing["bars"]]
    assert starts == sorted(starts)
    for bar in timing["bars"]:
        assert bar["end"] >= bar["start"]


def test_api_exposes_timing_route_and_export():
    from aitsm import api

    assert "timing" in api.EXPORT_FILES
    assert api.EXPORT_FILES["timing"][0] == "timing.json"
    paths = {getattr(route, "path", "") for route in api.app.routes}
    assert "/api/jobs/{job_id}/timing" in paths
    assert "/api/jobs/{job_id}/payload" in paths

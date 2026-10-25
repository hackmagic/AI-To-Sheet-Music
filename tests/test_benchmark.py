"""``aitsm.benchmark`` 多曲目基准的离线测试。

不跑真实音频：用注入的 ``runner`` 写两份最小产物，验证汇总、降级标记、
失败容忍与 Markdown 表格。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aitsm.benchmark import (
    SCHEMA,
    expand_sources,
    render_markdown,
    run_benchmark,
    summarize_item,
    write_report,
)

ANALYSIS = {
    "audio": {"duration": 203.4},
    "quality": {
        "overall": 0.66,
        "stages": {"beats": 0.8, "key": 0.75, "chords": 0.52, "melody": 0.9, "lyrics": 0.31},
    },
    "chords": [{"start": 0.0, "end": 2.0, "symbol": "C"}],
    "melody": {"notes": [{"start": 0.0, "end": 0.4, "midi": 60}]},
    "lyrics": {"lines": [{"start": 0.5, "end": 2.0, "text": "你好"}]},
    "tools": {"separate": {"requested": "htdemucs_ft", "engine": "demucs", "model": "htdemucs"}},
    "symbolic": {
        "provenance": [
            {"layer": "quantization", "engine": "hmm", "requested": "duet", "ok": True},
            {"layer": "separation", "engine": "demucs", "requested": "htdemucs_ft", "ok": True},
            {"layer": "vocal_transcription", "engine": "pyin", "requested": "VocalParse", "ok": False},
        ],
        "degraded_layers": ["vocal_transcription", "quantization"],
    },
}
SCORE = {"bars": [{"index": 1}, {"index": 2}, {"index": 3}], "sections": [{"label": "s1"}]}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _fake_runner(source_path: Path, *, out_dir: Path, options: dict) -> dict:
    analysis_path = _write(out_dir / "analysis.json", ANALYSIS)
    score_path = _write(out_dir / "score.json", SCORE)
    return {
        "analysis_path": analysis_path,
        "score_path": score_path,
        "manifest_path": out_dir / "manifest.json",
        "exports": {},
        "warnings": [{"level": "info", "code": "x", "message": "y"}],
        "stats": {"bars": 3, "sections": 1, "duration_sec": 203.4},
    }


def test_expand_sources_collects_audio_from_directories(tmp_path: Path) -> None:
    folder = tmp_path / "songs"
    folder.mkdir()
    (folder / "b.flac").write_bytes(b"")
    (folder / "a.mp3").write_bytes(b"")
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")
    missing = tmp_path / "missing.wav"

    paths = expand_sources([folder, folder, missing])

    # 目录只收一层、非音频跳过、重复目录只展开一次，缺失路径保留给报告
    assert [p.name for p in paths] == ["a.mp3", "b.flac", "missing.wav"]


def test_summarize_item_reads_layers_and_counts(tmp_path: Path) -> None:
    analysis_path = _write(tmp_path / "analysis.json", ANALYSIS)
    score_path = _write(tmp_path / "score.json", SCORE)

    item = summarize_item(
        tmp_path / "曲目.mp3",
        ok=True,
        elapsed_seconds=12.345,
        result={
            "analysis_path": analysis_path,
            "score_path": score_path,
            "warnings": [{}, {}],
            "stats": {"bars": 3, "sections": 1, "duration_sec": 203.4},
        },
    )

    assert item["name"] == "曲目"
    assert item["elapsed_seconds"] == 12.35
    assert item["duration_sec"] == 203.4
    # 层按架构顺序排好，不按 JSON 里的出现顺序
    assert [layer["layer"] for layer in item["layers"]] == [
        "separation",
        "vocal_transcription",
        "quantization",
    ]
    assert item["degraded_layers"] == ["vocal_transcription", "quantization"]
    assert item["quality"]["overall"] == pytest.approx(0.66)
    assert item["counts"] == {
        "bars": 3,
        "sections": 1,
        "chords": 1,
        "melody_notes": 1,
        "lyric_lines": 1,
    }
    assert item["warnings"] == 2


def test_summarize_item_without_result_is_failure(tmp_path: Path) -> None:
    item = summarize_item(tmp_path / "x.mp3", ok=False, elapsed_seconds=0.0, error="找不到文件")

    assert item["ok"] is False
    assert item["error"] == "找不到文件"
    assert item["layers"] == []
    assert item["counts"]["bars"] == 0


def test_run_benchmark_aggregates_and_tolerates_failures(tmp_path: Path) -> None:
    good_a = tmp_path / "a.mp3"
    good_b = tmp_path / "b.mp3"
    good_a.write_bytes(b"")
    good_b.write_bytes(b"")
    missing = tmp_path / "missing.mp3"

    report = run_benchmark(
        [good_a, good_b, missing],
        out_dir=tmp_path / "bench",
        options={"duration_sec": 30},
        runner=_fake_runner,
    )

    assert report["schema"] == SCHEMA
    assert report["options"] == {"duration_sec": 30}
    assert [item["ok"] for item in report["items"]] == [True, True, False]
    summary = report["summary"]
    assert summary["count"] == 3
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert summary["mean_overall"] == pytest.approx(0.66)
    assert summary["degraded_counts"] == {"vocal_transcription": 2, "quantization": 2}

    # 每首一个独立子目录，名字带序号
    assert (tmp_path / "bench" / "01-a" / "analysis.json").is_file()
    assert (tmp_path / "bench" / "02-b" / "score.json").is_file()

    outputs = report["outputs"]
    assert Path(outputs["json"]).is_file()
    assert Path(outputs["markdown"]).is_file()
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "曲目明细" in markdown
    assert "pyin ⚠" in markdown
    assert "找不到文件" in markdown


def test_render_markdown_lists_engines_and_degraded_layers(tmp_path: Path) -> None:
    analysis_path = _write(tmp_path / "analysis.json", ANALYSIS)
    report = {
        "created_at": "2026-10-24T01:00:00+08:00",
        "items": [
            summarize_item(
                tmp_path / "歌.mp3",
                ok=True,
                elapsed_seconds=61.0,
                result={"analysis_path": analysis_path, "stats": {"bars": 3, "sections": 1}},
            )
        ],
        "summary": {
            "count": 1,
            "succeeded": 1,
            "failed": 0,
            "mean_overall": 0.66,
            "degraded_counts": {"vocal_transcription": 1},
            "total_elapsed_seconds": 61.0,
        },
        "environment": {"separation": {"onnx": False, "demucs": True}, "vocalparse": False},
        "options": {"duration_sec": 30},
    }

    markdown = render_markdown(report)

    assert "# 转录基准" in markdown
    assert "平均总分：66.0 / 100" in markdown
    assert "separation：demucs" in markdown
    assert "vocalparse：不可用" in markdown
    assert "| 歌 | 3:23.4 | 1:01.0 |" in markdown
    assert "人声 ×1" in markdown
    assert "pyin ⚠" in markdown
    assert "demucs ⚠" in markdown


def test_write_report_creates_both_files(tmp_path: Path) -> None:
    report = {"schema": SCHEMA, "items": [], "summary": {}}

    paths = write_report(report, tmp_path / "out")

    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["schema"] == SCHEMA
    assert "曲目明细" in Path(paths["markdown"]).read_text(encoding="utf-8")

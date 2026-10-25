"""多曲目转录基准：把多次 ``run_job`` 的产物汇总成可对比的表。

用途：等 VocalParse / SheetSage2 / DuET / htdemucs_ft 权重到位后，用同一批
曲目跑一遍，看每层的 ``requested`` 与 ``engine`` 是否真的切到了专用模型，
以及质量分有没有随之上升。

设计约束：

* **不做任何音频处理**，只调用 ``aitsm.service.run_job`` 并汇总产物；
* 单曲失败不影响整批（失败信息进表，不中断）；
* 结果同时写 ``benchmark.json``（机器读）与 ``benchmark.md``（人读）。

用法::

    python -m aitsm.cli bench songs/ --out work/bench
    python -m aitsm.cli bench a.mp3 b.flac --duration 30 --out work/bench
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = ["SCHEMA", "run_benchmark", "summarize_item", "render_markdown", "write_report"]

SCHEMA = "aitsm.benchmark/v1"

AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
}

# 表格里层顺序固定，和文档里的架构图一致
_LAYER_ORDER = (
    "separation",
    "vocal_transcription",
    "accompaniment_transcription",
    "fusion",
    "quantization",
)

_LAYER_LABELS = {
    "separation": "分离",
    "vocal_transcription": "人声",
    "accompaniment_transcription": "伴奏",
    "fusion": "融合",
    "quantization": "量化",
}

#: 架构首选组件 -> 可接受的 engine 别名（与 symbolic_stage 的降级判定保持一致）
_ENGINE_ALIASES: dict[str, set[str]] = {
    "htdemucs_ft": {"htdemucs_ft", "demucs-onnx"},
    "vocalparse": {"vocalparse"},
    "sheetsage2": {"sheetsage2"},
    "duet": {"duet"},
    "fusion.skeleton": {"rule-skeleton", "skeleton"},
}

_QUALITY_LABELS = (
    ("beats", "节拍"),
    ("key", "调性"),
    ("chords", "和弦"),
    ("melody", "旋律"),
    ("lyrics", "歌词"),
)

_ROW_BEGIN = "| "
_ROW_END = " |"


# ---------------------------------------------------------------- 输入展开


def expand_sources(paths: Iterable[str | Path]) -> list[Path]:
    """把文件/目录列表展开成音频文件列表（目录只收一层，按名字排序）。"""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates: Sequence[Path] = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            )
        elif path.is_file():
            candidates = [path]
        else:
            # 不存在的路径也保留，让报告里明确写「找不到」
            candidates = [path]
        for item in candidates:
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(item)
    return out


# ---------------------------------------------------------------- 汇总


def _load_json(path: Any) -> dict[str, Any] | None:
    if not isinstance(path, (str, Path)):
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _layers(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """从符号契约的 provenance 里取每层的 requested/engine/ok。"""
    symbolic = analysis.get("symbolic")
    if not isinstance(symbolic, dict):
        return []
    provenance = symbolic.get("provenance")
    if not isinstance(provenance, list):
        return []
    layers: list[dict[str, Any]] = []
    for entry in provenance:
        if not isinstance(entry, dict):
            continue
        layer = str(entry.get("layer") or "").strip()
        if not layer:
            continue
        layers.append(
            {
                "layer": layer,
                "requested": str(entry.get("requested") or ""),
                "engine": str(entry.get("engine") or ""),
                "ok": bool(entry.get("ok", True)),
            }
        )
    # 分离层的 provenance 不在符号契约里（它先于符号阶段），从 tools 补回来
    if not any(item["layer"] == "separation" for item in layers):
        tools = analysis.get("tools")
        separation = tools.get("separate") if isinstance(tools, dict) else None
        if isinstance(separation, dict):
            layers.append(
                {
                    "layer": "separation",
                    "requested": str(separation.get("requested") or ""),
                    "engine": str(separation.get("engine") or ""),
                    "ok": True,
                }
            )
    order = {name: index for index, name in enumerate(_LAYER_ORDER)}
    layers.sort(key=lambda item: order.get(item["layer"], len(order)))
    return layers


def _quality(analysis: dict[str, Any], score: dict[str, Any] | None) -> dict[str, Any]:
    quality = analysis.get("quality")
    if not isinstance(quality, dict) and isinstance(score, dict):
        quality = score.get("quality")
    if not isinstance(quality, dict):
        return {}
    stages = quality.get("stages")
    stages = dict(stages) if isinstance(stages, dict) else {}
    overall = quality.get("overall")
    return {
        "overall": float(overall) if isinstance(overall, (int, float)) else None,
        "stages": {str(k): float(v) for k, v in stages.items() if isinstance(v, (int, float))},
    }


def _counts(analysis: dict[str, Any], score: dict[str, Any] | None, result: dict[str, Any]) -> dict[str, int]:
    melody = analysis.get("melody")
    notes = melody.get("notes") if isinstance(melody, dict) else None
    lyrics = analysis.get("lyrics")
    lines = lyrics.get("lines") if isinstance(lyrics, dict) else None
    chords = analysis.get("chords")
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    bars = stats.get("bars")
    if not isinstance(bars, int) and isinstance(score, dict) and isinstance(score.get("bars"), list):
        bars = len(score["bars"])
    return {
        "bars": int(bars or 0),
        "sections": int(stats.get("sections") or 0),
        "chords": len(chords) if isinstance(chords, list) else 0,
        "melody_notes": len(notes) if isinstance(notes, list) else 0,
        "lyric_lines": len(lines) if isinstance(lines, list) else 0,
    }


def _duration(analysis: dict[str, Any], result: dict[str, Any]) -> float | None:
    audio = analysis.get("audio")
    if isinstance(audio, dict):
        value = audio.get("duration") or audio.get("duration_sec")
        if isinstance(value, (int, float)):
            return round(float(value), 3)
    stats = result.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("duration_sec"), (int, float)):
        return round(float(stats["duration_sec"]), 3)
    return None


def summarize_item(
    source: str | Path,
    *,
    ok: bool,
    elapsed_seconds: float,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """把单次 run_job 的结果压成一行报告数据（纯函数，不读盘以外的副作用）。"""
    path = Path(source)
    result = result if isinstance(result, dict) else {}
    analysis = _load_json(result.get("analysis_path")) or {}
    score = _load_json(result.get("score_path")) or {}
    layers = _layers(analysis)
    symbolic = analysis.get("symbolic") if isinstance(analysis.get("symbolic"), dict) else {}
    degraded = symbolic.get("degraded_layers") if isinstance(symbolic, dict) else None
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []

    return {
        "name": path.stem or path.name,
        "source": str(path),
        "ok": bool(ok),
        "error": error,
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "duration_sec": _duration(analysis, result),
        "layers": layers,
        "degraded_layers": [str(item) for item in degraded] if isinstance(degraded, list) else [],
        "quality": _quality(analysis, score),
        "counts": _counts(analysis, score, result),
        "warnings": len(warnings),
    }


# ---------------------------------------------------------------- 执行


def run_benchmark(
    sources: Iterable[str | Path],
    *,
    out_dir: str | Path,
    options: dict[str, Any] | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    progress: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """逐首跑完整流水线并汇总。

    ``runner`` 默认是 ``aitsm.service.run_job``；测试可注入假实现。
    每首曲子写到 ``<out_dir>/<序号>-<名字>/`` 下，互不覆盖。
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if runner is None:
        from .service import run_job as runner  # 延迟导入，导入本模块不拉 torch

    items: list[dict[str, Any]] = []
    paths = expand_sources(sources)
    total = len(paths)
    for index, source in enumerate(paths, start=1):
        if progress is not None:
            progress("bench", (index - 1) / max(1, total) * 100, f"{index}/{total} {source.name}")
        job_dir = out_root / f"{index:02d}-{source.stem or 'track'}"
        started = time.time()
        if not source.is_file():
            items.append(
                summarize_item(
                    source,
                    ok=False,
                    elapsed_seconds=0.0,
                    error="找不到文件",
                )
            )
            continue
        try:
            result = runner(source_path=source, out_dir=job_dir, options=dict(options or {}))
        except Exception as exc:  # noqa: BLE001 - 单曲失败不中断整批
            items.append(
                summarize_item(
                    source,
                    ok=False,
                    elapsed_seconds=time.time() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        items.append(
            summarize_item(
                source,
                ok=True,
                elapsed_seconds=time.time() - started,
                result=result,
            )
        )

    report = {
        "schema": SCHEMA,
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "options": dict(options or {}),
        "items": items,
        "summary": _summary(items),
        "environment": _environment(),
    }
    report["outputs"] = write_report(report, out_root)
    if progress is not None:
        progress("bench", 100.0, f"完成 {len(items)} 首")
    return report


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [item for item in items if item.get("ok")]
    overalls = [
        item["quality"]["overall"]
        for item in succeeded
        if isinstance(item.get("quality"), dict) and isinstance(item["quality"].get("overall"), (int, float))
    ]
    degraded_counts: dict[str, int] = {}
    for item in succeeded:
        for layer in item.get("degraded_layers") or []:
            degraded_counts[layer] = degraded_counts.get(layer, 0) + 1
    return {
        "count": len(items),
        "succeeded": len(succeeded),
        "failed": len(items) - len(succeeded),
        "mean_overall": round(sum(overalls) / len(overalls), 4) if overalls else None,
        "degraded_counts": degraded_counts,
        "total_elapsed_seconds": round(sum(item.get("elapsed_seconds") or 0 for item in items), 2),
    }


def _environment() -> dict[str, Any]:
    """记录关键引擎是否可用，避免「换机器就变」的基准无法解释。"""
    env: dict[str, Any] = {}
    try:
        from .separation import engine as separation_engine

        env["separation"] = separation_engine.available_backends()
    except Exception:  # noqa: BLE001
        env["separation"] = None
    for key, module_name in (
        ("vocalparse", "aitsm.transcribe.vocal"),
        ("sheetsage2", "aitsm.transcribe.accompaniment"),
        ("duet", "aitsm.fusion.quantize"),
    ):
        try:
            module = __import__(module_name, fromlist=["is_available"])
            env[key] = bool(module.is_available())
        except Exception:  # noqa: BLE001
            env[key] = None
    return env


# ---------------------------------------------------------------- 输出


def _fmt_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    seconds = float(value)
    minutes, rest = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}:{rest:04.1f}"
    return f"{rest:.1f}s"


def _fmt_score(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) * 100:.0f}"


def _layer_degraded(entry: dict[str, Any], degraded: set[str]) -> bool:
    """某层是否没跑到架构首选组件。"""
    if not entry.get("ok", True):
        return True
    if str(entry.get("layer") or "") in degraded:
        return True
    requested = str(entry.get("requested") or "").strip().lower()
    engine = str(entry.get("engine") or "").strip().lower()
    if not requested or not engine:
        return False
    accepted = _ENGINE_ALIASES.get(requested)
    if accepted is not None:
        return engine not in accepted
    return requested not in engine


def _layers_cell(item: dict[str, Any]) -> dict[str, str]:
    by_layer = {
        layer["layer"]: layer for layer in item.get("layers") or [] if isinstance(layer, dict)
    }
    degraded = {str(name) for name in item.get("degraded_layers") or []}
    cells: dict[str, str] = {}
    for layer in _LAYER_ORDER:
        entry = by_layer.get(layer)
        if entry is None:
            cells[layer] = "—"
            continue
        engine = entry.get("engine") or "?"
        cells[layer] = f"{engine} ⚠" if _layer_degraded(entry, degraded) else engine
    return cells


def render_markdown(report: dict[str, Any]) -> str:
    """人读表格：每首曲子的引擎、降级层、质量分与规模。"""
    items = report.get("items") if isinstance(report.get("items"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    env = report.get("environment") if isinstance(report.get("environment"), dict) else {}

    lines: list[str] = []
    title = "# 转录基准"
    lines.append(title)
    lines.append("")
    lines.append(f"- 生成时间：{report.get('created_at') or '—'}")
    lines.append(
        f"- 曲目：{summary.get('count', len(items))} 首"
        f"（成功 {summary.get('succeeded', 0)}，失败 {summary.get('failed', 0)}）"
    )
    mean = summary.get("mean_overall")
    if isinstance(mean, (int, float)):
        lines.append(f"- 平均总分：{float(mean) * 100:.1f} / 100")
    degraded = summary.get("degraded_counts") or {}
    if degraded:
        parts = ", ".join(f"{_LAYER_LABELS.get(k, k)} ×{v}" for k, v in sorted(degraded.items()))
        lines.append(f"- 降级层：{parts}")
    options = report.get("options") or {}
    if options:
        lines.append(f"- 参数：`{json.dumps(options, ensure_ascii=False, sort_keys=True)}`")
    lines.append("")

    lines.append("## 环境")
    lines.append("")
    for key, value in env.items():
        if isinstance(value, dict):
            available = [str(name) for name, ok in value.items() if ok]
            value = ", ".join(available) or "无可用后端"
        elif isinstance(value, list):
            value = ", ".join(str(item) for item in value) or "—"
        elif isinstance(value, bool):
            value = "可用" if value else "不可用"
        elif value is None:
            value = "未知"
        lines.append(f"- {key}：{value}")
    lines.append("")

    lines.append("## 曲目明细")
    lines.append("")
    header = [
        "曲目",
        "时长",
        "耗时",
        "分离",
        "人声",
        "伴奏",
        "量化",
        "总分",
        "节拍",
        "调性",
        "和弦",
        "旋律",
        "歌词",
        "小节",
        "降级",
    ]
    lines.append(_ROW_BEGIN + " | ".join(header) + _ROW_END)
    lines.append(_ROW_BEGIN + " | ".join("---" for _ in header) + _ROW_END)
    for item in items:
        layers = _layers_cell(item)
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        stages = quality.get("stages") if isinstance(quality.get("stages"), dict) else {}
        counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
        if not item.get("ok"):
            row = [
                item.get("name") or "—",
                _fmt_seconds(item.get("duration_sec")),
                _fmt_seconds(item.get("elapsed_seconds")),
                "失败",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                item.get("error") or "—",
            ]
            lines.append(_ROW_BEGIN + " | ".join(str(cell) for cell in row) + _ROW_END)
            continue
        degraded = ", ".join(
            _LAYER_LABELS.get(layer, layer) for layer in item.get("degraded_layers") or []
        )
        row = [
            item.get("name") or "—",
            _fmt_seconds(item.get("duration_sec")),
            _fmt_seconds(item.get("elapsed_seconds")),
            layers.get("separation", "—"),
            layers.get("vocal_transcription", "—"),
            layers.get("accompaniment_transcription", "—"),
            layers.get("quantization", "—"),
            _fmt_score(quality.get("overall")),
            _fmt_score(stages.get("beats")),
            _fmt_score(stages.get("key")),
            _fmt_score(stages.get("chords")),
            _fmt_score(stages.get("melody")),
            _fmt_score(stages.get("lyrics")),
            str(counts.get("bars", 0)),
            degraded or "—",
        ]
        lines.append(_ROW_BEGIN + " | ".join(str(cell) for cell in row) + _ROW_END)

    lines.append("")
    lines.append("> 分值 0–100，来自 analysis.json 的 quality（转录层自评，非人工标注）。")
    lines.append("> 「⚠」表示该层没跑到架构首选组件，engine 是本机实际使用的降级实现。")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """写 benchmark.json + benchmark.md，返回两个路径。"""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "benchmark.json"
    md_path = root / "benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

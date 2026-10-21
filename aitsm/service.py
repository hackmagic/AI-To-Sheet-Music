"""端到端服务编排：上传文件 -> 分析 -> 编配 -> 校验 -> 导出。

这一层是 Lead 的集成边界：
- 只依赖各模块的公共接口（见 docs/02-architecture.md 第 3 节），不触碰算法细节；
- 每个阶段都可单独失败并降级，失败信息写入 ``quality.warnings``；
- 任务产物统一落在 ``work/jobs/<id>/`` 下，包含 analysis.json / score.json /
  manifest.json 与导出文件，便于复现和排查。
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import SETTINGS
from .jobs import Job
from .schema import validate_analysis, validate_score

ProgressFn = Callable[[str, float, str], None]


# ---------------------------------------------------------------- 工具


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _warn(
    target: list[dict[str, Any]],
    level: str,
    code: str,
    message: str,
    stage: str,
) -> None:
    target.append({"level": level, "code": code, "message": message, "stage": stage})


def _merge_manifests(
    base: dict[str, Any] | None,
    integration: dict[str, Any],
) -> dict[str, Any]:
    """把 service 层的集成信息并入分析层写好的 manifest。

    分析层（pipeline/run.py）写出的 manifest 带 ``input.sha256`` / ``degraded`` /
    ``tools`` / ``quality`` / ``schema_validation`` 等可复现性字段；service 层再补
    ``stages.arrange`` / ``stages.export`` / ``outputs`` / 总耗时。以前两边各写一份
    同路径文件，service 覆盖后这些字段就丢了，这里改为合并。

    ``stages`` 在两个来源里形状不同（分析层是 list，service 层是 dict），统一成
    ``{"analysis": [...], "arrange": {...}, "export": {...}}``。
    """
    merged: dict[str, Any] = dict(base or {})
    merged.setdefault("schema", "aitsm.manifest/v1")

    analysis_stages = merged.get("stages")
    if isinstance(analysis_stages, dict):
        analysis_stages = analysis_stages.get("analysis") or []
    merged["stages"] = {
        "analysis": analysis_stages if isinstance(analysis_stages, list) else [],
        **{k: v for k, v in (integration.get("stages") or {}).items()},
    }

    # 分析层的 outputs（分析 json / 播放音源 / 分轨）与导出产物都要保留
    outputs = dict(merged.get("outputs") or {})
    # 导出产物同时登记在 stages.export.outputs 里，两边都收，避免调用方只填一处
    export_stage = (integration.get("stages") or {}).get("export")
    if isinstance(export_stage, dict) and isinstance(export_stage.get("outputs"), dict):
        outputs.update(export_stage["outputs"])
    outputs.update(integration.get("outputs") or {})
    merged["outputs"] = outputs

    # 告警去重合并：按 (code, stage, message) 判重
    combined: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for item in list(merged.get("warnings") or []) + list(integration.get("warnings") or []):
        if not isinstance(item, dict):
            continue
        key = (item.get("code"), item.get("stage"), item.get("message"))
        if key in seen:
            continue
        seen.add(key)
        combined.append(item)
    merged["warnings"] = combined

    # service 层独有的字段直接覆盖
    for key in ("generator", "options", "started_at", "duration_sec"):
        if key in integration:
            merged[key] = integration[key]

    # degraded 若分析层没写，则由合并后的告警推出来，保证下游有两种读法都成立
    if not merged.get("degraded"):
        merged["degraded"] = [
            {
                "stage": item.get("stage"),
                "level": item.get("level"),
                "code": item.get("code"),
                "message": item.get("message"),
            }
            for item in combined
            if item.get("level") in ("warn", "error")
        ]
    return merged


# ---------------------------------------------------------------- 主流程


def _normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    """把 API/CLI 暴露的选项名统一成 pipeline 认识的键。

    CLI 用 ``--no-separation``，API 用 ``no_separation``，而 pipeline 读的是
    ``separate`` / ``no_separate``。这里做一次归一，避免选项被静默忽略。
    """
    opts = dict(options or {})
    if opts.get("no_separation") or opts.get("no_separate"):
        opts["separate"] = False
        opts["no_separate"] = True
    if opts.get("no_lyrics"):
        # pipeline 只认 skip_lyrics；enable_lyrics 仅供编配层决定是否写歌词行。
        # 之前只设了 enable_lyrics，导致 --no-lyrics 仍会跑 3 分钟 ASR。
        opts.setdefault("enable_lyrics", False)
        opts["skip_lyrics"] = True
    if opts.get("no_melody"):
        opts.setdefault("enable_melody", False)
    opts.pop("no_separation", None)
    return opts


def run_job(
    *,
    source_path: Path,
    out_dir: Path,
    options: dict[str, Any],
    progress: ProgressFn | None = None,
    on_warning: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """执行一次完整的“音频 -> 弹唱谱”流程，返回产物路径与结果摘要。

    返回::

        {
          "analysis_path": str, "score_path": str, "manifest_path": str,
          "exports": {"musicxml": str, "midi": str, "json": str},
          "warnings": [...], "stats": {...}
        }
    """
    warnings: list[dict[str, Any]] = []
    options = _normalize_options(options)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # 这一份只是 service 层的增量；最终写盘前会与 pipeline 写的 manifest 合并
    manifest: dict[str, Any] = {
        "schema": "aitsm.manifest/v1",
        "generator": {"name": "AI-To-Sheet-Music", "version": __version__},
        "options": options,
        "started_at": started,
        "stages": {},
        "warnings": warnings,
    }

    def emit(stage: str, percent: float, message: str = "") -> None:
        if progress is not None:
            progress(stage, percent, message)

    def note(warning: dict[str, Any]) -> None:
        warnings.append(warning)
        if on_warning is not None:
            on_warning(warning)

    # ---- 1) 分析流水线（audio-core / melody-lyrics 负责具体算法） ----
    from .pipeline.run import run_pipeline

    emit("ingesting", 2.0, "解析音频")
    analysis: dict[str, Any] = run_pipeline(
        source_path,
        out_dir,
        options,
        progress=lambda stage, percent, message="": emit(stage, percent, message),
    )
    manifest["stages"]["analysis_summary"] = {
        "duration_sec": round(time.time() - started, 2),
        "engines": analysis.get("tools", {}),
    }
    for warning in analysis.get("warnings", []):
        if isinstance(warning, dict):
            note(warning)

    analysis_errors = validate_analysis(analysis)
    if analysis_errors:
        note(
            {
                "level": "warn",
                "code": "analysis.schema",
                "message": "analysis.json 结构不合规：" + "; ".join(analysis_errors[:3]),
                "stage": "analyzing",
            }
        )
    analysis_path = _write_json(out_dir / "analysis.json", analysis)
    manifest["stages"]["analysis_summary"]["output"] = analysis_path

    # ---- 2) 编配（score-engine） ----
    emit("arranging", 0.0, "生成弹唱谱")
    arrange_started = time.time()
    from .score.arrange import arrange

    score = arrange(analysis, options)
    manifest["stages"]["arrange"] = {"duration_sec": round(time.time() - arrange_started, 2)}

    # 原曲音源：浏览器同步播放用；路径由 API 层挂载为静态资源
    playback = analysis.get("audio", {}).get("playback_audio")
    if playback:
        try:
            relative = Path(playback).resolve().relative_to(out_dir.resolve())
            score.setdefault("meta", {}).setdefault("source", {})["audio_url"] = f"/jobs/{out_dir.name}/{relative.as_posix()}"
        except ValueError:
            # 音源不在任务目录内，交给 API 层补 URL
            pass

    score.setdefault("meta", {})["generator"] = {
        "version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engines": analysis.get("tools", {}),
    }

    # ---- 3) 校验（结构 + 音乐合理性） ----
    emit("validating", 10.0, "校验谱面")
    from .score.validate import validate_score as inspect_score

    report = inspect_score(score) if callable(inspect_score) else {}
    score.setdefault("quality", {})
    score["quality"].setdefault("warnings", [])
    for warning in (report.get("warnings") or []) + warnings:
        if isinstance(warning, dict):
            score["quality"]["warnings"].append(warning)
    if report.get("errors"):
        note(
            {
                "level": "warn",
                "code": "score.validation",
                "message": "谱面校验发现 " + str(len(report["errors"])) + " 个问题",
                "stage": "validating",
            }
        )
    score_errors = validate_score(score)
    if score_errors:
        note(
            {
                "level": "warn",
                "code": "score.schema",
                "message": "score.json 结构不合规：" + "; ".join(score_errors[:3]),
                "stage": "validating",
            }
        )

    score_path = _write_json(out_dir / "score.json", score)

    # ---- 4) 导出 ----
    emit("validating", 60.0, "导出 MusicXML / MIDI")
    exports: dict[str, str] = {"json": score_path}
    export_errors: dict[str, str] = {}
    try:
        from .score.export.musicxml import export_musicxml

        exports["musicxml"] = str(out_dir / "score.musicxml")
        export_musicxml(score, out_dir / "score.musicxml")
    except Exception as exc:  # noqa: BLE001 - 导出失败不阻断主交付
        export_errors["musicxml"] = f"{type(exc).__name__}: {exc}"
        exports.pop("musicxml", None)
    try:
        from .score.export.midi import export_midi

        exports["midi"] = str(out_dir / "score.mid")
        export_midi(score, out_dir / "score.mid")
    except Exception as exc:  # noqa: BLE001
        export_errors["midi"] = f"{type(exc).__name__}: {exc}"
        exports.pop("midi", None)
    for fmt, error in export_errors.items():
        note(
            {
                "level": "warn",
                "code": f"export.{fmt}",
                "message": f"{fmt} 导出失败：{error}",
                "stage": "validating",
            }
        )
    manifest["stages"]["export"] = {"outputs": exports, "errors": export_errors}

    # ---- 5) 收尾 ----
    manifest["duration_sec"] = round(time.time() - started, 2)
    manifest["warnings"] = warnings
    manifest["outputs"] = {
        "analysis": analysis_path,
        "score": score_path,
        **exports,
    }
    # pipeline 已经把可复现性字段（input.sha256/degraded/tools/quality/
    # schema_validation/stage_timings/分轨路径）写进同一个 manifest.json，
    # 直接覆盖会把它们全丢掉 —— 先读回来合并。
    pipeline_manifest: dict[str, Any] | None = None
    pipeline_manifest_path = out_dir / "manifest.json"
    if pipeline_manifest_path.is_file():
        try:
            pipeline_manifest = _read_json(pipeline_manifest_path)
        except (OSError, json.JSONDecodeError):
            pipeline_manifest = None
    manifest = _merge_manifests(pipeline_manifest, manifest)
    manifest_path = _write_json(out_dir / "manifest.json", manifest)
    emit("validating", 100.0, "完成")

    return {
        "analysis_path": analysis_path,
        "score_path": score_path,
        "manifest_path": manifest_path,
        "exports": exports,
        "warnings": warnings,
        "stats": {
            "bars": len(score.get("bars") or []),
            "sections": len(score.get("sections") or []),
            "duration_sec": manifest["duration_sec"],
        },
    }


def run_job_for_job_record(job: Job, source_path: Path) -> dict[str, Any]:
    """供 JobManager 调用的适配器：把进度与告警回写到 Job。"""
    manager = None
    try:
        from .jobs import JOBS as manager  # 延迟导入，避免循环依赖
    except ImportError:  # pragma: no cover
        manager = None

    progress = manager.progress_callback(job) if manager is not None else None
    on_warning = (lambda w: manager.add_warning(job, w)) if manager is not None else None

    try:
        result = run_job(
            source_path=source_path,
            out_dir=job.dir,
            options=job.options,
            progress=progress,
            on_warning=on_warning,
        )
    except Exception:
        if manager is not None:
            manager.log_line(job, traceback.format_exc())
        raise

    if manager is not None:
        for key, value in result["exports"].items():
            manager.set_artifact(job, key, value)
        manager.set_artifact(job, "analysis", result["analysis_path"])
        manager.set_artifact(job, "manifest", result["manifest_path"])
        # 播放音源的位置由分析层决定（可能是 audio/playback.mp3，也可能直接在任务根目录），
        # 这里按实际文件存在性登记，不要硬编码子目录。
        for candidate in (job.dir / "audio" / "playback.mp3", job.dir / "playback.mp3"):
            if candidate.is_file():
                manager.set_artifact(job, "playback", candidate)
                break
    return result


def load_artifact(job: Job, name: str) -> Any:
    """读取任务产物 JSON。"""
    path = job.dir / name
    if not path.is_file():
        return None
    return _read_json(path)


__all__ = ["run_job", "run_job_for_job_record", "load_artifact", "SETTINGS"]

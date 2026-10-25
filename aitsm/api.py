"""FastAPI 服务：上传音频、创建任务、查询进度、取产物、导出与静态站点。

约定（详见 docs/02-architecture.md）：
- 任务创建后立即返回 job_id，前端轮询 ``GET /api/jobs/{id}``；
- 所有产物通过 ``/jobs/{job_id}/<file>`` 静态挂载，避免额外下载接口；
- 单个进程内串行分析，避免 GPU 争抢。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import REPO_ROOT, SAMPLES_DIR, SETTINGS, WEB_DIR, check_optional_engines
from .jobs import JOBS, Job
from .service import run_job_for_job_record

SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv"}
SUPPORTED = SUPPORTED_AUDIO | SUPPORTED_VIDEO

app = FastAPI(title="AI-To-Sheet-Music", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SETTINGS.ensure_dirs()
app.mount("/jobs", StaticFiles(directory=str(SETTINGS.work_root / "jobs")), name="jobs")
# 示例谱面同时以静态路径暴露，方便前端在 /api 不可用时（纯静态托管）仍能加载
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


# ---------------------------------------------------------------- 后端任务


def _options_from_form(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"options 不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="options 必须是 JSON 对象")
    return payload


def _run(job: Job, source_path: Path) -> None:
    result = run_job_for_job_record(job, source_path)
    JOBS.log_line(job, f"完成：{result['stats']}")


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
) -> JSONResponse:
    """上传音乐文件并创建分析任务。"""
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix or '(空)'}，支持：{', '.join(sorted(SUPPORTED))}",
        )

    opts = _options_from_form(options)
    job = JOBS.create(name, opts)
    upload_dir = job.dir / "input"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / name

    limit = SETTINGS.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {SETTINGS.max_upload_mb}MB 限制",
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    if written == 0:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="上传文件为空")

    JOBS.set_artifact(job, "upload", target)
    JOBS.log_line(job, f"已上传 {name}（{written / 1024 / 1024:.1f}MB）")
    JOBS.submit(job, lambda j: _run(j, target))
    return JSONResponse({"job_id": job.id, "status": job.status}, status_code=202)


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> JSONResponse:
    return JSONResponse({"jobs": [job.to_dict() for job in JOBS.list(limit=limit)]})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JSONResponse(job.to_dict())


@app.get("/api/jobs/{job_id}/score")
def get_score(job_id: str) -> JSONResponse:
    return _serve_json(job_id, "score.json")


@app.get("/api/jobs/{job_id}/analysis")
def get_analysis(job_id: str) -> JSONResponse:
    return _serve_json(job_id, "analysis.json")


@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str) -> JSONResponse:
    return _serve_json(job_id, "manifest.json")


@app.get("/api/jobs/{job_id}/symbolic")
def get_symbolic(job_id: str) -> JSONResponse:
    """v2 符号契约（转录层输出：节拍/调性/和弦/段落/旋律/歌词 + provenance）。"""
    return _serve_json(job_id, "symbolic.json")


@app.get("/api/jobs/{job_id}/payload")
def get_payload(job_id: str) -> JSONResponse:
    """前端渲染 payload（MusicXML + ABC + timing + 指法图）。"""
    return _serve_json(job_id, "payload.json")


@app.get("/api/jobs/{job_id}/timing")
def get_timing(job_id: str) -> JSONResponse:
    """绝对时间轴（小节/音符/和弦/歌词的秒），供同步播放与第三方消费。"""
    return _serve_json(job_id, "timing.json")


def _serve_json(job_id: str, filename: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    path = job.dir / filename
    if not path.is_file():
        raise HTTPException(status_code=409, detail=f"{filename} 尚未生成（当前阶段：{job.stage}）")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


EXPORT_FILES = {
    "musicxml": ("score.musicxml", "application/vnd.recordare.musicxml+xml"),
    "midi": ("score.mid", "audio/midi"),
    "json": ("score.json", "application/json"),
    "payload": ("payload.json", "application/json"),
    "timing": ("timing.json", "application/json"),
    # 转录层原生 MusicXML（含 <technical>/<harmony>/<lyric>，未经编配层增强）
    "symbolic": ("symbolic.musicxml", "application/vnd.recordare.musicxml+xml"),
}


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export(job_id: str, fmt: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    entry = EXPORT_FILES.get(fmt)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"未知导出格式 {fmt}")
    filename, media_type = entry
    path = job.dir / filename
    if not path.is_file():
        raise HTTPException(status_code=409, detail=f"{filename} 尚未生成")
    download_name = f"{Path(job.original_name).stem or 'score'}.{path.suffix.lstrip('.')}"
    return FileResponse(path, media_type=media_type, filename=download_name)


# ---------------------------------------------------------------- 静态站点


@app.get("/api/voicings")
def voicings() -> JSONResponse:
    path = WEB_DIR / "voicings.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="指法库尚未生成，请运行 python -m aitsm.cli voicings")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/health")
def health() -> JSONResponse:
    from .devices import detect_devices

    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "engines": check_optional_engines(),
            "ffmpeg": SETTINGS.ffmpeg,
            "work_root": str(SETTINGS.work_root),
            # 前端据此决定「CPU / GPU」下拉框的默认值与可用性
            "devices": detect_devices(),
        }
    )


@app.get("/api/demo")
def demo_score(sample: bool = False) -> JSONResponse:
    """示例谱面，供前端在后端任务不可用时自检。

    优先给**真曲交付版**（``artifacts/score.json``：整首 88 小节、真实英文歌词），
    这样打开首页看到的就是完整成品而不是手写样例。
    ``?sample=1`` 仍可取回 12 小节的手写样例（离线自检用，不依赖 artifacts）。

    交付文件里不写播放地址（它是产物，不该绑定某个本地目录）；但如果
    ``work/jobs/demo/playback.mp3`` 存在，就在**响应里**补一个 ``audio_url``，
    让首页可以直接对齐音频播放。只改内存副本，不回写文件。
    """
    candidates: list[Path] = []
    if not sample:
        candidates.append(REPO_ROOT / "artifacts" / "score.json")
    candidates.append(SAMPLES_DIR / "demo-score.json")
    for path in candidates:
        if path.is_file():
            score = json.loads(path.read_text(encoding="utf-8"))
            if not sample:
                _attach_demo_audio(score)
            return JSONResponse(score)
    raise HTTPException(status_code=404, detail="示例数据缺失")


def _attach_demo_audio(score: dict[str, Any]) -> None:
    """若演示音频存在，把地址挂到 ``meta.source.audio_url``（只改内存副本）。"""
    playback = SETTINGS.work_root / "jobs" / "demo" / "playback.mp3"
    if not playback.is_file():
        return
    meta = score.get("meta")
    if not isinstance(meta, dict):
        return
    source = meta.get("source")
    if not isinstance(source, dict):
        source = {}
        meta["source"] = source
    source.setdefault("audio_url", "/jobs/demo/playback.mp3")


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

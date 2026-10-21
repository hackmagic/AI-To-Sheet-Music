"""内存任务系统：上传 -> 后台分析线程 -> 阶段进度 -> 产物。

设计取舍：
- 单机单用户场景，任务状态放内存，产物落磁盘（work/jobs/<id>/）。
- 分析阶段用串行队列（max_workers=1），避免 Demucs / Whisper 同时抢显存。
- 每个任务目录自带 `job.json`，进程重启后可据此恢复历史任务列表。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import SETTINGS

# 阶段顺序用于计算总体进度下限
STAGE_ORDER = [
    "queued",
    "ingesting",
    "separating",
    "analyzing",
    "transcribing",
    "arranging",
    "validating",
    "done",
]
STAGE_LABELS = {
    "queued": "排队中",
    "ingesting": "解析音频",
    "separating": "分离音轨",
    "analyzing": "分析节奏与和声",
    "transcribing": "识别歌词与人声旋律",
    "arranging": "生成弹唱谱",
    "validating": "校验谱面",
    "done": "完成",
    "failed": "失败",
}

# pipeline 内细粒度阶段名 -> 对外粗粒度阶段名
STAGE_ALIASES = {
    "ingest": "ingesting",
    "ingesting": "ingesting",
    "separate": "separating",
    "separating": "separating",
    "beats": "analyzing",
    "key": "analyzing",
    "chords": "analyzing",
    "structure": "analyzing",
    "analyzing": "analyzing",
    "lyrics": "transcribing",
    "melody": "transcribing",
    "transcribing": "transcribing",
    "assemble": "arranging",
    "arrange": "arranging",
    "arranging": "arranging",
    "validate": "validating",
    "validating": "validating",
    "done": "done",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    original_name: str
    options: dict[str, Any]
    dir: Path
    status: str = "queued"
    stage: str = "queued"
    percent: float = 0.0
    message: str = ""
    error: str | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    started_at: float | None = None
    finished_at: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    log: deque[str] = field(default_factory=lambda: deque(maxlen=400))

    @property
    def root(self) -> Path:
        return self.dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, self.stage),
            "percent": round(self.percent, 1),
            "message": self.message,
            "error": self.error,
            "warnings": self.warnings,
            "original_name": self.original_name,
            "options": self.options,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_sec": round((self.finished_at or time.time()) - (self.started_at or time.time()), 2)
            if self.started_at
            else 0.0,
            "artifacts": self.artifacts,
            "log": list(self.log)[-60:],
        }

    def persist(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload = self.to_dict()
            payload["log"] = list(self.log)
            (self.dir / "job.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            # 持久化失败不应中断分析
            pass


class JobManager:
    """线程安全的任务注册表 + 单线程执行队列。"""

    def __init__(self, max_workers: int | None = None) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[tuple[str, Callable[["Job"], None]]] = deque()
        self._worker: threading.Thread | None = None
        self._max_workers = max(1, max_workers or SETTINGS.max_workers)
        SETTINGS.ensure_dirs()
        self._load_existing()

    # ---------- 生命周期 ----------

    def _load_existing(self) -> None:
        jobs_root = SETTINGS.work_root / "jobs"
        if not jobs_root.is_dir():
            return
        for job_dir in sorted(jobs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[
            : SETTINGS.keep_jobs
        ]:
            meta_path = job_dir / "job.json"
            if not meta_path.is_file():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            job = Job(
                id=str(payload.get("id") or job_dir.name),
                original_name=str(payload.get("original_name") or ""),
                options=payload.get("options") or {},
                dir=job_dir,
                status=str(payload.get("status") or "done"),
                stage=str(payload.get("stage") or "done"),
                percent=float(payload.get("percent") or 100.0),
                message=str(payload.get("message") or ""),
                error=payload.get("error"),
                warnings=payload.get("warnings") or [],
                created_at=str(payload.get("created_at") or _now_iso()),
                updated_at=str(payload.get("updated_at") or _now_iso()),
                artifacts=payload.get("artifacts") or {},
            )
            # 进程重启时未完成的任务标记为失败，避免前端无限轮询
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.stage = "failed"
                job.error = "服务重启，任务中断"
                job.persist()
            self._jobs[job.id] = job

    def create(self, original_name: str, options: dict[str, Any]) -> Job:
        job_id = uuid.uuid4().hex[:16]
        job_dir = SETTINGS.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, original_name=original_name, options=options, dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()
        job.persist()
        return job

    def _prune_locked(self) -> None:
        if len(self._jobs) <= SETTINGS.keep_jobs:
            return
        removable = [
            job
            for job in self._jobs.values()
            if job.status in {"done", "failed"} and job.id not in {jid for jid, _ in self._queue}
        ]
        removable.sort(key=lambda j: j.updated_at)
        for job in removable[: max(0, len(self._jobs) - SETTINGS.keep_jobs)]:
            self._jobs.pop(job.id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    # ---------- 执行 ----------

    def submit(self, job: Job, task: Callable[[Job], None]) -> None:
        with self._lock:
            self._queue.append((job.id, task))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop, name="aitsm-worker", daemon=True)
                self._worker.start()

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                job_id, task = self._queue.popleft()
                job = self._jobs.get(job_id)
            if job is None:
                continue
            self._run_one(job, task)

    def _run_one(self, job: Job, task: Callable[[Job], None]) -> None:
        self.update(job, stage="ingesting", percent=1.0, status="running", message="开始处理")
        job.started_at = time.time()
        try:
            task(job)
            if job.status != "failed":
                self.update(job, stage="done", percent=100.0, status="done", message="完成")
        except Exception as exc:  # noqa: BLE001 - 任务边界必须兜住所有异常
            job.status = "failed"
            job.stage = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "处理失败"
            job.log.append(traceback.format_exc())
            job.updated_at = _now_iso()
            job.persist()
        finally:
            job.finished_at = time.time()

    # ---------- 进度上报 ----------

    def update(
        self,
        job: Job,
        *,
        stage: str | None = None,
        percent: float | None = None,
        message: str | None = None,
        status: str | None = None,
    ) -> None:
        with self._lock:
            changed = False
            if stage and stage != job.stage:
                job.stage = stage
                changed = True
            if status and status != job.status:
                job.status = status
                changed = True
            if percent is not None:
                # 进度单调不回退
                job.percent = max(job.percent, min(100.0, float(percent)))
            if message:
                job.message = str(message)
            if message or changed:
                job.log.append(f"[{job.stage}] {message or ''}".strip())
            job.updated_at = _now_iso()
        job.persist()

    def progress_callback(self, job: Job) -> Callable[[str, float, str], None]:
        """把 pipeline 的阶段事件映射到 Job 的对外阶段。

        pipeline 的 ``percent`` 已经是整条流水线的全局百分比（0-100），因此这里
        直接采信，不再按阶段重新缩放；阶段名需要做一次别名映射，因为算法层
        用的是细粒度名字（ingest/separate/beats/key/chords/structure/lyrics/
        melody/assemble），而对前端暴露的是粗粒度阶段。
        """

        def callback(stage: str, percent: float, message: str = "") -> None:
            clamped = max(0.0, min(100.0, float(percent)))
            mapped = STAGE_ALIASES.get(stage, stage)
            if mapped not in STAGE_LABELS:
                # 未知阶段名：保持当前阶段，只更新进度与文案
                mapped = job.stage
            self.update(job, stage=mapped, percent=clamped, message=message)

        return callback

    def add_warning(self, job: Job, warning: dict[str, Any]) -> None:
        with self._lock:
            job.warnings.append(warning)

    def set_artifact(self, job: Job, key: str, path: str | Path) -> None:
        with self._lock:
            job.artifacts[key] = str(path)

    def log_line(self, job: Job, message: str) -> None:
        with self._lock:
            job.log.append(str(message))
            job.updated_at = _now_iso()


JOBS = JobManager()

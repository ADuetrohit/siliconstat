"""In-process background job manager for long-running Monte Carlo runs.

A 10 000-sample run takes minutes, so the HTTP request that starts it must
return immediately with a job handle the UI can poll.  Jobs run on a bounded
thread pool: the Monte Carlo engine itself decides whether to spread work over
*processes*, so the worker thread here is only a supervisor that stays
responsive to progress callbacks and cancellation.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

__all__ = ["Job", "JobManager", "JobStatus"]


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = JobStatus.QUEUED
    run_id: str | None = None
    completed: int = 0
    total: int = 0
    successful: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def eta_s(self) -> float | None:
        if self.status != JobStatus.RUNNING or self.completed <= 0 or not self.total:
            return None
        rate = self.completed / self.elapsed_s if self.elapsed_s > 0 else 0.0
        return (self.total - self.completed) / rate if rate > 0 else None

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "kind": self.kind, "status": self.status,
            "run_id": self.run_id, "completed": self.completed,
            "total": self.total, "successful": self.successful,
            "failed": self.failed, "elapsed_s": self.elapsed_s,
            "eta_s": self.eta_s, "message": self.message, "error": self.error,
            "result": self.result, "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """Bounded pool of background jobs with progress reporting."""

    def __init__(self, max_workers: int = 2, retain: int = 200) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="siliconstat-job")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._retain = retain

    def submit(self, kind: str, target: Callable[[Job], dict[str, Any]], *,
               total: int = 0, message: str = "") -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], kind=kind, total=total,
                  message=message)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._evict()
        self._pool.submit(self._run, job, target)
        return job

    def _run(self, job: Job, target: Callable[[Job], dict[str, Any]]) -> None:
        job.status = JobStatus.RUNNING
        started = time.perf_counter()
        try:
            result = target(job)
            job.result = result
            job.status = (JobStatus.CANCELLED if job.cancelled
                          else JobStatus.DONE)
        except Exception as exc:  # surface, never swallow
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = traceback.format_exc(limit=6)
        finally:
            job.elapsed_s = time.perf_counter() - started
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i] for i in ids if i in self._jobs]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.status in (JobStatus.QUEUED, JobStatus.RUNNING))

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job.cancel()
        job.message = "cancellation requested"
        return True

    def _evict(self) -> None:
        while len(self._order) > self._retain:
            old = self._order.pop(0)
            job = self._jobs.get(old)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                self._order.append(old)  # never evict a live job
                break
            self._jobs.pop(old, None)

    def shutdown(self) -> None:  # pragma: no cover - lifecycle
        self._pool.shutdown(wait=False, cancel_futures=True)

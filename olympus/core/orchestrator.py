"""Central OLYMPUS orchestrator — coordinates all 10 modules."""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from olympus.core.base_module import BaseModule, ModuleResult
from olympus.core.config import CONFIG
from olympus.core.logger import AUDIT, get_logger

log = get_logger("orchestrator")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OlympusTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    module_id: str = ""
    name: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[ModuleResult] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def duration(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 3)
        return None


class Orchestrator:
    """Manages module registry, task queue, and concurrent execution."""

    def __init__(self, max_workers: int = 4) -> None:
        self._modules: dict[str, BaseModule] = {}
        self._tasks: dict[str, OlympusTask] = {}
        self._lock = threading.RLock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._callbacks: list[Callable[[OlympusTask], None]] = []
        log.info("OLYMPUS Orchestrator initialized (workers=%d)", max_workers)

    # ── module registry ───────────────────────────────────────────────────────

    def register(self, module: BaseModule) -> None:
        with self._lock:
            self._modules[module.MODULE_ID] = module
        log.info("Registered module: %s (%s)", module.MODULE_NAME, module.MODULE_ID)

    def get_module(self, module_id: str) -> Optional[BaseModule]:
        return self._modules.get(module_id)

    def list_modules(self) -> list[dict[str, str]]:
        return [
            {
                "id": m.MODULE_ID,
                "name": m.MODULE_NAME,
                "type": m.MODULE_TYPE,
                "status": "ok" if m._enabled else "disabled",
            }
            for m in self._modules.values()
        ]

    # ── task management ───────────────────────────────────────────────────────

    def submit(self, module_id: str, name: str = "", **kwargs: Any) -> OlympusTask:
        module = self._modules.get(module_id)
        if not module:
            raise ValueError(f"Unknown module: {module_id}")

        task = OlympusTask(module_id=module_id, name=name or module.MODULE_NAME, kwargs=kwargs)
        with self._lock:
            self._tasks[task.task_id] = task

        AUDIT.log("orchestrator", "task_submit", {"task_id": task.task_id, "module": module_id})
        future = self._executor.submit(self._run_task, task, module)
        future.add_done_callback(lambda f: self._on_done(task, f))
        return task

    def _run_task(self, task: OlympusTask, module: BaseModule) -> ModuleResult:
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        try:
            return module.run(**task.kwargs)
        except Exception as exc:
            task.error = str(exc)
            log.error("Module %s failed: %s", module.MODULE_ID, exc, exc_info=True)
            raise

    def _on_done(self, task: OlympusTask, future: concurrent.futures.Future) -> None:
        task.finished_at = time.time()
        if future.exception():
            task.status = TaskStatus.FAILED
            task.error = str(future.exception())
        else:
            task.status = TaskStatus.COMPLETED
            task.result = future.result()
        AUDIT.log("orchestrator", "task_done", {
            "task_id": task.task_id,
            "status": task.status.value,
            "duration_s": task.duration(),
        })
        for cb in self._callbacks:
            try:
                cb(task)
            except Exception as exc:
                log.warning("Callback error: %s", exc)

    def get_task(self, task_id: str) -> Optional[OlympusTask]:
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> list[OlympusTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status.value == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def on_task_complete(self, callback: Callable[[OlympusTask], None]) -> None:
        self._callbacks.append(callback)

    # ── run all modules ───────────────────────────────────────────────────────

    def run_all(self, **kwargs: Any) -> list[OlympusTask]:
        tasks = []
        for module_id in self._modules:
            try:
                task = self.submit(module_id, **kwargs)
                tasks.append(task)
            except Exception as exc:
                log.error("Failed to submit %s: %s", module_id, exc)
        return tasks

    # ── health ────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        return {
            "modules_registered": len(self._modules),
            "tasks_total": len(self._tasks),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "modules": self.list_modules(),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        log.info("Orchestrator shut down")


ORCHESTRATOR = Orchestrator()

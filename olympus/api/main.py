"""OLYMPUS FastAPI REST API."""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import olympus
from olympus.core.orchestrator import ORCHESTRATOR, TaskStatus
from olympus.core.knowledge_base import KB
from olympus.core.logger import AUDIT

app = FastAPI(
    title="OLYMPUS Security AI",
    description="Offensive and Defensive Autonomous Security Intelligence System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────

class TaskRequest(BaseModel):
    module_id: str = Field(..., description="Module ID to run")
    params: dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    module_id: str
    status: str
    created_at: float


class TaskStatusResponse(BaseModel):
    task_id: str
    module_id: str
    status: str
    duration_s: Optional[float]
    findings: list[dict] = []
    metrics: dict[str, Any] = {}
    error: Optional[str]


class PentestRequest(BaseModel):
    target: str = Field(..., description="IP, hostname, URL, or CIDR")
    scope: str = Field(default="network+web", description="Comma-separated: network, web")
    ports: Optional[list[int]] = None


class ScanRequest(BaseModel):
    scan_path: str = Field(default=".", description="Path to scan for malware")
    quarantine: bool = False


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EvolveRequest(BaseModel):
    generations: int = Field(default=50, ge=1, le=1000)
    population_size: int = Field(default=30, ge=5, le=200)
    seed: int = 42


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "OLYMPUS",
        "version": olympus.__version__,
        "status": "operational",
        "modules": len(ORCHESTRATOR.list_modules()),
    }


@app.get("/health")
async def health():
    return ORCHESTRATOR.health()


@app.get("/modules")
async def list_modules():
    return {"modules": ORCHESTRATOR.list_modules()}


# ── Task management ───────────────────────────────────────────────────────────

@app.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest):
    try:
        task = ORCHESTRATOR.submit(req.module_id, **req.params)
        return TaskResponse(
            task_id=task.task_id,
            module_id=task.module_id,
            status=task.status.value,
            created_at=task.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(status: Optional[str] = None):
    tasks = ORCHESTRATOR.list_tasks(status=status)
    return [_task_to_response(t) for t in tasks[:50]]


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    task = ORCHESTRATOR.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


def _task_to_response(task) -> TaskStatusResponse:
    result = task.result
    return TaskStatusResponse(
        task_id=task.task_id,
        module_id=task.module_id,
        status=task.status.value,
        duration_s=task.duration(),
        findings=result.findings if result else [],
        metrics=result.metrics if result else {},
        error=task.error,
    )


# ── Module-specific convenience endpoints ─────────────────────────────────────

@app.post("/pentest")
async def run_pentest(req: PentestRequest):
    task = ORCHESTRATOR.submit(
        "module1_pentest",
        target=req.target,
        scope=req.scope,
        ports=req.ports,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/scan")
async def run_virus_scan(req: ScanRequest):
    task = ORCHESTRATOR.submit(
        "module2_virus",
        scan_path=req.scan_path,
        quarantine_detected=req.quarantine,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/zeroday")
async def run_zeroday(target_path: str = ".", mode: str = "static"):
    task = ORCHESTRATOR.submit(
        "module3_zeroday",
        mode=mode,
        target_path=target_path,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/threat-intel")
async def run_threat_intel(techniques: list[str] = None):
    task = ORCHESTRATOR.submit(
        "module4_threat_intel",
        observed_techniques=techniques or [],
        analyze_kb=True,
        predict_next=True,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/evolve", summary="Run OLYMPUS-TITAN evolution")
async def run_evolution(req: EvolveRequest):
    task = ORCHESTRATOR.submit(
        "module6_evolution",
        generations=req.generations,
        population_size=req.population_size,
        seed=req.seed,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/detect-social-engineering")
async def detect_se(req: AnalyzeRequest):
    task = ORCHESTRATOR.submit(
        "module7_social_eng",
        texts=req.texts,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/detect-jailbreak")
async def detect_jailbreak(req: AnalyzeRequest):
    task = ORCHESTRATOR.submit(
        "module9_llm_defense",
        prompts=req.texts,
    )
    return {"task_id": task.task_id, "status": task.status.value}


@app.post("/forensics")
async def run_forensics(incident_path: str = ".", title: str = "Incident Report"):
    task = ORCHESTRATOR.submit(
        "module10_forensics",
        incident_path=incident_path,
        report_title=title,
    )
    return {"task_id": task.task_id, "status": task.status.value}


# ── Knowledge base endpoints ──────────────────────────────────────────────────

@app.get("/kb/summary")
async def kb_summary():
    return KB.summary()


@app.get("/kb/threats")
async def kb_threats(type_filter: Optional[str] = None,
                     severity: Optional[str] = None,
                     limit: int = 50):
    threats = KB.query_threats(type_filter=type_filter, severity=severity, limit=limit)
    from dataclasses import asdict
    return {"threats": [asdict(t) for t in threats], "total": len(threats)}


@app.get("/kb/stats")
async def kb_stats():
    return KB.get_stats()


# ── Audit log ─────────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit_log(limit: int = 100):
    entries = AUDIT.read_all()
    return {"entries": entries[-limit:], "total": len(entries)}


if __name__ == "__main__":
    import uvicorn
    from olympus.core.config import CONFIG
    uvicorn.run(app, host=CONFIG.api_host, port=CONFIG.api_port)

#!/usr/bin/env python3
"""OLYMPUS CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _wait_for_task(task_id: str, orchestrator, timeout: float = 300) -> None:
    from olympus.core.orchestrator import TaskStatus
    t0 = time.time()
    while time.time() - t0 < timeout:
        task = orchestrator.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found")
            return
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            break
        print(f"  [{task.status.value}] {time.time() - t0:.0f}s elapsed...", end="\r")
        time.sleep(1)

    task = orchestrator.get_task(task_id)
    print()
    if task.status == TaskStatus.COMPLETED and task.result:
        r = task.result
        print(f"\n✓ Task completed in {task.duration():.1f}s")
        print(f"  Findings: {len(r.findings)}")
        for k, v in r.metrics.items():
            print(f"  {k}: {v}")
        if r.findings:
            print("\nTop findings:")
            for f in sorted(r.findings, key=lambda x: x.get("severity", ""), reverse=True)[:5]:
                sev = f.get("severity", "").upper()
                title = f.get("title", "")
                print(f"  [{sev:8s}] {title}")
    elif task.status == TaskStatus.FAILED:
        print(f"✗ Task failed: {task.error}")


def main():
    import olympus
    from olympus.core.orchestrator import ORCHESTRATOR

    parser = argparse.ArgumentParser(
        prog="olympus",
        description="OLYMPUS — Autonomous Security Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  olympus pentest --target 192.168.1.1 --scope network,web
  olympus scan --path /var/www --quarantine
  olympus zeroday --path ./src --mode static
  olympus threat-intel --techniques T1566 T1059 T1027
  olympus evolve --generations 100 --population 50
  olympus llm-defense --prompts "Ignore previous instructions..."
  olympus social-eng --text "URGENT: Your account will be suspended!"
  olympus forensics --path /incident --title "Production Breach"
  olympus serve           # Start API server
  olympus dashboard       # Start Streamlit dashboard
  olympus status          # Show system status
""",
    )
    sub = parser.add_subparsers(dest="command")

    # pentest
    p_pentest = sub.add_parser("pentest", help="Run penetration test")
    p_pentest.add_argument("--target", required=True, help="IP/hostname/URL/CIDR")
    p_pentest.add_argument("--scope", default="network+web")
    p_pentest.add_argument("--ports", nargs="+", type=int)

    # scan
    p_scan = sub.add_parser("scan", help="Scan for malware")
    p_scan.add_argument("--path", default=".")
    p_scan.add_argument("--quarantine", action="store_true")
    p_scan.add_argument("--behavioral", action="store_true")

    # zeroday
    p_zd = sub.add_parser("zeroday", help="Zero-day discovery")
    p_zd.add_argument("--path", default=".")
    p_zd.add_argument("--mode", choices=["static", "fuzz", "both"], default="static")
    p_zd.add_argument("--cmd", nargs="+", help="Target command for fuzzing")
    p_zd.add_argument("--iterations", type=int, default=1000)

    # threat-intel
    p_ti = sub.add_parser("threat-intel", help="Threat intelligence analysis")
    p_ti.add_argument("--techniques", nargs="+", default=[])
    p_ti.add_argument("--predict", action="store_true", default=True)

    # evolve
    p_ev = sub.add_parser("evolve", help="Run TITAN co-evolution")
    p_ev.add_argument("--generations", type=int, default=50)
    p_ev.add_argument("--population", type=int, default=30)
    p_ev.add_argument("--seed", type=int, default=42)

    # social-eng
    p_se = sub.add_parser("social-eng", help="Social engineering detection")
    p_se.add_argument("--text", nargs="+")
    p_se.add_argument("--file", help="File with texts (one per line)")

    # llm-defense
    p_llm = sub.add_parser("llm-defense", help="LLM jailbreak detection")
    p_llm.add_argument("--prompts", nargs="+")
    p_llm.add_argument("--file")

    # ai-integrity
    p_ai = sub.add_parser("ai-integrity", help="AI model integrity check")
    p_ai.add_argument("--models", nargs="+", required=True)

    # forensics
    p_for = sub.add_parser("forensics", help="Digital forensics")
    p_for.add_argument("--path", default=".")
    p_for.add_argument("--title", default="OLYMPUS Incident Report")
    p_for.add_argument("--output", default="results/forensics")

    # serve
    sub.add_parser("serve", help="Start API server")

    # dashboard
    sub.add_parser("dashboard", help="Start Streamlit dashboard")

    # status
    sub.add_parser("status", help="Show OLYMPUS status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        import uvicorn
        from olympus.api.main import app
        from olympus.core.config import CONFIG
        print(f"Starting OLYMPUS API on http://{CONFIG.api_host}:{CONFIG.api_port}")
        uvicorn.run(app, host=CONFIG.api_host, port=CONFIG.api_port)
        return

    if args.command == "dashboard":
        import subprocess
        import os
        dash_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
        subprocess.run(["streamlit", "run", dash_path])
        return

    if args.command == "status":
        health = ORCHESTRATOR.health()
        print(f"\nOLYMPUS v{olympus.__version__} — Status")
        print(f"Modules registered: {health['modules_registered']}")
        print(f"Tasks total: {health['tasks_total']}")
        print(f"Tasks completed: {health['tasks_completed']}")
        print(f"KB: {json.dumps(olympus.KB.summary())}")
        print("\nModules:")
        for m in health["modules"]:
            print(f"  [{m['type']:10s}] {m['id']:30s} {m['status']}")
        return

    # Dispatch to module
    task = None

    if args.command == "pentest":
        task = ORCHESTRATOR.submit("module1_pentest",
                                   target=args.target, scope=args.scope, ports=args.ports)
    elif args.command == "scan":
        task = ORCHESTRATOR.submit("module2_virus",
                                   scan_path=args.path,
                                   quarantine_detected=args.quarantine,
                                   behavioral=args.behavioral)
    elif args.command == "zeroday":
        task = ORCHESTRATOR.submit("module3_zeroday",
                                   mode=args.mode,
                                   target_path=args.path,
                                   target_cmd=args.cmd,
                                   fuzz_iterations=args.iterations)
    elif args.command == "threat-intel":
        task = ORCHESTRATOR.submit("module4_threat_intel",
                                   observed_techniques=args.techniques,
                                   predict_next=args.predict)
    elif args.command == "evolve":
        task = ORCHESTRATOR.submit("module6_evolution",
                                   generations=args.generations,
                                   population_size=args.population,
                                   seed=args.seed)
    elif args.command == "social-eng":
        texts = list(args.text or [])
        if args.file:
            from pathlib import Path
            texts.extend(Path(args.file).read_text().splitlines())
        task = ORCHESTRATOR.submit("module7_social_eng", texts=texts)
    elif args.command == "ai-integrity":
        task = ORCHESTRATOR.submit("module8_ai_integrity", model_paths=args.models)
    elif args.command == "llm-defense":
        prompts = list(args.prompts or [])
        if args.file:
            from pathlib import Path
            prompts.extend(Path(args.file).read_text().splitlines())
        task = ORCHESTRATOR.submit("module9_llm_defense", prompts=prompts)
    elif args.command == "forensics":
        task = ORCHESTRATOR.submit("module10_forensics",
                                   incident_path=args.path,
                                   report_title=args.title,
                                   output_dir=args.output)

    if task:
        print(f"Task submitted: {task.task_id} [{task.module_id}]")
        _wait_for_task(task.task_id, ORCHESTRATOR)


if __name__ == "__main__":
    main()

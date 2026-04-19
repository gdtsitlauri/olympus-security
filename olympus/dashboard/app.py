"""OLYMPUS Streamlit Dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import streamlit as st
    import pandas as pd
    _ST = True
except ImportError:
    _ST = False
    raise ImportError("streamlit and pandas required for dashboard")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import olympus
from olympus.core.orchestrator import ORCHESTRATOR, TaskStatus
from olympus.core.knowledge_base import KB
from olympus.core.logger import AUDIT

st.set_page_config(
    page_title="OLYMPUS Security AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1a1a2e; }
    .metric-card { background: #16213e; color: white; padding: 1rem; border-radius: 8px; }
    .severity-critical { color: #ff4444; font-weight: bold; }
    .severity-high { color: #ff8800; font-weight: bold; }
    .severity-medium { color: #ffcc00; }
    .severity-low { color: #44ff44; }
    .severity-info { color: #4488ff; }
    .stMetric label { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


def severity_color(sev: str) -> str:
    colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}
    return colors.get(sev, "⚪")


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.image("https://via.placeholder.com/200x60/1a1a2e/gold?text=OLYMPUS", use_column_width=True)
st.sidebar.title("OLYMPUS v" + olympus.__version__)

pages = [
    "🏠 Overview",
    "⚔️ Penetration Testing",
    "🦠 Malware Detection",
    "🔍 Zero-Day Discovery",
    "🧠 Threat Intelligence",
    "🪤 Deception",
    "🧬 TITAN Evolution",
    "📧 Social Engineering",
    "🤖 AI Integrity",
    "💬 LLM Defense",
    "🔬 Forensics",
    "📋 Tasks",
    "📚 Knowledge Base",
    "📊 Experiments",
]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
kb_sum = KB.summary()
st.sidebar.metric("Threats in KB", kb_sum["threats"])
st.sidebar.metric("Attack Patterns", kb_sum["attack_patterns"])
st.sidebar.metric("Defense Records", kb_sum["defenses"])


# ── Overview ──────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown('<p class="main-header">⚡ OLYMPUS Security AI</p>', unsafe_allow_html=True)
    st.markdown("*Offensive and Defensive Autonomous Security Intelligence System*")

    health = ORCHESTRATOR.health()
    cols = st.columns(4)
    cols[0].metric("Modules Active", health["modules_registered"])
    cols[1].metric("Tasks Total", health["tasks_total"])
    cols[2].metric("Completed", health["tasks_completed"])
    cols[3].metric("Failed", health["tasks_failed"])

    st.subheader("Module Status")
    module_data = []
    for m in health["modules"]:
        emoji = {"offensive": "⚔️", "defensive": "🛡️", "core": "⚙️"}.get(m["type"], "❓")
        module_data.append({
            "Module": f"{emoji} {m['name']}",
            "Type": m["type"],
            "Status": "✅ " + m["status"],
        })
    st.dataframe(pd.DataFrame(module_data), use_container_width=True)

    # Recent audit events
    st.subheader("Recent Activity")
    events = AUDIT.read_all()[-20:]
    if events:
        df = pd.DataFrame(events)
        st.dataframe(df[["ts", "module", "action", "severity"]
                       if all(c in df.columns for c in ["ts", "module", "action", "severity"])
                       else df.columns.tolist()],
                     use_container_width=True)
    else:
        st.info("No audit events yet. Run a module to see activity.")


# ── Penetration Testing ───────────────────────────────────────────────────────

elif page == "⚔️ Penetration Testing":
    st.header("⚔️ Penetration Testing")
    st.warning("⚠️ Only scan systems you own or have explicit written authorization to test.")

    with st.form("pentest_form"):
        target = st.text_input("Target (IP/hostname/URL/CIDR)", placeholder="192.168.1.1")
        scope = st.multiselect("Scope", ["network", "web"], default=["network", "web"])
        submitted = st.form_submit_button("🚀 Run Pentest")

    if submitted and target:
        with st.spinner(f"Scanning {target}..."):
            task = ORCHESTRATOR.submit(
                "module1_pentest",
                target=target,
                scope="+".join(scope),
            )
            # Wait (blocking for demo; production use async)
            deadline = time.time() + 120
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(2)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            metrics = r.metrics
            cols = st.columns(4)
            cols[0].metric("Risk Score", f"{metrics.get('risk_score', 0):.1f}")
            cols[1].metric("Open Ports", metrics.get("open_ports_total", 0))
            cols[2].metric("Vulnerabilities", metrics.get("vulnerabilities_found", 0))
            cols[3].metric("Critical", metrics.get("critical", 0))

            if r.findings:
                st.subheader("Findings")
                df = pd.DataFrame([{
                    "Severity": severity_color(f.get("severity", "")) + " " + f.get("severity", ""),
                    "Title": f.get("title", ""),
                    "Detail": f.get("detail", "")[:100],
                } for f in r.findings])
                st.dataframe(df, use_container_width=True)


# ── Malware Detection ─────────────────────────────────────────────────────────

elif page == "🦠 Malware Detection":
    st.header("🦠 Virus Detection & Fighting")

    with st.form("scan_form"):
        scan_path = st.text_input("Scan Path", value=".")
        quarantine = st.checkbox("Quarantine detected files")
        behavioral = st.checkbox("Behavioral monitoring")
        submitted = st.form_submit_button("🔍 Scan")

    if submitted:
        with st.spinner("Scanning..."):
            task = ORCHESTRATOR.submit("module2_virus",
                                       scan_path=scan_path,
                                       quarantine_detected=quarantine,
                                       behavioral=behavioral)
            deadline = time.time() + 120
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(2)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            cols = st.columns(3)
            cols[0].metric("Files Scanned", r.metrics.get("files_scanned", 0))
            cols[1].metric("Malicious", r.metrics.get("malicious_detected", 0))
            cols[2].metric("Quarantined", r.metrics.get("quarantined", 0))

            if r.findings:
                for f in r.findings:
                    sev = f.get("severity", "info")
                    icon = severity_color(sev)
                    with st.expander(f"{icon} {f.get('title', '')}"):
                        st.write(f.get("detail", ""))
                        if "indicators" in f:
                            st.write("**Indicators:**", f["indicators"])


# ── Zero-Day Discovery ────────────────────────────────────────────────────────

elif page == "🔍 Zero-Day Discovery":
    st.header("🔍 Zero-Day Discovery")

    with st.form("zeroday_form"):
        mode = st.selectbox("Mode", ["static", "fuzz", "both"])
        target_path = st.text_input("Target Path", value=".")
        fuzz_iterations = st.slider("Fuzz Iterations", 100, 10000, 1000)
        submitted = st.form_submit_button("🔬 Discover")

    if submitted:
        with st.spinner("Analyzing..."):
            task = ORCHESTRATOR.submit("module3_zeroday",
                                       mode=mode,
                                       target_path=target_path,
                                       fuzz_iterations=fuzz_iterations)
            deadline = time.time() + 180
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(2)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            cols = st.columns(3)
            cols[0].metric("Static Findings", r.metrics.get("static_findings", 0))
            cols[1].metric("Crashes Found", r.metrics.get("crashes_found", 0))
            cols[2].metric("Critical", r.metrics.get("critical_findings", 0))

            if r.findings:
                findings_df = pd.DataFrame([{
                    "Severity": f.get("severity", ""),
                    "Title": f.get("title", ""),
                    "CWE": f.get("cwe", ""),
                } for f in r.findings])
                st.dataframe(findings_df, use_container_width=True)


# ── TITAN Evolution ───────────────────────────────────────────────────────────

elif page == "🧬 TITAN Evolution":
    st.header("🧬 OLYMPUS-TITAN Co-Evolutionary Algorithm")
    st.markdown("""
    **TITAN** (Total Intelligent Threat Analysis and Neutralization) runs a co-evolutionary
    genetic algorithm where attack and defense populations compete and evolve simultaneously.
    """)

    with st.form("titan_form"):
        cols = st.columns(3)
        generations = cols[0].number_input("Generations", 10, 1000, 50)
        population = cols[1].number_input("Population Size", 5, 200, 30)
        seed = cols[2].number_input("Random Seed", 0, 999, 42)
        submitted = st.form_submit_button("🧬 Evolve")

    if submitted:
        with st.spinner(f"Running TITAN for {generations} generations..."):
            task = ORCHESTRATOR.submit("module6_evolution",
                                       generations=int(generations),
                                       population_size=int(population),
                                       seed=int(seed))
            deadline = time.time() + 600
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(3)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            cols = st.columns(4)
            cols[0].metric("Best Attack", f"{r.metrics.get('best_attack_fitness', 0):.4f}")
            cols[1].metric("Best Defense", f"{r.metrics.get('best_defense_fitness', 0):.4f}")
            cols[2].metric("Convergence", f"{r.metrics.get('convergence_score', 0):.4f}")
            cols[3].metric("Total Evals", r.metrics.get("total_evaluations", 0))

        # Load history from results file
        hist_path = Path("results/titan_evolution.json")
        if hist_path.exists():
            data = json.loads(hist_path.read_text())
            history = data.get("history", [])
            if history:
                df = pd.DataFrame(history)
                st.subheader("Evolution Progress")
                st.line_chart(df.set_index("generation")[
                    ["attack_best", "defense_best", "attack_mean", "defense_mean"]
                ])


# ── LLM Defense ───────────────────────────────────────────────────────────────

elif page == "💬 LLM Defense":
    st.header("💬 LLM Defense & Jailbreak Detection")

    with st.form("llm_form"):
        prompt_text = st.text_area("Enter prompt(s) to analyze (one per line)", height=150)
        submitted = st.form_submit_button("🔍 Analyze")

    if submitted and prompt_text:
        prompts = [p.strip() for p in prompt_text.splitlines() if p.strip()]
        with st.spinner("Analyzing prompts..."):
            task = ORCHESTRATOR.submit("module9_llm_defense", prompts=prompts)
            deadline = time.time() + 30
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(0.5)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            cols = st.columns(4)
            cols[0].metric("Analyzed", r.metrics.get("prompts_analyzed", 0))
            cols[1].metric("Jailbreaks", r.metrics.get("jailbreaks_detected", 0))
            cols[2].metric("Blocked", r.metrics.get("blocked", 0))
            cols[3].metric("Avg Risk", f"{r.metrics.get('avg_risk_score', 0):.1f}/100")

            for f in r.findings:
                sev = f.get("severity", "info")
                st.error(f"🚨 {f.get('title', '')}" if sev == "critical" else
                         f"⚠️ {f.get('title', '')}")
                with st.expander("Details"):
                    st.write(f.get("detail", ""))
                    if "sanitized_preview" in f:
                        st.write("**Sanitized:**", f["sanitized_preview"])


# ── Social Engineering ────────────────────────────────────────────────────────

elif page == "📧 Social Engineering":
    st.header("📧 Social Engineering Detection")

    with st.form("se_form"):
        text_input = st.text_area("Paste email/message content", height=200)
        submitted = st.form_submit_button("🔍 Analyze")

    if submitted and text_input:
        with st.spinner("Analyzing..."):
            task = ORCHESTRATOR.submit("module7_social_eng", texts=[text_input])
            deadline = time.time() + 30
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(0.5)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            for f in r.findings:
                sev = f.get("severity", "info")
                if sev in ("critical", "high"):
                    st.error(f"🚨 {f.get('title', '')}")
                else:
                    st.warning(f"⚠️ {f.get('title', '')}")
                st.write("**Indicators:**")
                for ind in f.get("indicators", []):
                    st.write(f"  • {ind}")
                st.write("**Action:**", f.get("recommended_action", ""))


# ── Threat Intelligence ───────────────────────────────────────────────────────

elif page == "🧠 Threat Intelligence":
    st.header("🧠 Threat Intelligence")

    with st.form("ti_form"):
        techniques = st.text_input("MITRE ATT&CK Technique IDs (comma-separated)",
                                   placeholder="T1566, T1059, T1027")
        analyze_kb = st.checkbox("Include KB patterns", value=True)
        submitted = st.form_submit_button("🧠 Analyze")

    if submitted:
        tids = [t.strip() for t in techniques.split(",") if t.strip()] if techniques else []
        with st.spinner("Analyzing threat intelligence..."):
            task = ORCHESTRATOR.submit("module4_threat_intel",
                                       observed_techniques=tids,
                                       analyze_kb=analyze_kb,
                                       predict_next=True)
            deadline = time.time() + 60
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(1)

        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            r = t.result
            cols = st.columns(4)
            cols[0].metric("Techniques", r.metrics.get("techniques_analyzed", 0))
            cols[1].metric("Groups Attributed", r.metrics.get("groups_attributed", 0))
            cols[2].metric("Signatures", r.metrics.get("signatures_generated", 0))
            cols[3].metric("Prediction Conf.", f"{r.metrics.get('prediction_confidence', 0):.1%}")
            for f in r.findings:
                with st.expander(f"{severity_color(f.get('severity',''))} {f.get('title','')}"):
                    st.write(f.get("detail", ""))


# ── Tasks ─────────────────────────────────────────────────────────────────────

elif page == "📋 Tasks":
    st.header("📋 Task Monitor")
    if st.button("🔄 Refresh"):
        st.rerun()

    tasks = ORCHESTRATOR.list_tasks()
    if not tasks:
        st.info("No tasks yet.")
    else:
        task_data = []
        for t in tasks:
            icon = {"completed": "✅", "running": "⏳", "failed": "❌", "pending": "🔵"}.get(
                t.status.value, "❓"
            )
            task_data.append({
                "Status": f"{icon} {t.status.value}",
                "Module": t.module_id,
                "Task ID": t.task_id[:8],
                "Duration (s)": f"{t.duration():.1f}" if t.duration() else "-",
                "Findings": len(t.result.findings) if t.result else "-",
            })
        st.dataframe(pd.DataFrame(task_data), use_container_width=True)


# ── Knowledge Base ────────────────────────────────────────────────────────────

elif page == "📚 Knowledge Base":
    st.header("📚 Knowledge Base")
    summary = KB.summary()
    cols = st.columns(3)
    cols[0].metric("Threats", summary["threats"])
    cols[1].metric("Attack Patterns", summary["attack_patterns"])
    cols[2].metric("Defense Records", summary["defenses"])

    severity_filter = st.selectbox("Filter by severity",
                                   ["all", "critical", "high", "medium", "low"])
    type_filter = st.selectbox("Filter by type",
                               ["all", "vulnerability", "malware", "ttp", "ioc", "signature"])

    threats = KB.query_threats(
        severity=None if severity_filter == "all" else severity_filter,
        type_filter=None if type_filter == "all" else type_filter,
        limit=100,
    )

    if threats:
        df = pd.DataFrame([{
            "Severity": severity_color(t.severity) + " " + t.severity,
            "Type": t.type,
            "Name": t.name[:60],
            "Source": t.source_module,
            "Confidence": f"{t.confidence:.0%}",
        } for t in threats])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No threats in knowledge base yet.")


# ── Other pages ───────────────────────────────────────────────────────────────

elif page in ("🪤 Deception", "🤖 AI Integrity", "🔬 Forensics"):
    module_map = {
        "🪤 Deception": ("module5_deception", "Deception & Honeypots",
                          "Configure and deploy honeypot services to track attackers."),
        "🤖 AI Integrity": ("module8_ai_integrity", "AI Model Integrity",
                            "Check ML models for poisoning and backdoors."),
        "🔬 Forensics": ("module10_forensics", "Digital Forensics & IR",
                         "Reconstruct attack timelines and generate incident reports."),
    }
    mid, mname, mdesc = module_map[page]
    st.header(page)
    st.markdown(mdesc)

    with st.form(f"{mid}_form"):
        path = st.text_input("Path / Target", value=".")
        submitted = st.form_submit_button(f"▶ Run {mname}")

    if submitted:
        with st.spinner(f"Running {mname}..."):
            task = ORCHESTRATOR.submit(mid, **{"incident_path" if "forensics" in mid else "scan_path": path})
            deadline = time.time() + 120
            while time.time() < deadline:
                t = ORCHESTRATOR.get_task(task.task_id)
                if t and t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    break
                time.sleep(2)
        t = ORCHESTRATOR.get_task(task.task_id)
        if t and t.result:
            st.json(t.result.metrics)
            for f in t.result.findings[:10]:
                st.write(f"{severity_color(f.get('severity', ''))} **{f.get('title', '')}**")
                st.caption(f.get("detail", ""))

elif page == "📊 Experiments":
    st.header("📊 Experiment Results")
    st.markdown("Run experiments from the command line: `python experiments/baseline_comparison.py`")
    results_dir = Path("results")
    if results_dir.exists():
        json_files = list(results_dir.glob("**/*.json"))
        for jf in json_files[:20]:
            with st.expander(jf.name):
                try:
                    data = json.loads(jf.read_text())
                    st.json(data)
                except Exception:
                    st.write(jf.read_text()[:1000])
    else:
        st.info("No results yet. Run experiments first.")

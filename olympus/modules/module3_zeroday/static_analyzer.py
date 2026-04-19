"""Static code analysis for zero-day discovery — AST + pattern matching."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olympus.core.logger import get_logger

log = get_logger("module3.static")


@dataclass
class StaticFinding:
    rule_id: str
    category: str
    severity: str
    title: str
    file: str
    line: int
    column: int
    snippet: str
    cwe: str = ""
    cve_pattern: str = ""


# ── Python AST-based checks ───────────────────────────────────────────────────

_DANGEROUS_CALLS = {
    # Command injection
    "os.system": ("CWE-78", "Command injection via os.system", "critical"),
    "subprocess.call": ("CWE-78", "Command injection risk in subprocess.call", "high"),
    "subprocess.Popen": ("CWE-78", "Command injection risk in subprocess.Popen", "high"),
    "eval": ("CWE-95", "Code injection via eval()", "critical"),
    "exec": ("CWE-94", "Code injection via exec()", "critical"),
    "pickle.loads": ("CWE-502", "Unsafe deserialization via pickle", "critical"),
    "marshal.loads": ("CWE-502", "Unsafe deserialization via marshal", "high"),
    "yaml.load": ("CWE-502", "Unsafe YAML load (use yaml.safe_load)", "high"),
    # SQL injection
    "cursor.execute": ("CWE-89", "Potential SQL injection (verify parameterization)", "medium"),
    # Path traversal
    "open": ("CWE-22", "Potential path traversal in file open", "low"),
    # Crypto
    "hashlib.md5": ("CWE-327", "Weak hash function MD5", "medium"),
    "hashlib.sha1": ("CWE-327", "Weak hash function SHA1", "medium"),
    "random.random": ("CWE-338", "Insecure random number generator", "medium"),
}

_HARDCODED_SECRETS = re.compile(
    r'(?:password|passwd|secret|api[_-]?key|token|private[_-]?key'
    r'|access[_-]?key|auth[_-]?token)\s*=\s*["\'][^"\']{4,}["\']',
    re.I,
)

_SQL_FORMAT_STRING = re.compile(
    r'(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER).*%[s|d|i]',
    re.I,
)

_INSECURE_CRYPTO = re.compile(
    r'\b(?:DES|3DES|RC4|MD5|SHA1|ECB|CBC)\b',
    re.I,
)


class PythonASTAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str, filename: str) -> None:
        self.source = source
        self.filename = filename
        self.lines = source.splitlines()
        self.findings: list[StaticFinding] = []

    def _snippet(self, lineno: int) -> str:
        if 0 < lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:120]
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        call_str = None
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                call_str = f"{node.func.value.id}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            call_str = node.func.id

        if call_str and call_str in _DANGEROUS_CALLS:
            cwe, title, severity = _DANGEROUS_CALLS[call_str]
            self.findings.append(StaticFinding(
                rule_id=f"PY-{call_str.upper().replace('.', '_')}",
                category="dangerous_api",
                severity=severity,
                title=title,
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                snippet=self._snippet(node.lineno),
                cwe=cwe,
            ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        snippet = self._snippet(node.lineno)
        if _HARDCODED_SECRETS.search(snippet):
            self.findings.append(StaticFinding(
                rule_id="PY-HARDCODED-SECRET",
                category="hardcoded_secret",
                severity="high",
                title="Hardcoded credential or secret",
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                snippet=snippet,
                cwe="CWE-798",
            ))
        self.generic_visit(node)


def _regex_scan(content: str, filename: str) -> list[StaticFinding]:
    findings = []
    for i, line in enumerate(content.splitlines(), 1):
        if _SQL_FORMAT_STRING.search(line):
            findings.append(StaticFinding(
                rule_id="REGEX-SQL-FORMAT",
                category="sql_injection",
                severity="critical",
                title="SQL query built with format string — injection risk",
                file=filename,
                line=i,
                column=0,
                snippet=line.strip()[:120],
                cwe="CWE-89",
            ))
        if _INSECURE_CRYPTO.search(line):
            findings.append(StaticFinding(
                rule_id="REGEX-INSECURE-CRYPTO",
                category="cryptographic_weakness",
                severity="medium",
                title=f"Insecure cryptographic algorithm reference",
                file=filename,
                line=i,
                column=0,
                snippet=line.strip()[:120],
                cwe="CWE-327",
            ))
    return findings


# ── C/C++ pattern checks (regex-based, no compiler needed) ────────────────────

_C_DANGEROUS = [
    (re.compile(r'\bstrcpy\s*\('), "CWE-121", "strcpy — no bounds check (buffer overflow)", "critical"),
    (re.compile(r'\bgets\s*\('), "CWE-121", "gets() — unconditional buffer overflow", "critical"),
    (re.compile(r'\bsprintf\s*\('), "CWE-134", "sprintf without length limit", "high"),
    (re.compile(r'\bprintf\s*\(\s*[a-zA-Z_]\w*\s*\)'), "CWE-134", "Uncontrolled format string", "critical"),
    (re.compile(r'\bmalloc\s*\([^)]*\+[^)]*\)'), "CWE-190", "Integer overflow in malloc size", "high"),
    (re.compile(r'\bfree\s*\(.*\).*\bfree\s*\('), "CWE-415", "Double free detected", "critical"),
    (re.compile(r'\bsystem\s*\('), "CWE-78", "Command injection via system()", "critical"),
    (re.compile(r'\bstrcat\s*\('), "CWE-120", "strcat — no bounds check", "high"),
]


def analyze_c_file(content: str, filename: str) -> list[StaticFinding]:
    findings = []
    for i, line in enumerate(content.splitlines(), 1):
        for pattern, cwe, title, severity in _C_DANGEROUS:
            if pattern.search(line):
                findings.append(StaticFinding(
                    rule_id=f"C-{cwe.replace('-', '_')}",
                    category="memory_safety",
                    severity=severity,
                    title=title,
                    file=filename,
                    line=i,
                    column=0,
                    snippet=line.strip()[:120],
                    cwe=cwe,
                ))
    return findings


def analyze_file(path: str | Path) -> list[StaticFinding]:
    path = Path(path)
    if not path.exists():
        return []

    content = path.read_text(errors="replace")
    findings: list[StaticFinding] = []

    ext = path.suffix.lower()
    if ext == ".py":
        findings.extend(_regex_scan(content, str(path)))
        try:
            tree = ast.parse(content)
            analyzer = PythonASTAnalyzer(content, str(path))
            analyzer.visit(tree)
            findings.extend(analyzer.findings)
        except SyntaxError as exc:
            log.debug("AST parse error in %s: %s", path, exc)
    elif ext in (".c", ".cpp", ".h", ".cc"):
        findings.extend(analyze_c_file(content, str(path)))
        findings.extend(_regex_scan(content, str(path)))
    else:
        findings.extend(_regex_scan(content, str(path)))

    log.debug("Static analysis of %s: %d findings", path, len(findings))
    return findings

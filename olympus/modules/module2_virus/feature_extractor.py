"""Static + behavioral feature extraction from files for malware detection."""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olympus.core.logger import get_logger

log = get_logger("module2.features")

# PE header magic
_PE_MAGIC = b"MZ"
_ELF_MAGIC = b"\x7fELF"

# Suspicious strings commonly found in malware
_SUSPICIOUS_STRINGS = [
    # Persistence
    b"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    b"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
    b"schtasks", b"at.exe", b"crontab",
    # Network
    b"socket", b"connect", b"recv", b"send", b"WSAStartup",
    b"InternetOpenUrl", b"WinHttpOpen",
    # Process injection
    b"VirtualAlloc", b"WriteProcessMemory", b"CreateRemoteThread",
    b"NtCreateThreadEx", b"RtlCreateUserThread",
    # Evasion
    b"IsDebuggerPresent", b"CheckRemoteDebuggerPresent", b"NtQueryInformationProcess",
    b"GetTickCount", b"Sleep",
    # Crypto / ransomware
    b"CryptEncrypt", b"CryptGenKey", b"BCryptEncrypt",
    b"bitcoin", b"monero", b"ransom",
    # Info stealing
    b"chrome\\User Data", b"firefox\\profiles", b"wallet.dat",
    b"keylogger", b"screenshot",
]

_SUSPICIOUS_URLS = re.compile(
    rb"https?://(?:\d{1,3}\.){3}\d{1,3}|"    # IP-based URLs
    rb"\.onion|"                               # Tor
    rb"pastebin\.com|"                         # common C2 staging
    rb"bit\.ly|t\.co",                         # URL shorteners
)


@dataclass
class FileFeatures:
    # File identity
    path: str = ""
    size_bytes: int = 0
    md5: str = ""
    sha256: str = ""

    # Entropy
    byte_entropy: float = 0.0
    section_entropies: list[float] = field(default_factory=list)

    # File type
    file_type: str = "unknown"   # "pe", "elf", "pdf", "script", "other"
    is_packed: bool = False

    # Suspicious indicators
    suspicious_string_count: int = 0
    suspicious_strings_found: list[str] = field(default_factory=list)
    suspicious_url_count: int = 0
    import_count: int = 0
    suspicious_imports: list[str] = field(default_factory=list)

    # PE-specific
    pe_sections: int = 0
    has_overlay: bool = False
    is_dotnet: bool = False
    compilation_timestamp: int = 0

    # Metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Convert to normalized feature vector for ML models."""
        v = [
            min(self.size_bytes / 1e7, 1.0),
            self.byte_entropy / 8.0,
            min(self.suspicious_string_count / 20.0, 1.0),
            min(self.suspicious_url_count / 5.0, 1.0),
            min(self.import_count / 200.0, 1.0),
            min(len(self.suspicious_imports) / 20.0, 1.0),
            1.0 if self.is_packed else 0.0,
            1.0 if self.has_overlay else 0.0,
            1.0 if self.is_dotnet else 0.0,
            min(self.pe_sections / 10.0, 1.0),
        ]
        # File type one-hot
        types = ["pe", "elf", "pdf", "script", "other"]
        for t in types:
            v.append(1.0 if self.file_type == t else 0.0)
        # Section entropy stats
        if self.section_entropies:
            v.append(max(self.section_entropies) / 8.0)
            v.append(sum(self.section_entropies) / len(self.section_entropies) / 8.0)
        else:
            v.extend([0.0, 0.0])
        # Pad to 32 dims
        while len(v) < 32:
            v.append(0.0)
        return v[:32]


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    entropy = 0.0
    n = len(data)
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _parse_pe(data: bytes, feats: FileFeatures) -> None:
    feats.file_type = "pe"
    try:
        # e_lfanew at offset 0x3c
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if e_lfanew + 4 > len(data):
            return
        pe_sig = data[e_lfanew:e_lfanew + 4]
        if pe_sig != b"PE\x00\x00":
            return

        # COFF header
        coff_off = e_lfanew + 4
        machine, num_sections, ts, _, _, opt_size, _ = struct.unpack_from("<HHIIIHH", data, coff_off)
        feats.compilation_timestamp = ts
        feats.pe_sections = num_sections

        # Optional header
        opt_off = coff_off + 20
        if opt_size >= 2:
            magic = struct.unpack_from("<H", data, opt_off)[0]
            feats.is_dotnet = False  # will check import dir

        # Section entropies
        section_table_off = opt_off + opt_size
        for i in range(min(num_sections, 96)):
            sec_off = section_table_off + i * 40
            if sec_off + 40 > len(data):
                break
            raw_size = struct.unpack_from("<I", data, sec_off + 16)[0]
            raw_ptr = struct.unpack_from("<I", data, sec_off + 20)[0]
            if raw_ptr and raw_size:
                sec_data = data[raw_ptr:raw_ptr + raw_size]
                feats.section_entropies.append(_byte_entropy(sec_data))

        # Overlay detection
        if num_sections > 0:
            last_sec_off = section_table_off + (num_sections - 1) * 40
            if last_sec_off + 40 <= len(data):
                last_raw = struct.unpack_from("<I", data, last_sec_off + 20)[0]
                last_size = struct.unpack_from("<I", data, last_sec_off + 16)[0]
                if last_raw + last_size < len(data):
                    feats.has_overlay = True

        # Packing heuristic: high entropy in sections
        if feats.section_entropies and max(feats.section_entropies) > 7.0:
            feats.is_packed = True

    except Exception as exc:
        log.debug("PE parse error: %s", exc)


def _parse_elf(data: bytes, feats: FileFeatures) -> None:
    feats.file_type = "elf"
    try:
        e_class = data[4]  # 1=32bit, 2=64bit
        e_type = struct.unpack_from(">H" if data[5] == 2 else "<H", data, 16)[0]
        feats.extra["elf_class"] = "64bit" if e_class == 2 else "32bit"
        feats.extra["elf_type"] = {2: "exec", 3: "dyn", 4: "core"}.get(e_type, "other")
        feats.byte_entropy = _byte_entropy(data)
        if feats.byte_entropy > 7.0:
            feats.is_packed = True
    except Exception as exc:
        log.debug("ELF parse error: %s", exc)


def _find_suspicious_content(data: bytes, feats: FileFeatures) -> None:
    for sig in _SUSPICIOUS_STRINGS:
        if sig in data:
            feats.suspicious_string_count += 1
            feats.suspicious_strings_found.append(sig.decode("utf-8", errors="replace")[:60])

    url_matches = _SUSPICIOUS_URLS.findall(data)
    feats.suspicious_url_count = len(url_matches)


def extract_features(path: str | Path) -> FileFeatures:
    path = Path(path)
    feats = FileFeatures(path=str(path))

    if not path.exists():
        log.warning("File not found: %s", path)
        return feats

    data = path.read_bytes()
    feats.size_bytes = len(data)
    feats.md5 = hashlib.md5(data).hexdigest()
    feats.sha256 = hashlib.sha256(data).hexdigest()
    feats.byte_entropy = _byte_entropy(data)

    # Detect file type
    if data[:2] == _PE_MAGIC:
        _parse_pe(data, feats)
    elif data[:4] == _ELF_MAGIC:
        _parse_elf(data, feats)
    elif data[:4] == b"%PDF":
        feats.file_type = "pdf"
    elif path.suffix.lower() in (".py", ".js", ".sh", ".ps1", ".bat", ".vbs"):
        feats.file_type = "script"

    _find_suspicious_content(data, feats)
    return feats

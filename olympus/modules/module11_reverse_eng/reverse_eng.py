"""
module11_reverse_eng — Static Reverse Engineering Module

Performs static binary analysis without execution:
  - PE/ELF header parsing and metadata extraction
  - String extraction (ASCII/Unicode)
  - Import/export table analysis
  - Entropy calculation (packing/obfuscation detection)
  - Control-flow graph skeleton via opcode frequency analysis
  - Disassembly via capstone (optional)
  - YARA rule matching (optional)

Author: George David Tsitlauri
Project: OLYMPUS — Unified AI Security Research Platform
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from olympus.core.base_module import BaseModule, ModuleResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BinaryMetadata:
    file_path: str
    file_size: int
    md5: str
    sha256: str
    file_type: str           # 'PE', 'ELF', 'Mach-O', 'unknown'
    architecture: str        # 'x86', 'x64', 'ARM', 'unknown'
    entry_point: int
    sections: List[Dict]
    imports: List[str]
    exports: List[str]
    strings: List[str]
    entropy: float
    is_packed: bool
    suspicious_indicators: List[str]
    extra: Dict = field(default_factory=dict)


@dataclass
class DisassemblyResult:
    instructions: List[Dict]   # [{address, mnemonic, op_str}, ...]
    opcode_freq: Dict[str, int]
    call_targets: List[int]
    jump_targets: List[int]


# ---------------------------------------------------------------------------
# PE Parser (pure Python, no external deps)
# ---------------------------------------------------------------------------

class PEParser:
    """Minimal PE32/PE64 header parser."""

    DOS_MAGIC = b'MZ'
    PE_MAGIC = b'PE\x00\x00'

    def parse(self, data: bytes) -> Dict:
        result = {
            'file_type': 'unknown',
            'architecture': 'unknown',
            'entry_point': 0,
            'sections': [],
            'imports': [],
            'exports': [],
        }
        if len(data) < 64 or data[:2] != self.DOS_MAGIC:
            return result

        pe_offset = struct.unpack_from('<I', data, 0x3C)[0]
        if pe_offset + 4 > len(data):
            return result
        if data[pe_offset:pe_offset + 4] != self.PE_MAGIC:
            return result

        result['file_type'] = 'PE'
        machine = struct.unpack_from('<H', data, pe_offset + 4)[0]
        result['architecture'] = {
            0x014c: 'x86',
            0x8664: 'x64',
            0x01c0: 'ARM',
            0xaa64: 'ARM64',
        }.get(machine, f'unknown(0x{machine:04x})')

        num_sections = struct.unpack_from('<H', data, pe_offset + 6)[0]
        opt_header_size = struct.unpack_from('<H', data, pe_offset + 20)[0]

        # Optional header magic
        opt_offset = pe_offset + 24
        if opt_offset + 2 <= len(data):
            opt_magic = struct.unpack_from('<H', data, opt_offset)[0]
            ep_offset = opt_offset + 16
            if ep_offset + 4 <= len(data):
                result['entry_point'] = struct.unpack_from('<I', data, ep_offset)[0]

        # Section headers
        section_offset = pe_offset + 24 + opt_header_size
        for i in range(min(num_sections, 32)):
            off = section_offset + i * 40
            if off + 40 > len(data):
                break
            name = data[off:off + 8].rstrip(b'\x00').decode('ascii', errors='replace')
            vsize = struct.unpack_from('<I', data, off + 8)[0]
            vaddr = struct.unpack_from('<I', data, off + 12)[0]
            raw_size = struct.unpack_from('<I', data, off + 16)[0]
            raw_ptr = struct.unpack_from('<I', data, off + 20)[0]
            characteristics = struct.unpack_from('<I', data, off + 36)[0]

            sec_data = data[raw_ptr:raw_ptr + raw_size] if raw_ptr + raw_size <= len(data) else b''
            entropy = _entropy(sec_data) if sec_data else 0.0

            result['sections'].append({
                'name': name,
                'virtual_address': vaddr,
                'virtual_size': vsize,
                'raw_size': raw_size,
                'entropy': round(entropy, 3),
                'executable': bool(characteristics & 0x20000000),
                'writable': bool(characteristics & 0x80000000),
            })

        return result


class ELFParser:
    """Minimal ELF header parser."""

    ELF_MAGIC = b'\x7fELF'

    def parse(self, data: bytes) -> Dict:
        result = {
            'file_type': 'unknown',
            'architecture': 'unknown',
            'entry_point': 0,
            'sections': [],
            'imports': [],
            'exports': [],
        }
        if len(data) < 16 or data[:4] != self.ELF_MAGIC:
            return result

        result['file_type'] = 'ELF'
        ei_class = data[4]
        ei_data = data[5]
        e_machine = struct.unpack_from('<H' if ei_data == 1 else '>H', data, 18)[0]

        result['architecture'] = {
            0x03: 'x86',
            0x3e: 'x64',
            0x28: 'ARM',
            0xb7: 'ARM64',
            0x08: 'MIPS',
        }.get(e_machine, f'unknown(0x{e_machine:04x})')

        if ei_class == 1:   # 32-bit
            if len(data) >= 24:
                result['entry_point'] = struct.unpack_from('<I' if ei_data == 1 else '>I', data, 24)[0]
        elif ei_class == 2:  # 64-bit
            if len(data) >= 32:
                result['entry_point'] = struct.unpack_from('<Q' if ei_data == 1 else '>Q', data, 24)[0]

        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _extract_strings(data: bytes, min_len: int = 5) -> List[str]:
    """Extract printable ASCII and wide (UTF-16LE) strings."""
    ascii_pattern = re.compile(rb'[ -~]{' + str(min_len).encode() + rb',}')
    ascii_strings = [m.group().decode('ascii') for m in ascii_pattern.finditer(data)]

    wide_pattern = re.compile(rb'(?:[ -~]\x00){' + str(min_len).encode() + rb',}')
    wide_strings = []
    for m in wide_pattern.finditer(data):
        try:
            wide_strings.append(m.group().decode('utf-16-le'))
        except Exception:
            pass

    return list(dict.fromkeys(ascii_strings + wide_strings))  # deduplicate, preserve order


def _suspicious_strings(strings: List[str]) -> List[str]:
    """Flag strings matching common malicious patterns."""
    indicators = []
    patterns = [
        (r'(?i)(cmd\.exe|powershell|wscript|cscript)', 'Shell execution'),
        (r'(?i)(CreateRemoteThread|VirtualAllocEx|WriteProcessMemory)', 'Process injection API'),
        (r'(?i)(RegSetValue|RegCreateKey|HKLM|HKCU)', 'Registry manipulation'),
        (r'(?i)(WinExec|ShellExecute|CreateProcess)', 'Process creation API'),
        (r'(?i)(socket|connect|recv|send|WSAStartup)', 'Network API'),
        (r'(?i)(CryptEncrypt|CryptDecrypt|AES|RC4)', 'Cryptographic API'),
        (r'(?i)(DeleteFile|MoveFile|CopyFile)', 'File manipulation'),
        (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'Hardcoded IP address'),
        (r'(?i)(http|https|ftp)://\S+', 'Hardcoded URL'),
        (r'(?i)(password|passwd|secret|token|apikey)', 'Credential string'),
        (r'(?i)(UPX|MPRESS|PECompact|Themida)', 'Known packer string'),
    ]
    seen = set()
    for s in strings:
        for pattern, label in patterns:
            if re.search(pattern, s) and label not in seen:
                indicators.append(f'{label}: {s[:80]}')
                seen.add(label)
                break
    return indicators


YARA_RULES_BUILTIN = [
    {
        'name': 'UPX_packed',
        'condition': lambda strings: any('UPX' in s for s in strings),
        'description': 'Binary appears to be UPX-packed',
    },
    {
        'name': 'high_entropy_section',
        'condition': lambda sections: any(s.get('entropy', 0) > 7.0 for s in sections),
        'description': 'Section entropy > 7.0 — possible packing or encryption',
    },
    {
        'name': 'process_injection_api',
        'condition': lambda strings: any(
            api in s for s in strings
            for api in ('VirtualAllocEx', 'WriteProcessMemory', 'CreateRemoteThread')
        ),
        'description': 'Process injection APIs detected',
    },
    {
        'name': 'network_and_crypt',
        'condition': lambda strings: (
            any(api in s for s in strings for api in ('socket', 'connect', 'recv', 'send')) and
            any(api in s for s in strings for api in ('CryptEncrypt', 'CryptDecrypt', 'AES'))
        ),
        'description': 'Network + cryptographic APIs — possible C2 or ransomware behavior',
    },
]


# ---------------------------------------------------------------------------
# Optional disassembly via capstone
# ---------------------------------------------------------------------------

def _disassemble(data: bytes, arch: str, entry_point: int, max_bytes: int = 4096) -> Optional[DisassemblyResult]:
    try:
        import capstone

        arch_map = {
            'x86': (capstone.CS_ARCH_X86, capstone.CS_MODE_32),
            'x64': (capstone.CS_ARCH_X86, capstone.CS_MODE_64),
            'ARM': (capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM),
            'ARM64': (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM),
        }
        if arch not in arch_map:
            return None

        cs_arch, cs_mode = arch_map[arch]
        md = capstone.Cs(cs_arch, cs_mode)
        md.detail = False

        snippet = data[:max_bytes]
        instructions = []
        opcode_freq: Dict[str, int] = {}
        call_targets = []
        jump_targets = []

        for insn in md.disasm(snippet, entry_point):
            instructions.append({
                'address': insn.address,
                'mnemonic': insn.mnemonic,
                'op_str': insn.op_str,
            })
            opcode_freq[insn.mnemonic] = opcode_freq.get(insn.mnemonic, 0) + 1

            if insn.mnemonic.startswith('call'):
                try:
                    call_targets.append(int(insn.op_str, 16))
                except ValueError:
                    pass
            elif insn.mnemonic.startswith('j'):
                try:
                    jump_targets.append(int(insn.op_str, 16))
                except ValueError:
                    pass

        return DisassemblyResult(
            instructions=instructions,
            opcode_freq=opcode_freq,
            call_targets=call_targets,
            jump_targets=jump_targets,
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class ReverseEngineeringModule(BaseModule):
    """
    Module 11 — Static Reverse Engineering

    Analyzes binary files without execution. Produces:
      - File metadata and hashes
      - PE/ELF header parsing
      - String extraction and triage
      - Entropy analysis (packing detection)
      - Import/export table
      - Optional disassembly (requires capstone)
      - Built-in YARA-style rule matching
    """

    MODULE_ID = "module11_reverse_eng"
    MODULE_NAME = "Reverse Engineering"
    MODULE_TYPE = "forensic"

    def run(self, binary_path: str = '', binary_data: Optional[bytes] = None, **kwargs) -> ModuleResult:
        result, t0 = self._start_result()

        if binary_data is None:
            if not binary_path or not os.path.isfile(binary_path):
                result.status = 'error'
                result.error = f'File not found: {binary_path}'
                return self._finish_result(result, t0)
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            file_path = binary_path
        else:
            file_path = binary_path or '<in-memory>'

        metadata = self._analyze(file_path, binary_data)
        yara_hits = self._run_yara(metadata)
        disasm = _disassemble(binary_data, metadata.architecture, metadata.entry_point)
        risk = self._risk_score(metadata, yara_hits)

        disasm_data = {
            'instruction_count': len(disasm.instructions),
            'top_opcodes': sorted(disasm.opcode_freq.items(), key=lambda x: -x[1])[:15],
            'call_targets': disasm.call_targets[:20],
            'jump_targets': disasm.jump_targets[:20],
            'sample_instructions': disasm.instructions[:30],
        } if disasm else {'note': 'capstone not installed — disassembly skipped'}

        severity = 'critical' if risk >= 75 else 'high' if risk >= 50 else 'medium' if risk >= 25 else 'low'

        result.add_finding(
            severity=severity,
            title=f'Static analysis: {metadata.file_type} ({metadata.architecture})',
            detail=f'Entropy={metadata.entropy}, packed={metadata.is_packed}, risk={risk}/100',
            metadata={
                'file_path': metadata.file_path,
                'file_size': metadata.file_size,
                'md5': metadata.md5,
                'sha256': metadata.sha256,
                'file_type': metadata.file_type,
                'architecture': metadata.architecture,
                'entry_point': hex(metadata.entry_point),
                'entropy': metadata.entropy,
                'is_packed': metadata.is_packed,
                'sections': metadata.sections,
                'imports': metadata.imports[:50],
                'exports': metadata.exports[:50],
                'strings_count': len(metadata.strings),
                'strings_sample': metadata.strings[:30],
            },
            suspicious_indicators=metadata.suspicious_indicators,
            yara_hits=yara_hits,
            risk_score=risk,
            disassembly=disasm_data,
        )

        for indicator in metadata.suspicious_indicators[:5]:
            result.add_finding(
                severity='high',
                title='Suspicious indicator',
                detail=indicator,
            )

        result.metrics = {
            'risk_score': float(risk),
            'entropy': metadata.entropy,
            'strings_count': float(len(metadata.strings)),
            'yara_hits': float(len(yara_hits)),
            'suspicious_indicators': float(len(metadata.suspicious_indicators)),
            'sections_count': float(len(metadata.sections)),
        }

        return self._finish_result(result, t0)

    def _analyze(self, file_path: str, data: bytes) -> BinaryMetadata:
        md5 = hashlib.md5(data).hexdigest()
        sha256 = hashlib.sha256(data).hexdigest()
        entropy = _entropy(data)

        pe_parser = PEParser()
        elf_parser = ELFParser()

        pe_info = pe_parser.parse(data)
        if pe_info['file_type'] == 'PE':
            parsed = pe_info
        else:
            elf_info = elf_parser.parse(data)
            parsed = elf_info if elf_info['file_type'] == 'ELF' else pe_info

        strings = _extract_strings(data)
        suspicious = _suspicious_strings(strings)

        is_packed = (
            entropy > 7.2 or
            any(s.get('entropy', 0) > 7.0 for s in parsed.get('sections', [])) or
            any('UPX' in s or 'MPRESS' in s or 'Themida' in s for s in strings)
        )

        return BinaryMetadata(
            file_path=file_path,
            file_size=len(data),
            md5=md5,
            sha256=sha256,
            file_type=parsed['file_type'],
            architecture=parsed['architecture'],
            entry_point=parsed['entry_point'],
            sections=parsed.get('sections', []),
            imports=parsed.get('imports', []),
            exports=parsed.get('exports', []),
            strings=strings,
            entropy=round(entropy, 4),
            is_packed=is_packed,
            suspicious_indicators=suspicious,
        )

    def _run_yara(self, metadata: BinaryMetadata) -> List[Dict]:
        hits = []
        for rule in YARA_RULES_BUILTIN:
            try:
                if rule['name'] == 'high_entropy_section':
                    matched = rule['condition'](metadata.sections)
                else:
                    matched = rule['condition'](metadata.strings)
                if matched:
                    hits.append({'rule': rule['name'], 'description': rule['description']})
            except Exception:
                pass
        return hits

    def _risk_score(self, metadata: BinaryMetadata, yara_hits: List[Dict]) -> int:
        score = 0
        score += len(metadata.suspicious_indicators) * 10
        score += len(yara_hits) * 15
        if metadata.is_packed:
            score += 25
        if metadata.entropy > 7.5:
            score += 20
        return min(score, 100)

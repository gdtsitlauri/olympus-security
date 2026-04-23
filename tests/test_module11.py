"""Tests for module11 — Reverse Engineering"""
import pytest
from olympus.modules.module11_reverse_eng import ReverseEngineeringModule
from olympus.modules.module11_reverse_eng.reverse_eng import (
    _entropy, _extract_strings, _suspicious_strings, PEParser, ELFParser,
)


class TestHelpers:
    def test_entropy_empty(self):
        assert _entropy(b'') == 0.0

    def test_entropy_uniform(self):
        data = bytes(range(256)) * 4
        assert abs(_entropy(data) - 8.0) < 0.01

    def test_entropy_constant(self):
        assert _entropy(b'\x00' * 100) == 0.0

    def test_extract_strings_ascii(self):
        data = b'hello world this is a test string here'
        strings = _extract_strings(data, min_len=5)
        assert any('hello' in s for s in strings)

    def test_extract_strings_min_len(self):
        data = b'hello world this is a test string'
        strings = _extract_strings(data, min_len=5)
        assert all(len(s) >= 5 for s in strings)

    def test_suspicious_strings_injection_api(self):
        strings = ['VirtualAllocEx is called', 'WriteProcessMemory here', 'CreateRemoteThread']
        hits = _suspicious_strings(strings)
        assert any('injection' in h.lower() for h in hits)

    def test_suspicious_strings_url(self):
        strings = ['download from http://evil.com/payload.exe']
        hits = _suspicious_strings(strings)
        assert any('URL' in h or 'url' in h.lower() for h in hits)

    def test_suspicious_strings_clean(self):
        strings = ['Hello World', 'This is clean text with nothing bad']
        hits = _suspicious_strings(strings)
        assert len(hits) == 0


class TestPEParser:
    def test_non_pe_returns_unknown(self):
        parser = PEParser()
        result = parser.parse(b'\x00' * 256)
        assert result['file_type'] == 'unknown'

    def test_mz_header_detected(self):
        parser = PEParser()
        data = b'MZ' + b'\x00' * 62 + b'\x3c\x00\x00\x00' + b'\x00' * 200
        result = parser.parse(data)
        assert 'file_type' in result


class TestELFParser:
    def test_non_elf_returns_unknown(self):
        parser = ELFParser()
        result = parser.parse(b'\x00' * 64)
        assert result['file_type'] == 'unknown'

    def test_elf_magic_detected(self):
        parser = ELFParser()
        data = b'\x7fELF' + bytes(60)
        result = parser.parse(data)
        assert result['file_type'] == 'ELF'


class TestReverseEngineeringModule:
    def setup_method(self):
        self.module = ReverseEngineeringModule()

    def _first_finding(self, result):
        return result.findings[0] if result.findings else {}

    def test_missing_file_returns_error(self):
        result = self.module.run(binary_path='/nonexistent/file.exe')
        assert result.status == 'error'

    def test_binary_data_analysis_completes(self):
        data = b'hello world this is a test binary with some content in it for analysis'
        result = self.module.run(binary_path='test.bin', binary_data=data)
        assert result.status == 'success'
        assert len(result.findings) > 0

    def test_high_entropy_detected(self):
        import os
        data = os.urandom(4096)
        result = self.module.run(binary_path='packed.bin', binary_data=data)
        assert result.metrics['entropy'] > 6.0

    def test_yara_process_injection(self):
        data = b'VirtualAllocEx WriteProcessMemory CreateRemoteThread some padding here'
        result = self.module.run(binary_path='injector.exe', binary_data=data)
        finding = self._first_finding(result)
        yara_hits = finding.get('yara_hits', [])
        yara_names = [h['rule'] for h in yara_hits]
        assert 'process_injection_api' in yara_names

    def test_strings_extracted(self):
        payload = b'Hello World This Is A Long Enough String For Extraction Indeed'
        result = self.module.run(binary_path='test.bin', binary_data=payload)
        finding = self._first_finding(result)
        assert finding.get('metadata', {}).get('strings_count', 0) > 0

    def test_risk_score_range(self):
        data = b'clean binary with no suspicious content at all'
        result = self.module.run(binary_path='clean.bin', binary_data=data)
        assert 0 <= result.metrics['risk_score'] <= 100

    def test_elf_binary_detected(self):
        data = b'\x7fELF' + bytes(200)
        result = self.module.run(binary_path='elf_binary', binary_data=data)
        finding = self._first_finding(result)
        assert finding.get('metadata', {}).get('file_type') == 'ELF'

    def test_metrics_populated(self):
        data = b'some binary data for testing metrics population here'
        result = self.module.run(binary_path='test.bin', binary_data=data)
        assert 'risk_score' in result.metrics
        assert 'entropy' in result.metrics
        assert 'strings_count' in result.metrics

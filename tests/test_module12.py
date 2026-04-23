"""Tests for module12 — Malware Analysis"""
import pytest
from olympus.modules.module12_malware_analysis import MalwareAnalysisModule
from olympus.modules.module12_malware_analysis.malware_analysis import (
    BehavioralClassifier, SandboxSimulator, _extract_iocs,
)


class TestIOCExtraction:
    def test_ip_extraction(self):
        iocs = _extract_iocs(['connect to 10.0.0.1 port 4444'])
        ips = [i for i in iocs if i['type'] == 'ip']
        assert any(i['value'] == '10.0.0.1' for i in ips)

    def test_url_extraction(self):
        iocs = _extract_iocs(['download from http://evil.com/payload.exe'])
        urls = [i for i in iocs if i['type'] == 'url']
        assert len(urls) > 0

    def test_domain_extraction(self):
        iocs = _extract_iocs(['beacon to malware.ru every 60 seconds'])
        domains = [i for i in iocs if i['type'] == 'domain']
        assert any('malware.ru' in i['value'] for i in domains)

    def test_email_extraction(self):
        iocs = _extract_iocs(['send logs to attacker@evil.net'])
        emails = [i for i in iocs if i['type'] == 'email']
        assert len(emails) > 0

    def test_clean_strings(self):
        iocs = _extract_iocs(['hello world', 'no indicators here'])
        assert len(iocs) == 0


class TestBehavioralClassifier:
    def setup_method(self):
        self.clf = BehavioralClassifier()

    def test_ransomware_classification(self):
        apis = ['CryptEncrypt', 'CryptGenKey', 'FindFirstFile', 'FindNextFile',
                'WriteFile', 'DeleteFile', 'GetLogicalDrives']
        result = self.clf.classify(apis, [], [], [])
        assert result.family == 'ransomware'
        assert result.severity == 'critical'

    def test_trojan_classification(self):
        apis = ['CreateRemoteThread', 'VirtualAllocEx', 'WriteProcessMemory',
                'socket', 'connect', 'send', 'recv']
        result = self.clf.classify(apis, [], [], [])
        assert result.family == 'trojan'

    def test_clean_classification(self):
        apis = ['MessageBoxA', 'GetLocalTime', 'ExitProcess']
        result = self.clf.classify(apis, [], [], [])
        assert result.family == 'clean'
        assert result.threat_score == 0

    def test_confidence_range(self):
        apis = ['CryptEncrypt', 'FindFirstFile', 'WriteFile']
        result = self.clf.classify(apis, [], [], [])
        assert 0.0 <= result.confidence <= 1.0

    def test_threat_score_range(self):
        apis = ['socket', 'connect', 'CryptEncrypt', 'CreateRemoteThread']
        result = self.clf.classify(apis, [], [], [])
        assert 0 <= result.threat_score <= 100

    def test_evidence_populated(self):
        apis = ['VirtualAllocEx', 'WriteProcessMemory', 'CreateRemoteThread']
        result = self.clf.classify(apis, [], [], [])
        assert len(result.evidence) > 0


class TestSandboxSimulator:
    def setup_method(self):
        self.sandbox = SandboxSimulator()

    def test_profile_structure(self):
        profile = self.sandbox.simulate([], [], [], 'abc123')
        assert hasattr(profile, 'api_calls')
        assert hasattr(profile, 'iocs')
        assert hasattr(profile, 'ttp_mapping')

    def test_network_inference(self):
        strings = ['connect to http://evil.com/c2 for updates']
        profile = self.sandbox.simulate(strings, [], [], 'hash1')
        assert len(profile.network_connections) > 0

    def test_file_op_inference(self):
        strings = [r'C:\Windows\Temp\payload.exe']
        profile = self.sandbox.simulate(strings, [], [], 'hash2')
        assert len(profile.file_operations) > 0

    def test_api_from_imports(self):
        imports = ['VirtualAllocEx', 'CreateRemoteThread']
        profile = self.sandbox.simulate([], imports, [], 'hash3')
        assert 'VirtualAllocEx' in profile.api_calls


class TestMalwareAnalysisModule:
    def setup_method(self):
        self.module = MalwareAnalysisModule()

    def _first_finding(self, result):
        return result.findings[0] if result.findings else {}

    def test_basic_run_completes(self):
        result = self.module.run(strings=[], imports=[])
        assert result.status == 'success'
        assert len(result.findings) > 0

    def test_ransomware_detected(self):
        strings = ['CryptEncrypt keys', 'FindFirstFile C:\\Users', 'README_DECRYPT.txt']
        imports = ['CryptEncrypt', 'FindFirstFile', 'WriteFile', 'DeleteFile']
        result = self.module.run(strings=strings, imports=imports)
        finding = self._first_finding(result)
        assert finding['classification']['family'] == 'ransomware'
        assert finding['classification']['severity'] == 'critical'

    def test_iocs_extracted(self):
        strings = ['beacon http://c2.evil.ru/gate.php', 'connect 45.33.32.156']
        result = self.module.run(strings=strings, imports=[])
        finding = self._first_finding(result)
        assert len(finding.get('iocs', [])) > 0

    def test_ttp_mapping(self):
        imports = ['CreateRemoteThread', 'VirtualAllocEx', 'WriteProcessMemory']
        result = self.module.run(strings=[], imports=imports)
        finding = self._first_finding(result)
        ttps = finding.get('ttp_mapping', [])
        techniques = [t['technique'] for t in ttps]
        assert 'T1055' in techniques

    def test_summary_string(self):
        result = self.module.run(strings=[], imports=[])
        finding = self._first_finding(result)
        assert isinstance(finding.get('summary', ''), str)
        assert 'classified' in finding.get('summary', '').lower()

    def test_metrics_populated(self):
        result = self.module.run(strings=[], imports=[])
        assert 'threat_score' in result.metrics
        assert 'iocs_count' in result.metrics
        assert 'ttp_count' in result.metrics

    def test_integration_with_static_result(self):
        static_result = {
            'metadata': {
                'strings_sample': ['VirtualAllocEx', 'socket', 'connect', '192.168.1.1'],
                'imports': ['VirtualAllocEx', 'WriteProcessMemory'],
                'sha256': 'abc123def456',
            },
            'suspicious_indicators': ['Process injection API: VirtualAllocEx'],
        }
        result = self.module.run(static_result=static_result)
        assert result.status == 'success'
        finding = self._first_finding(result)
        assert finding['classification']['family'] != ''

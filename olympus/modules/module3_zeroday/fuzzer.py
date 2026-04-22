"""AI-guided fuzzer for zero-day discovery — mutation + generation strategies."""

from __future__ import annotations

import hashlib
import os
import random
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from olympus.core.device import get_device
from olympus.core.logger import get_logger

log = get_logger("module3.fuzzer")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False
    nn = type("_TorchNNFallback", (), {"Module": object})()  # type: ignore[assignment]


# ── Crash result ──────────────────────────────────────────────────────────────

@dataclass
class CrashResult:
    crash_id: str
    input_hash: str
    crash_type: str         # "segfault", "abort", "timeout", "oom", "assertion"
    signal: Optional[int]
    exit_code: int
    input_bytes: bytes
    backtrace: str = ""
    severity: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    is_unique: bool = True

    def to_poc(self) -> str:
        """Generate minimal PoC description."""
        hex_input = self.input_bytes[:64].hex()
        return (
            f"# PoC for {self.crash_id}\n"
            f"# Crash type: {self.crash_type}\n"
            f"# Signal: {self.signal}\n"
            f"# Input (hex): {hex_input}{'...' if len(self.input_bytes) > 64 else ''}\n"
        )


# ── Mutation strategies ───────────────────────────────────────────────────────

class MutationEngine:
    """Grammar-free mutation engine with AFL-style bit flipping + havoc."""

    _INTERESTING_8  = [0, 1, 0x7F, 0x80, 0xFF, 0xFE]
    _INTERESTING_16 = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x100, 0x200]
    _INTERESTING_32 = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100, 0x1000]

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def mutate(self, data: bytes, strategy: str = "havoc") -> bytes:
        strategies = {
            "bit_flip": self._bit_flip,
            "byte_flip": self._byte_flip,
            "interesting_values": self._interesting_values,
            "arithmetic": self._arithmetic,
            "splice": self._splice,
            "havoc": self._havoc,
            "dictionary": self._dictionary,
        }
        fn = strategies.get(strategy, self._havoc)
        return fn(data)

    def _bit_flip(self, data: bytes, num_bits: int = 1) -> bytes:
        if not data:
            return data
        arr = bytearray(data)
        for _ in range(num_bits):
            pos = self._rng.randrange(len(arr) * 8)
            arr[pos // 8] ^= 1 << (pos % 8)
        return bytes(arr)

    def _byte_flip(self, data: bytes) -> bytes:
        if not data:
            return data
        arr = bytearray(data)
        pos = self._rng.randrange(len(arr))
        arr[pos] ^= 0xFF
        return bytes(arr)

    def _interesting_values(self, data: bytes) -> bytes:
        if len(data) < 4:
            return data
        arr = bytearray(data)
        pos = self._rng.randrange(len(arr))
        choice = self._rng.choice(["8", "16", "32"])
        if choice == "8":
            arr[pos] = self._rng.choice(self._INTERESTING_8)
        elif choice == "16" and pos + 2 <= len(arr):
            val = self._rng.choice(self._INTERESTING_16)
            struct.pack_into("<H", arr, pos, val & 0xFFFF)
        elif pos + 4 <= len(arr):
            val = self._rng.choice(self._INTERESTING_32)
            struct.pack_into("<I", arr, pos, val & 0xFFFFFFFF)
        return bytes(arr)

    def _arithmetic(self, data: bytes) -> bytes:
        if not data:
            return data
        arr = bytearray(data)
        pos = self._rng.randrange(len(arr))
        delta = self._rng.randint(-35, 35)
        arr[pos] = (arr[pos] + delta) & 0xFF
        return bytes(arr)

    def _splice(self, data: bytes) -> bytes:
        """Splice two halves from independently mutated versions."""
        if len(data) < 4:
            return data
        mid = self._rng.randrange(1, len(data))
        left = data[:mid]
        right = self._bit_flip(data[mid:], num_bits=self._rng.randint(1, 4))
        return left + right

    def _havoc(self, data: bytes) -> bytes:
        """Apply random combination of mutations."""
        n_mutations = self._rng.randint(1, 8)
        ops = ["bit_flip", "byte_flip", "interesting_values", "arithmetic"]
        result = data
        for _ in range(n_mutations):
            op = self._rng.choice(ops)
            result = self.mutate(result, op)
        return result

    def _dictionary(self, data: bytes) -> bytes:
        """Inject format-string or boundary tokens."""
        tokens = [
            b"%s", b"%n", b"%x", b"../../../etc/passwd",
            b"A" * 256, b"\x00" * 16, b"\xff" * 16,
            b"'; DROP TABLE", b"<script>",
            b"\x7fELF", b"MZ\x90\x00",
        ]
        if not data:
            return self._rng.choice(tokens)
        arr = bytearray(data)
        token = self._rng.choice(tokens)
        pos = self._rng.randrange(max(1, len(arr) - len(token)))
        arr[pos:pos + len(token)] = token
        return bytes(arr)

    def generate_seeds(self, n: int = 10, min_size: int = 4, max_size: int = 1024) -> list[bytes]:
        """Generate initial random seed corpus."""
        seeds = []
        for _ in range(n):
            size = self._rng.randint(min_size, max_size)
            seeds.append(bytes(self._rng.getrandbits(8) for _ in range(size)))
        return seeds


# ── Neural seed generator ─────────────────────────────────────────────────────

class NeuralSeedGenerator(nn.Module):  # type: ignore[misc]
    """VAE-based seed generator that learns the input space distribution."""

    def __init__(self, input_dim: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        import torch
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, n: int = 1) -> "torch.Tensor":
        import torch
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decode(z)

    def loss(self, recon, x, mu, log_var) -> "torch.Tensor":
        import torch
        recon_loss = torch.nn.functional.binary_cross_entropy(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld


# ── Target executor ───────────────────────────────────────────────────────────

class TargetExecutor:
    """Execute fuzzing targets and detect crashes."""

    TIMEOUT = 5.0

    def __init__(self, target_cmd: list[str], stdin_mode: bool = True,
                 file_ext: str = ".bin") -> None:
        self.target_cmd = target_cmd
        self.stdin_mode = stdin_mode
        self.file_ext = file_ext
        self._crash_hashes: set[str] = set()

    def execute(self, data: bytes) -> Optional[CrashResult]:
        try:
            if self.stdin_mode:
                r = subprocess.run(
                    self.target_cmd,
                    input=data,
                    capture_output=True,
                    timeout=self.TIMEOUT,
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=self.file_ext, delete=False) as f:
                    f.write(data)
                    fname = f.name
                try:
                    r = subprocess.run(
                        self.target_cmd + [fname],
                        capture_output=True,
                        timeout=self.TIMEOUT,
                    )
                finally:
                    try:
                        os.unlink(fname)
                    except Exception:
                        pass

            return self._analyze_result(data, r)
        except subprocess.TimeoutExpired:
            h = hashlib.md5(data).hexdigest()
            return CrashResult(
                crash_id=h[:8],
                input_hash=h,
                crash_type="timeout",
                signal=None,
                exit_code=-1,
                input_bytes=data,
                severity="medium",
            )
        except Exception as exc:
            log.debug("Execution error: %s", exc)
            return None

    def _analyze_result(self, data: bytes, r: subprocess.CompletedProcess) -> Optional[CrashResult]:
        crash_type = None
        signal_num = None

        if r.returncode < 0:
            signal_num = -r.returncode
            crash_map = {
                11: "segfault",    # SIGSEGV
                6:  "abort",       # SIGABRT
                4:  "illegal_instr",  # SIGILL
                8:  "fpe",         # SIGFPE
                7:  "bus_error",   # SIGBUS
            }
            crash_type = crash_map.get(signal_num, f"signal_{signal_num}")
        elif r.returncode != 0:
            stderr = r.stderr.decode("utf-8", errors="replace")
            if "assertion" in stderr.lower():
                crash_type = "assertion"
            elif "runtime error" in stderr.lower():
                crash_type = "runtime_error"

        if not crash_type:
            return None

        h = hashlib.sha256(data).hexdigest()
        # Deduplicate by hash
        if h in self._crash_hashes:
            return None
        self._crash_hashes.add(h)

        severity = "critical" if signal_num in (11, 6) else "high"

        return CrashResult(
            crash_id=h[:8],
            input_hash=h,
            crash_type=crash_type,
            signal=signal_num,
            exit_code=r.returncode,
            input_bytes=data[:4096],
            backtrace=r.stderr.decode("utf-8", errors="replace")[:2000],
            severity=severity,
        )

"""Tests for A100 GPU boundary and benchmark scripts (plan §4.2).

Ensures scripts run without crashing on CPU (via --tiny or no-CUDA fallbacks),
so CI and local developers catch interface drifts immediately.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_microbench_tiny_runs():
    """microbench_a100.py --tiny runs on CPU and exits 0."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "microbench_a100.py"), "--tiny"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"
    assert "PASS (CPU self-check completed)" in res.stdout


def test_microbench_no_cuda_fallback():
    """microbench_a100.py without args exits 0 cleanly on CPU."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "microbench_a100.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"


def test_step_time_tiny_runs():
    """step_time_a100.py --tiny runs on CPU and exits 0."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "step_time_a100.py"), "--tiny", "--steps", "2", "--warmup", "1"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"
    assert "PASS (CPU self-check completed)" in res.stdout


def test_step_time_no_cuda_fallback():
    """step_time_a100.py without args exits 0 cleanly on CPU."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "step_time_a100.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"


def test_e2e_gpu_smoke_tiny_runs(tmp_path):
    """e2e_gpu_smoke.py --tiny executes multi-step training, checkpointing, and generation."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "e2e_gpu_smoke.py"), "--tiny", "--steps", "6", "--workdir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}\nStdout: {res.stdout}"
    assert "PASS" in res.stdout
    assert "generate output shape" in res.stdout


def test_launch_script_exists_and_executable():
    """launch_a100.sh exists and has execute bit set."""
    script = ROOT / "scripts" / "launch_a100.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK), "scripts/launch_a100.sh must be executable"

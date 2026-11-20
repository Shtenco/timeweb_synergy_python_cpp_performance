"""Equivalent Python/C++ sum-of-squares demo; C++ runs only if a compiler exists."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def run(seed: int = 42, n: int = 10000) -> dict:
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 1_000_000:
        raise ValueError("n must be an integer between 0 and 1000000")
    started = time.perf_counter()
    python_result = sum(i * i for i in range(n))
    python_seconds = time.perf_counter() - started
    expected = n * (n - 1) * (2 * n - 1) // 6
    if python_result != expected:
        raise AssertionError("Python result disagrees with the closed-form reference")
    source = Path(__file__).resolve().parents[1] / "cpp" / "sum_squares.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"C++ prototype source is missing: {source}")
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    native = {"compiled": False, "compiler": None, "result": None, "seconds": None}
    if compiler:
        with tempfile.TemporaryDirectory(prefix="hiai-article07-") as temporary:
            exe = Path(temporary) / ("sum_squares.exe" if __import__("os").name == "nt" else "sum_squares")
            compile_run = subprocess.run(
                [compiler, "-O2", "-std=c++17", str(source), "-o", str(exe)],
                capture_output=True, text=True, timeout=120,
            )
            if compile_run.returncode:
                raise RuntimeError(f"C++ compilation failed: {compile_run.stderr.strip()}")
            started = time.perf_counter()
            native_run = subprocess.run([str(exe), str(n)], capture_output=True, text=True, timeout=30, check=True)
            native_seconds = time.perf_counter() - started
            native_result = int(native_run.stdout.strip())
            if native_result != expected:
                raise AssertionError("C++ result disagrees with the closed-form reference")
            native = {
                "compiled": True, "compiler": Path(compiler).name,
                "result": native_result, "seconds": round(native_seconds, 8),
            }
    return {
        "evidence_type": "computed",
        "seed": seed,
        "n": n,
        "python_result": python_result,
        "reference_result": expected,
        "python_seconds": round(python_seconds, 8),
        "cpp_source_available": True,
        "native": native,
        "python_correctness": "PASS",
        "native_correctness": "PASS" if native["compiled"] else "NOT_RUN",
        "limitations": [
            "The benchmark is a single integer loop and does not represent typical application workloads.",
            "Each runtime is measured once; results are not a stable performance comparison.",
            "If no C++ compiler is installed, the source is included but the native run is marked unavailable.",
        ],
    }

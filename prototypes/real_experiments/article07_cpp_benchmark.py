"""Reproducible in-process Python vs compiled C++ sum-of-squares benchmark."""

from __future__ import annotations

import argparse
import ctypes
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK64 = (1 << 64) - 1
WORKLOADS = ((1_000, 1_000), (10_000, 100), (100_000, 10), (1_000_000, 1))


def validate_workload(n: int, repeats: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1_000_000:
        raise ValueError("n must be an integer between 1 and 1000000")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100_000:
        raise ValueError("repeats must be an integer between 1 and 100000")
    if n * repeats > 100_000_000:
        raise ValueError("one measurement may contain at most 100000000 loop iterations")


def reference_checksum(n: int, repeats: int) -> int:
    validate_workload(n, repeats)
    one_sum = n * (n - 1) * (2 * n - 1) // 6
    return (one_sum * repeats) & MASK64


def python_checksum(n: int, repeats: int) -> int:
    validate_workload(n, repeats)
    checksum = 0
    for _ in range(repeats):
        subtotal = 0
        for value in range(n):
            subtotal += value * value
        checksum = (checksum + subtotal) & MASK64
    return checksum


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one measurement is required")
    ordered = sorted(samples)
    return {
        "median_seconds": round(statistics.median(ordered), 8),
        "p25_seconds": round(statistics.quantiles(ordered, n=4, method="inclusive")[0], 8) if len(ordered) > 1 else round(ordered[0], 8),
        "p75_seconds": round(statistics.quantiles(ordered, n=4, method="inclusive")[2], 8) if len(ordered) > 1 else round(ordered[0], 8),
        "p95_seconds": round(statistics.quantiles(ordered, n=20, method="inclusive")[18], 8) if len(ordered) > 1 else round(ordered[0], 8),
        "min_seconds": round(min(ordered), 8),
        "max_seconds": round(max(ordered), 8),
    }


def _find_compiler(explicit: str | None = None) -> str:
    candidates = [explicit, os.environ.get("CXX"), "g++", "clang++", "c++"]
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
            if resolved:
                return str(Path(resolved).resolve())
    raise FileNotFoundError("C++ compiler not found; pass --compiler or set CXX / PATH")


def native_filesystem_path(path: str | Path) -> str:
    """Use 8.3 paths on Windows so MinGW tools can handle non-ASCII user directories."""
    value = str(path)
    if os.name != "nt":
        return value
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, len(buffer))
    if length and length < len(buffer):
        return buffer.value
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_native(executable: Path, compiler_path: str, n: int, repeats: int, expected: int) -> float:
    environment = os.environ.copy()
    compiler_bin = native_filesystem_path(Path(compiler_path).parent)
    environment["PATH"] = compiler_bin + os.pathsep + environment.get("PATH", "")
    executable_path = native_filesystem_path(executable)
    proc = subprocess.run(
        [executable_path, str(n), str(repeats)], check=True, capture_output=True, text=True,
        timeout=60, env=environment, cwd=native_filesystem_path(executable.parent),
    )
    fields = proc.stdout.strip().split()
    if len(fields) != 2:
        raise RuntimeError(f"unexpected C++ output: {proc.stdout!r}")
    elapsed_ns, checksum = map(int, fields)
    if checksum != expected:
        raise AssertionError(f"C++ checksum {checksum} does not match reference {expected}")
    return elapsed_ns / 1_000_000_000


def run_experiment(compiler: str | None = None, rounds: int = 7) -> dict:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 3 <= rounds <= 101 or rounds % 2 == 0:
        raise ValueError("rounds must be an odd integer from 3 to 101")
    compiler_path = _find_compiler(compiler)
    compiler_native = native_filesystem_path(compiler_path)
    source = Path(__file__).resolve().parents[1] / "cpp" / "sum_squares_benchmark.cpp"
    with tempfile.TemporaryDirectory(prefix="hiai-article07-") as temp_dir:
        executable = Path(temp_dir) / ("sum_squares_benchmark.exe" if os.name == "nt" else "sum_squares_benchmark")
        compiler_root = Path(compiler_path).parent.parent
        gcc_version_dirs = list((compiler_root / "lib" / "gcc" / "x86_64-w64-mingw32").glob("*"))
        gcc_prefixes = [gcc_version_dirs[-1], compiler_root / "x86_64-w64-mingw32" / "lib"] if gcc_version_dirs else []
        path_overrides = []
        for prefix in gcc_prefixes:
            if prefix.is_dir():
                path_overrides.extend(["-B" + native_filesystem_path(prefix) + os.sep, "-L" + native_filesystem_path(prefix)])
        common_lib = compiler_root / "lib"
        if common_lib.is_dir():
            path_overrides.append("-L" + native_filesystem_path(common_lib))
        compile_proc = subprocess.run(
            [compiler_native, *path_overrides, "-O2", "-std=c++17", "-DNDEBUG", native_filesystem_path(source), "-o", native_filesystem_path(executable)],
            capture_output=True, text=True, timeout=180, cwd=native_filesystem_path(temp_dir),
        )
        if compile_proc.returncode:
            raise RuntimeError(f"C++ compilation failed:\n{compile_proc.stderr}")
        version_proc = subprocess.run([compiler_native, "--version"], capture_output=True, text=True, timeout=15, check=True)
        compiler_version = version_proc.stdout.splitlines()[0].replace(Path(compiler_native).name, Path(compiler_path).name, 1)
        rows = []
        for n, repeats in WORKLOADS:
            expected = reference_checksum(n, repeats)
            if python_checksum(n, repeats) != expected:
                raise AssertionError("Python checksum disagrees with closed-form reference")
            _ = python_checksum(n, repeats)
            _run_native(executable, compiler_path, n, repeats, expected)
            py_samples, cpp_samples = [], []
            for index in range(rounds):
                order = ("python", "cpp") if index % 2 == 0 else ("cpp", "python")
                for runtime in order:
                    if runtime == "python":
                        started = time.perf_counter()
                        checksum = python_checksum(n, repeats)
                        elapsed = time.perf_counter() - started
                        if checksum != expected:
                            raise AssertionError("Python checksum changed during measurement")
                        py_samples.append(elapsed)
                    else:
                        cpp_samples.append(_run_native(executable, compiler_path, n, repeats, expected))
            py = summarize_samples(py_samples)
            cpp = summarize_samples(cpp_samples)
            rows.append({
                "n_per_sum": n,
                "repeats": repeats,
                "loop_iterations_per_sample": n * repeats,
                "checksum": expected,
                "python": py,
                "cpp": cpp,
                "python_samples_seconds": [round(value, 8) for value in py_samples],
                "cpp_samples_seconds": [round(value, 8) for value in cpp_samples],
                "speedup_cpp_over_python": round(py["median_seconds"] / cpp["median_seconds"], 3),
            })
    return {
        "experiment": "Python vs C++ sum of squares; in-process loop timing",
        "evidence_type": "measured",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "compiler_path": compiler_path,
        "compiler_version": compiler_version,
        "compiler_flags": ["-O2", "-std=c++17", "-DNDEBUG"],
        "link_search_overrides": [str(prefix.relative_to(compiler_root)).replace("\\", "/") for prefix in gcc_prefixes if prefix.is_dir()] + (["lib"] if common_lib.is_dir() else []),
        "compiler_binary_sha256": file_sha256(Path(compiler_path)),
        "cpp_source_sha256": file_sha256(source),
        "python_harness_sha256": file_sha256(Path(__file__)),
        "rounds": rounds,
        "warmup_runs_per_runtime_and_workload": 1,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "workloads": rows,
        "method": "Both runtimes execute the same nested integer loops in their own process. C++ reports steady-state loop time from std::chrono; Python uses perf_counter. Compilation and process startup are excluded. Runtime order alternates, each measurement is correctness-checked, raw times are retained, and medians/IQR are reported.",
        "limitations": [
            "A sum-of-squares loop is a narrow CPU-bound microbenchmark, not a representative application workload.",
            "The result describes this machine, compiler, optimization level, Python version, and implementation only.",
            "The C++ executable starts once per sample, but its internal timer excludes process startup; Python is measured within the already-running harness.",
            "Power management, thermal state, and background processes can affect results; this is a single-session exploratory experiment.",
        ],
    }


def write_artifacts(result: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"article-07-python-cpp-{stamp}.json"
    csv_path = output_dir / f"article-07-python-cpp-{stamp}.csv"
    raw_csv_path = output_dir / f"article-07-python-cpp-{stamp}-raw.csv"
    plot_path = output_dir / f"article-07-python-cpp-{stamp}.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n_per_sum", "repeats", "loop_iterations_per_sample", "checksum", "python_median_seconds", "python_p25_seconds", "python_p75_seconds", "cpp_median_seconds", "cpp_p25_seconds", "cpp_p75_seconds", "speedup_cpp_over_python"])
        writer.writeheader()
        for row in result["workloads"]:
            writer.writerow({
                "n_per_sum": row["n_per_sum"], "repeats": row["repeats"],
                "loop_iterations_per_sample": row["loop_iterations_per_sample"], "checksum": row["checksum"],
                "python_median_seconds": row["python"]["median_seconds"], "python_p25_seconds": row["python"]["p25_seconds"], "python_p75_seconds": row["python"]["p75_seconds"],
                "cpp_median_seconds": row["cpp"]["median_seconds"], "cpp_p25_seconds": row["cpp"]["p25_seconds"], "cpp_p75_seconds": row["cpp"]["p75_seconds"],
                "speedup_cpp_over_python": row["speedup_cpp_over_python"],
            })
    with raw_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["workload_n", "repeats", "loop_iterations_per_sample", "runtime", "sample_index", "seconds", "checksum"])
        writer.writeheader()
        for row in result["workloads"]:
            for runtime in ("python", "cpp"):
                for sample_index, seconds in enumerate(row[f"{runtime}_samples_seconds"], start=1):
                    writer.writerow({
                        "workload_n": row["n_per_sum"], "repeats": row["repeats"],
                        "loop_iterations_per_sample": row["loop_iterations_per_sample"], "runtime": runtime,
                        "sample_index": sample_index, "seconds": seconds, "checksum": row["checksum"],
                    })
    iterations = [row["n_per_sum"] for row in result["workloads"]]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for runtime, label, color, marker in (("python", "Python", "#1f77b4", "o"), ("cpp", "C++17 -O2", "#d95f02", "s")):
        medians = [row[runtime]["median_seconds"] for row in result["workloads"]]
        lower = [median - row[runtime]["p25_seconds"] for median, row in zip(medians, result["workloads"])]
        upper = [row[runtime]["p75_seconds"] - median for median, row in zip(medians, result["workloads"])]
        axis.errorbar(iterations, medians, yerr=[lower, upper], label=label, color=color, marker=marker, capsize=4, linewidth=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(iterations, [f"{row['n_per_sum']:,}\n×{row['repeats']} passes" for row in result["workloads"]])
    axis.set_xlabel("N per sum-of-squares pass (1M total iterations per sample)")
    axis.set_ylabel("Median execution time (seconds; IQR bars)")
    axis.set_title("Python vs C++ sum-of-squares microbenchmark")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return json_path, csv_path, raw_csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="C++ compiler executable or path (otherwise CXX/PATH is searched)")
    parser.add_argument("--rounds", type=int, default=7, help="odd number of paired samples, 3..101")
    parser.add_argument("--out", type=Path, default=Path("evidence/07_real"))
    args = parser.parse_args()
    result = run_experiment(args.compiler, args.rounds)
    paths = write_artifacts(result, args.out)
    print(json.dumps({"compiler": result["compiler_version"], "artifacts": [str(path) for path in paths], "workloads": result["workloads"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

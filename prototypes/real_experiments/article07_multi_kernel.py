"""Compare Python, NumPy, Numba, and C++ on two equivalent CPU workloads."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from prototypes.real_experiments.article07_cpp_benchmark import (
    MASK64,
    _find_compiler,
    native_filesystem_path,
    summarize_samples,
    file_sha256,
)

HASH_WORKLOADS = ((1_000, 1_000), (10_000, 100), (100_000, 10), (1_000_000, 1))
SIEVE_WORKLOADS = ((10_000, 10), (100_000, 3), (1_000_000, 1))
SORT_WORKLOADS = ((1_000, 100), (10_000, 10), (100_000, 1))
BACKENDS = ("python", "numpy", "numba", "cpp")
HASH_SEED = 0x9E3779B97F4A7C15
HASH_MUL1 = 0xBF58476D1CE4E5B9
HASH_MUL2 = 0x94D049BB133111EB
HASH_MASK = 0xFFFFFFFF
SORT_MULTIPLIER = 2_654_435_761


def _sort_checksum(values) -> tuple[int, int]:
    checksum = 0
    weighted = 0
    for index, value in enumerate(values):
        checksum = (checksum + int(value)) & MASK64
        weighted = (weighted + (index + 1) * int(value)) & MASK64
    return checksum, weighted


def sort_python(n: int, repeats: int = 1) -> tuple[int, int]:
    _validate(n, repeats)
    total = [0, 0]
    for _ in range(repeats):
        values = sorted((index * SORT_MULTIPLIER + 12_345) & HASH_MASK for index in range(n))
        part = _sort_checksum(values)
        total[0] = (total[0] + part[0]) & MASK64
        total[1] = (total[1] + part[1]) & MASK64
    return tuple(total)


def sort_numpy(n: int, repeats: int = 1) -> tuple[int, int]:
    _validate(n, repeats)
    total = [0, 0]
    positions = np.arange(1, n + 1, dtype=np.uint64)
    for _ in range(repeats):
        values = (np.arange(n, dtype=np.uint64) * np.uint64(SORT_MULTIPLIER) + np.uint64(12_345)) & np.uint64(HASH_MASK)
        values.sort()
        total[0] = (total[0] + int(values.sum(dtype=np.uint64))) & MASK64
        weighted = int(np.sum(positions * values, dtype=np.uint64))
        total[1] = (total[1] + weighted) & MASK64
    return tuple(total)


def _validate(n: int, repeats: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 1_000_000:
        raise ValueError("n must be an integer between 0 and 1000000")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100_000:
        raise ValueError("repeats must be an integer between 1 and 100000")
    if n * repeats > 100_000_000:
        raise ValueError("one measurement may contain at most 100000000 input elements")


def sum_squares_python(n: int, repeats: int = 1) -> int:
    _validate(n, repeats)
    checksum = 0
    for _ in range(repeats):
        subtotal = 0
        for value in range(n):
            subtotal += value * value
        checksum = (checksum + subtotal) & MASK64
    return checksum


def sum_squares_numpy(n: int, repeats: int = 1) -> int:
    _validate(n, repeats)
    values = np.arange(n, dtype=np.int64)
    subtotal = int(np.dot(values, values))
    return (subtotal * repeats) & MASK64


def hash_sum_python(n: int, repeats: int = 1) -> int:
    """Sum squared low words of SplitMix64-finalized indices modulo 2**64."""
    _validate(n, repeats)
    checksum = 0
    for _ in range(repeats):
        subtotal = 0
        for value in range(n):
            mixed = (value + HASH_SEED) & MASK64
            mixed = ((mixed ^ (mixed >> 30)) * HASH_MUL1) & MASK64
            mixed = ((mixed ^ (mixed >> 27)) * HASH_MUL2) & MASK64
            mixed ^= mixed >> 31
            sample = mixed & HASH_MASK
            subtotal += sample * sample
        checksum = (checksum + subtotal) & MASK64
    return checksum


def hash_sum_numpy(n: int, repeats: int = 1) -> int:
    _validate(n, repeats)
    checksum = 0
    for _ in range(repeats):
        indices = np.arange(n, dtype=np.uint64)
        mixed = indices + np.uint64(HASH_SEED)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(HASH_MUL1)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(HASH_MUL2)
        mixed ^= mixed >> np.uint64(31)
        samples = mixed & np.uint64(HASH_MASK)
        checksum = (checksum + int(np.sum(samples * samples, dtype=np.uint64))) & MASK64
    return checksum


def sieve_python(n: int, repeats: int = 1) -> tuple[int, int]:
    _validate(n, repeats)
    total_count = 0
    total_sum = 0
    for _ in range(repeats):
        flags = bytearray(b"\x01") * (n + 1)
        if n >= 0:
            flags[0] = 0
        if n >= 1:
            flags[1] = 0
        p = 2
        while p * p <= n:
            if flags[p]:
                for composite in range(p * p, n + 1, p):
                    flags[composite] = 0
            p += 1
        total_count += sum(flags)
        total_sum += sum(value for value, flag in enumerate(flags) if flag)
    return total_count, total_sum


def sieve_numpy(n: int, repeats: int = 1) -> tuple[int, int]:
    _validate(n, repeats)
    total_count = 0
    total_sum = 0
    for _ in range(repeats):
        flags = np.ones(n + 1, dtype=np.bool_)
        flags[: min(2, n + 1)] = False
        for p in range(2, int(n**0.5) + 1):
            if flags[p]:
                flags[p * p :: p] = False
        prime_values = np.flatnonzero(flags)
        total_count += int(prime_values.size)
        total_sum += int(prime_values.sum(dtype=np.int64))
    return total_count, total_sum


def numba_kernels():
    try:
        from numba import njit
    except ImportError as exc:
        raise RuntimeError("Numba is required for this experiment; install prototypes/requirements.txt") from exc

    @njit(cache=False)
    def hash_sum_numba(n, repeats):
        checksum = np.uint64(0)
        for _ in range(repeats):
            subtotal = np.uint64(0)
            for value in range(n):
                mixed = np.uint64(value) + np.uint64(HASH_SEED)
                mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(HASH_MUL1)
                mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(HASH_MUL2)
                mixed ^= mixed >> np.uint64(31)
                sample = mixed & np.uint64(HASH_MASK)
                subtotal += sample * sample
            checksum += subtotal
        return checksum

    @njit(cache=False)
    def sieve_numba(n, repeats):
        total_count = 0
        total_sum = 0
        for _ in range(repeats):
            flags = np.ones(n + 1, dtype=np.uint8)
            if n >= 0:
                flags[0] = 0
            if n >= 1:
                flags[1] = 0
            p = 2
            while p * p <= n:
                if flags[p] != 0:
                    for composite in range(p * p, n + 1, p):
                        flags[composite] = 0
                p += 1
            for value in range(2, n + 1):
                if flags[value] != 0:
                    total_count += 1
                    total_sum += value
        return total_count, total_sum

    @njit(cache=False)
    def sort_numba(n, repeats):
        total = np.uint64(0)
        weighted_total = np.uint64(0)
        for _ in range(repeats):
            values = np.empty(n, dtype=np.uint64)
            for index in range(n):
                values[index] = (np.uint64(index) * np.uint64(SORT_MULTIPLIER) + np.uint64(12_345)) & np.uint64(HASH_MASK)
            values.sort()
            checksum = np.uint64(0)
            weighted = np.uint64(0)
            for index in range(n):
                checksum += values[index]
                weighted += np.uint64(index + 1) * values[index]
            total += checksum
            weighted_total += weighted
        return total, weighted_total

    return hash_sum_numba, sieve_numba, sort_numba


def _cpp_run(executable: Path, compiler_path: str, task: str, n: int, repeats: int):
    environment = os.environ.copy()
    environment["PATH"] = native_filesystem_path(Path(compiler_path).parent) + os.pathsep + environment.get("PATH", "")
    proc = subprocess.run(
        [native_filesystem_path(executable), task, str(n), str(repeats)],
        check=True, capture_output=True, text=True, timeout=60, env=environment,
        cwd=native_filesystem_path(executable.parent),
    )
    fields = [int(value) for value in proc.stdout.strip().split()]
    if task == "hashsum" and len(fields) == 2:
        elapsed_ns, checksum = fields
        return elapsed_ns / 1e9, checksum
    if task == "sieve" and len(fields) == 3:
        elapsed_ns, count, prime_sum = fields
        return elapsed_ns / 1e9, (count, prime_sum)
    if task == "sort" and len(fields) == 3:
        elapsed_ns, checksum, weighted = fields
        return elapsed_ns / 1e9, (checksum, weighted)
    raise RuntimeError(f"unexpected C++ output for {task}: {proc.stdout!r}")


def _timed_python(function, n: int, repeats: int):
    started = time.perf_counter()
    result = function(n, repeats)
    return time.perf_counter() - started, result


def _processor_name() -> str:
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    return platform.processor() or platform.machine()


def run_experiment(compiler: str | None = None, rounds: int = 7) -> dict:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 5 or rounds % 2 == 0:
        raise ValueError("rounds must be an odd integer of at least 5")
    compiler_path = _find_compiler(compiler)
    compiler_native = native_filesystem_path(compiler_path)
    source = Path(__file__).resolve().parents[1] / "cpp" / "article07_multi_benchmark.cpp"
    numba_sum, numba_sieve, numba_sort = numba_kernels()
    results = []
    compile_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hiai-article07-multi-") as temp_dir:
        executable = Path(temp_dir) / ("article07_multi_benchmark.exe" if os.name == "nt" else "article07_multi_benchmark")
        compiler_root = Path(compiler_path).parent.parent
        gcc_versions = sorted((compiler_root / "lib" / "gcc" / "x86_64-w64-mingw32").glob("*"))
        prefixes = ([gcc_versions[-1], compiler_root / "x86_64-w64-mingw32" / "lib"] if gcc_versions else [])
        link_options = []
        for prefix in prefixes:
            if prefix.is_dir():
                link_options.extend(["-B" + native_filesystem_path(prefix) + os.sep, "-L" + native_filesystem_path(prefix)])
        if (compiler_root / "lib").is_dir():
            link_options.append("-L" + native_filesystem_path(compiler_root / "lib"))
        compile_proc = subprocess.run(
            [compiler_native, *link_options, "-O3", "-std=c++17", "-DNDEBUG", native_filesystem_path(source), "-o", native_filesystem_path(executable)],
            capture_output=True, text=True, timeout=180, cwd=native_filesystem_path(temp_dir),
        )
        if compile_proc.returncode:
            raise RuntimeError(f"C++ compilation failed:\n{compile_proc.stderr}")
        cpp_compile_seconds = time.perf_counter() - compile_started
        compiler_version = subprocess.run([compiler_native, "--version"], capture_output=True, text=True, timeout=15, check=True).stdout.splitlines()[0]
        compiler_version = compiler_version.replace(Path(compiler_native).name, Path(compiler_path).name, 1)

        with threadpool_limits(limits=1):
            for task, workloads in (("hashsum", HASH_WORKLOADS), ("sieve", SIEVE_WORKLOADS), ("sort", SORT_WORKLOADS)):
                functions = {
                    "python": hash_sum_python if task == "hashsum" else sieve_python if task == "sieve" else sort_python,
                    "numpy": hash_sum_numpy if task == "hashsum" else sieve_numpy if task == "sieve" else sort_numpy,
                    "numba": numba_sum if task == "hashsum" else numba_sieve if task == "sieve" else numba_sort,
                }
                task_records = []
                numba_compile_seconds = None
                for n, repeats in workloads:
                    expected = functions["python"](n, repeats)
                    numba_started = time.perf_counter()
                    numba_warm_result = functions["numba"](n, repeats)
                    numba_elapsed = time.perf_counter() - numba_started
                    if task_records == []:
                        numba_compile_seconds = numba_elapsed
                    if hasattr(numba_warm_result, "item"):
                        numba_warm_result = int(numba_warm_result)
                    if numba_warm_result != expected:
                        raise AssertionError(f"Numba correctness mismatch for {task} N={n}")
                    _ = functions["numpy"](n, repeats)
                    cpp_warm = _cpp_run(executable, compiler_path, task, n, repeats)[1]
                    if cpp_warm != expected:
                        raise AssertionError(f"C++ correctness mismatch for {task} N={n}")
                    timing_samples = {backend: [] for backend in BACKENDS}
                    for sample_idx in range(rounds):
                        order = list(BACKENDS)
                        shift = sample_idx % len(order)
                        order = order[shift:] + order[:shift]
                        for backend in order:
                            if backend == "cpp":
                                elapsed, result = _cpp_run(executable, compiler_path, task, n, repeats)
                            else:
                                elapsed, result = _timed_python(functions[backend], n, repeats)
                            if hasattr(result, "item"):
                                result = int(result)
                            if result != expected:
                                raise AssertionError(f"{backend} checksum mismatch for {task} N={n}")
                            timing_samples[backend].append(elapsed)
                    summaries = {backend: summarize_samples(values) for backend, values in timing_samples.items()}
                    cpp_ratio = {backend: round(summaries[backend]["median_seconds"] / summaries["cpp"]["median_seconds"], 3) for backend in BACKENDS if backend != "cpp"}
                    task_records.append({
                        "n": n,
                        "repeats": repeats,
                        "checksum": expected,
                        "timings": summaries,
                        "samples_seconds": {backend: [round(value, 8) for value in values] for backend, values in timing_samples.items()},
                        "python_over_cpp_ratio": cpp_ratio,
                    })
                results.append({"task": task, "numba_first_call_seconds": round(numba_compile_seconds or 0.0, 8), "workloads": task_records})

    try:
        import numba
        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    return {
        "experiment": "Python / NumPy / Numba / C++ equivalent workloads",
        "evidence_type": "measured",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "compiler_path": compiler_path,
        "compiler_version": compiler_version,
        "compiler_flags": ["-O3", "-std=c++17", "-DNDEBUG"],
        "link_search_overrides": [str(prefix.relative_to(compiler_root)).replace("\\", "/") for prefix in prefixes if prefix.is_dir()] + (["lib"] if (compiler_root / "lib").is_dir() else []),
        "cpp_compile_seconds": round(cpp_compile_seconds, 6),
        "cpp_binary_sha256": file_sha256(Path(compiler_path)),
        "cpp_source_sha256": file_sha256(source),
        "python_harness_sha256": file_sha256(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": numba_version,
        "threadpool_limit": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "processor_name": _processor_name(),
        "logical_cpu_count": os.cpu_count(),
        "rounds": rounds,
        "warmup_runs_per_backend_and_workload": 1,
        "tasks": results,
        "method": "Each backend computes the same checksum for the same workload. NumPy, Python, and Numba are timed in-process; C++ reports steady-state time from std::chrono and excludes process startup. NumPy input/setup allocations are included in the measured function. Numba first-call/JIT time and C++ compile time are reported separately.",
        "limitations": [
            "Three CPU microbenchmarks do not represent all application workloads.",
            "This is a single-machine exploratory run; power state and background load can still affect timings.",
            "Peak resident memory is not measured here; the report does not make total-RAM claims.",
            "The arithmetic C++ loop uses exact uint64 arithmetic, the sieve uses byte flags, and sorting uses std::sort; no FFI marshaling cost is included.",
        ],
    }


def write_artifacts(result: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"article-07-multikernel-{stamp}.json"
    csv_path = output_dir / f"article-07-multikernel-{stamp}.csv"
    plot_path = output_dir / f"article-07-multikernel-{stamp}.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "n", "repeats", "backend", "sample", "seconds", "checksum"])
        writer.writeheader()
        for task in result["tasks"]:
            for workload in task["workloads"]:
                for backend, samples in workload["samples_seconds"].items():
                    for index, seconds in enumerate(samples, start=1):
                        writer.writerow({"task": task["task"], "n": workload["n"], "repeats": workload["repeats"], "backend": backend, "sample": index, "seconds": seconds, "checksum": workload["checksum"]})
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(result["tasks"]), figsize=(7 * len(result["tasks"]), 5.2), squeeze=False, constrained_layout=True)
    colors = {"python": "#333333", "numpy": "#1f77b4", "numba": "#2ca02c", "cpp": "#d95f02"}
    labels = {"python": "Python", "numpy": "NumPy", "numba": "Numba (warm)", "cpp": "C++17 -O3"}
    for axis, task in zip(axes[0], result["tasks"]):
        workloads = task["workloads"]
        x_values = [workload["n"] for workload in workloads]
        for backend in BACKENDS:
            medians = [workload["timings"][backend]["median_seconds"] for workload in workloads]
            lower = [median - workload["timings"][backend]["p25_seconds"] for median, workload in zip(medians, workloads)]
            upper = [workload["timings"][backend]["p75_seconds"] - median for median, workload in zip(medians, workloads)]
            axis.errorbar(x_values, medians, yerr=[lower, upper], label=labels[backend], color=colors[backend], marker="o", capsize=3)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(x_values, [f"{value:,}" for value in x_values])
        axis.set_xlabel("Input size N")
        axis.set_ylabel("Median kernel time (s; P25–P75 bars)")
        axis.set_title({"hashsum": "64-bit hash-and-sum", "sieve": "Sieve of Eratosthenes", "sort": "Sort + checksums"}[task["task"]])
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return json_path, csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("evidence/07_real"))
    args = parser.parse_args()
    result = run_experiment(args.compiler, args.rounds)
    paths = write_artifacts(result, args.out)
    print(json.dumps({"artifacts": [str(p) for p in paths], "tasks": result["tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

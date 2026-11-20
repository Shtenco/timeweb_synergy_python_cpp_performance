from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from prototypes.real_experiments.article07_cpp_benchmark import (
    reference_checksum,
    summarize_samples,
    validate_workload,
    native_filesystem_path,
    write_artifacts,
)


class Article07RealBenchmarkTests(unittest.TestCase):
    def test_reference_checksum_matches_repeated_sum_of_squares(self):
        self.assertEqual(reference_checksum(4, 3), 42)

    def test_workload_rejects_zero_and_overflowing_dimensions(self):
        for n, repeats in ((0, 1), (4, 0), (1_000_001, 1), (1, 100_001)):
            with self.subTest(n=n, repeats=repeats), self.assertRaises(ValueError):
                validate_workload(n, repeats)

    def test_native_filesystem_path_resolves_existing_unicode_workspace_file(self):
        source_file = Path(__file__).resolve()
        self.assertTrue(Path(native_filesystem_path(source_file)).is_file())

    def test_summary_reports_median_and_interquartile_range(self):
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(summary["median_seconds"], 3.0)
        self.assertEqual(summary["p25_seconds"], 2.0)
        self.assertEqual(summary["p75_seconds"], 4.0)
        self.assertEqual(summary["p95_seconds"], 80.8)

    def test_artifact_writer_creates_machine_readable_data_and_plot(self):
        result = {
            "compiler_version": "GCC test",
            "workloads": [{
                "n_per_sum": 10, "repeats": 100, "loop_iterations_per_sample": 1000, "checksum": 28500,
                "python": {"median_seconds": 0.01, "p25_seconds": 0.009, "p75_seconds": 0.011},
                "cpp": {"median_seconds": 0.001, "p25_seconds": 0.0009, "p75_seconds": 0.0011},
                "python_samples_seconds": [0.01, 0.011, 0.009],
                "cpp_samples_seconds": [0.001, 0.0011, 0.0009],
                "speedup_cpp_over_python": 10.0,
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            paths = write_artifacts(result, Path(temp))
            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from prototypes.real_experiments.article07_multi_kernel import (
    sieve_numpy,
    sieve_python,
    sum_squares_numpy,
    sum_squares_python,
    numba_kernels,
    hash_sum_python,
    hash_sum_numpy,
    sort_numpy,
    sort_python,
    write_artifacts,
)
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np


class Article07MultiKernelTests(unittest.TestCase):
    def test_numpy_and_python_sum_squares_match_exactly(self):
        for n, repeats in ((1, 1), (19, 7), (1000, 2)):
            with self.subTest(n=n, repeats=repeats):
                self.assertEqual(sum_squares_numpy(n, repeats), sum_squares_python(n, repeats))

    def test_numpy_and_python_hash_workload_match_exactly(self):
        for n, repeats in ((0, 1), (19, 7), (1000, 2)):
            with self.subTest(n=n, repeats=repeats):
                self.assertEqual(hash_sum_numpy(n, repeats), hash_sum_python(n, repeats))

    def test_numpy_hash_recomputes_each_requested_repeat(self):
        original_arange = np.arange
        with patch("prototypes.real_experiments.article07_multi_kernel.np.arange", wraps=original_arange) as arange:
            hash_sum_numpy(19, 7)
        self.assertEqual(arange.call_count, 7)

    def test_numpy_and_python_sieve_return_prime_count_and_sum(self):
        for n, expected in ((0, (0, 0)), (1, (0, 0)), (10, (4, 17)), (100, (25, 1060))):
            with self.subTest(n=n):
                self.assertEqual(sieve_python(n), expected)
                self.assertEqual(sieve_numpy(n), expected)

    def test_python_numpy_and_numba_sort_checksums_match(self):
        sort_numba = numba_kernels()[2]
        for n, repeats in ((0, 1), (19, 3), (1000, 2)):
            with self.subTest(n=n, repeats=repeats):
                expected = sort_python(n, repeats)
                self.assertEqual(sort_numpy(n, repeats), expected)
                self.assertEqual(tuple(map(int, sort_numba(n, repeats))), expected)

    def test_input_bounds_are_checked(self):
        for function, args in ((sum_squares_python, (-1, 1)), (sum_squares_numpy, (-1, 1)), (sieve_python, (-1,)), (sieve_numpy, (-1,))):
            with self.subTest(function=function.__name__, args=args), self.assertRaises(ValueError):
                function(*args)

    def test_numba_kernels_match_python_results(self):
        sum_numba, sieve_numba, _ = numba_kernels()
        self.assertEqual(int(sum_numba(1000, 7)), hash_sum_python(1000, 7))
        self.assertEqual(sieve_numba(100, 3), (75, 3180))

    def test_writer_creates_raw_csv_and_comparison_chart(self):
        timings = {name: {"median_seconds": value, "p25_seconds": value * 0.9, "p75_seconds": value * 1.1, "p95_seconds": value * 1.2} for name, value in (("python", 1.0), ("numpy", 0.1), ("numba", 0.01), ("cpp", 0.02))}
        result = {"tasks": [{"task": "hashsum", "workloads": [{"n": 100, "repeats": 2, "checksum": 123, "timings": timings, "samples_seconds": {name: [value["median_seconds"]] for name, value in timings.items()}}]}]}
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifacts(result, Path(directory))
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()

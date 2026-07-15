# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Off-bench tests for the profiler ground-truth check (tools/profiling/ground_truth.py)
and the synthetic workload's split partitioning.

The real ground-truth RUN needs py-spy on a live process (bench-only), but the
check LOGIC -- does a recovered split within error PASS, and does a wrong split
or a busy idle thread FAIL -- is pure and testable here against synthetic
artifacts. A seeded 50/50 artifact must go red where the true split is 70/30.
"""

import pytest

from tools.profiling._synthetic_workload import ITERS_PER_CYCLE, iter_counts
from tools.profiling.ground_truth import check_identity, recovered_split


def _artifact(functions, total_cpu_cores=1.0):
    # functions: list of (leaf_name, cpu_cores, err_cores_95)
    return {
        'manifest': {'total_process_cpu_cores': total_cpu_cores},
        'functions': [{'function': n, 'cpu_cores': c, 'err_cores_95': e} for n, c, e in functions],
    }


# py-spy leaf strings carry the source location, so the check must substring-match.
_A = '_hot_a (tools/profiling/_synthetic_workload.py:63)'
_B = '_hot_b (tools/profiling/_synthetic_workload.py:71)'
_SLEEP = '_idle_sleeper (tools/profiling/_synthetic_workload.py:79)'


class TestIterCounts:
    def test_partitions_at_split(self):
        a, b = iter_counts(0.7, total=1000)
        assert (a, b) == (700, 300)

    def test_sums_to_total(self):
        for split in (0.0, 0.3, 0.5, 0.7, 1.0):
            a, b = iter_counts(split, total=ITERS_PER_CYCLE)
            assert a + b == ITERS_PER_CYCLE

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            iter_counts(1.5)


class TestRecoveredSplit:
    def test_normalizes_over_hot_pair_ignoring_overhead(self):
        # 0.70 / 0.30 in the hot pair, plus unrelated overhead leaves that must
        # not shift the ratio.
        art = _artifact(
            [
                (_A, 0.70, 0.02),
                (_B, 0.30, 0.02),
                ('run_workload (...)', 0.05, 0.01),
                ('range (builtin)', 0.03, 0.01),
            ]
        )
        assert recovered_split(art) == pytest.approx(0.70, abs=1e-9)

    def test_zero_when_no_hot_samples(self):
        art = _artifact([('other (...)', 1.0, 0.02)])
        assert recovered_split(art) == 0.0


class TestCheckIdentity:
    def test_pass_when_recovered_matches_known(self):
        art = _artifact([(_A, 0.70, 0.02), (_B, 0.30, 0.02), (_SLEEP, 0.0, 0.0)])
        result = check_identity(art, expected_a_share=0.70)
        assert result.passed
        assert result.recovered_a_share == pytest.approx(0.70, abs=1e-9)

    def test_fail_when_split_wrong(self):
        # Seeded 50/50 where the truth is 70/30: the profiler would be lying.
        art = _artifact([(_A, 0.50, 0.02), (_B, 0.50, 0.02), (_SLEEP, 0.0, 0.0)])
        result = check_identity(art, expected_a_share=0.70)
        assert not result.passed
        assert 'outside tolerance' in result.report

    def test_fail_when_idle_thread_burns_cpu(self):
        # Correct split, but the sleeper shows real CPU -> idle miscounted.
        art = _artifact([(_A, 0.70, 0.02), (_B, 0.30, 0.02), (_SLEEP, 0.30, 0.02)])
        result = check_identity(art, expected_a_share=0.70)
        assert not result.passed
        assert 'idle thread counted as CPU' in result.report

    def test_fail_when_no_hot_samples(self):
        art = _artifact([('other (...)', 1.0, 0.02)])
        result = check_identity(art, expected_a_share=0.70)
        assert not result.passed

    def test_tolerance_widens_with_sampling_error(self):
        # A recovered 0.66 vs known 0.70 passes only because the reported error
        # is large; the same 0.04 miss with tiny error fails.
        wide = _artifact([(_A, 0.66, 0.10), (_B, 0.34, 0.10), (_SLEEP, 0.0, 0.0)])
        tight = _artifact([(_A, 0.66, 0.005), (_B, 0.34, 0.005), (_SLEEP, 0.0, 0.0)])
        assert check_identity(wide, 0.70).passed
        assert not check_identity(tight, 0.70).passed

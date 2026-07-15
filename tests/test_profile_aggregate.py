# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the profiling aggregation core (tools/profiling/aggregate.py).

This is the deterministic, off-bench half of the instrument's accuracy proof:
given folded samples with a KNOWN split, the math must recover that split and
the absolute per-function CPU. The other half -- that py-spy's real samples map
to CPU (idle/GIL handling) -- is validated on the bench against a live process.
"""

import math

from tools.profiling.aggregate import compute_absolute_cpu, parse_folded


class TestParseFolded:
    def test_leaf_aggregation_and_total(self):
        text = 'main;hot_a 700\nmain;hot_b 300\n'
        counts, total, skipped = parse_folded(text)
        assert counts == {'hot_a': 700, 'hot_b': 300}
        assert total == 1000
        assert skipped == 0

    def test_count_split_from_right_preserves_spaces_in_frame(self):
        # py-spy frames carry spaces: "hot_a (synth.py:6)". The count is split
        # off the RIGHT so the frame's internal space is preserved.
        text = 'main (m.py:1);hot_a (synth.py:6) 42\n'
        counts, total, skipped = parse_folded(text)
        assert counts == {'hot_a (synth.py:6)': 42}
        assert total == 42
        assert skipped == 0

    def test_same_leaf_via_different_stacks_sums(self):
        # A function reached by two call paths accumulates its self-samples.
        text = 'a;work 10\nb;work 15\n'
        counts, _, _ = parse_folded(text)
        assert counts == {'work': 25}

    def test_malformed_lines_are_skipped_and_counted(self):
        # A line without a trailing integer count is a format drift -- skipped
        # and tallied, never silently folded into a real function.
        text = 'main;hot_a 700\ngarbage-with-no-count\nmain;hot_b notanumber\n'
        counts, total, skipped = parse_folded(text)
        assert counts == {'hot_a': 700}
        assert total == 700
        assert skipped == 2

    def test_blank_lines_ignored(self):
        text = '\n\nmain;hot_a 5\n\n'
        counts, total, skipped = parse_folded(text)
        assert counts == {'hot_a': 5}
        assert total == 5
        assert skipped == 0


class TestComputeAbsoluteCpu:
    def test_recovers_known_split_as_shares(self):
        counts = {'hot_a': 700, 'hot_b': 300}
        rows = compute_absolute_cpu(counts, total_samples=1000, total_process_cpu_cores=1.64)
        by_fn = {r.function: r for r in rows}
        assert math.isclose(by_fn['hot_a'].share, 0.70)
        assert math.isclose(by_fn['hot_b'].share, 0.30)

    def test_absolute_is_share_times_total(self):
        counts = {'hot_a': 700, 'hot_b': 300}
        rows = compute_absolute_cpu(counts, total_samples=1000, total_process_cpu_cores=1.64)
        by_fn = {r.function: r for r in rows}
        assert math.isclose(by_fn['hot_a'].cpu_cores, 0.70 * 1.64)
        assert math.isclose(by_fn['hot_b'].cpu_cores, 0.30 * 1.64)
        # ms/sec is cores * 1000 (1 core == 1000 ms of CPU per wall-second).
        assert math.isclose(by_fn['hot_a'].cpu_ms_per_s, 0.70 * 1.64 * 1000.0)

    def test_ranked_descending_by_absolute_cpu(self):
        counts = {'small': 100, 'big': 800, 'mid': 100}
        rows = compute_absolute_cpu(counts, total_samples=1000, total_process_cpu_cores=2.0)
        assert rows[0].function == 'big'
        cores = [r.cpu_cores for r in rows]
        assert cores == sorted(cores, reverse=True)

    def test_sleeping_thread_stays_small(self):
        # With idle samples excluded (py-spy's job on the bench), a mostly-
        # sleeping thread contributes few leaf samples -> a small CPU share,
        # never dominating. Here the math simply reflects the low count.
        counts = {'hot_a': 700, 'hot_b': 300, 'time.sleep': 20}
        rows = compute_absolute_cpu(counts, total_samples=1020, total_process_cpu_cores=1.64)
        by_fn = {r.function: r for r in rows}
        assert by_fn['time.sleep'].share < 0.03
        assert rows[0].function == 'hot_a'

    def test_error_shrinks_with_more_samples(self):
        # 95% band scales as 1/sqrt(N): the honest "few samples = a hint" signal.
        few = compute_absolute_cpu({'f': 70, 'g': 30}, 100, 1.0)[0]
        many = compute_absolute_cpu({'f': 7000, 'g': 3000}, 10000, 1.0)[0]
        assert few.function == 'f' and many.function == 'f'
        assert few.err_cores_95 > many.err_cores_95
        assert math.isclose(few.err_cores_95 / many.err_cores_95, 10.0, rel_tol=0.05)

    def test_zero_samples_returns_empty(self):
        assert compute_absolute_cpu({}, 0, 1.64) == []

    def test_end_to_end_folded_to_absolute(self):
        # The deterministic 1a proof: folded text with a known 70/30 split (plus
        # a small idle leaf) parses and resolves to the expected absolute cores.
        folded = (
            'main (m.py:9);hot_a (m.py:6) 700\n'
            'main (m.py:9);hot_b (m.py:12) 300\n'
            'idle (m.py:18);time.sleep 20\n'
        )
        counts, total, skipped = parse_folded(folded)
        assert skipped == 0 and total == 1020
        rows = compute_absolute_cpu(counts, total, total_process_cpu_cores=1.64)
        top = rows[0]
        assert top.function == 'hot_a (m.py:6)'
        assert math.isclose(top.share, 700 / 1020, rel_tol=1e-9)
        assert math.isclose(top.cpu_cores, (700 / 1020) * 1.64, rel_tol=1e-9)

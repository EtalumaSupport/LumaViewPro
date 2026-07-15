# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the profile compare / regression engine (tools/profiling/compare.py).

Models the before/after acceptance case: the pre-fix build spends ~0.85 cores in
the host resize; the post-fix build (GPU minify) does not. The engine must show
that as a significant improvement, and must REFUSE to compare runs that differ in
more than the code under test.
"""

from tools.profiling.compare import compare, compatibility_issues


def _artifact(*, scenario, machine, git_sha, config, functions):
    # functions: list of (name, cpu_cores, err_cores_95)
    return {
        'manifest': {
            'scenario': scenario,
            'machine': machine,
            'git_sha': git_sha,
            'version': '4.0.0-beta20',
            'total_process_cpu_cores': sum(c for _, c, _ in functions),
            'config': config,
        },
        'functions': [{'function': n, 'cpu_cores': c, 'err_cores_95': e} for n, c, e in functions],
    }


_CFG = {'live_view_fps': 30, 'preview_host_downscale': False}


class TestCompatibility:
    def test_same_scenario_machine_config_different_sha_is_comparable(self):
        # The before/after case: SHA differs (the variable under test), all else
        # matches -> comparable.
        before = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='bebf28be',
            config=_CFG,
            functions=[('f', 1.0, 0.02)],
        )
        after = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='91a5abd4',
            config=_CFG,
            functions=[('f', 1.0, 0.02)],
        )
        assert compatibility_issues(before, after) == []

    def test_config_mismatch_blocks(self):
        before = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='a',
            config={'live_view_fps': 30, 'preview_host_downscale': False},
            functions=[('f', 1.0, 0.02)],
        )
        after = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='b',
            config={'live_view_fps': 15, 'preview_host_downscale': False},
            functions=[('f', 1.0, 0.02)],
        )
        issues = compatibility_issues(before, after)
        assert any('live_view_fps' in i for i in issues)

    def test_machine_mismatch_blocks(self):
        before = _artifact(
            scenario='liveview-fit',
            machine='bench-A',
            git_sha='a',
            config=_CFG,
            functions=[('f', 1.0, 0.02)],
        )
        after = _artifact(
            scenario='liveview-fit',
            machine='bench-B',
            git_sha='b',
            config=_CFG,
            functions=[('f', 1.0, 0.02)],
        )
        assert any('machine' in i for i in compatibility_issues(before, after))


class TestCompare:
    def test_detects_vanished_resize_as_significant_improvement(self):
        # Pre-fix: host resize burns 0.85 cores. Post-fix: gone (not in artifact).
        before = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='bebf28be',
            config=_CFG,
            functions=[('resize (imgproc)', 0.85, 0.03), ('render', 0.40, 0.02)],
        )
        after = _artifact(
            scenario='liveview-fit',
            machine='bench',
            git_sha='91a5abd4',
            config=_CFG,
            functions=[('render', 0.40, 0.02)],
        )
        deltas = {d.function: d for d in compare(before, after)}
        resize = deltas['resize (imgproc)']
        assert resize.after_cores == 0.0
        assert resize.delta_cores < 0  # improvement
        assert resize.significant  # 0.85 drop >> combined error
        # Largest-change function ranks first.
        assert compare(before, after)[0].function == 'resize (imgproc)'

    def test_flags_regression_beyond_noise(self):
        before = _artifact(
            scenario='s', machine='m', git_sha='a', config=_CFG, functions=[('f', 0.20, 0.01)]
        )
        after = _artifact(
            scenario='s', machine='m', git_sha='b', config=_CFG, functions=[('f', 0.50, 0.01)]
        )
        d = compare(before, after)[0]
        assert d.delta_cores > 0 and d.significant

    def test_small_change_within_noise_not_significant(self):
        before = _artifact(
            scenario='s', machine='m', git_sha='a', config=_CFG, functions=[('f', 0.500, 0.05)]
        )
        after = _artifact(
            scenario='s', machine='m', git_sha='b', config=_CFG, functions=[('f', 0.510, 0.05)]
        )
        d = compare(before, after)[0]
        assert not d.significant  # 0.01 delta < combined error ~0.07

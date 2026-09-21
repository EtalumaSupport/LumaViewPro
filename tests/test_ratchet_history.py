# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The ratchet-history tool's two load-bearing invariants.

Both failures this pins are silent: they produce a plausible series rather
than an error, which is the shape the ratchets themselves exist to catch.
"""

import pytest

from tests.guards import test_architecture_fixes as _guards
from tools import ratchet_history as rh


def _boom():
    raise FileNotFoundError('modules/config_ui_getters.py')


def test_a_detector_that_cannot_run_records_none_never_zero(monkeypatch, tmp_path):
    """A detector reading a file that did not exist yet must not read as 0.

    Zero means "no violations", which would render as the migration's best
    week in exactly the era where the code predates the instrument.
    """
    monkeypatch.setattr(rh, 'DETECTORS', (('boom', _boom),))
    row = rh.measure_tree(tmp_path)
    assert row['boom'] is None, 'a failed detector must record None, not a count'
    assert 'FileNotFoundError' in row['boom (why)']


def test_measuring_a_tree_does_not_leave_the_guards_redirected(monkeypatch, tmp_path):
    """The redirection must not outlive the call.

    `REPO_ROOT` belongs to the guard module the suite imports; if a
    measurement leaves it pointing at a sampled tree, every guard afterwards
    in the same process measures that tree instead of the checkout.
    """
    before = _guards.REPO_ROOT
    monkeypatch.setattr(rh, 'DETECTORS', (('boom', _boom),))
    rh.measure_tree(tmp_path)
    assert before == _guards.REPO_ROOT


def test_the_redirection_is_undone_even_when_a_detector_raises_hard(monkeypatch, tmp_path):
    """A BaseException must not strand the redirection either."""

    def _hard():
        raise KeyboardInterrupt

    before = _guards.REPO_ROOT
    monkeypatch.setattr(rh, 'DETECTORS', (('hard', _hard),))
    with pytest.raises(KeyboardInterrupt):
        rh.measure_tree(tmp_path)
    assert before == _guards.REPO_ROOT


def test_sample_points_does_not_repeat_a_commit():
    """A quiet interval yields one point, not the same commit twice."""
    points = rh.sample_points('2026-06-22', '2026-09-21', 7, 'dev/4.0.0')
    shas = [sha for _date, sha in points]
    assert len(shas) == len(set(shas)), f'repeated commits in the series: {shas}'


def test_render_prints_na_for_an_unmeasurable_detector(monkeypatch):
    """An unmeasurable detector renders as n/a, never as a number."""
    monkeypatch.setattr(rh, 'DETECTORS', (('boom', _boom),))
    out = rh.render([{'date': '2026-06-22', 'sha': 'abc123', 'boom': None}])
    row = next(line for line in out.splitlines() if 'abc123' in line)
    assert row.split('abc123')[1].strip() == 'n/a'

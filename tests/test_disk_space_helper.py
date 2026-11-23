# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: shared check_disk_space_ok helper + 3-site consolidation.

Bug
---
Three call sites rolled disk-space checks by hand with shutil.disk_usage
(protocol_image_writer.py:498-510, protocol_run_loop.py:113-133,
main_display.py:236-254) while src/_lumascope/modules/common_utils.py:824 already
exposed check_disk_space using psutil.disk_usage. On Windows partition
mounts, shutil.disk_usage and psutil.disk_usage can disagree (different
free-bytes calculation on network mounts / VSS / sparse filesystems);
the three sites silently diverged on backend choice + unit conversion
+ threshold semantics. Rule-35 semantic-duplicate audit 2026-05-19,
finding 3.

Fix
---
common_utils.check_disk_space_ok(path, required_mb) -> (bool, float)
performs the canonical psutil-backed probe + threshold compare. Whether
there is room is decided by its two judges, room_to_start and
room_to_write, whose arithmetic is pinned here.

Test approach
-------------
1. Behavioral: helper returns the right tuple by monkeypatching
   psutil.disk_usage. A real filesystem probe reports this machine's
   free space, not the case's; the monkeypatch shapes the probe to
   return realistic free-bytes values per case.
2. That nothing else reads free space, a reintroduced hand-rolled read
   included, is the guard's: tests/guards/test_free_space_has_two_judges.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _lumascope.modules import common_utils
from _lumascope.modules.common_utils import check_disk_space_ok
from tests.disk_fakes import free_space_is, free_space_unreadable


def _stub_disk_usage(free_bytes: float):
    """Build a psutil.disk_usage stub that returns the given free-bytes."""
    return lambda path: SimpleNamespace(total=10**12, used=10**11, free=free_bytes)


# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------


def test_returns_tuple_of_bool_and_float(monkeypatch):
    """The helper returns (ok: bool, free_mb: float)."""
    free_mb_truth = 1500.0
    free_bytes = free_mb_truth * 1024 * 1024
    monkeypatch.setattr(common_utils.psutil, 'disk_usage', _stub_disk_usage(free_bytes))

    result = check_disk_space_ok('/', 500)

    assert isinstance(result, tuple) and len(result) == 2
    ok, free_mb = result
    assert isinstance(ok, bool)
    assert isinstance(free_mb, float)
    assert ok is True
    assert free_mb == pytest.approx(free_mb_truth)


def test_ok_true_when_threshold_met(monkeypatch):
    """Threshold below free space => ok=True."""
    monkeypatch.setattr(
        common_utils.psutil,
        'disk_usage',
        _stub_disk_usage(2_000_000_000),  # 2 GB free
    )
    ok, _free_mb = check_disk_space_ok('/', 500)
    assert ok is True


def test_ok_false_when_threshold_exceeds_free(monkeypatch):
    """Threshold above free space => ok=False."""
    monkeypatch.setattr(
        common_utils.psutil,
        'disk_usage',
        _stub_disk_usage(100 * 1024 * 1024),  # 100 MB free
    )
    ok, free_mb = check_disk_space_ok('/', 500)
    assert ok is False
    assert free_mb == pytest.approx(100.0)


def test_str_coercion_of_path_argument(monkeypatch, tmp_path):
    """psutil receives the path as str even if a pathlib.Path was passed."""
    received = {}

    def spy_disk_usage(path):
        received['path'] = path
        return SimpleNamespace(total=0, used=0, free=10**9)

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', spy_disk_usage)
    check_disk_space_ok(tmp_path, 0)
    assert received['path'] == str(tmp_path)
    assert isinstance(received['path'], str)


def test_a_folder_not_made_yet_is_measured_on_its_nearest_existing_ancestor(monkeypatch, tmp_path):
    """A run's output folder is often made only when its first file is
    written; the check before that must measure the volume it will land
    on, not raise and leave the run unchecked."""
    received = {}

    def spy_disk_usage(path):
        received['path'] = path
        return SimpleNamespace(total=0, used=0, free=10**9)

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', spy_disk_usage)
    unborn = tmp_path / 'Autofocus Characterization' / '20261005_095105'

    ok, _free_mb = check_disk_space_ok(unborn, 0)

    assert ok is True
    assert received['path'] == str(tmp_path)
    assert not unborn.exists()


def test_a_folder_not_made_yet_reads_the_real_free_space(tmp_path):
    """The real probe, no stand-in: a missing folder under a real directory
    answers that directory's free space instead of raising."""
    ok, free_mb = check_disk_space_ok(tmp_path / 'not' / 'made', 0)

    expected_mb = common_utils.psutil.disk_usage(str(tmp_path)).free / (1024**2)
    assert ok is True
    assert free_mb == pytest.approx(expected_mb, rel=0.01)


def test_propagates_oserror(monkeypatch):
    """psutil.disk_usage raises => helper raises (caller decides swallow)."""

    def boom(path):
        raise OSError('disk probe failed')

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', boom)
    with pytest.raises(OSError):
        check_disk_space_ok('/', 0)


# ---------------------------------------------------------------------------
# The running judge: a write needs its own size plus the headroom
# ---------------------------------------------------------------------------


def test_a_write_needs_its_size_plus_the_headroom(monkeypatch):
    # 900 MB free and a 600 MB scan: the scan fits alone, but writing it
    # would land in the last 500 MB, where each still's check would end the
    # run partway through the scan. The scan's check refuses it first.
    needs = 600.0 + common_utils.MIN_PER_WRITE_DISK_MB
    free_space_is(monkeypatch, 900.0)
    short = common_utils.room_to_write('/', 'a scan of 3 steps', 600.0)
    free_space_is(monkeypatch, needs)
    fits = common_utils.room_to_write('/', 'a scan of 3 steps', 600.0)

    assert short is not None and '900.0 MB is free at /' in short, short
    assert 'a scan of 3 steps needs 1.1 GB' in short, short
    assert fits is None


def test_a_running_check_that_cannot_read_the_disk_lets_the_write_go(monkeypatch, caplog):
    free_space_unreadable(monkeypatch)

    assert common_utils.room_to_write('/', 'a still', 10.0) is None
    assert any('could not be read' in r.getMessage() for r in caplog.records), caplog.records


def test_a_recordings_frames_keep_the_room_a_start_needs(monkeypatch):
    # What a recording writes after a stop (its backlog in memory, a stack
    # of its frames) is not known, so its check keeps a start's room.
    needs = common_utils.MIN_REQUIRED_DISK_MB + common_utils.MIN_PER_WRITE_DISK_MB
    check = common_utils.DiskCheckInterval(elapsed_clock=lambda: 100.0)
    free_space_is(monkeypatch, needs - 1)
    short = check.room_to_write('/', 'the recording')
    free_space_is(monkeypatch, needs)
    again = common_utils.DiskCheckInterval(elapsed_clock=lambda: 100.0)

    assert short is not None and 'needs 2.5 GB' in short, short
    assert again.room_to_write('/', 'the recording') is None


def test_a_recordings_check_reads_once_an_interval(monkeypatch):
    readings = []
    monkeypatch.setattr(
        common_utils, 'check_disk_space_ok', lambda path, mb: readings.append(path) or (False, 1.0)
    )
    check = common_utils.DiskCheckInterval(elapsed_clock=lambda: 100.0)

    first = check.room_to_write('/', 'the recording')
    second = check.room_to_write('/', 'the recording')

    assert first is not None and second is None
    assert readings == ['/']

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: shared check_disk_space_ok helper + 3-site consolidation.

Bug
---
Three call sites rolled disk-space checks by hand with shutil.disk_usage
(protocol_image_writer.py:498-510, protocol_run_loop.py:113-133,
main_display.py:236-254) while modules/common_utils.py:824 already
exposed check_disk_space using psutil.disk_usage. On Windows partition
mounts, shutil.disk_usage and psutil.disk_usage can disagree (different
free-bytes calculation on network mounts / VSS / sparse filesystems);
the three sites silently diverged on backend choice + unit conversion
+ threshold semantics. Rule-35 semantic-duplicate audit 2026-05-19,
finding 3.

Fix
---
common_utils.check_disk_space_ok(path, required_mb) -> (bool, float)
performs the canonical psutil-backed probe + threshold compare. The
three call sites use the helper; each keeps its own threshold + abort
+ notification policy (the audit's "shared probe, separate policy"
recommendation).

Test approach
-------------
1. Behavioral: helper returns the right tuple by monkeypatching
   psutil.disk_usage. conftest globally mocks psutil so a real
   filesystem probe is impossible; the monkeypatch shapes the mock to
   return realistic free-bytes values per case.
2. Source-text guards over the three call sites: shutil.disk_usage is
   gone; check_disk_space_ok is the new shape. Catches a reintroduction
   of either hand-rolled call.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from modules import common_utils
from modules.common_utils import check_disk_space_ok


REPO = pathlib.Path(__file__).resolve().parent.parent


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


def test_str_coercion_of_path_argument(monkeypatch):
    """psutil receives the path as str even if a pathlib.Path was passed."""
    received = {}

    def spy_disk_usage(path):
        received['path'] = path
        return SimpleNamespace(total=0, used=0, free=10**9)

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', spy_disk_usage)
    check_disk_space_ok(pathlib.Path('/tmp/example'), 0)
    assert received['path'] == '/tmp/example'
    assert isinstance(received['path'], str)


def test_propagates_oserror(monkeypatch):
    """psutil.disk_usage raises => helper raises (caller decides swallow)."""

    def boom(path):
        raise OSError('disk probe failed')

    monkeypatch.setattr(common_utils.psutil, 'disk_usage', boom)
    with pytest.raises(OSError):
        check_disk_space_ok('/', 0)


# ---------------------------------------------------------------------------
# Source-text guards: 3 call sites use the helper, not shutil.disk_usage
# ---------------------------------------------------------------------------

SITES = {
    'modules/protocol_image_writer.py': 'common_utils.check_disk_space_ok',
    'modules/protocol_run_loop.py': 'check_disk_space_ok',
    'modules/manual_recording.py': 'check_disk_space_ok',
}


@pytest.mark.parametrize('relpath, needle', list(SITES.items()))
def test_site_uses_helper(relpath, needle):
    # pin-justified: structural cross-site guard that all three callers use
    # the one canonical helper; the disk-abort behavior itself has a
    # behavioral twin (test_protocol_image_writer_disk_exhaustion_aborts),
    # and the main_display caller is Kivy-bound (no headless drive).
    src = (REPO / relpath).read_text()
    assert needle in src, f'{relpath} no longer calls {needle}; the consolidated probe is gone'


@pytest.mark.parametrize('relpath', list(SITES.keys()))
def test_site_no_longer_calls_shutil_disk_usage(relpath):
    src = (REPO / relpath).read_text()
    assert 'shutil.disk_usage' not in src, (
        f'{relpath} still calls shutil.disk_usage; this is exactly the '
        'Windows-disagree-with-psutil case the consolidation prevents'
    )

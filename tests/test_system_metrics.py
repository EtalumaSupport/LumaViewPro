# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the resource health metrics expansion.

`modules/common_utils.py::system_metrics` was extended in session 18 to
collect GDI/USER objects (Windows), OS handles, open files count,
thread count, GC objects, swap, private memory, and per-process I/O
counters. Goal: catch long-run leaks (steady upward trends across
hourly snapshots) before they degrade the whole Windows desktop.

This file pins the contract (all expected keys present, types correct,
graceful Windows-feature fallback on POSIX).
"""

import pytest

# system_metrics has no heavy deps — psutil only — so no conftest mock
# scaffolding needed. Just import and call.

from modules import common_utils


class TestSystemMetricsKeys:
    """Every key the rest of the app reads from system_metrics() must
    be present in the returned dict, even when the underlying call
    fails — fields fall back to -1 / 0.0 / [] in that case.

    Note: tests/conftest.py installs a MagicMock for psutil at module
    import time, so psutil-derived values here will be MagicMock
    instances, not numbers. We assert only on keys + types that come
    from real stdlib (gc, threading, ctypes), not psutil-derived
    values. End-to-end value validation happens at runtime in the
    actual log lines.
    """

    REQUIRED_KEYS = (
        # Pre-existing fields
        'cpu_percent_total', 'cpu_percent_python',
        'cpu_cores_logical', 'cpu_cores_physical',
        'ram_available_gb', 'ram_percent_total',
        'ram_used_python_percent', 'ram_used_python_mb', 'ram_used_total_mb',
        'disk_free_gb', 'disk_used_percent',
        # Session-18 additions
        'ram_private_mb',
        'swap_percent', 'swap_used_gb',
        'os_handles', 'open_files_count',
        'io_read_mb', 'io_write_mb',
        'gdi_objects', 'user_objects',
        'thread_count', 'thread_names',
        'gc_objects',
        'gc_gen0_collections', 'gc_gen1_collections', 'gc_gen2_collections',
    )

    def test_all_keys_present(self):
        m = common_utils.system_metrics(path='.')
        missing = [k for k in self.REQUIRED_KEYS if k not in m]
        assert not missing, f"Missing keys: {missing}"

    def test_thread_names_is_list(self):
        """threading.enumerate() is real (not mocked) — we sort its
        names into a list."""
        m = common_utils.system_metrics(path='.')
        assert isinstance(m['thread_names'], list)

    def test_thread_count_is_real_int(self):
        """threading.active_count() is real — should always be >= 1."""
        m = common_utils.system_metrics(path='.')
        assert isinstance(m['thread_count'], int)
        assert m['thread_count'] >= 1

    def test_gdi_user_objects_unavailable_on_non_windows_returns_negative(self):
        """On macOS / Linux the GDI object count cannot be queried —
        fields must be -1 (set explicitly), not raise."""
        import platform
        m = common_utils.system_metrics(path='.')
        if platform.system() != 'Windows':
            assert m['gdi_objects'] == -1
            assert m['user_objects'] == -1

    def test_gc_object_count_real_int_positive(self):
        """gc.get_objects() is real (not mocked) — always positive."""
        m = common_utils.system_metrics(path='.')
        assert isinstance(m['gc_objects'], int)
        assert m['gc_objects'] > 0

    def test_gc_collection_counts_are_ints(self):
        """gc.get_stats() is real — all three generation counts present."""
        m = common_utils.system_metrics(path='.')
        for gen in ('gc_gen0_collections', 'gc_gen1_collections', 'gc_gen2_collections'):
            assert isinstance(m[gen], int), f"{gen} = {m[gen]!r}"
            assert m[gen] >= 0


class TestLogSystemMetricsBackwardCompat:
    """The pre-existing `log_system_metrics` test in test_scope_api.py
    mocks `system_metrics` to return a dict with only the legacy keys.
    The new code in `log_system_metrics` must not crash when the new
    keys are missing — it uses `.get(key, -1)` and skips logging when
    the value is -1."""

    def test_log_runs_with_legacy_only_metrics(self):
        from unittest.mock import patch
        from modules import config_helpers
        legacy_metrics = {
            'cpu_percent_total': 25.0,
            'ram_available_gb': 8.0,
            'ram_percent_total': 50.0,
            'disk_free_gb': 100.0,
            'disk_used_percent': 30.0,
            'cpu_percent_python': 5.0,
            'ram_used_python_mb': 200.0,
            'ram_used_python_percent': 2.5,
        }
        with patch('modules.common_utils.system_metrics', return_value=legacy_metrics), \
             patch('modules.common_utils.check_disk_space', return_value=100000), \
             patch('modules.common_utils.get_extra_disks_info', return_value=None):
            # Must not raise even with missing new keys
            config_helpers.log_system_metrics({'live_folder': '/tmp'})

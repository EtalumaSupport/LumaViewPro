# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the gated memory profiler + its settings gate.

The profiler defaults OFF and must be a cheap no-op when disabled (the
common case in shipping builds). These cover the gate helper (current.json /
settings.json merged read, default-off) and the disabled-path no-ops.
"""

import json

from lib import memory_profile
from modules.settings_init import load_memory_profile_setting


def _write_settings(directory, payload):
    data = directory / 'data'
    data.mkdir(parents=True, exist_ok=True)
    (data / 'settings.json').write_text(json.dumps(payload))


class TestGate:
    def test_missing_settings_defaults_off(self, tmp_path):
        cfg = load_memory_profile_setting(tmp_path)  # no data/ at all
        assert cfg['enabled'] is False
        assert cfg['interval_s'] == 5.0

    def test_absent_key_defaults_off(self, tmp_path):
        _write_settings(tmp_path, {'debug_mode': True})
        cfg = load_memory_profile_setting(tmp_path)
        assert cfg['enabled'] is False

    def test_enabled_flag_read(self, tmp_path):
        _write_settings(tmp_path, {'memory_profile_enabled': True})
        cfg = load_memory_profile_setting(tmp_path)
        assert cfg['enabled'] is True
        assert cfg['interval_s'] == 5.0

    def test_interval_override_read(self, tmp_path):
        _write_settings(
            tmp_path, {'memory_profile_enabled': True, 'memory_profile_interval_s': 2.0}
        )
        cfg = load_memory_profile_setting(tmp_path)
        assert cfg['interval_s'] == 2.0

    def test_current_json_wins_over_settings(self, tmp_path):
        # current.json is the live merged file; it should be the read source
        # when present (the same precedence the other profiling gates use).
        data = tmp_path / 'data'
        data.mkdir(parents=True, exist_ok=True)
        (data / 'settings.json').write_text(json.dumps({'memory_profile_enabled': False}))
        (data / 'current.json').write_text(json.dumps({'memory_profile_enabled': True}))
        cfg = load_memory_profile_setting(tmp_path)
        assert cfg['enabled'] is True


class TestDisabledNoOps:
    def test_snapshot_is_noop_when_disabled(self):
        # Without start() enabling it, snapshot must not raise (and must not
        # require tracemalloc to be running).
        assert memory_profile.is_enabled() is False
        memory_profile.snapshot('any_state')  # no throw, no output

    def test_start_disabled_leaves_profiler_off(self, tmp_path):
        # Point start() at a settings dir with the gate off; it must stay off.
        _write_settings(tmp_path, {'memory_profile_enabled': False})
        # start() is idempotent via a module flag; this test only asserts the
        # disabled outcome, which holds regardless of prior start() calls.
        memory_profile.start(str(tmp_path))
        assert memory_profile.is_enabled() is False

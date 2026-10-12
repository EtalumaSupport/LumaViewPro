# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the gated memory profiler + its settings gate.

The profiler defaults OFF and must be a cheap no-op when disabled (the
common case in shipping builds). These cover the gate helper (current.json /
settings.json merged read, default-off) and the disabled-path no-ops.
"""

import json

import pytest

from lib import memory_profile
from modules.settings_init import load_memory_profile_setting

_SHIPPED = {'memory_profile_enabled': False, 'memory_profile_interval_s': 5.0}


def _write(directory, name, payload):
    data = directory / 'data'
    data.mkdir(parents=True, exist_ok=True)
    (data / name).write_text(json.dumps(payload))


class TestGate:
    """Read before preparation: current.json, else the shipped settings.json."""

    def test_an_installation_with_no_settings_is_an_error_not_off(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_memory_profile_setting(tmp_path)

    def test_the_shipped_values_answer_with_no_current_json(self, tmp_path):
        _write(tmp_path, 'settings.json', _SHIPPED)
        assert load_memory_profile_setting(tmp_path) == {'enabled': False, 'interval_s': 5.0}

    def test_current_json_wins_over_settings(self, tmp_path):
        _write(tmp_path, 'settings.json', _SHIPPED)
        _write(
            tmp_path,
            'current.json',
            {'memory_profile_enabled': True, 'memory_profile_interval_s': 2.0},
        )
        assert load_memory_profile_setting(tmp_path) == {'enabled': True, 'interval_s': 2.0}

    def test_a_key_current_json_lacks_takes_the_shipped_value(self, tmp_path):
        _write(tmp_path, 'settings.json', {**_SHIPPED, 'memory_profile_interval_s': 7.0})
        _write(tmp_path, 'current.json', {'memory_profile_enabled': True})
        assert load_memory_profile_setting(tmp_path) == {'enabled': True, 'interval_s': 7.0}

    def test_an_unusable_current_json_takes_the_shipped_values(self, tmp_path):
        _write(tmp_path, 'settings.json', _SHIPPED)
        (tmp_path / 'data' / 'current.json').write_text('{"memory_profile_enabled": true,,,')
        assert load_memory_profile_setting(tmp_path) == {'enabled': False, 'interval_s': 5.0}


class TestDisabledNoOps:
    def test_snapshot_is_noop_when_disabled(self):
        # Without start() enabling it, snapshot must not raise (and must not
        # require tracemalloc to be running).
        assert memory_profile.is_enabled() is False
        memory_profile.snapshot('any_state')  # no throw, no output

    def test_start_disabled_leaves_profiler_off(self, tmp_path):
        # Point start() at a settings dir with the gate off; it must stay off.
        _write(tmp_path, 'settings.json', _SHIPPED)
        # start() is idempotent via a module flag; this test only asserts the
        # disabled outcome, which holds regardless of prior start() calls.
        memory_profile.start(str(tmp_path))
        assert memory_profile.is_enabled() is False

# Copyright Etaluma, Inc.
"""Regression test: current.json is flushed periodically, not only at exit.

current.json holds runtime state and was written only at clean shutdown
(on_stop), so a hard kill or crash left a tech-support bundle with no
runtime-state file (observed on a weekend soak whose app was killed, not
closed). on_start now schedules a periodic flush that mirrors the on_stop
save; the hardware-presence gate inside save_settings keeps it from
overwriting real per-channel values when no hardware is present.

Source-scan: lumaviewpro.py is the Kivy app entry point (not importable
under the test harness), so this asserts the schedule + flush wiring by
inspecting the source.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = (REPO_ROOT / 'lumaviewpro.py').read_text()


def _method_body(name):
    start = SRC.find(f'def {name}(')
    assert start != -1, f'{name} not found in lumaviewpro.py'
    body = SRC[start : start + 2500]
    end = body.find('\n    def ', 1)
    return body if end == -1 else body[:end]


def test_on_start_schedules_periodic_flush():
    body = _method_body('on_start')
    assert 'self._flush_current_json' in body, (
        'on_start must schedule the periodic current.json flush'
    )
    assert '_CURRENT_JSON_FLUSH_INTERVAL_S' in body, (
        'the flush must be scheduled on the named interval constant'
    )


def test_flush_writes_current_json():
    body = _method_body('_flush_current_json')
    assert 'save_settings(' in body and 'current.json' in body, (
        '_flush_current_json must save_settings to current.json'
    )


def test_flush_interval_is_positive():
    # Constant defined at module scope with a sane (multi-minute) cadence.
    assert '_CURRENT_JSON_FLUSH_INTERVAL_S = ' in SRC
    line = next(ln for ln in SRC.splitlines() if ln.startswith('_CURRENT_JSON_FLUSH_INTERVAL_S = '))
    value = int(line.split('=')[1].strip())
    assert value >= 60, 'flush cadence should be at least a minute to avoid churn'

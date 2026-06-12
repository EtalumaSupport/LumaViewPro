# Copyright Etaluma, Inc.
"""Regression test: periodic connection-check happy-path logs at debug.

are_all_connected() runs on a periodic timer during a protocol scan, so its
"Performing connection check" and "All components connected" lines flooded
the main log (~1.2k/soak). Those happy-path lines are debug (routine, per
Rule 5); the "<board> not connected" lines stay at info -- a board dropping
out mid-run is a real degraded-path event the user should see.

lvp_logger is mocked under the test conftest, so levels are asserted by
swapping a recording logger into the module under test rather than caplog.
"""

import pytest

from modules.lumascope_api import Lumascope
from modules.lumascope_api import _lumascope as lumascope_mod


class _RecordingLogger:
    """Minimal logger stand-in capturing (level, message) pairs."""

    def __init__(self):
        self.records = []

    def debug(self, msg, *args, **kwargs):
        self.records.append(('DEBUG', str(msg)))

    def info(self, msg, *args, **kwargs):
        self.records.append(('INFO', str(msg)))

    def warning(self, msg, *args, **kwargs):
        self.records.append(('WARNING', str(msg)))

    def error(self, msg, *args, **kwargs):
        self.records.append(('ERROR', str(msg)))

    def exception(self, msg, *args, **kwargs):
        self.records.append(('ERROR', str(msg)))

    def critical(self, msg, *args, **kwargs):
        self.records.append(('CRITICAL', str(msg)))

    def levels_for(self, substring):
        return [lvl for lvl, msg in self.records if substring in msg]


@pytest.fixture
def sim_scope_with_log(monkeypatch):
    scope = Lumascope(simulate=True)
    recorder = _RecordingLogger()
    monkeypatch.setattr(lumascope_mod, 'logger', recorder)
    yield scope, recorder
    scope.disconnect()


def test_happy_path_lines_are_debug(sim_scope_with_log):
    scope, log = sim_scope_with_log
    assert scope.are_all_connected() is True, (
        'precondition: simulated scope reports all components connected'
    )
    assert log.levels_for('Performing connection check') == ['DEBUG'], (
        'the periodic "Performing connection check" line must be debug'
    )
    assert log.levels_for('All components connected') == ['DEBUG'], (
        'the happy-path "All components connected" line must be debug'
    )
    assert log.levels_for('not connected') == [], (
        'no degraded-path lines may fire when everything is connected'
    )


def test_not_connected_lines_stay_info(sim_scope_with_log):
    from drivers.null_ledboard import NullLEDBoard
    from drivers.null_motorboard import NullMotionBoard

    scope, log = sim_scope_with_log
    scope._led_driver = NullLEDBoard()
    scope._motion_driver = NullMotionBoard()
    scope._camera_driver = None

    assert scope.are_all_connected() is False

    for board in ('LED Board', 'Motion Board', 'Camera'):
        assert log.levels_for(f'{board} not connected') == ['INFO'], (
            f'the "{board} not connected" degraded-path line must stay at info'
        )
    assert log.levels_for('All components connected') == [], (
        'the happy-path line must not fire when components are missing'
    )

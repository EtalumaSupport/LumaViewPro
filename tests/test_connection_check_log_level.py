# Copyright Etaluma, Inc.
"""Regression test: periodic connection-check happy-path logs at debug.

are_all_connected() runs on a periodic timer during a protocol scan, so its
"Performing connection check" and "All components connected" lines flooded
the main log (~1.2k/soak). Those happy-path lines are debug (routine, per
Rule 5); the "<board> not connected" lines stay at info -- a board dropping
out mid-run is a real degraded-path event the user should see.

Levels are read from caplog: the lines the scope module writes through
its main logger, by the file that wrote them.
"""

import logging

import pytest

from modules.lumascope_api import _lumascope as lumascope_mod
from tests.log_capture import records_of
from tests.scope_fakes import build_scope


class _ScopeLog:
    """The levels of the scope module's main-logger lines that name a text."""

    def __init__(self, caplog):
        self._caplog = caplog

    def levels_for(self, substring):
        return [
            r.levelname
            for r in records_of(self._caplog, lumascope_mod)
            if r.name == lumascope_mod.logger.name and substring in r.getMessage()
        ]


@pytest.fixture
def sim_scope_with_log(caplog):
    scope = build_scope(simulate=True)
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger=lumascope_mod.logger.name)
    yield scope, _ScopeLog(caplog)
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

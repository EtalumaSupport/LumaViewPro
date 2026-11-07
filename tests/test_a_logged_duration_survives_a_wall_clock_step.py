# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A logged duration is measured on the steady clock, whatever the wall clock does.

A protocol scan's "completed in" line and a post-processing build's
"Complete ... in" line state how long the work took. Read off the wall
clock, a step of that clock while the work runs -- an NTP correction, a
daylight-saving change on a host whose clock keeps local time, a user
setting the time -- is added to the duration: an hour's step logs a
two-second scan as an hour. Measured on the host's steady clock
(``host_clock.steady_ns``) the step moves nothing.

The log line is the product here: the duration exists nowhere else.

The step is made where every wall read is made, ``host_clock.wall_ns``,
so any wall read between the work's start and end sees it. A step of the
host's clock is a fault the simulator cannot produce; this is its
stand-in.
"""

import re

from _lumascope.drivers import host_clock
from _lumascope.modules import protocol_post_processor, protocol_run_loop
from _lumascope.modules.run_outcome import PendingRunOutcome

import pytest

from tests.stock_channels import LED0
from tests.test_a_timelapse_keeps_its_period import FakeClock, _make_parent
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    turret_in_a_known_slot,
)
from tests.test_an_add_is_one_channel_group import _added, _composite, _run_folder
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture

_AN_HOUR_S = 3600.0
# The work under test takes well under this; a wall step of an hour logs
# far over it.
_UNDER_S = 60.0


class SteppedWallClock:
    """The host's wall clock, stepped forward by ``step`` from the call on."""

    def __init__(self):
        self._real = host_clock.wall_ns
        self._offset_ns = 0

    def __call__(self) -> int:
        return self._real() + self._offset_ns

    def step(self, seconds: float) -> None:
        self._offset_ns += int(seconds * 1e9)


@pytest.fixture
def wall(monkeypatch):
    clock = SteppedWallClock()
    monkeypatch.setattr(host_clock, 'wall_ns', clock)
    return clock


def _told(monkeypatch, module, on_line=None):
    told = []

    def info(message, *_a, **_k):
        told.append(message)
        if on_line is not None:
            on_line(message)

    monkeypatch.setattr(module.logger, 'info', info)
    return told


def test_a_scans_logged_duration_does_not_take_a_wall_clock_step(wall, monkeypatch):
    clock = FakeClock()
    parent = _make_parent(clock, n_scans=1, period_s=0.0)
    scan = parent._step_executor.scan_loop.side_effect

    def scan_across_a_step():
        scan()
        wall.step(_AN_HOUR_S)

    parent._step_executor.scan_loop.side_effect = scan_across_a_step
    told = _told(monkeypatch, protocol_run_loop)

    loop = protocol_run_loop.ProtocolRunLoop(parent)
    # The loop's own name for time, as the pacing tests fake it: the
    # period wait passes no real time.
    monkeypatch.setattr(protocol_run_loop, 'time', clock)
    loop._run_loop_inner(PendingRunOutcome())

    (line,) = [m for m in told if m.startswith('Protocol scan 1 completed in')]
    logged_s = float(re.search(r'completed in ([\d.]+) seconds', line).group(1))
    assert logged_s < _UNDER_S, line


@pytest.mark.usefixtures('turret_in_a_known_slot')
def test_a_builds_logged_duration_does_not_take_a_wall_clock_step(
    scope, tmp_path, wall, monkeypatch
):
    folder = _run_folder(tmp_path, _added(scope, adds=(('BF', LED0),)))

    def step_while_building(message):
        if 'group start' in message:
            wall.step(_AN_HOUR_S)

    told = _told(monkeypatch, protocol_post_processor, on_line=step_while_building)

    _composite(folder)

    (line,) = [m for m in told if 'Complete - Created' in m]
    # The duration is logged as a timedelta: H:MM:SS[.ffffff].
    hours, minutes, seconds = re.search(r' in (\d+):(\d\d):(\d\d(?:\.\d+)?)\.', line).groups()
    logged_s = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    assert logged_s < _UNDER_S, line

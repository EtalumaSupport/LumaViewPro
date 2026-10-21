# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A change records the earliest moment it could have reached the hardware.

A write is recorded once it returns, and a move once its axis is seen
settled; the change itself lands somewhere before that. ``earliest_ns``
is the other end: a write's start, or the start of the last read that saw
the axis still moving. A delay measured from it is never short, which is
what the camera characterization tool measures from, while the rule
judges from the record, so a frame is never accepted early.
"""

import sys
import threading
import time

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope, home_sim_scope
from tests.settings_fixtures import complete_settings

_SLOW_S = 0.05


def test_a_camera_writes_change_is_earliest_at_the_writes_start(monkeypatch):
    # a stand-in by design: the simulator cannot slow a camera write (a
    # simulator gap), so its gain write waits before it writes.
    write = SimulatedCamera.gain

    def slow(self, value):
        time.sleep(_SLOW_S)
        return write(self, value)

    monkeypatch.setattr(SimulatedCamera, 'gain', slow)
    scope = build_scope(simulate=True, register_atexit=False)
    try:
        before = time.monotonic_ns()
        scope.camera.set_gain_db(4.0)
        change = scope.imaging.recorded_changes()['gain']
        assert before <= change.earliest_ns
        assert change.written_ns - change.earliest_ns >= _SLOW_S * 1e9
    finally:
        scope.disconnect()


@pytest.mark.skipif(
    not (sys.platform == 'darwin' or sys.platform.startswith('linux')),
    reason='the firmware-backed simulator runs on macOS and Linux only',
)
def test_an_led_writes_change_is_earliest_at_the_writes_start():
    # Answered inside the LED board's 0.3 s wait, so the write succeeds and
    # its change is recorded the answer's time after it began.
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        scope = session.scope
        scope.simulation.delay_reply('led', 0.15)
        before = time.monotonic_ns()
        scope.illumination.led_on(1, 10)
        change = scope.imaging.recorded_changes()['led']
        assert before <= change.earliest_ns
        assert change.written_ns - change.earliest_ns >= 0.15e9
    finally:
        session.shutdown()


def test_a_moves_change_is_earliest_at_the_last_read_that_saw_it_moving():
    scope = home_sim_scope(build_scope(simulate=True, register_atexit=False))
    try:
        halted = scope.simulation.hold_axis('Z')
        handle = scope.motion.start_move_absolute('Z', 500.0)
        assert halted.wait(10.0), 'the held move never halted'
        # The monitor tells the position listeners after each poll: one
        # after the halt has read the held axis still moving.
        polled = threading.Event()
        scope.motion.add_position_listener(lambda axis, _at: axis == 'Z' and polled.set())
        assert polled.wait(5.0), 'the monitor never read the held axis'
        scope.simulation.clear()
        handle.result()
        change = scope.imaging.recorded_changes()['z_move']
        # Not the record's time: a read after the move began saw it moving.
        assert change.written_ns < change.earliest_ns < change.in_force_ns
    finally:
        scope.disconnect()

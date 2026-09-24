# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The production motor driver homes and moves the real firmware against the
simulated TMC5072 pair.

Instant mode is the suite's tier: a home completes in tens of milliseconds
and lands where the firmware puts it. Realistic mode is measured once: the
ramp runs at the fitted chip clock, and a STOP mid-move stops the stage
short, which the fast tier cannot show.
"""

import sys
import time

import pytest

from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import DEFAULT_DIALECT, MotorBoardSpec, SimWireBackend

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


@pytest.fixture
def board(request):
    model, axes, timing, *named = getattr(request, 'param', ('LS850T', 'XYZT', 'instant'))
    dialect = named[0] if named else DEFAULT_DIALECT
    spec = MotorBoardSpec(model, frozenset(axes), timing=timing, dialect=dialect)
    b = MotorBoard(backend=SimWireBackend(spec))
    try:
        yield b
    finally:
        b.disconnect()


class TestHomingInInstantMode:
    def test_a_full_board_homes_every_axis_and_lands_at_the_firmwares_home(self, board):
        started = time.monotonic()
        assert board.home() is True
        assert time.monotonic() - started < 2.0
        assert board.has_homed() and board.has_thomed()
        # The firmware's own 'Initial Position after home' from the unit config.
        cfg = board.motorconfig
        assert board.current_pos('X') == pytest.approx(
            1221838 / cfg.usteps_per_mm('X') * 1000, abs=1
        )
        assert board.current_pos('Y') == pytest.approx(
            818342 / cfg.usteps_per_mm('Y') * 1000, abs=1
        )
        assert board.current_pos('T') == 1

    @pytest.mark.parametrize('board', [('LS820', 'Z', 'instant')], indirect=True)
    def test_a_z_only_board_homes_z_and_reports_the_missing_stage(self, board):
        assert board.detect_present_axes() == ['Z']
        assert board.home() is True  # the driver counts the firmware's partial home as success
        assert board.has_homed()
        # The firmware zeroes Z at its reference and stops there when it finds
        # no stage to home; the park position is only reached after X and Y.
        assert board.current_pos('Z') == 0.0


class TestMovesInInstantMode:
    def test_a_move_reaches_its_target(self, board):
        assert board.home()
        board.move_abs_pos('X', 20000.0, overshoot_enabled=False)
        assert board.wait_for_position('X', timeout=2.0)
        assert board.current_pos('X') == pytest.approx(20000.0, abs=0.1)

    def test_a_move_into_the_reference_switch_never_reaches_its_target(self, board):
        assert board.home()
        board.move('X', -400_000)  # microsteps, far past the flag
        assert not board.wait_for_position('X', timeout=0.3)
        left, _right = board.limit_switch_status('X')
        assert left == 1
        assert board.current_pos('X') < 0


@pytest.mark.parametrize('board', [('LS850T', 'XYZT', 'realistic')], indirect=True)
class TestRealisticMode:
    def test_a_20_mm_move_takes_the_ramps_time(self, board):
        # 20 mm on the shipped INI at 16 MHz: 0.13 s up, 0.40 s at speed,
        # 0.13 s down, plus the driver's 10 ms poll.
        assert board.home()
        start = board.current_pos('X')
        started = time.monotonic()
        board.move_abs_pos('X', start + 20000.0, overshoot_enabled=False)
        assert board.wait_for_position('X', timeout=5.0)
        elapsed = time.monotonic() - started
        assert 0.6 <= elapsed <= 1.0, elapsed


# The field firmware has no STOP; the 3.0 firmware, which has one, is named.
@pytest.mark.parametrize('board', [('LS850T', 'XYZT', 'realistic', '3.0')], indirect=True)
def test_a_stop_mid_move_leaves_the_stage_short_of_its_target(board):
    assert board.home()
    start = board.current_pos('X')
    target = start + 60000.0
    board.move_abs_pos('X', target, overshoot_enabled=False)
    time.sleep(0.4)
    assert board.motor_stop() is True
    time.sleep(0.3)
    stopped = board.current_pos('X')
    assert start < stopped < target - 10000.0, (start, stopped, target)
    assert board.limit_switch_status('X') == (0, 0)

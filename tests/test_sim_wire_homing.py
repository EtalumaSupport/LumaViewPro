# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The motor driver's home on the firmware that ships: what each reply
means, and that a home returns when the stage has stopped.

Each of ``zhome``, ``home`` and ``thome`` returns on a reply saying the
home is done, or that an axis it would home is absent, and raises
HardwareError on no reply or a reply naming a failure.

The field firmware answers HOME and then drives X, Y and Z to the centre
of travel, and answers THOME and then drives Z back to where it was,
without waiting for either move. What the driver returns after a home is
therefore where the stage IS only if the driver waits: those tests run the
real field firmware in realistic timing and read the stage the moment the
home returns.
"""

import pytest

from drivers.exceptions import HardwareError
from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend
from drivers.sim_wire.mp import tmc5072
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS

pytestmark = pytest.mark.firmware_tier


@pytest.fixture
def board():
    spec = MotorBoardSpec('LS850T', frozenset('XYZT'), dialect='field', timing='realistic')
    b = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimWireBackend(spec))
    try:
        yield b
    finally:
        b.disconnect()


def _firmware_board(model: str, axes: str, dialect: str = '3.0'):
    """(the production driver, the simulated board) on the real firmware."""
    backend = SimWireBackend(MotorBoardSpec(model, frozenset(axes), dialect=dialect))
    driver = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    return driver, backend.motor_board


@pytest.fixture(scope='module')
def homing_boards():
    """Boards shared by the reply tests; each test clears the faults it injects."""
    boards = {'full': _firmware_board('LS850T', 'XYZT'), 'z_only': _firmware_board('LS820', 'Z')}
    yield boards
    for driver, _sim in boards.values():
        driver.disconnect()


@pytest.mark.parametrize(
    ('model', 'axes', 'dialect'),
    [
        ('LS850T', 'XYZT', '3.0'),  # 'XYZ home complete'
        ('LS820', 'Z', '3.0'),  # 'ERROR: X not present'
        ('LS820', 'Z', 'field'),  # 'X not present'
        ('LS850', 'XZ', '3.0'),  # 'ERROR: Y not present'
    ],
)
def test_a_home_done_or_with_x_or_y_absent_returns_and_the_board_is_homed(model, axes, dialect):
    """The firmware homes Z (and T) before it reports a missing X or Y, so
    that reply is a home, not a failure."""
    driver, _sim = _firmware_board(model, axes, dialect)
    try:
        assert driver.initial_homing_complete is False
        assert driver.home() is None
        assert driver.initial_homing_complete is True
    finally:
        driver.disconnect()


@pytest.mark.parametrize(
    ('method', 'board_key'),
    [
        ('zhome', 'full'),  # 'Z home successful'
        ('thome', 'full'),  # 'T home successful'
        ('thome', 'z_only'),  # 'T not present'
    ],
)
def test_a_z_or_turret_home_done_or_with_no_turret_returns(homing_boards, method, board_key):
    driver, _sim = homing_boards[board_key]
    assert getattr(driver, method)() is None


def test_a_turret_home_marks_the_turret_homed(homing_boards):
    driver, _sim = homing_boards['full']
    driver.thome()
    assert driver.has_thomed() is True


@pytest.mark.parametrize('method', ['zhome', 'home', 'thome'])
def test_a_home_with_no_reply_raises(method):
    """The cable is pulled before the command is sent."""
    driver, sim = _firmware_board('LS850T', 'XYZT')
    try:
        sim.unplug()
        with pytest.raises(HardwareError, match='no response'):
            getattr(driver, method)()
        assert driver.has_homed() is False
        assert driver.has_thomed() is False
    finally:
        driver.disconnect()


@pytest.mark.parametrize(
    ('method', 'axis', 'fault'),
    [
        ('zhome', 'Z', tmc5072.SWITCH_NEVER_TRIPS),  # 'ERROR: Z home timeout'
        ('home', 'X', tmc5072.STALL),  # 'ERROR: XY home timeout'
        ('thome', 'T', tmc5072.SWITCH_NEVER_TRIPS),  # 'ERROR: T home timeout'
    ],
)
def test_a_home_the_firmware_reports_failed_raises(method, axis, fault):
    """A failed home leaves the board unhomed, so its own board: a shared one
    may already be homed by another test."""
    driver, sim = _firmware_board('LS850T', 'XYZT')
    try:
        sim.inject(axis, fault)
        with pytest.raises(HardwareError, match='firmware error'):
            getattr(driver, method)()
        assert driver.has_homed() is False
        assert driver.has_thomed() is False
    finally:
        driver.disconnect()


def _raw(board, query, axis):
    return int(board.exchange_command(f'{query}_R{axis}'))


@pytest.mark.slow
def test_after_home_every_axis_is_at_its_target(board):
    assert board.home() is None
    for axis in 'XYZT':
        assert _raw(board, 'ACTUAL', axis) == _raw(board, 'TARGET', axis), axis


@pytest.mark.slow
def test_after_a_turret_home_z_is_back_at_its_target(board):
    assert board.home() is None
    board.move_abs_pos('Z', 3000.0)
    assert board.wait_for_position('Z', timeout=10.0)
    assert board.thome() is None
    for axis in 'ZT':
        assert _raw(board, 'ACTUAL', axis) == _raw(board, 'TARGET', axis), axis


@pytest.fixture
def faulted():
    """(the production driver, the simulated board), field firmware, realistic timing."""
    spec = MotorBoardSpec('LS850T', frozenset('XYZT'), dialect='field', timing='realistic')
    backend = SimWireBackend(spec)
    b = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    try:
        yield b, backend.motor_board
    finally:
        b.disconnect()


# A failed home is reported in the board's own words, which the board sends
# only when its own sequence ends: the host waits that long.


@pytest.mark.slow
def test_a_stuck_z_switch_on_a_z_home_is_the_boards_own_timeout(faulted):
    board, sim = faulted
    sim.inject('Z', tmc5072.SWITCH_NEVER_TRIPS)
    with pytest.raises(HardwareError, match='Z home timeout'):
        board.zhome()

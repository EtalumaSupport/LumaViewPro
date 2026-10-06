# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The LS720's TMCM-6110 read at the wire, on the bench.

The wire-level rows of Stage 4's bench run, driven through the production
driver, ``drivers/tmcm6110.py``: what the board answers, how fast, what its
lid and supply inputs read, and how arrival and a stop look in its
registers. Everything the API asks of the stage rests on these readings,
and none of them can be taken from the simulator, which answers what it was
written to answer.

Run on a Mac with the LS720's 6110 on USB and nothing else holding its
port (no LumaViewPro running):

    python3 -m pytest tests/test_tmcm6110_hardware.py --run-hardware --run-tmcm6110-hardware \
        -s --driver-log -p no:randomly

``--run-hardware`` lets a test reach a real serial port at all. The
``SAP 1`` test moves X's origin; every test after it homes first.

Each test writes what it read to ``$TMCM6110_RECORD_DIR`` (default
``build/tmcm6110_bench_<date>/``) as ``<test>.json``, for the bench record.

The lid and supply test reads the inputs in the state the operator states in
``TMCM6110_EXPECT`` (``lid=closed,power=on`` and so on) and checks the board
reads that state; run it once per state.

The motion tests home the stage first. Rows 2 and 7 move only well inside
the travel an LS720 is known to have; rows 5 and 6 drive onto the limit
switches, and the far-corner home and row 6 need the plate off the stage.
"""

import datetime
import json
import logging
import os
import pathlib
import statistics
import time

import pytest

from drivers import tmcm6110
from drivers.serial_backend import PYSERIAL
from drivers.tmcm6110 import (
    AP_ACTUAL_POSITION,
    AP_ACTUAL_VELOCITY,
    AP_TARGET_POSITION,
    GAP,
    GIO,
    LID_INPUT,
    MOTORS,
    MST,
    POWER_INPUT,
    POWER_PRESENT_ABOVE,
    SAP,
    USB_IDS,
    Tmcm6110Board,
)

pytestmark = pytest.mark.tmcm6110_hardware

logger = logging.getLogger('LVP.tests.tmcm6110_hardware')

REPO = pathlib.Path(__file__).resolve().parents[1]
RECORD_DIR = pathlib.Path(
    os.environ.get('TMCM6110_RECORD_DIR')
    or REPO / 'build' / f'tmcm6110_bench_{datetime.date.today().isoformat()}'
)

# The motion monitor's cycle, which its four reads per moving axis must fit.
MONITOR_CYCLE_S = 0.020
ROUND_TRIPS = 100

# Targets well inside the LS720's travel (the measured far switches are
# X 123.71, Y 79.79 mm); the mixed lengths of row 7, in micrometres.
XY_TARGETS_UM = {
    'X': [20_000, 21_000, 40_000, 40_050, 60_000, 25_000, 25_010, 50_000, 30_000, 45_000],
    'Y': [20_000, 20_500, 35_000, 35_020, 50_000, 22_000, 22_005, 45_000, 30_000, 40_000],
}
ARRIVAL_TIMEOUT_S = 30.0


def _record(name, data):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORD_DIR / f'{name}.json'
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f'[6110 bench] {name}: {json.dumps(data, default=str)}')
    return path


def _defaults():
    return json.loads((REPO / 'data' / 'motorconfig_defaults.json').read_text())


def _identified_ports():
    return [info for info in PYSERIAL.comports() if (info.vid, info.pid) in USB_IDS]


@pytest.fixture(scope='module')
def board():
    board = Tmcm6110Board(motorconfig_defaults=_defaults())
    assert board.found, 'no TMCM-6110 answered on any port with its USB identity'
    yield board
    board.disconnect()


@pytest.fixture(scope='module')
def homed(board):
    started = time.monotonic()
    assert board.home()
    logger.info(f'[6110 bench] home took {time.monotonic() - started:.1f} s')
    return board


def _gap(board, parameter, axis):
    return board._exchange(GAP, parameter, MOTORS[axis])


# ---------------------------------------------------------------------------
# Row 1: enumeration, identity, both baud rates
# ---------------------------------------------------------------------------


def test_row1_the_board_enumerates_and_answers_at_both_baud_rates():
    ports = _identified_ports()
    assert len(ports) == 1, f'expected one port with a 6110 identity, found {ports}'
    info = ports[0]
    answers = {}
    for baud in (9600, 115200):
        port = PYSERIAL.open(
            port=info.device,
            baudrate=baud,
            timeout=tmcm6110.REPLY_TIMEOUT_S,
            write_timeout=tmcm6110.REPLY_TIMEOUT_S,
        )
        try:
            answers[baud] = Tmcm6110Board._query_version(port)
        finally:
            port.close()
    _record(
        'row1_identity',
        {
            'device': info.device,
            'vid': f'0x{info.vid:04X}',
            'pid': f'0x{info.pid:04X}',
            'serial_number': info.serial_number,
            'version_by_baud': answers,
        },
    )
    assert answers[9600] == answers[115200]
    assert answers[9600] is not None and answers[9600].startswith('6110')


# ---------------------------------------------------------------------------
# Row 2: one round trip, and one monitor cycle's reads per moving axis
# ---------------------------------------------------------------------------


def _timed(call, count):
    times = []
    for _ in range(count):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter() - started)
    return times


def _summary_ms(times):
    ms = sorted(t * 1000 for t in times)
    return {
        'n': len(ms),
        'median': round(statistics.median(ms), 3),
        'p95': round(ms[int(0.95 * (len(ms) - 1))], 3),
        'max': round(ms[-1], 3),
    }


def _monitor_cycle(board, axis):
    """The four reads the motion monitor makes of one moving axis."""
    _gap(board, AP_ACTUAL_POSITION, axis)
    _gap(board, AP_ACTUAL_POSITION, axis)
    _gap(board, AP_ACTUAL_VELOCITY, axis)
    _gap(board, AP_TARGET_POSITION, axis)


def test_row2_a_round_trip_and_a_monitor_cycle_fit_the_cycle(homed):
    board = homed
    idle_trip = _timed(lambda: _gap(board, AP_ACTUAL_POSITION, 'X'), ROUND_TRIPS)
    idle_cycle = _timed(lambda: _monitor_cycle(board, 'X'), ROUND_TRIPS)

    board.move_abs_pos('X', 60_000)
    moving_cycle = []
    while len(moving_cycle) < ROUND_TRIPS and not board.target_status('X'):
        moving_cycle.extend(_timed(lambda: _monitor_cycle(board, 'X'), 1))
    _wait_arrived(board, 'X')
    board.move_abs_pos('X', 20_000)
    _wait_arrived(board, 'X')

    data = {
        'round_trip_idle_ms': _summary_ms(idle_trip),
        'monitor_cycle_idle_ms': _summary_ms(idle_cycle),
        'monitor_cycle_moving_ms': _summary_ms(moving_cycle) if moving_cycle else None,
        'cycle_budget_ms': MONITOR_CYCLE_S * 1000,
    }
    _record('row2_timing', data)
    assert moving_cycle, 'X arrived before one cycle was read while it moved'
    assert statistics.median(moving_cycle) < MONITOR_CYCLE_S


# ---------------------------------------------------------------------------
# Row 3: the lid and the stage supply, in the state the operator states
# ---------------------------------------------------------------------------


def _expected_state():
    raw = os.environ.get('TMCM6110_EXPECT', '')
    expected = dict(part.split('=', 1) for part in raw.split(',') if '=' in part)
    if set(expected) != {'lid', 'power'}:
        pytest.skip('set TMCM6110_EXPECT=lid=<open|closed>,power=<on|off> to run this row')
    return expected


def test_row3_the_lid_and_the_supply_read_as_the_operator_set_them(board):
    expected = _expected_state()
    lid_raw = board._exchange(GIO, *LID_INPUT)
    power_raw = board._exchange(GIO, *POWER_INPUT)
    interlocks = board.interlocks()
    _record(
        f'row3_lid_{expected["lid"]}_power_{expected["power"]}',
        {
            'expected': expected,
            'lid_raw': lid_raw,
            'power_raw': power_raw,
            'power_present_above': POWER_PRESENT_ABOVE,
            'interlocks': sorted(interlocks),
        },
    )
    assert ('lid_open' in interlocks) == (expected['lid'] == 'open')
    assert ('stage_unpowered' in interlocks) == (expected['power'] == 'off')


# ---------------------------------------------------------------------------
# Row 7: arrival and stop, as the registers show them
# ---------------------------------------------------------------------------


def _wait_arrived(board, axis):
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_S
    while time.monotonic() < deadline:
        if board.target_status(axis):
            return
        time.sleep(0.01)
    raise AssertionError(f'{axis} did not arrive within {ARRIVAL_TIMEOUT_S} s')


def _registers(board, axis):
    return {
        'target': _gap(board, AP_TARGET_POSITION, axis),
        'actual': _gap(board, AP_ACTUAL_POSITION, axis),
        'velocity': _gap(board, AP_ACTUAL_VELOCITY, axis),
    }


def test_row7_twenty_moves_arrive_with_velocity_0_and_position_at_target(homed):
    board = homed
    moves = []
    for axis, targets in XY_TARGETS_UM.items():
        for target_um in targets:
            board.move_abs_pos(axis, target_um)
            right_after = _registers(board, axis)
            started = time.monotonic()
            _wait_arrived(board, axis)
            arrived = _registers(board, axis)
            moves.append(
                {
                    'axis': axis,
                    'target_um': target_um,
                    'right_after_mvp': right_after,
                    'velocity_0_before_start': right_after['velocity'] == 0
                    and right_after['actual'] != right_after['target'],
                    'arrived': arrived,
                    'seconds': round(time.monotonic() - started, 3),
                }
            )
    _record('row7_arrivals', {'moves': moves})
    for move in moves:
        assert move['arrived']['velocity'] == 0
        assert move['arrived']['actual'] == move['arrived']['target'], move


def test_row7_target_after_a_home(homed):
    board = homed
    data = {axis: _registers(board, axis) for axis in MOTORS}
    _record('row7_target_after_home', data)
    for axis, registers in data.items():
        assert registers['target'] == registers['actual'], (axis, registers)


def test_row7_a_stop_mid_move_and_no_motion_after_it(homed):
    board = homed
    motor = MOTORS['X']
    board.move_abs_pos('X', 20_000)
    _wait_arrived(board, 'X')
    board.move_abs_pos('X', 90_000)
    time.sleep(0.5)
    before_stop = _registers(board, 'X')
    board._exchange(MST, 0, motor)
    at_mst = _registers(board, 'X')

    deadline = time.monotonic() + tmcm6110.STOP_SETTLE_S
    while _gap(board, AP_ACTUAL_VELOCITY, 'X') != 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    stopped = _registers(board, 'X')

    board._exchange(SAP, AP_TARGET_POSITION, motor, stopped['actual'])
    after_sap0 = []
    for _ in range(20):
        after_sap0.append(_registers(board, 'X'))
        time.sleep(0.05)

    _record(
        'row7_stop',
        {
            'before_stop': before_stop,
            'at_mst': at_mst,
            'stopped': stopped,
            'after_sap0_to_actual': after_sap0,
        },
    )
    assert before_stop['velocity'] != 0, 'X was not moving when the stop was sent'
    assert stopped['velocity'] == 0
    assert all(r['velocity'] == 0 for r in after_sap0)
    assert all(r['actual'] == stopped['actual'] for r in after_sap0)


def test_row7_target_after_setting_the_actual_position(homed):
    """Last: ``SAP 1 = 0`` moves X's coordinate origin, so the stage must be
    homed again before any later use."""
    board = homed
    motor = MOTORS['X']
    board.move_abs_pos('X', 30_000)
    _wait_arrived(board, 'X')
    before = _registers(board, 'X')
    board._exchange(SAP, AP_ACTUAL_POSITION, motor, 0)
    after = _registers(board, 'X')
    time.sleep(0.5)
    settled = _registers(board, 'X')
    _record('row7_target_after_sap1', {'before': before, 'after': after, 'settled': settled})
    logger.warning('[6110 bench] X actual set to 0 by SAP 1: home the stage before using it')
    assert after['actual'] == 0


# ---------------------------------------------------------------------------
# Rows 5 and 6: homes timed, the index's repeat, and the ends of travel
# ---------------------------------------------------------------------------

HOMES = 10
# The ends are approached at a fraction of the axis's configured speed, and
# driven toward a target this far past the configured travel, so a switch
# ends the drive, not the target.
END_SPEED_FRACTION = {'X': 0.25, 'Y': 0.25, 'Z': 0.5}
PAST_TRAVEL_UM = {'X': 30_000, 'Y': 30_000, 'Z': 4_000}
END_DRIVE_TIMEOUT_S = 240.0


def _switches(board, axis):
    return {
        'left': _gap(board, tmcm6110.AP_LEFT_SWITCH, axis),
        'right': _gap(board, tmcm6110.AP_RIGHT_SWITCH, axis),
    }


def _drive_to_switch(board, axis, toward_reference):
    """Drive ``axis`` slowly until a limit switch stops it; return where.

    The board stops a motor at an engaged switch itself (the switches are
    enabled at a home). The lid is read first under the driver's lock, as
    every command the driver sends to start X or Y does. Whatever happens,
    the axis is stopped and its speed restored before this returns or
    raises.
    """
    motor = MOTORS[axis]
    params = board.motorconfig.axis_parameters(axis)
    speed = params['Max Positioning Speed']
    sense = -1 if toward_reference else 1
    distance_um = board.motorconfig.travel_limit_um(axis) + PAST_TRAVEL_UM[axis]
    steps = board._to_board(axis, sense * board._um2ustep(axis, distance_um))
    started = time.monotonic()
    try:
        with board._lock:
            board._refuse_if_lid_open(axis)
            board._exchange(
                SAP,
                tmcm6110.AP_MAX_POSITIONING_SPEED,
                motor,
                max(1, int(speed * END_SPEED_FRACTION[axis])),
            )
            board._exchange(tmcm6110.MVP, tmcm6110.MVP_REL, motor, steps)
        time.sleep(0.2)
        while _gap(board, AP_ACTUAL_VELOCITY, axis) != 0:
            if time.monotonic() - started > END_DRIVE_TIMEOUT_S:
                raise AssertionError(f'{axis} still moving after {END_DRIVE_TIMEOUT_S} s')
            time.sleep(0.05)
        actual = _gap(board, AP_ACTUAL_POSITION, axis)
        return {
            'toward_reference': toward_reference,
            'switches': _switches(board, axis),
            'board_usteps': actual,
            'api_um': round(board._usteps_to_um(axis, board._to_board(axis, actual)), 2),
            'seconds': round(time.monotonic() - started, 1),
        }
    finally:
        board.motor_stop()
        board._exchange(SAP, tmcm6110.AP_MAX_POSITIONING_SPEED, motor, speed)


def test_row5_ten_homes_timed_and_the_index_repeat(board):
    homes = []
    try:
        for _ in range(HOMES):
            started = time.monotonic()
            assert board.home()
            seconds = round(time.monotonic() - started, 1)
            # Where the home switch sits from the index the home set as 0:
            # its spread across homes is the index's repeat.
            switch = {axis: _drive_to_switch(board, axis, True) for axis in ('X', 'Y')}
            homes.append({'seconds': seconds, 'home_switch_from_index': switch})
            _record('row5_homes', {'homes': homes})
    finally:
        board.home()
    spread = {
        axis: round(
            max(h['home_switch_from_index'][axis]['api_um'] for h in homes)
            - min(h['home_switch_from_index'][axis]['api_um'] for h in homes),
            2,
        )
        for axis in ('X', 'Y')
    }
    _record(
        'row5_homes',
        {
            'homes': homes,
            'home_seconds': [h['seconds'] for h in homes],
            'switch_from_index_spread_um': spread,
        },
    )
    for home in homes:
        for axis in ('X', 'Y'):
            assert any(home['home_switch_from_index'][axis]['switches'].values()), (axis, home)


def test_row6_each_axis_driven_to_both_ends(board):
    """The plate must be off the stage: Z is driven up to its top end."""
    assert board.home()
    ends = {}
    try:
        for axis in ('Z', 'X', 'Y'):
            far = _drive_to_switch(board, axis, False)
            near = _drive_to_switch(board, axis, True)
            ends[axis] = {'far': far, 'near': near}
            _record('row6_ends', ends)
    finally:
        board.home()
    for axis, end in ends.items():
        assert any(end['far']['switches'].values()), (axis, 'no switch at the far end', end)
        assert any(end['near']['switches'].values()), (axis, 'no switch at the near end', end)


def test_row5_a_home_from_the_far_corner(board):
    """The longest home: X and Y at their travel limits, Z near its top.
    The plate must be off the stage."""
    assert board.home()
    for axis, fraction in (('Z', 0.95), ('X', 1.0), ('Y', 1.0)):
        board.move_abs_pos(axis, board.get_axis_limits(axis)['max'] * fraction)
        _wait_arrived(board, axis)
    start = {axis: _registers(board, axis) for axis in MOTORS}
    started = time.monotonic()
    assert board.home()
    seconds = round(time.monotonic() - started, 1)
    _record('row5_home_from_far_corner', {'start': start, 'seconds': seconds})

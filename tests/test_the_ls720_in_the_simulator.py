# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The LS720 in the simulator: the production drivers over a simulated FX2
and a simulated TMCM-6110, driven through the API as an integrator would.

The stage starts unknown, as every LS720 does, homes with Classic's
sequence, moves on all three axes, refuses X and Y with the lid open while
Z still moves, and runs a protocol with tiling and a Z-stack.
"""

import pytest

from drivers.tmcm6110 import Tmcm6110Board
from modules.exceptions import (
    HardwareCommandRefusedError,
    MoveNotCompletedError,
    RunCheckFailedError,
)
from tests.scope_fakes import bind_settings_like_a_session, build_scope, record_turret_answer
from tests.test_integration import (  # noqa: F401 -- the fixtures are used by name
    _make_protocol,
    _run_and_wait,
    executor,
    executors,
)


@pytest.fixture
def scope():
    s = build_scope(simulate=True, sim_model='LS720', source_path='.', register_atexit=False)
    # Bring-up records that the scope has no turret, asks which objective is
    # mounted and writes the stage offset; a bare scope skipped all three,
    # and a run reads each.
    record_turret_answer(s)
    # The shipped stage offset, which the bench LS720's index positions
    # were measured against.
    bind_settings_like_a_session(s, objective_id='10x Oly', stage_offset={'x': 5500.0, 'y': 4000.0})
    s.imaging.start_streaming()
    yield s
    s.disconnect()


@pytest.fixture
def homed(scope):
    scope.motion.home()
    return scope


def _board(scope):
    return scope._motion_driver._backend.board


def test_the_ls720_runs_the_6110_driver_and_starts_unknown(scope):
    assert isinstance(scope._motion_driver, Tmcm6110Board)
    assert set(scope.capabilities.axes) == {'X', 'Y', 'Z'}
    assert not scope.motion.has_homed()


def test_a_home_makes_every_axis_known_at_its_index_position(homed):
    assert homed.motion.has_homed()
    assert homed.motion.get_current_position() == {'X': 117_690.0, 'Y': 950.0, 'Z': 0.0}


# Where the bench LS720's board stood with A1 and H12 centred by eye
# (2026-10-06, row 4a), in microsteps from its index: the API positions
# then were distances from the index, 6400 microsteps per mm.
BENCH_BOARD_USTEPS = {
    'A1': {'X': round(-9_760.0 * 6.4), 'Y': round(-69_203.6 * 6.4)},
    'H12': {'X': round(-108_860.0 * 6.4), 'Y': round(-6_380.0 * 6.4)},
}
# How closely a well was centred by eye on the bench: 0.3 mm.
BY_EYE_USTEPS = 0.3 * 6400


@pytest.mark.parametrize(('well', 'column', 'row'), [('A1', 0, 0), ('H12', 11, 7)])
def test_a_map_click_on_a_well_drives_the_board_where_the_bench_found_it(homed, well, column, row):
    """The plate frame is the one every model shares; the LS720's index
    positions register its board against it, so a click on a well's centre
    drives the board's registers to where the bench centred that well, X
    from the column-12 end as on an LS850. Read at the board: a plate
    position round-trips through any frame."""
    plate_x, plate_y = homed.runtime_state.get_labware().get_well_position(column, row)
    homed.motion.move_absolute('X', plate_x, frame='plate')
    homed.motion.move_absolute('Y', plate_y, frame='plate')
    board = _board(homed)
    for axis in ('X', 'Y'):
        assert board.position(axis) == pytest.approx(
            BENCH_BOARD_USTEPS[well][axis], abs=BY_EYE_USTEPS
        ), axis


def _at(scope, axis):
    """Where the board says the axis is: the API's position at arrival is
    the one it read just before it saw the arrival."""
    return scope._motion_driver.current_pos(axis)


def test_each_axis_moves_through_the_api(homed):
    homed.motion.move_absolute('X', 12_000)
    homed.motion.move_absolute('Y', 8_000)
    homed.motion.move_absolute('Z', 3_000)
    homed.motion.move_relative('Z', -500)
    assert _at(homed, 'X') == pytest.approx(12_000, abs=0.2)
    assert _at(homed, 'Y') == pytest.approx(8_000, abs=0.2)
    assert _at(homed, 'Z') == pytest.approx(2_500, abs=0.1)


def test_the_lid_refuses_x_and_y_and_lets_z_move(homed):
    _board(homed).lid_open = True
    assert homed.motion.interlocks() == {'lid_open'}
    for axis in ('X', 'Y'):
        with pytest.raises(HardwareCommandRefusedError) as refused:
            homed.motion.move_absolute(axis, 1_000)
        assert refused.value.reason == 'lid_open'
    homed.motion.move_absolute('Z', 1_000)
    assert _at(homed, 'Z') == pytest.approx(1_000, abs=0.1)
    assert _at(homed, 'X') == 117_690.0


def _ended(handle):
    with pytest.raises(MoveNotCompletedError) as raised:
        handle.wait()
    return raised.value.reason


def test_a_lid_refusal_ends_a_waited_move_on_another_axis_as_a_stop_does(homed):
    """The stage stops all three axes in refusing an X move for the lid, so
    a Z move in flight learns it was stopped, as after stop_motion, instead
    of reading its halt as arrival."""
    z = homed.motion.start_move_absolute('Z', 8_000)
    _board(homed).lid_open = True
    with pytest.raises(HardwareCommandRefusedError):
        homed.motion.move_absolute('X', 1_000)
    assert _ended(z) == 'stopped'
    assert 0 < _at(homed, 'Z') < 8_000


def test_a_lid_refused_home_ends_a_waited_move_as_a_stop_does(homed):
    """A home the lid refuses before it moves anything still stops every
    axis, and the X move in flight says so."""
    x = homed.motion.start_move_absolute('X', 100_000)
    _board(homed).lid_open = True
    with pytest.raises(HardwareCommandRefusedError):
        homed.motion.home('Z')
    assert _ended(x) == 'stopped'
    # X set out from its index position, down toward 100 mm.
    assert 100_000 < _at(homed, 'X') < 117_690


def test_a_home_with_the_lid_open_is_refused_and_moves_nothing(homed):
    homed.motion.move_absolute('Z', 1_000)
    _board(homed).lid_open = True
    with pytest.raises(HardwareCommandRefusedError) as refused:
        homed.motion.home()
    assert refused.value.reason == 'lid_open'
    assert _at(homed, 'Z') == pytest.approx(1_000, abs=0.1)
    assert homed.motion.has_homed()


def test_a_run_whose_lid_read_fails_is_not_checked_not_refused(homed, executor, tmp_path):
    """A board that does not answer the lid read leaves the stage's state
    unknown: the run is neither admitted nor refused, as when the
    connection read crashes."""
    _board(homed).silent = True
    try:
        with pytest.raises(RunCheckFailedError) as failed:
            _run_and_wait(executor, _make_protocol([_BF_STEP]), tmp_path)
    finally:
        _board(homed).silent = False
    assert failed.value.reason == 'hardware_state_unknown'
    assert "read the stage's interlocks" in failed.value.message
    assert 'no reply from the TMCM-6110' in str(failed.value.__cause__)


_BF_STEP = {'color': 'BF', 'x': 10.0, 'y': 20.0, 'illumination_ma': 50.0}


def test_a_protocol_with_tiling_and_a_z_stack_completes(homed, executor, tmp_path):
    tiles = [
        {
            'color': 'BF',
            'x': 10.0 + i * 1.0,
            'y': 20.0,
            'tile': f'T{i}',
            'tile_group_id': 1,
            'illumination_ma': 50.0,
        }
        for i in range(3)
    ]
    stack = [
        {'color': 'BF', 'z': z, 'z_slice': i, 'zstack_group_id': 2, 'illumination_ma': 50.0}
        for i, z in enumerate((1_000.0, 1_500.0, 2_000.0))
    ]
    completed, _ = _run_and_wait(executor, _make_protocol(tiles + stack), tmp_path)
    assert completed
    assert _at(homed, 'Z') == pytest.approx(2_000, abs=0.1)


def test_bring_up_on_an_ls720_host_finds_the_6110():
    """The bring-up's own selection: the EL-0940 motor board is tried first
    and finds none, then the 6110 is found by its identity."""
    from drivers.registry import motor_registry
    from drivers.simulated_tmcm6110 import SimulatedTmcm6110, SimulatedTmcm6110Backend
    from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS

    board, fallback = motor_registry.create_with_fallback(
        'auto',
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS,
        backend=SimulatedTmcm6110Backend(
            SimulatedTmcm6110(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
        ),
    )
    try:
        assert isinstance(board, Tmcm6110Board)
        assert fallback is None
    finally:
        board.disconnect()


def test_a_simulated_ls720s_stage_runs_faster_than_the_bench(scope):
    """So a simulated home takes seconds, not the minute and a half the
    simulated board takes at real speed."""
    import time

    from drivers.simulated_tmcm6110 import SCOPE_SPEEDUP

    clock = _board(scope)._clock
    before, wall = clock(), time.monotonic()
    time.sleep(0.05)
    assert (clock() - before) / (time.monotonic() - wall) == pytest.approx(SCOPE_SPEEDUP, rel=0.2)
    assert SCOPE_SPEEDUP >= 10


def test_the_api_registers_the_6110_with_the_other_motor_drivers():
    """Bring-up tries only the drivers registered by the time it runs, which
    are the ones the API imports; a fresh interpreter shows it on its own."""
    import subprocess
    import sys

    names = subprocess.run(
        [
            sys.executable,
            '-c',
            'import modules.lumascope_api._lumascope\n'
            'from drivers.registry import motor_registry\n'
            'print(" ".join(motor_registry.registered_names()))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert 'tmcm6110' in names

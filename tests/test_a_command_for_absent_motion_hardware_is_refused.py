# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motion command for hardware the scope does not have is refused, and names what is missing.

A Z-only scope asked to move X, a scope with no turret asked to turn one, a
manual scope asked for any motion: each answered as if the command had
worked. Moves returned a started move that drove nothing, the stop, the
precision mode and the acceleration limit returned silently, a turret
command on a scope with no turret parked Z and recorded a slot, and a
position read of an absent axis answered 0.0. The one refusal there was
said "check the USB cable" to a scope that never had a cable to check.

Every motion command now asks one presence question: with no motor
controller connected, a model that has one is ``not_connected`` and a
manual model has no motors; with one, an axis the scope lacks is
``axis_absent``. The refusal carries the missing part, and its words come
from it. A read of a position with no hardware behind it answers None.
The scope's own writes ask first, and the step moves of a run and of Go
To Step drive only the axes the scope has, through one conversion.
"""

from __future__ import annotations

import pytest

from modules.exceptions import (
    HardwareCommandRefusedError,
    MissingPart,
    ProtocolRunRefusedError,
)
from modules.lumascope_api._constants import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

MANUAL = ('LS620', 'LS560')
Z_ONLY = ('LS820', 'Lumi')
EVERY_MODEL = (*MANUAL, *Z_ONLY, 'LS850', 'LS850T')
_RUN_WAIT_S = 120.0


def _session(
    tmp_path, model: str, *, homed: bool = True, acquiring: tuple = (), **extra
) -> ScopeSession:
    settings = _settings(tmp_path, microscope=model, **extra)
    if model.startswith('LS850T'):
        settings['turret_objectives'] = dict(TEST_TURRET_OBJECTIVES)
    settings = complete_settings(**settings)
    for layer in acquiring:
        settings[layer]['acquire'] = 'image'
    session = ScopeSession.create(settings, simulate=True)
    if homed and session.scope.motor_connected:
        home_sim_scope(session.scope)
    return session


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model, **kwargs):
        session = _session(tmp_path, model, **kwargs)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def _spy_wire(monkeypatch, scope) -> list:
    """Every command sent to the motor board, in order."""
    sent = []
    driver = scope._motion_driver
    if not hasattr(driver, 'exchange_command'):
        return sent
    real = driver.exchange_command

    def exchange(command, *args, **kwargs):
        sent.append(command)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(driver, 'exchange_command', exchange)
    return sent


def _refused(call) -> HardwareCommandRefusedError:
    with pytest.raises(HardwareCommandRefusedError) as exc:
        call()
    return exc.value


# --- The refusal names what is missing ----------------------------------------


@pytest.mark.parametrize('model', MANUAL)
@pytest.mark.parametrize(
    'command',
    [
        lambda m: m.move_absolute('Z', 1000.0),
        lambda m: m.move_relative('X', 10.0),
        lambda m: m.start_move_absolute('Y', 1000.0),
        lambda m: m.move_turret(2),
        lambda m: m.home('ALL'),
        lambda m: m.home('Z'),
        lambda m: m.home('T'),
        lambda m: m.stop_motion(),
        lambda m: m.set_precision_mode('Z', True),
        lambda m: m.set_acceleration_limit(val_pct=50),
    ],
    ids=[
        'move_absolute',
        'move_relative',
        'start_move_absolute',
        'move_turret',
        'home_all',
        'home_z',
        'home_t',
        'stop_motion',
        'precision',
        'acceleration',
    ],
)
def test_a_manual_scope_has_no_motors(make_session, model, command):
    scope = make_session(model).scope

    refusal = _refused(lambda: command(scope.motion))

    assert refusal.reason == 'axis_absent'
    assert refusal.missing is MissingPart.MOTORS
    assert str(refusal) == 'This microscope has no motors.'


@pytest.mark.parametrize('model', Z_ONLY)
@pytest.mark.parametrize(
    ('command', 'part'),
    [
        (lambda m: m.move_absolute('X', 1000.0), MissingPart.X),
        (lambda m: m.start_move_relative('Y', 10.0), MissingPart.Y),
        (lambda m: m.set_precision_mode('X', True), MissingPart.X),
        (lambda m: m.move_turret(2), MissingPart.TURRET),
        (lambda m: m.home('T'), MissingPart.TURRET),
    ],
    ids=['move_x', 'move_y', 'precision_x', 'move_turret', 'home_t'],
)
def test_a_z_only_scope_names_the_axis_it_lacks(make_session, monkeypatch, model, command, part):
    scope = make_session(model).scope
    z_before = scope.motion.get_current_position('Z')
    sent = _spy_wire(monkeypatch, scope)

    refusal = _refused(lambda: command(scope.motion))

    assert (refusal.reason, refusal.missing) == ('axis_absent', part)
    assert sent == []
    # A turret command parks no Z and records no slot for a turret that is not there.
    assert scope.motion.get_current_position('Z') == z_before
    assert scope.motion.get_turret_slot() is None
    assert scope.motion.get_preferred_turret_slot() is None


def test_a_scope_with_no_turret_names_the_turret(make_session, monkeypatch):
    scope = make_session('LS850').scope
    sent = _spy_wire(monkeypatch, scope)

    refusal = _refused(lambda: scope.motion.move_turret(3))

    assert str(refusal) == 'This microscope has no turret.'
    assert sent == []
    assert scope.motion.get_turret_slot() is None


@pytest.mark.parametrize('model', EVERY_MODEL)
def test_a_name_that_is_no_axis_stays_a_value_error(make_session, model):
    scope = make_session(model).scope
    with pytest.raises(ValueError, match='Axis must be one of'):
        scope.motion.move_absolute('Q', 0)


# --- A controller out of reach --------------------------------------------------


def test_a_pulled_cable_refuses_the_move_and_shutdown_closes_clean(make_session, monkeypatch):
    session = make_session('LS850')
    scope = session.scope
    sent = _spy_wire(monkeypatch, scope)
    monkeypatch.setattr(scope._motion_driver, 'is_connected', lambda: False)

    refusal = _refused(lambda: scope.motion.move_absolute('X', 1000.0))

    assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.MOTOR_CONTROLLER)
    assert 'Check the USB cable' in str(refusal)
    assert sent == []
    session.shutdown()


def test_go_to_step_on_a_scope_whose_board_did_not_come_up_is_refused(make_session, monkeypatch):
    from drivers.null_motorboard import NullMotionBoard

    session = make_session('LS850')
    monkeypatch.setattr(session.scope, '_motion_driver', NullMotionBoard())
    protocol = _build_real_protocol([_make_single_step_protocol().step(idx=0)])

    refusal = _refused(lambda: session.start_go_to_step(protocol, 0))

    assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.MOTOR_CONTROLLER)


@pytest.mark.parametrize('model', EVERY_MODEL)
def test_stop_after_disconnect_says_the_scope_is_disconnected(make_session, model):
    session = make_session(model)
    session.shutdown()
    # A second disconnect is clean on every model: teardown asks first.
    session.scope.disconnect()

    refusal = _refused(session.scope.motion.stop_motion)

    assert refusal.reason == 'scope_disconnected'


# --- Reads ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('model', 'absent'), [('LS820', ('X', 'Y', 'T')), ('LS850', ('T',)), ('LS620', 'XYZT')]
)
def test_a_position_read_of_an_absent_axis_answers_none(make_session, model, absent):
    scope = make_session(model).scope
    for axis in absent:
        assert scope.motion.get_current_position(axis) is None
        assert scope.motion.get_target_position(axis) is None
        assert scope.motion.get_actual_position(axis) is None
    assert set(scope.motion.get_current_position()) == set(scope.capabilities.axes) - set(absent)


def test_after_disconnect_every_read_answers_none(make_session):
    session = make_session('LS850T')
    session.shutdown()
    motion = session.scope.motion
    for axis in 'XYZT':
        assert motion.get_current_position(axis) is None
        assert motion.get_target_position(axis) is None
    assert motion.get_current_position() == {}


def test_a_present_axis_that_lost_its_reference_keeps_its_number(make_session):
    scope = make_session('LS820').scope
    scope.motion.move_absolute('Z', 3000.0)
    scope.motion._set_axis_state('Z', AxisState.UNKNOWN)

    assert scope.motion.get_current_position('Z') == pytest.approx(3000.0, abs=1.0)


# --- The flows that drive only what the scope has --------------------------------


def _wait(run):
    """The run's outcome, once its files are done, so the next run is admitted."""
    outcome = run.wait(timeout_s=_RUN_WAIT_S)
    run.wait_for_files(timeout_s=_RUN_WAIT_S)
    return outcome.status, outcome.reason


def _protocol(n_scans: int | None = None, rows=None):
    step = {**_make_single_step_protocol().step(idx=0), 'Z': 3000.0}
    rows = rows or [step]
    if n_scans is None:
        return _build_real_protocol(rows)
    # A period of 1.2 s, and a duration that fits n_scans of them.
    return _build_real_protocol(rows, period_min=0.02, duration_hrs=0.02 * n_scans / 60)


@pytest.mark.slow
@pytest.mark.parametrize('model', ('LS620', 'LS820'))
def test_a_time_lapse_returns_to_its_first_step_on_the_axes_it_has(
    make_session, monkeypatch, tmp_path, model
):
    import modules.protocol_step_runner as step_runner

    # The return between scans fails as a transient scan failure: the run
    # completes regardless, so the return's own outcome is what is read.
    returns = []
    real = step_runner.ProtocolStepRunner.default_move

    def default_move(self, *args, **kwargs):
        try:
            real(self, *args, **kwargs)
        except Exception as ex:
            returns.append(ex)
            raise
        returns.append(None)

    monkeypatch.setattr(step_runner.ProtocolStepRunner, 'default_move', default_move)
    session = make_session(model)
    runner = session.create_protocol_runner()

    run = runner.run_protocol(
        protocol=_protocol(n_scans=3),
        sequence_name='timelapse',
        parent_dir=str(tmp_path),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )

    assert _wait(run) == ('completed', 'completed')
    assert returns and all(r is None for r in returns), returns


@pytest.mark.slow
@pytest.mark.parametrize('model', MANUAL)
def test_a_single_scan_and_a_composite_on_a_manual_scope_complete(make_session, tmp_path, model):
    session = make_session(model, acquiring=('BF', 'Blue', 'Green'))
    runner = session.create_protocol_runner()

    run = runner.run_single_scan(
        protocol=_protocol(),
        sequence_name='scan',
        parent_dir=str(tmp_path),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    assert _wait(run) == ('completed', 'completed')
    assert runner.run_composite(parent_dir=str(tmp_path)).status == 'completed'


def test_a_z_stack_on_a_z_only_scope_returns_z_to_its_start(make_session, tmp_path, caplog):
    session = make_session(
        'LS820',
        zstack={'step_size': 50, 'range': 200, 'position': 'Current Position at Center'},
    )
    scope = session.scope
    scope.motion.move_absolute('Z', 3000.0)
    runner = session.create_protocol_runner()

    assert _wait(runner.run_zstack(layer='BF', parent_dir=str(tmp_path))) == (
        'completed',
        'completed',
    )
    scope.motion.wait_until_finished_moving(timeout_s=30)

    assert scope.motion.get_current_position('Z') == pytest.approx(3000.0, abs=1.0)
    assert [r.getMessage() for r in caplog.records if r.levelname in ('ERROR', 'CRITICAL')] == []


def test_go_to_step_on_a_z_only_scope_moves_z(make_session):
    session = make_session('LS820')

    session.go_to_step(_protocol(), 0)

    assert session.scope.motion.get_current_position('Z') == pytest.approx(3000.0, abs=1.0)


@pytest.mark.parametrize('model', MANUAL)
def test_go_to_step_on_a_manual_scope_moves_nothing_and_loads_the_layer(make_session, model):
    session = make_session(model)

    assert session.start_go_to_step(_protocol(), 0) == ()
    assert session.get_setting('BF')['focus'] == 3000.0


# --- The run asks presence before it asks about the home --------------------------


def test_an_unhomed_scope_asked_to_run_is_told_to_home(make_session, tmp_path):
    session = make_session('LS850', homed=False)
    runner = session.create_protocol_runner()
    far = {**_make_single_step_protocol().step(idx=0), 'X': 500.0, 'Y': 500.0}

    with pytest.raises(ProtocolRunRefusedError) as exc:
        runner.run_single_scan(
            protocol=_build_real_protocol([far]),
            sequence_name='s',
            parent_dir=str(tmp_path),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )

    assert exc.value.reason == 'position_unknown'


def test_an_unhomed_z_only_scope_with_an_xy_protocol_is_told_it_has_no_xy(make_session, tmp_path):
    session = make_session('LS820', homed=False)
    runner = session.create_protocol_runner()
    a = {**_make_single_step_protocol().step(idx=0), 'Label': 'a', 'X': 10.0, 'Y': 10.0}
    b = {**a, 'Label': 'b', 'X': 30.0, 'Y': 30.0}

    with pytest.raises(ProtocolRunRefusedError) as exc:
        runner.run_single_scan(
            protocol=_build_real_protocol([a, b]),
            sequence_name='s',
            parent_dir=str(tmp_path),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )

    assert exc.value.reason == 'positions_unreachable'

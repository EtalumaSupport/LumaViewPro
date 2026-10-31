# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scope.simulation``: each fault, injected by name, is answered by the API
as the real driver's channel is, on every board family that produces it,
and refused by name on one that cannot.

The answers asserted are the production ones: the exception and reason a
caller meets on hardware. Nothing here reaches a simulated board privately.

On the wire every member but ``hold_axis`` and ``exposed_under`` is a
route, and each injection and take-back is a guarded write: a held
control session refuses another caller's fault, and a real scope refuses
the fault before the session is asked.
"""

import pytest
from fastapi.testclient import TestClient

from drivers import tmcm6110
from drivers.exceptions import CameraReadFailedError, CameraWriteFailedError, HardwareError
from modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingRejected,
    HardwareCommandRefusedError,
    HomingFailedError,
    MotorStopFailedError,
    MoveNotCompletedError,
)
from modules.caller import Caller, calling
from modules.lumascope_api import _lumascope
from modules.lumascope_api.simulation import InjectedFault
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import bind_settings_like_a_session, build_scope, home_sim_scope
from tests.settings_fixtures import complete_settings, installation_of_its_own


@pytest.fixture
def scope():
    """A homed LS850T on the fast tier."""
    return home_sim_scope(build_scope(simulate=True))


@pytest.fixture
def firmware_scope():
    """An unhomed LS850T on the firmware tier: the production EL-0940
    drivers over the board firmware in simulation."""
    return build_scope(simulate=True, sim_tier='firmware')


@pytest.fixture
def ls720():
    """A homed LS720: the production TMCM-6110 driver over its simulated board."""
    scope = build_scope(simulate=True, sim_model='LS720', register_atexit=False)
    scope.motion.home()
    return scope


@pytest.fixture
def ls620():
    """An LS620: the production FX2 drivers over a simulated FX2."""
    return build_scope(simulate=True, sim_model='LS620', register_atexit=False)


def _refused(call, *args, **kwargs) -> ArgumentRefusedError:
    with pytest.raises(ArgumentRefusedError) as refused:
        call(*args, **kwargs)
    return refused.value


# --- the fast tier ----------------------------------------------------------


def test_a_dropped_position_read_refuses_the_z_move_that_needed_it(scope):
    """The backlash leg reads Z's position first; with no reply the move is
    refused before anything is written."""
    scope.motion.move_z(3000.0)
    scope.simulation.drop_replies('motor', command='ACTUAL_RZ')
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion.move_z(1000.0, overshoot_enabled=True)
    assert refused.value.reason == 'position_unread'


def test_a_counted_drop_takes_that_many_replies_and_no_more(scope):
    scope.simulation.drop_replies('motor', count=1, command='ACTUAL_RZ')
    with pytest.raises(HardwareError):
        scope.motion.get_actual_position('Z')
    assert scope.motion.get_actual_position('Z') == pytest.approx(0.0, abs=0.1)


def test_an_uncounted_drop_lasts_until_cleared(scope):
    scope.simulation.drop_replies('motor', count=None, command='ACTUAL_RZ')
    for _ in range(3):
        with pytest.raises(HardwareError):
            scope.motion.get_actual_position('Z')
    scope.simulation.clear()
    assert scope.motion.get_actual_position('Z') == pytest.approx(0.0, abs=0.1)


def test_a_stop_the_board_does_not_answer_raises(scope):
    scope.simulation.drop_replies('motor', command='STOP')
    with pytest.raises(MotorStopFailedError):
        scope.motion.stop_motion()


def test_an_unplugged_motor_board_refuses_a_z_move_then_reads_not_connected(scope):
    """The pulled cable is found by the move's first read, which refuses the
    move as the board not connected; after it the board is not connected.
    What brings it back after ``replug`` is the reconnect, which is not the
    simulator's."""
    scope.simulation.unplug('motor')
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion.move_z(1000.0, overshoot_enabled=True)
    assert refused.value.reason == 'not_connected'
    assert not scope.motor_connected


def test_a_silenced_motor_board_answers_nothing_until_unsilenced(scope):
    scope.simulation.silence('motor')
    with pytest.raises(HardwareError):
        scope.motion.get_actual_position('X')
    scope.simulation.unsilence('motor')
    assert scope.motion.get_actual_position('X') == pytest.approx(0.0, abs=0.1)


def test_a_held_axis_never_arrives_and_its_move_is_not_completed(scope, monkeypatch):
    """The waiter and the monitor give the move up on one bound, whichever
    is first, and both name it ``'stalled'``: a move that did not arrive."""
    # The settle bound is 120 s; the patch stays until it has a seam.
    monkeypatch.setattr(scope.motion, '_MOVE_STALL_WINDOW_S', 0.3)
    scope.simulation.hold_axis('X')
    with pytest.raises(MoveNotCompletedError) as not_completed:
        scope.motion.move_xy(x_um=20_000.0)
    assert not_completed.value.reason == 'stalled'
    assert 0.0 < scope.motion.get_actual_position('X') < 20_000.0


def test_a_removed_camera_refuses_the_next_write(scope):
    scope.simulation.remove_camera()
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.camera.set_gain_db(3.0)
    assert refused.value.reason == 'not_connected'


def test_a_refused_gain_write_is_rejected_and_records_no_target(scope):
    scope.camera.set_gain_db(3.0)
    scope.simulation.refuse_camera_write('gain', 'false')
    with pytest.raises(CameraSettingRejected) as rejected:
        scope.camera.set_gain_db(9.0)
    assert rejected.value.setting == 'gain_db'
    assert scope.imaging.frame_validity.target('gain') == 3.0


def test_a_raising_binning_write_is_a_chained_rejection(scope):
    # The binning reads the framed region it keeps from the settings.
    bind_settings_like_a_session(scope)
    scope.simulation.refuse_camera_write('binning', 'raise')
    with pytest.raises(CameraSettingRejected) as rejected:
        scope.camera.set_binning(2)
    assert rejected.value.setting == 'binning'
    assert isinstance(rejected.value.__cause__, CameraWriteFailedError)


def test_a_refused_write_is_one_write(scope):
    scope.simulation.refuse_camera_write('gain', 'false')
    with pytest.raises(CameraSettingRejected):
        scope.camera.set_gain_db(9.0)
    scope.camera.set_gain_db(9.0)
    assert scope.imaging.frame_validity.target('gain') == 9.0


def test_a_refused_write_counted_twice_is_two_writes(scope):
    scope.simulation.refuse_camera_write('gain', 'false', count=2)
    for _ in range(2):
        with pytest.raises(CameraSettingRejected):
            scope.camera.set_gain_db(9.0)
    scope.camera.set_gain_db(9.0)
    assert scope.imaging.frame_validity.target('gain') == 9.0


@pytest.mark.parametrize('answer', ['failed', 'removed'])
def test_a_failed_frame_size_read_fails_the_snapshot_that_reads_it(scope, answer):
    """Each failed-read answer a driver gives fails the read that met it, once."""
    bind_settings_like_a_session(scope)
    scope.simulation.refuse_camera_read('frame_size', answer)
    with pytest.raises(CameraReadFailedError):
        scope.camera.save_camera_state('test')
    if answer != 'removed':
        assert scope.camera.save_camera_state('test')['frame_size']


def test_a_failed_read_counted_twice_is_two_reads(scope):
    scope.simulation.refuse_camera_read('black_level', 'failed', count=2)
    for _ in range(2):
        with pytest.raises(CameraReadFailedError):
            scope.camera.read_black_level()
    assert scope.camera.read_black_level() is not None


def test_the_fast_led_board_refuses_a_link_fault_by_name(scope):
    for member in (scope.simulation.drop_replies, scope.simulation.unplug):
        refused = _refused(member, 'led')
        assert refused.reason == 'not_injectable_on_board'
        assert refused.board == 'the fast simulated LED board'


def test_the_fast_boards_refuse_a_late_reply_by_name(scope):
    for board in ('motor', 'led'):
        refused = _refused(scope.simulation.delay_reply, board, 0.5)
        assert refused.reason == 'not_injectable_on_board'
        assert refused.board == f'the fast simulated {"motor" if board == "motor" else "LED"} board'


def test_the_fast_motor_board_cannot_fail_a_home(scope):
    refused = _refused(scope.simulation.fail_home, 'X', 'stall')
    assert (refused.reason, refused.board) == (
        'not_injectable_on_board',
        'the fast simulated motor board',
    )


@pytest.mark.parametrize(
    ('member', 'args', 'reason', 'argument'),
    [
        ('unplug', ('camera',), 'fault_unknown', 'board'),
        ('drop_replies', ('motor', 0), 'fault_bound', 'count'),
        ('hold_axis', ('X', 1.0), 'fault_bound', 'at_fraction'),
        ('hold_axis', ('x',), 'not_offered', 'axis'),
        ('fail_home', ('X', 'jammed'), 'not_offered', 'how'),
        ('refuse_camera_write', ('focus', 'false'), 'not_offered', 'node'),
        ('refuse_camera_write', ('gain', 'no'), 'not_offered', 'answer'),
        # A frame-size write answers the window it delivers, never a bare True.
        ('refuse_camera_write', ('frame_size', 'true'), 'not_offered', 'answer'),
        ('refuse_camera_write', ('gain', 'false', 0), 'fault_bound', 'count'),
        ('refuse_camera_read', ('high_conversion_gain', 'failed'), 'not_offered', 'node'),
        ('refuse_camera_read', ('gain', 'false'), 'not_offered', 'answer'),
        ('refuse_camera_read', ('gain', 'failed', 0), 'fault_bound', 'count'),
        ('stall_stream', (0.0, 0.0), 'fault_bound', 'for_s'),
        ('reboot', ('motor', -1.0), 'fault_bound', 'booting_for_s'),
        ('delay_reply', ('motor', 0.0), 'fault_bound', 'delay_s'),
    ],
)
def test_an_argument_the_fault_does_not_take_is_refused(scope, member, args, reason, argument):
    refused = _refused(getattr(scope.simulation, member), *args)
    assert (refused.reason, refused.argument) == (reason, argument)


def test_a_camera_fault_on_a_scope_whose_camera_did_not_come_up_names_the_boards_it_has(
    monkeypatch,
):
    # The camera is a board the scope lacks, refused as a missing motor
    # board is: the boards it has are offered.
    def _no_camera(*args, **kwargs):
        raise RuntimeError('no camera found')

    # a stand-in by design: the simulator has no member that fails the
    # camera at bring-up, so its factory raises as a camera not found does.
    monkeypatch.setattr(_lumascope.camera_registry, 'create', _no_camera)
    scope = build_scope(simulate=True, register_atexit=False)

    refused = _refused(scope.simulation.stall_stream, 0.0, 1.0)

    assert (refused.reason, refused.argument, refused.value, refused.offered) == (
        'fault_unknown',
        'board',
        'camera',
        ('motor', 'led'),
    )


def test_injected_lists_what_was_asked_and_clear_takes_it_all_back(scope):
    scope.simulation.silence('motor')
    scope.simulation.refuse_camera_write('gain', 'none')
    scope.simulation.refuse_camera_read('black_level', 'failed')
    assert scope.simulation.injected() == (
        InjectedFault('silence', {'board': 'motor'}),
        InjectedFault('refuse_camera_write', {'node': 'gain', 'answer': 'none', 'count': 1}),
        InjectedFault(
            'refuse_camera_read', {'node': 'black_level', 'answer': 'failed', 'count': 1}
        ),
    )
    scope.simulation.clear()
    assert scope.simulation.injected() == ()
    assert scope.motion.get_actual_position('X') == pytest.approx(0.0, abs=0.1)
    scope.camera.set_gain_db(4.0)
    assert scope.imaging.frame_validity.target('gain') == 4.0
    assert scope.camera.read_black_level() is not None


def test_a_held_axis_says_when_it_halted_and_clear_releases_it(scope):
    halted = scope.simulation.hold_axis('X')
    handle = scope.motion.start_move_xy(x_um=20_000.0)
    assert halted.wait(10.0), 'the held move never halted'
    assert 0.0 < scope.motion.get_actual_position('X') < 20_000.0
    scope.simulation.clear()
    handle.result()
    assert scope.motion.get_actual_position('X') == pytest.approx(20_000.0, abs=0.1)


def test_an_answered_model_is_what_the_board_reports_until_cleared(scope):
    scope.simulation.answer_model('LS999')
    assert scope.diagnostics.get_motor_info().model == 'LS999'
    assert scope.capabilities.has_turret, 'the board kept the axes it was built with'
    scope.simulation.answer_model(None)
    assert scope.diagnostics.get_motor_info().model is None
    scope.simulation.clear()
    assert scope.diagnostics.get_motor_info().model == 'LS850T'


def test_every_member_refuses_on_a_real_scope(diagnostic_scope):
    refused = _refused(diagnostic_scope.simulation.unplug, 'motor')
    assert (refused.reason, refused.argument) == ('needs_simulated_scope', 'unplug')
    refused = _refused(diagnostic_scope.simulation.clear)
    assert refused.reason == 'needs_simulated_scope'


# --- the firmware tier -------------------------------------------------------


def test_a_firmware_home_fault_raises_from_the_firmwares_reply(firmware_scope):
    firmware_scope.simulation.fail_home('X', 'absent')
    with pytest.raises(HomingFailedError) as failed:
        firmware_scope.motion.home()
    assert failed.value.reason == 'error'
    assert isinstance(failed.value.__cause__, HardwareError)


def test_a_firmware_board_takes_no_command_to_drop_and_holds_no_axis(firmware_scope):
    refused = _refused(firmware_scope.simulation.drop_replies, 'motor', command='STOP')
    assert (refused.reason, refused.argument) == ('argument_not_on_board', 'command')
    refused = _refused(firmware_scope.simulation.drop_replies, 'motor', count=None)
    assert (refused.reason, refused.argument) == ('argument_not_on_board', 'count')
    refused = _refused(firmware_scope.simulation.hold_axis, 'X')
    assert refused.reason == 'not_injectable_on_board'


def test_a_firmware_board_answers_with_the_model_it_was_built_as(firmware_scope):
    refused = _refused(firmware_scope.simulation.answer_model, 'LS999')
    assert (refused.reason, refused.argument) == ('not_injectable_on_board', 'answer_model')


def test_a_rebooting_board_fails_a_read_as_a_dead_port(firmware_scope):
    firmware_scope.simulation.reboot('motor', booting_for_s=2.0)
    with pytest.raises(HardwareError):
        firmware_scope.motion.get_actual_position('Z')


# --- the Classic boards -------------------------------------------------------


def test_the_ls720s_open_lid_refuses_an_x_move_until_cleared(ls720):
    ls720.simulation.interlock(lid_open=True)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        ls720.motion.move_xy(x_um=1_000)
    assert refused.value.reason == 'lid_open_before_motion'
    ls720.simulation.clear()
    ls720.motion.move_xy(x_um=1_000)


def test_the_ls720_fails_a_home_by_its_switch_only(ls720):
    refused = _refused(ls720.simulation.fail_home, 'X', 'stall')
    # The board and what it takes are two fields, so a client reads either.
    assert (refused.reason, refused.argument, refused.board, refused.takes) == (
        'argument_not_on_board',
        'how',
        'the simulated TMCM-6110',
        "'switch_never_trips' only: its home fails by its switch",
    )
    refused = _refused(ls720.simulation.hold_axis, 'X')
    assert (refused.reason, refused.board) == ('not_injectable_on_board', 'the simulated TMCM-6110')


def test_the_ls720s_broken_home_switch_fails_the_home_until_cleared(ls720, monkeypatch):
    # The driver waits its phase bound for a switch that never closes.
    monkeypatch.setattr(tmcm6110, 'HOME_PHASE_TIMEOUT_S', 0.5)
    ls720.simulation.fail_home('Z', 'switch_never_trips')
    with pytest.raises(HomingFailedError) as failed:
        ls720.motion.home()
    assert failed.value.reason == 'error'
    ls720.simulation.clear()
    ls720.motion.home()
    assert not (ls720.motion.axes_without_position().keys() - {'T'})


def test_a_silenced_ls720_board_answers_nothing_until_unsilenced(ls720):
    ls720.simulation.silence('motor')
    with pytest.raises(HardwareError):
        ls720.motion.get_actual_position('X')
    ls720.simulation.unsilence('motor')
    ls720.motion.get_actual_position('X')


def test_the_fx2_refuses_what_its_device_cannot_show(ls620):
    for member, args in (
        (ls620.simulation.refuse_camera_write, ('gain', 'false')),
        (ls620.simulation.stall_stream, (0.0, 1.0)),
        (ls620.simulation.replug, ('led',)),
    ):
        refused = _refused(member, *args)
        assert (refused.reason, refused.board) == ('not_injectable_on_board', 'the simulated FX2')


# --- on the wire -------------------------------------------------------------


@pytest.fixture
def client():
    """A REST client of a simulated session, on one event loop as the server runs."""
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        with TestClient(build_app(session)) as client:
            yield client
    finally:
        session.shutdown()


def test_a_fault_injected_over_the_wire_meets_the_next_call(client):
    injected = client.post(
        '/api/v1/scope/simulation/refuse_camera_write', json={'node': 'gain', 'answer': 'false'}
    )
    assert injected.status_code == 200, injected.text
    assert client.get('/api/v1/scope/simulation/injected').json() == [
        {
            'member': 'refuse_camera_write',
            'arguments': {'node': 'gain', 'answer': 'false', 'count': 1},
        }
    ]

    refused = client.put('/api/v1/scope/camera/gain_db', json={'gain_db': 2.0})
    assert (refused.status_code, refused.json()['reason']) == (500, 'camera_rejected')

    assert client.post('/api/v1/scope/simulation/clear', json={}).status_code == 200
    assert client.get('/api/v1/scope/simulation/injected').json() == []


def test_a_held_control_session_refuses_another_callers_fault_and_admits_its_holders(client):
    session_id = client.post('/api/v1/take_control', json={'name': 'plate reader'}).json()

    refused = client.post('/api/v1/scope/simulation/unplug', json={'board': 'motor'})
    assert (refused.status_code, refused.json()['reason']) == (409, 'control_session_held')
    taken_back = client.post('/api/v1/scope/simulation/clear', json={})
    assert (taken_back.status_code, taken_back.json()['reason']) == (409, 'control_session_held')
    # A read changes nothing, so it is answered whoever holds the session.
    assert client.get('/api/v1/scope/simulation/injected').json() == []

    held = {'Control-Session': session_id}
    admitted = client.post('/api/v1/scope/simulation/unplug', json={'board': 'motor'}, headers=held)
    assert admitted.status_code == 200, admitted.text
    assert [f['member'] for f in client.get('/api/v1/scope/simulation/injected').json()] == [
        'unplug'
    ]
    # Plugged back in before any command finds it pulled, so the session closes cleanly.
    cleared = client.post('/api/v1/scope/simulation/clear', json={}, headers=held)
    assert cleared.status_code == 200, cleared.text
    assert client.get('/api/v1/scope/simulation/injected').json() == []


def test_a_real_scope_refuses_a_fault_over_the_wire_before_a_held_session_is_asked(no_hardware):
    session = ScopeSession.create(
        complete_settings(), source_path=str(installation_of_its_own()), warn_pre_release=False
    )
    try:
        with calling(Caller.GUI):
            session.take_control('operator')
        with TestClient(build_app(session)) as client:
            refused = client.post('/api/v1/scope/simulation/unplug', json={'board': 'motor'})
    finally:
        session.shutdown()
    # The scope cannot have the fault, which no session's holder could change.
    assert (refused.status_code, refused.json()['reason']) == (422, 'needs_simulated_scope')

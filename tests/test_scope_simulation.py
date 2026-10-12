# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scope.simulation``: each fault, injected by name, is answered by the API
as the real driver's channel is, on every board family that produces it,
and refused by name on one that cannot.

The answers asserted are the production ones: the exception and reason a
caller meets on hardware. Nothing here reaches a simulated board privately.
"""

import pytest

from drivers import tmcm6110
from drivers.exceptions import HardwareError
from modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingRejected,
    HardwareCommandRefusedError,
    HomingFailedError,
    MotorStopFailedError,
    MoveNotCompletedError,
)
from modules.lumascope_api.simulation import InjectedFault
from tests.scope_fakes import build_scope, home_sim_scope


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
    scope.motion.move_absolute('Z', 3000.0)
    scope.simulation.drop_replies('motor', command='ACTUAL_RZ')
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion.move_absolute('Z', 1000.0, overshoot_enabled=True)
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
    """The pulled cable is found by the move's first read; after it the
    board is not connected. What brings it back after ``replug`` is the
    reconnect, which is not the simulator's."""
    scope.simulation.unplug('motor')
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion.move_absolute('Z', 1000.0, overshoot_enabled=True)
    assert refused.value.reason == 'position_unread'
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
    monkeypatch.setattr(scope.motion, '_MOTION_SETTLE_TIMEOUT_S', 0.3)
    scope.simulation.hold_axis('X')
    with pytest.raises(MoveNotCompletedError) as not_completed:
        scope.motion.move_absolute('X', 20_000.0)
    assert not_completed.value.reason == 'stalled'
    assert 0.0 < scope.motion.get_actual_position('X') < 20_000.0


def test_a_removed_camera_refuses_the_next_write(scope):
    scope.simulation.remove_camera()
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.imaging.set_gain_db(3.0)
    assert refused.value.reason == 'not_connected'


def test_a_refused_gain_write_is_rejected_and_records_no_target(scope):
    scope.imaging.set_gain_db(3.0)
    scope.simulation.refuse_camera_write('gain', 'false')
    with pytest.raises(CameraSettingRejected) as rejected:
        scope.imaging.set_gain_db(9.0)
    assert rejected.value.setting == 'gain_db'
    assert scope.imaging.frame_validity.target('gain') == 3.0


def test_a_raising_binning_write_is_a_chained_rejection(scope):
    scope.simulation.refuse_camera_write('binning', 'raise')
    with pytest.raises(CameraSettingRejected) as rejected:
        scope.imaging.set_binning_size(2)
    assert rejected.value.setting == 'binning'
    assert isinstance(rejected.value.__cause__, RuntimeError)


def test_a_refused_write_is_one_write(scope):
    scope.simulation.refuse_camera_write('gain', 'false')
    with pytest.raises(CameraSettingRejected):
        scope.imaging.set_gain_db(9.0)
    scope.imaging.set_gain_db(9.0)
    assert scope.imaging.frame_validity.target('gain') == 9.0


def test_the_fast_led_board_refuses_a_link_fault_by_name(scope):
    for member in (scope.simulation.drop_replies, scope.simulation.unplug):
        refused = _refused(member, 'led')
        assert refused.reason == 'not_injectable_on_board'
        assert refused.kind == 'the fast simulated LED board'


def test_the_fast_motor_board_cannot_fail_a_home(scope):
    refused = _refused(scope.simulation.fail_home, 'X', 'stall')
    assert (refused.reason, refused.kind) == (
        'not_injectable_on_board',
        'the fast simulated motor board',
    )


@pytest.mark.parametrize(
    ('member', 'args', 'reason', 'argument'),
    [
        ('unplug', ('camera',), 'fault_unknown', 'board'),
        ('drop_replies', ('motor', 0), 'fault_bound', 'count'),
        ('hold_axis', ('X', 1.0), 'fault_bound', 'at_fraction'),
        ('hold_axis', ('x',), 'axis_unknown', 'axis'),
        ('fail_home', ('X', 'jammed'), 'fault_unknown', 'how'),
        ('refuse_camera_write', ('focus', 'false'), 'fault_unknown', 'node'),
        ('refuse_camera_write', ('gain', 'no'), 'fault_unknown', 'answer'),
        ('stall_stream', (0.0, 0.0), 'fault_bound', 'for_s'),
        ('reboot', ('motor', -1.0), 'fault_bound', 'booting_for_s'),
    ],
)
def test_an_argument_the_fault_does_not_take_is_refused(scope, member, args, reason, argument):
    refused = _refused(getattr(scope.simulation, member), *args)
    assert (refused.reason, refused.argument) == (reason, argument)


def test_injected_lists_what_was_asked_and_clear_takes_it_all_back(scope):
    scope.simulation.silence('motor')
    scope.simulation.refuse_camera_write('gain', 'none')
    assert scope.simulation.injected() == (
        InjectedFault('silence', {'board': 'motor'}),
        InjectedFault('refuse_camera_write', {'node': 'gain', 'answer': 'none'}),
    )
    scope.simulation.clear()
    assert scope.simulation.injected() == ()
    assert scope.motion.get_actual_position('X') == pytest.approx(0.0, abs=0.1)
    scope.imaging.set_gain_db(4.0)
    assert scope.imaging.frame_validity.target('gain') == 4.0


def test_a_held_axis_says_when_it_halted_and_clear_releases_it(scope):
    halted = scope.simulation.hold_axis('X')
    handle = scope.motion.start_move_absolute('X', 20_000.0)
    assert halted.wait(10.0), 'the held move never halted'
    assert 0.0 < scope.motion.get_actual_position('X') < 20_000.0
    scope.simulation.clear()
    handle.wait()
    assert scope.motion.get_actual_position('X') == pytest.approx(20_000.0, abs=0.1)


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


def test_a_rebooting_board_fails_a_read_as_a_dead_port(firmware_scope):
    firmware_scope.simulation.reboot('motor', booting_for_s=2.0)
    with pytest.raises(HardwareError):
        firmware_scope.motion.get_actual_position('Z')


# --- the Classic boards -------------------------------------------------------


def test_the_ls720s_open_lid_refuses_an_x_move_until_cleared(ls720):
    ls720.simulation.interlock(lid_open=True)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        ls720.motion.move_absolute('X', 1_000)
    assert refused.value.reason == 'lid_open'
    ls720.simulation.clear()
    ls720.motion.move_absolute('X', 1_000)


def test_the_ls720_fails_a_home_by_its_switch_only(ls720):
    refused = _refused(ls720.simulation.fail_home, 'X', 'stall')
    assert (refused.reason, refused.argument) == ('argument_not_on_board', 'how')
    refused = _refused(ls720.simulation.hold_axis, 'X')
    assert (refused.reason, refused.kind) == ('not_injectable_on_board', 'the simulated TMCM-6110')


def test_the_ls720s_broken_home_switch_fails_the_home_until_cleared(ls720, monkeypatch):
    # The driver waits its phase bound for a switch that never closes.
    monkeypatch.setattr(tmcm6110, 'HOME_PHASE_TIMEOUT_S', 0.5)
    ls720.simulation.fail_home('Z', 'switch_never_trips')
    with pytest.raises(HomingFailedError) as failed:
        ls720.motion.home()
    assert failed.value.reason == 'error'
    ls720.simulation.clear()
    ls720.motion.home()
    assert ls720.motion.has_homed()


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
        assert (refused.reason, refused.kind) == ('not_injectable_on_board', 'the simulated FX2')

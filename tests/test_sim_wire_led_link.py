# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the API believes about the LEDs when the link to the LED board fails.

The scope is a simulated LS850T on the firmware tier, its LED board the
field LED firmware with the DAC oracle on, so each test compares what the
illumination API believes is lit with what the DAC drives. Each link fault
is the port's own: a reply dropped or garbled on its way back, the host
cable pulled (the firmware runs on), or the board rebooted (its DAC resets).

The field firmware echoes each command (`RE: LED0_10`) before its answer,
and a reply fault takes the first line the board sends after the write, so
it is the echo that is dropped or garbled, not the answer.

Where the API's belief or its report is wrong today, the check is a strict
xfail naming the difference: the demonstration is built here, and the fix
is triage's.
"""

import contextlib
import functools
import time

import pytest

import drivers.sim_wire.backend as sim_backend
from drivers.exceptions import BoardNoReplyError
from lib.lumascope_error import Fault
from modules.exceptions import HardwareCommandRefusedError
from modules.lumascope_api.bring_up import Part
from modules.lumascope_api.illumination import (
    IlluminationAPI,
    LedState,
    LedTransition,
    LedTransitionCtx,
)
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.protocol_drives import held_run_claim
from tests.settings_fixtures import complete_settings

pytestmark = pytest.mark.firmware_tier


@pytest.fixture
def scope(centre_posts):
    """(the illumination API, the simulated LED board, the centre's posts,
    the scope's faults)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sim_backend, 'LedBoardSpec', functools.partial(sim_backend.LedBoardSpec, oracle=True)
        )
        session = ScopeSession.create(
            complete_settings(simulator_tier='firmware', microscope='LS850T'),
            simulate=True,
            warn_pre_release=False,
        )
    try:
        yield (
            session.scope.illumination,
            session.scope._led_driver._backend.led_board,
            centre_posts,
            session.scope.simulation,
        )
    finally:
        session.shutdown()


def _warnings(posts) -> list:
    """(category, title) of each warning posted."""
    return [(n.category, n.title) for n in posts if n.severity == Severity.WARNING]


def _believed(illumination) -> dict:
    """Address -> mA, for each address the API believes is lit."""
    return {
        r.address: r.illumination_ma
        for r in illumination.get_led_states()
        if r.state is LedState.LIT
    }


def _driven(board) -> dict:
    """Address -> DAC code, for each address the DAC drives."""
    return {
        address: code
        for address, (enabled, powered, code) in enumerate(board.state('DAC'))
        if enabled and powered and code
    }


def test_a_dropped_echo_leaves_the_answer_and_the_api_is_right(scope):
    illumination, board, posts, simulation = scope
    simulation.drop_replies('led')
    illumination.led_on_by_address(0, 10)
    assert board.replies_dropped == 1, 'the drop must have taken the echo'
    assert _believed(illumination) == {0: 10.0}
    # The firmware's mA_to_dac: int(60.936 * 10 + 890).
    assert _driven(board) == {0: 1499}
    assert _warnings(posts) == []


def _state(illumination, address: int) -> LedState:
    return next(r.state for r in illumination.get_led_states() if r.address == address)


def test_a_write_the_board_did_not_answer_raises_and_its_led_is_unknown(scope):
    # The firmware lights the channel and its answer comes after the
    # driver's wait: the LED is on, and nothing the host was told says so.
    illumination, board, posts, simulation = scope
    simulation.delay_reply('led', 0.5)
    with pytest.raises(BoardNoReplyError):
        illumination.led_on_by_address(1, 10)
    assert 1 in _driven(board)
    assert _state(illumination, 1) is LedState.UNKNOWN
    # The raise is the report: nothing below the GUI posts one.
    assert _warnings(posts) == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='K84: any non-empty reply confirms an LED write; a garbled echo is taken as the '
    "answer, and the board's own answer is discarded unread",
)
def test_a_garbled_reply_does_not_confirm_a_write(scope):
    illumination, board, _posts, _simulation = scope
    board.garble_next_reply()
    # Which fault an unconfirmed write raises is K84's to decide; what is
    # pinned is that the write is not believed.
    with contextlib.suppress(Fault):
        illumination.led_on_by_address(0, 10)
    assert _state(illumination, 0) is not LedState.LIT


def test_a_write_the_board_never_got_is_refused_and_not_believed(scope):
    # The write is what finds the board gone, so the call is answered as one
    # made with no LED controller, chained to the no-answer.
    illumination, board, posts, simulation = scope
    simulation.unplug('led')
    with pytest.raises(HardwareCommandRefusedError) as refused:
        illumination.led_on_by_address(1, 10)
    assert refused.value.reason == 'not_connected'
    assert isinstance(refused.value.__cause__, BoardNoReplyError)
    assert _driven(board) == {}
    assert _believed(illumination) == {}
    assert _state(illumination, 1) is LedState.UNKNOWN
    assert _warnings(posts) == []


def test_a_write_to_a_lost_board_fails_out_loud(scope):
    illumination, _board, _posts, simulation = scope
    simulation.unplug('led')
    with pytest.raises(HardwareCommandRefusedError):
        illumination.led_on_by_address(1, 10)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        illumination.led_on_by_address(2, 10)
    assert refused.value.reason == 'not_connected'


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='after the board reboots the API goes on believing the channel it lit is lit; '
    'the rebooted board drives nothing',
)
def test_after_a_reboot_the_api_does_not_believe_a_channel_lit(scope):
    illumination, board, _posts, simulation = scope
    illumination.led_on_by_address(0, 10)
    simulation.reboot('led')
    with pytest.raises(HardwareCommandRefusedError):
        illumination.led_on_by_address(1, 10)  # the write that finds the link gone
    # The firmware's boot clears the DAC.
    deadline = time.monotonic() + 10
    while _driven(board):
        assert time.monotonic() < deadline, 'the rebooted board never cleared its DAC'
        time.sleep(0.05)
    assert _believed(illumination) == {}


def _states(illumination) -> set:
    return {r.state for r in illumination.get_led_states()}


def test_an_all_off_the_board_did_not_answer_raises_and_every_led_is_unknown(scope):
    illumination, _board, _posts, simulation = scope
    illumination.led_on_by_address(0, 10)
    simulation.delay_reply('led', 0.5)
    with pytest.raises(BoardNoReplyError):
        illumination.leds_off()
    assert _states(illumination) == {LedState.UNKNOWN}


def test_the_safety_off_the_board_did_not_answer_says_so_without_raising(scope):
    illumination, _board, _posts, simulation = scope
    illumination.led_on_by_address(0, 10)
    simulation.delay_reply('led', 0.5)
    assert illumination.force_off() is False
    assert _states(illumination) == {LedState.UNKNOWN}


def test_a_lease_release_darkens_the_led_its_unanswered_write_left_unknown(scope):
    illumination, board, _posts, simulation = scope
    lease = illumination.acquire_led_lease('protocol', claim=held_run_claim())
    simulation.delay_reply('led', 0.5)
    with pytest.raises(BoardNoReplyError):
        lease.apply(
            LedTransition.MANUAL_STEP,
            LedTransitionCtx(channel=1, illumination_ma=10.0, preview_on=True),
        )
    assert 1 in _driven(board)
    lease.release()
    assert _state(illumination, 1) is LedState.DARK
    assert 1 not in _driven(board)


def test_a_bring_up_all_off_the_board_did_not_answer_leaves_every_led_unknown(monkeypatch):
    # a stand-in by design: a fault is armed for the board's next command,
    # and bring-up writes others first (the firmware STIM probe), so the
    # delay is armed as bring-up's all-off starts, the first scope off after
    # connect.
    scope_off = IlluminationAPI._leds_off_if_present
    armed = []

    def armed_once(self):
        if not armed:
            armed.append(True)
            self._scope.simulation.delay_reply('led', 0.5)
        return scope_off(self)

    monkeypatch.setattr(IlluminationAPI, '_leds_off_if_present', armed_once)
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        led = next(p for p in session.scope.bring_up_record().parts if p.part == Part.LED)
        assert (led.up, led.cause) == (True, 'safety_off_failed')
        assert _states(session.scope.illumination) == {LedState.UNKNOWN}
    finally:
        session.shutdown()

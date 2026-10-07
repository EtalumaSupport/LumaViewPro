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

import functools
import sys
import time

import pytest

import drivers.sim_wire.backend as sim_backend
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )

_UNCONFIRMED = ('LED Safety', 'LED command did not confirm')


@pytest.fixture
def scope(centre_posts):
    """(the illumination API, the simulated LED board, the centre's posts)."""
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
        )
    finally:
        session.shutdown()


def _warnings(posts) -> list:
    """(category, title) of each warning posted."""
    return [(n.category, n.title) for n in posts if n.severity == Severity.WARNING]


def _believed(illumination) -> dict:
    """Colour -> mA, for each channel the API believes is lit."""
    return {
        color: state['illumination_ma']
        for color, state in illumination.get_led_states().items()
        if state['enabled']
    }


def _driven(illumination, board) -> dict:
    """Colour -> DAC code, for each channel the DAC drives."""
    return {
        illumination.state_ch2color(channel): code
        for channel, (enabled, powered, code) in enumerate(board.state('DAC')[:6])
        if enabled and powered and code
    }


def test_a_dropped_echo_leaves_the_answer_and_the_api_is_right(scope):
    illumination, board, posts = scope
    board.drop_next_reply()
    illumination.led_on(0, 10)
    assert _believed(illumination) == {'Blue': 10.0}
    # The firmware's mA_to_dac: int(60.936 * 10 + 890).
    assert _driven(illumination, board) == {'Blue': 1499}
    assert _warnings(posts) == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='K84: any non-empty reply confirms an LED write; a garbled echo is taken as the '
    "answer, and the board's own answer is discarded unread",
)
def test_a_garbled_reply_does_not_confirm_a_write(scope):
    illumination, board, posts = scope
    board.garble_next_reply()
    illumination.led_on(0, 10)
    assert _warnings(posts) == [_UNCONFIRMED]


def test_a_write_the_board_never_got_is_reported(scope):
    illumination, board, posts = scope
    board.unplug()
    illumination.led_on(1, 10)
    assert _warnings(posts) == [_UNCONFIRMED]
    assert _driven(illumination, board) == {}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F16: the API caches a failed led_on as lit; the popup is the write's only report, "
    'and led_on returns as if it had succeeded',
)
def test_a_write_the_board_never_got_is_not_believed(scope):
    illumination, board, _posts = scope
    board.unplug()
    illumination.led_on(1, 10)
    assert _believed(illumination) == {}


def test_a_write_to_a_lost_board_fails_out_loud(scope):
    from modules.exceptions import HardwareCommandRefusedError

    illumination, board, _posts = scope
    board.unplug()
    illumination.led_on(1, 10)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        illumination.led_on(2, 10)
    assert refused.value.reason == 'not_connected'


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='after the board reboots the API goes on believing the channel it lit is lit; '
    'the rebooted board drives nothing',
)
def test_after_a_reboot_the_api_does_not_believe_a_channel_lit(scope):
    illumination, board, _posts = scope
    illumination.led_on(0, 10)
    board.reboot()
    illumination.led_on(1, 10)  # the write that finds the link gone
    # The firmware's boot clears the DAC.
    deadline = time.monotonic() + 10
    while _driven(illumination, board):
        assert time.monotonic() < deadline, 'the rebooted board never cleared its DAC'
        time.sleep(0.05)
    assert _believed(illumination) == {}

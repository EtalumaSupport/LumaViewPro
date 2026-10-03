"""An LED request lights at the board's step, and the scope records what it commanded.

Each board drives its LEDs in steps: one brightness byte (840 / 255 mA) on
the FX2's Classic peripheral, whole mA on the EL-0940 LED firmware. A
request is commanded at the nearest step, a request above zero but below one
step at one step, and the API skips, sends, stores and returns that commanded
current. An FX2 request of 1.0 mA was byte 0, dark, while the record said
1.0; a serial request of 0.5 mA was sent as ``LED0_0``, which the field
firmware drives at its DAC offset instead of off.
"""

import functools
import logging
import sys
import time

import pytest

import drivers.sim_wire.backend as sim_backend
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_FX2_STEP_MA = 840 / 255


@pytest.fixture(scope='module')
def fx2_session():
    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


@pytest.fixture
def fx2(fx2_session):
    """(the illumination API, the simulated LED peripheral), all LEDs off after."""
    scope = fx2_session.scope
    yield scope.illumination, scope._led_driver._fx2._transport.device.leds
    scope.illumination.leds_off()


def test_an_fx2_request_below_one_step_lights_at_one_step(fx2, caplog):
    illumination, peripheral = fx2
    with caplog.at_level(logging.INFO, logger='LVP.api'):
        commanded = illumination.led_on('BF', 1.0)
    assert any('(requested 1.0)' in r.getMessage() for r in caplog.records)
    assert peripheral.commands[-1] == ('D', 1)
    assert commanded == pytest.approx(_FX2_STEP_MA)
    assert illumination.get_led_state('BF')['illumination_ma'] == pytest.approx(_FX2_STEP_MA)


def test_an_fx2_request_repeated_is_written_once(fx2):
    illumination, peripheral = fx2
    illumination.led_on('BF', 5.0)
    written = len(peripheral.commands)
    assert illumination.led_on('BF', 5.0) == pytest.approx(2 * _FX2_STEP_MA)
    assert len(peripheral.commands) == written
    assert peripheral.commands[-1] == ('D', 2)


def test_an_fx2_frame_records_the_commanded_current(fx2_session, fx2):
    illumination, _peripheral = fx2
    illumination.led_on('BF', 5.0)
    imaging = fx2_session.scope.imaging
    assert imaging.capture_and_wait(accept_dark=True, timeout_s=2.0) is not None
    record = imaging.last_capture_info['frame_record']
    assert record.illumination_ma['BF'] == pytest.approx(2 * _FX2_STEP_MA)


def test_the_simulated_serial_board_steps_as_the_board_does():
    session = ScopeSession.create(
        complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
    )
    try:
        illumination, board = session.scope.illumination, session.scope._led_driver
        sent = []
        exchange = board.exchange_command
        board.exchange_command = lambda command, *a, **k: (
            sent.append(command),
            exchange(command, *a, **k),
        )[1]
        assert illumination.led_on(0, 0.5) == 1.0
        assert illumination.led_on(0, 0) == 0.0
        assert sent == ['LED0_1', 'LED0_OFF']
    finally:
        session.shutdown()


_firmware_tier = pytest.mark.skipif(
    not (sys.platform == 'darwin' or sys.platform.startswith('linux')),
    reason='the firmware-backed simulator runs on macOS and Linux only',
)


@pytest.fixture
def serial():
    """(the illumination API, the simulated LED board running the field firmware)."""
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
        yield session.scope.illumination, session.scope._led_driver._backend.led_board
    finally:
        session.shutdown()


def _dac(board, channel: int) -> int:
    """The DAC code the board drives on a channel; 0 when the channel is dark."""
    enabled, powered, code = board.state('DAC')[channel]
    return code if enabled and powered else 0


@_firmware_tier
def test_a_serial_request_below_1_ma_lights_at_1_ma(serial):
    illumination, board = serial
    commanded = illumination.led_on(0, 0.5)
    # The field firmware's mA_to_dac: int(60.936 * 1 + 890).
    assert _dac(board, 0) == 950
    assert illumination.get_led_state('Blue')['illumination_ma'] == 1.0
    assert commanded == 1.0


@_firmware_tier
def test_a_blocking_serial_request_off_the_step_is_confirmed_at_once(serial):
    illumination, board = serial
    started = time.monotonic()
    commanded = illumination.led_on(0, 1.7, block=True)
    assert _dac(board, 0) == 1011
    assert time.monotonic() - started < 1.0
    assert commanded == 2.0


@_firmware_tier
def test_a_serial_request_of_0_ma_is_dark_and_stays_on_at_0(serial):
    illumination, board = serial
    illumination.led_on(0, 10)
    commanded = illumination.led_on(0, 0)
    assert _dac(board, 0) == 0
    assert illumination.get_led_state('Blue') == {'enabled': True, 'illumination_ma': 0.0}
    assert commanded == 0.0

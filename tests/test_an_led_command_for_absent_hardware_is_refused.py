# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LED command for hardware the scope does not have is refused, and names what is missing.

With no LED board every LED command logged one WARNING and returned the
None a successful off returns, so an L2 ``led_on`` on a board-less scope
answered as if it had lit. After ``disconnect()`` the state reads still
said the last channel was lit, the capabilities took the null board's 0 mA
as a real cap (a run was refused "Illumination must be 0-0 mA"), an LS560
asked to light channel 2 lit Red, and a run whose LED board was pulled
captured on, dark, to its end.

Every LED command now asks for the LED controller on the io lane, and an
on asks the model for the LED; an off whose end state already holds is
satisfied with no write. The reads answer None with no board installed,
the capabilities carry no cap and no channels, and a run is refused for
the missing LED controller before its steps are judged.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from drivers.null_ledboard import NullLEDBoard
from modules import common_utils
from modules.exceptions import (
    ArgumentRefusedError,
    MissingPart,
    ProtocolRunRefusedError,
)
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx
from tests.test_a_command_for_absent_motion_hardware_is_refused import (
    EVERY_MODEL,
    _protocol,
    _refused,
    _wait,
    make_session,  # noqa: F401 -- the fixture this file's tests take
)
from tests.log_capture import capture_module_log, messages
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol


def _spy_led_writes(monkeypatch, scope) -> list:
    """Every on, off and all-off sent to the LED driver, in order."""
    sent = []
    driver = scope._led_driver
    for name in ('led_on', 'led_off', 'leds_off'):
        real = getattr(driver, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            sent.append((_name, *args))
            return _real(*args, **kwargs)

        monkeypatch.setattr(driver, name, spy)
    return sent


def _remove_board(monkeypatch, scope) -> None:
    """The scope as one whose LED board never came up: the null board installed."""
    monkeypatch.setattr(scope, '_led_driver', NullLEDBoard())


def _led_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and 'led' in r.getMessage().lower()
    ]


def _preview(lit: bool) -> LedTransitionCtx:
    return LedTransitionCtx(channel=3, illumination_ma=10.0, preview_on=lit)


# --- Board present: nothing changes but an LED the model lacks --------------------


@pytest.mark.parametrize('model', EVERY_MODEL)
def test_with_its_board_an_led_lights_and_goes_dark_as_before(make_session, caplog, model):
    illumination = make_session(model).scope.illumination

    commanded = illumination.led_on('BF', 10.0)
    assert illumination.get_led_state('BF') == {'enabled': True, 'illumination_ma': commanded}
    illumination.led_off('BF')
    assert illumination.get_led_state('BF')['enabled'] is False
    illumination.led_on('BF', 10.0)
    illumination.leds_off()

    assert illumination.get_led_states()['BF']['enabled'] is False
    assert _led_lines(caplog) == []


@pytest.mark.parametrize(
    ('led', 'part'), [(2, MissingPart.led(2)), ('Red', MissingPart.led('Red'))]
)
def test_an_led_the_model_lacks_is_refused_by_name_and_lights_nothing(
    make_session, monkeypatch, led, part
):
    scope = make_session('LS560').scope
    sent = _spy_led_writes(monkeypatch, scope)

    refusal = _refused(lambda: scope.illumination.led_on(led, 10.0))

    assert (refusal.reason, refusal.missing) == ('axis_absent', part)
    assert sent == []
    assert scope.illumination.get_led_state('Red')['enabled'] is False


@pytest.mark.parametrize(('model', 'led'), [('LS560', 'Red'), ('LS560', 2), ('Lumi', 'Lumi')])
def test_an_off_of_an_led_the_model_lacks_is_silent(make_session, monkeypatch, caplog, model, led):
    scope = make_session(model).scope
    sent = _spy_led_writes(monkeypatch, scope)

    scope.illumination.led_off(led)

    assert sent == []
    assert _led_lines(caplog) == []


def test_a_name_that_is_no_layer_and_a_number_off_the_board_are_the_requests(make_session):
    """Neither is absent hardware: no scope has a Purple layer, and an FX2 board has no channel 5."""
    illumination = make_session('LS620').scope.illumination

    with pytest.raises(ArgumentRefusedError) as purple:
        illumination.led_on('Purple', 10.0)
    with pytest.raises(ArgumentRefusedError) as five:
        illumination.led_on(5, 10.0)

    assert purple.value.reason == 'layer_unknown'
    assert (five.value.reason, five.value.offered) == ('led_channel_unknown', (0, 1, 2, 3))


# --- No LED board ----------------------------------------------------------------


@pytest.mark.parametrize('model', EVERY_MODEL)
def test_without_its_board_a_light_is_refused_and_an_off_is_satisfied(
    make_session, monkeypatch, caplog, model
):
    scope = make_session(model).scope
    _remove_board(monkeypatch, scope)
    illumination = scope.illumination

    for light in (
        lambda: illumination.led_on('BF', 10.0),
        lambda: illumination.apply_transition(LedTransition.MANUAL_STEP, _preview(True)),
    ):
        refusal = _refused(light)
        assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.LED_CONTROLLER)
        assert str(refusal) == 'The LED controller is not connected.'
    illumination.led_off('BF')
    illumination.leds_off()
    illumination.apply_transition(LedTransition.MANUAL_STEP, _preview(False))

    assert _led_lines(caplog) == []


def test_the_reads_answer_none_and_nothing_is_believed_lit_after_disconnect(make_session):
    session = make_session('LS850')
    illumination = session.scope.illumination
    heard = []
    illumination.add_led_listener(lambda color, enabled, ma: heard.append((color, enabled)))
    illumination.led_on('BF', 10.0)

    session.shutdown()

    assert illumination.get_led_state('BF') is None
    assert illumination.get_led_states() is None
    assert illumination.save_led_state('after') is None
    assert illumination.state_ch2color(3) is None
    assert ('BF', False) in heard
    assert common_utils.resolve_channel_identity(illumination, 'Green') == 'Green'


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_a_light_says_the_scope_is_disconnected(make_session, model):
    session = make_session(model)
    session.shutdown()

    refusal = _refused(lambda: session.scope.illumination.led_on('BF', 10.0))

    assert refusal.reason == 'scope_disconnected'


@pytest.mark.parametrize('model', ('LS820', 'LS620'))
def test_go_to_step_after_disconnect_says_the_scope_is_disconnected(make_session, model):
    session = make_session(model)
    session.shutdown()

    refusal = _refused(lambda: session.start_go_to_step(_protocol(), 0))

    assert refusal.reason == 'scope_disconnected'


def test_a_shutdown_after_disconnect_logs_nothing_for_the_led(make_session, caplog):
    session = make_session('LS850')
    session.scope.disconnect()

    session.shutdown()
    session.scope.disconnect()

    assert _led_lines(caplog) == []


def test_a_scope_that_came_up_without_its_board_has_no_led_cap_or_channels(make_session):
    from modules.scope_capabilities import ScopeCapabilities

    scope = make_session('LS620').scope

    caps = ScopeCapabilities.from_drivers(
        motion=scope._motion_driver,
        led=NullLEDBoard(),
        camera=None,
        layer_identity=scope.layer_identity,
        scope_models=scope.scope_models,
    )

    assert (caps.led_max_ma, caps.led_channels) == (None, None)
    assert caps.led_colors == ('BF', 'Blue', 'Green', 'Red')


def test_a_run_without_the_led_board_is_refused_for_the_led_controller(
    make_session, monkeypatch, tmp_path
):
    session = make_session('LS850')
    scope = session.scope
    _remove_board(monkeypatch, scope)
    monkeypatch.setattr(
        scope,
        'capabilities',
        dataclasses.replace(scope.capabilities, led_max_ma=None, led_channels=None),
    )
    runner = session.create_protocol_runner()

    with pytest.raises(ProtocolRunRefusedError) as exc:
        runner.run_single_scan(
            protocol=_protocol(),
            sequence_name='s',
            parent_dir=str(tmp_path),
        )

    assert exc.value.reason == 'hardware_disconnected'
    assert str(exc.value).startswith('The LED controller is not connected.')


@pytest.mark.parametrize(('preview_on', 'moves'), [(False, True), (True, False)])
def test_go_to_step_without_the_board_moves_only_when_its_preview_is_dark(
    make_session, monkeypatch, preview_on, moves
):
    session = make_session('LS820', protocol_led_on=preview_on)
    scope = session.scope
    _remove_board(monkeypatch, scope)
    before = scope.motion.get_current_position('Z')

    if moves:
        session.go_to_step(_protocol(), 0)
        assert scope.motion.get_current_position('Z') == pytest.approx(3000.0, abs=1.0)
    else:
        refusal = _refused(lambda: session.start_go_to_step(_protocol(), 0))
        assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.LED_CONTROLLER)
        assert scope.motion.get_current_position('Z') == before


# --- A board pulled mid-run ------------------------------------------------------


def test_a_board_pulled_mid_run_ends_the_run_at_its_next_led_write(
    make_session, monkeypatch, tmp_path
):
    import modules.sequenced_capture_runner as runner_module

    runner_log = capture_module_log(monkeypatch, runner_module)
    session = make_session('LS850')
    scope = session.scope
    driver = scope._led_driver
    real_on = driver.led_on

    def pulled_after_this(*args, **kwargs):
        result = real_on(*args, **kwargs)
        monkeypatch.setattr(driver, 'is_connected', lambda: False)
        return result

    monkeypatch.setattr(driver, 'led_on', pulled_after_this)
    first = {**_make_single_step_protocol().step(idx=0), 'Z': 3000.0}
    second = {**first, 'Name': 'A1_second', 'Label': 'A1_second', 'Color': 'Blue', 'X': 12.0}
    runner = session.create_protocol_runner()

    run = runner.run_single_scan(
        protocol=_build_real_protocol([first, second]),
        sequence_name='pulled',
        parent_dir=str(tmp_path),
    )

    assert _wait(run) == ('failed', 'hardware_disconnected')
    # The channel lit before the pull may still be lit on the board; the
    # cleanup says so rather than that it forced it dark.
    cleanup = [m for m in messages(runner_log) if 'Cleanup: LED end-state undecided' in m]
    assert cleanup and all('could be forced dark' in m for m in cleanup), cleanup

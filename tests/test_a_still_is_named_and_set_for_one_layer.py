# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A still is named for the light, and taken at that layer's stored settings.

The still's name is the light: the lit LED, else the shown layer
(``common_utils.read_light``). The LEDs are written on the IO
lane and the camera on the camera lane, so the light could change without
the camera following, and a still was named for one layer and taken at
another's: after a run (named for its last step at the pre-run layer's
settings), after a raw LED, after a typed value, and pressed between a
show's LED and camera halves. The still now has the camera hold its
name's layer inside its own camera-lane task (``manual_capture``,
``ImagingAPI._hold_layer_for_capture``), unless a raw camera write came
last: then the raw values stand and the frame's record says no layer was
held.
"""

from __future__ import annotations

import logging

import pytest

from modules.exceptions import CaptureFailedError, HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0, LED1

# Distinct from each other and from the simulated camera's default (10 ms, 1 dB).
_BF = {'exposure_ms': 2.0, 'gain_db': 1.0, 'auto_gain': False}
_BLUE = {'exposure_ms': 88.0, 'gain_db': 3.0, 'auto_gain': False}
_GREEN = {'exposure_ms': 55.0, 'gain_db': 4.0, 'auto_gain': False}


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(
        complete_settings(
            BF=_BF,
            **{LED0: _BLUE, LED1: _GREEN},
            live_folder=str(tmp_path),
            separate_folder_per_channel=False,
        ),
        simulate=True,
    )
    home_sim_scope(built.scope)
    built.show_channel('BF')
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _still(session):
    """The still's channel (from its file name) and its frame's record."""
    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30)
    return still.file.name, still.frame


def _settings_of(record) -> tuple[float, float]:
    return round(record.exposure_ms, 3), round(record.gain_db, 3)


def _stored(layer) -> tuple[float, float]:
    values = {'BF': _BF, LED0: _BLUE, LED1: _GREEN}[layer]
    return values['exposure_ms'], values['gain_db']


def test_a_raw_led_names_the_still_and_the_camera_takes_its_layer(session):
    session.scope.illumination.led_on(LED0, 100.0)

    name, record = _still(session)

    assert f'_{LED0}_' in name
    assert _settings_of(record) == _stored(LED0)
    assert record.held_channel == LED0


def test_a_still_after_a_run_is_one_layers(session, tmp_path):
    """BF shown before a one-step LED 0 scan: the still after it is named for
    the layer shown and taken at that layer's settings, whichever it is."""
    for layer in session.get_layer_configs():
        session.set_channel_acquire(layer, None)
    session.set_channel_acquire(LED0, 'image')
    protocol = session.new_protocol()
    while protocol.num_steps() > 1:
        session.delete_step(protocol, protocol.num_steps() - 1)
    runner = session.create_protocol_runner()
    runner.start_single_scan(
        protocol=protocol,
        sequence_name='scan',
        parent_dir=str(tmp_path / 'runs'),
        enable_image_saving=False,
    ).result(timeout=120)

    name, record = _still(session)

    shown = session.shown_channel.channel
    assert f'_{shown}_' in name
    assert _settings_of(record) == _stored(shown)


def test_a_typed_value_is_the_next_stills(session):
    session.update_settings('BF.exposure_ms', 12.0)

    _name, record = _still(session)

    assert _settings_of(record) == (12.0, _BF['gain_db'])


def test_a_still_pressed_as_the_shown_layer_changes_is_one_layers(session):
    """Pressed from the shown-channel listener: after the show has changed the
    name and before its camera half has run. Taken first on the camera lane,
    the still holds the new layer itself."""
    pressed = []

    def press(layer, _lit):
        pressed.append(session.manual_capture.capture(false_color_on=False))

    session.add_shown_channel_listener(press)
    try:
        session.show_channel(LED0)
    finally:
        session.remove_shown_channel_listener(press)
    (still,) = pressed[0].result(timeout=30)
    record = still.frame

    assert f'_{LED0}_' in still.file.name
    assert _settings_of(record) == _stored(LED0)


def test_a_raw_camera_write_stands_and_the_record_says_no_layer(session):
    session.scope.camera.set_exposure_ms(30.0)

    name, record = _still(session)

    assert '_BF_' in name
    assert _settings_of(record) == (30.0, _BF['gain_db'])
    assert record.held_channel is None


def test_the_saved_file_states_the_layer_the_camera_held(session):
    """The file, not only the frame's record: a reader of the image knows
    whether its settings were a layer's, and whose."""
    from modules.image_utils import read_file_record

    session.scope.illumination.led_on(LED0, 100.0)
    held = session.manual_capture.capture(false_color_on=False).result(timeout=30)[0].file
    session.scope.illumination.leds_off()
    session.scope.camera.set_exposure_ms(30.0)
    raw = session.manual_capture.capture(false_color_on=False).result(timeout=30)[0].file

    held_saved = read_file_record(held)
    raw_saved = read_file_record(raw)
    assert (held_saved.get('camera_held_a_channel'), held_saved.get('camera_held_channel')) == (
        True,
        LED0,
    ), held_saved
    assert (raw_saved.get('camera_held_a_channel'), 'camera_held_channel' in raw_saved) == (
        False,
        False,
    ), raw_saved


def test_a_still_two_leds_lit_is_named_for_the_first_and_its_file_states_both(session):
    from modules.image_utils import read_file_record

    session.scope.illumination.led_on(LED0, 100.0)
    session.scope.illumination.led_on(LED1, 50.0)

    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30)

    saved = read_file_record(still.file)
    assert f'_{LED0}_' in still.file.name, 'the first lit LED in address order names it'
    assert saved.get('illumination_by_channel_ma') == {LED0: 100.0, LED1: 50.0}, saved
    assert saved.get('illumination_ma') == 100.0, 'Illumination is the named channel alone'


def test_a_still_one_led_lit_states_its_current_in_both_fields(session):
    from modules.image_utils import read_file_record

    session.scope.illumination.led_on(LED1, 75.0)

    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30)

    saved = read_file_record(still.file)
    assert saved.get('illumination_by_channel_ma') == {LED1: 75.0}, saved
    assert saved.get('illumination_ma') == 75.0


def test_an_led_left_on_at_0_ma_neither_names_a_still_nor_is_recorded(session):
    """The board takes 0 mA as off, and so does the frame's record: the
    still is named for the LED that lit it, which its file states."""
    from modules.image_utils import read_file_record

    session.scope.illumination.led_on(LED0, 0.0)
    session.scope.illumination.led_on(LED1, 100.0)

    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30)

    saved = read_file_record(still.file)
    assert f'_{LED1}_' in still.file.name, still.file.name
    assert saved.get('channel') == LED1
    assert saved.get('illumination_by_channel_ma') == {LED1: 100.0}, saved


def test_a_composite_states_no_lit_layer_of_its_inputs(session):
    from modules import image_utils
    from modules.image_utils import build_postproc_output_metadata

    session.scope.illumination.led_on(LED0, 100.0)
    session.scope.illumination.led_on(LED1, 50.0)
    (still,) = session.manual_capture.capture(false_color_on=False).result(timeout=30)

    composite = build_postproc_output_metadata(
        [image_utils.read_file_record(still.file)],
        'Composite',
        significant_bits=8,
        adds_frames=False,
    )

    assert 'illumination_by_channel_ma' not in composite
    assert 'illumination_ma' not in composite


def test_a_still_that_would_change_a_recordings_layer_is_refused(session):
    recording = session.manual_recording.start(false_color_on=False)
    try:
        session.scope.illumination.led_on(LED0, 100.0)

        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.manual_capture.capture(false_color_on=False).result(timeout=30)

        assert refused.value.reason == 'exclusive_activity_running'
        assert refused.value.holder == 'recording'
        assert session.scope.camera.get_held_layer() == 'BF'
    finally:
        recording.stop()
        recording.result(timeout=60)


def test_a_still_during_a_recording_on_the_held_layer_is_taken(session):
    recording = session.manual_recording.start(false_color_on=False)
    try:
        name, record = _still(session)
    finally:
        recording.stop()
        recording.result(timeout=60)

    assert '_BF_' in name
    assert _settings_of(record) == _stored('BF')


@pytest.mark.slow
def test_a_light_that_never_settles_fails_the_still_within_one_budget(session, monkeypatch):
    import modules.manual_capture as manual_capture

    # a stand-in by design: an LED toggled faster than a capture takes is a
    # race the simulated board cannot be made to win on every run
    lights = iter([LED0, LED1] * 1000)
    real = manual_capture.common_utils.read_light
    monkeypatch.setattr(
        manual_capture.common_utils,
        'read_light',
        lambda illumination, shown: manual_capture.common_utils.Light(next(lights), {}),
    )
    try:
        with pytest.raises(CaptureFailedError) as failed:
            session.manual_capture.capture(false_color_on=False).result(timeout=60)
    finally:
        monkeypatch.setattr(manual_capture.common_utils, 'read_light', real)

    assert failed.value.reason == 'deadline_expired'


def test_a_running_still_holds_its_layer_when_the_scope_is_taken_meanwhile(session, monkeypatch):
    """A still whose lane task has started is taken when an activity takes the
    scope meanwhile: its hold is part of the still the lane admitted, not a
    new request admitted again against the holder. (A still still queued when
    the scope is taken is refused by the lane, as ``capture`` says.)"""
    controller = session.manual_capture
    taken = []
    reads = []
    real_shown_layer = controller._shown_layer

    def shown_layer_after_the_scope_is_taken():
        # a stand-in by design: the scope is taken at the one point inside the
        # running still where the timing can be made exact. The first read
        # sizes the caller's bound at the call; the second is the still's own,
        # on the lane.
        reads.append(1)
        if len(reads) == 2:
            taken.append(session.activity_claim.try_claim('diagnostic'))
        return real_shown_layer()

    monkeypatch.setattr(controller, '_shown_layer', shown_layer_after_the_scope_is_taken)
    session.scope.illumination.led_on(LED0, 100.0)
    try:
        (still,) = controller.capture(false_color_on=False).result(timeout=30)
    finally:
        taken[0].release()
    record = still.frame

    assert taken[0] is not None
    assert f'_{LED0}_' in still.file.name
    assert _settings_of(record) == _stored(LED0)

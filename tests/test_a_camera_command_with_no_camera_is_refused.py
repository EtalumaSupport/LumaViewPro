# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera command on a scope with no camera is refused, and names the camera.

With no camera every camera command returned the None or False a write
that worked returns -- an L2 ``set_gain_db`` on a camera-less scope
answered as if it had set the gain -- and six of them posted their own
"Camera not connected" warning from inside the API, so bring-up reported
the missing camera three times. The value reads answered stand-ins (-1 dB,
0 ms, 0 px) a caller could take for the camera's. After ``disconnect()``
the cache kept the departed camera's values, and a frame edit at the old
size was compared against it, skipped, and stored with no camera behind it.

Every camera command now asks for the camera on the camera lane, after the
lane admits it: with none it is refused ``not_connected``, naming the
camera, unless it is an off whose end state already holds. The value reads
answer None, ``disconnect()`` returns the cache to the no-camera state,
bring-up reports the absence once, Go To Step is refused before anything
moves, and the GUI sends no camera apply from a layer.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
from modules.exceptions import (
    HardwareCommandRefusedError,
    MissingPart,
    ProtocolRunRefusedError,
    RecordingRefusedError,
)
from modules.lumascope_api import _lumascope
from tests.test_a_command_for_absent_motion_hardware_is_refused import (
    _protocol,
    _refused,
    _wait,
    make_session,  # noqa: F401 -- the fixture this file's tests take
)
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

MODELS = ('LS850', 'LS850T', 'LS820', 'Lumi', 'LS620', 'LS560')
FX2_MODELS = ('LS620', 'LS560')

_AUTO_GAIN = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 100.0,
    'min_exposure_ms': 0.1,
}

# Every camera command an outside caller sends that needs a camera, by name.
REFUSED = {
    'set_gain_db': lambda im: im.set_gain_db(5.0),
    'set_exposure_ms': lambda im: im.set_exposure_ms(20.0),
    'set_black_level': lambda im: im.set_black_level(10.0),
    'set_auto_gain_on': lambda im: im.set_auto_gain(True, _AUTO_GAIN),
    'set_auto_exposure_time_on': lambda im: im.set_auto_exposure_time(True),
    'update_auto_gain_target_brightness': lambda im: im.update_auto_gain_target_brightness(0.5),
    'auto_gain_once': lambda im: im.auto_gain_once(True, 0.5, 0.0, 20.0),
    'start_streaming': lambda im: im.start_streaming(),
    'set_frame_size': lambda im: im.set_frame_size(1024, 1024),
    'set_binning_size': lambda im: im.set_binning_size(1),
    'set_pixel_format': lambda im: im.set_pixel_format('Mono8'),
    'set_conversion_gain_mode': lambda im: im.set_conversion_gain_mode('Low'),
    'set_line_noise_reduction': lambda im: im.set_line_noise_reduction(False),
    'apply_layer_camera_settings': lambda im: im.apply_layer_camera_settings(5.0, 20.0, layer='BF'),
    'capture_and_wait': lambda im: im.capture_and_wait(),
}

# The offs: satisfied with no camera, answering what each answers with one.
SATISFIED = {
    'stop_streaming': lambda im: im.stop_streaming(),
    'set_auto_gain_off': lambda im: im.set_auto_gain(False, _AUTO_GAIN),
    'set_auto_exposure_time_off': lambda im: im.set_auto_exposure_time(False),
    'lock_auto_gain': lambda im: im.lock_auto_gain(),
}

DIAGNOSTIC_RUNNERS = {
    'run_grab_lifecycle_benchmark': lambda d: d.run_grab_lifecycle_benchmark(2),
    'run_pylon_diagnostic_probe': lambda d: d.run_pylon_diagnostic_probe(0.1),
}


def _without_a_camera(monkeypatch, model: str) -> None:
    """Bring the next scope up as one whose camera never came up: its factory raises."""
    if model in FX2_MODELS:
        import drivers.fx2driver as fx2driver

        def _no_fx2_camera(**kwargs):
            raise RuntimeError('no FX2 camera on the bus')

        monkeypatch.setattr(fx2driver, 'FX2Camera', _no_fx2_camera)
    else:

        def _no_camera(*args, **kwargs):
            raise RuntimeError('no camera found')

        monkeypatch.setattr(_lumascope.camera_registry, 'create', _no_camera)


def _camera_less(make_session, monkeypatch, model: str, **kwargs):
    _without_a_camera(monkeypatch, model)
    session = make_session(model, **kwargs)
    assert not session.scope.camera_connected
    return session


def _camera_posts(posts) -> list:
    return [n for n in posts if n.category == 'Camera' and n.severity.value >= logging.WARNING]


def _assert_names_the_camera(refusal: HardwareCommandRefusedError) -> None:
    assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.CAMERA)
    assert str(refusal) == 'The camera is not connected.'


def _camera_store(session) -> dict:
    s = session.settings
    return {
        'frame': dict(s['frame']),
        'binning': dict(s['binning']),
        'BF': {k: s['BF'][k] for k in ('gain_db', 'exposure_ms', 'auto_gain')},
    }


# --- Camera present: nothing changes ---------------------------------------------


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_its_camera_every_command_and_read_answers_as_before(make_session, model):
    scope = make_session(model).scope
    im = scope.imaging

    assert im.set_gain_db(5.0) == pytest.approx(5.0, abs=0.5)
    assert im.set_exposure_ms(20.0) == pytest.approx(20.0, rel=0.1)
    assert im.get_gain_db() == pytest.approx(5.0, abs=0.5)
    assert im.get_exposure_ms() == pytest.approx(20.0, rel=0.1)
    assert im.get_width() > 0 and im.get_height() > 0
    assert im.capture_and_wait() is not None
    im.stop_streaming()
    im.start_streaming()
    im.restore_camera_state(im.save_camera_state('present'))


# --- No camera -------------------------------------------------------------------


@pytest.mark.parametrize('model', MODELS)
def test_a_scope_with_no_camera_comes_up_and_says_so_once(
    make_session, monkeypatch, centre_posts, model
):
    _camera_less(make_session, monkeypatch, model)

    reports = _camera_posts(centre_posts)
    assert len(reports) == 1, [(n.title, n.message) for n in reports]


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
@pytest.mark.parametrize('command', list(REFUSED.values()), ids=list(REFUSED))
def test_with_no_camera_a_command_is_refused_naming_it(
    make_session, monkeypatch, centre_posts, model, command
):
    session = _camera_less(make_session, monkeypatch, model)
    before = _camera_store(session)
    centre_posts.clear()

    _assert_names_the_camera(_refused(lambda: command(session.scope.imaging)))

    assert _camera_store(session) == before
    assert _camera_posts(centre_posts) == []


@pytest.mark.parametrize('runner', list(DIAGNOSTIC_RUNNERS.values()), ids=list(DIAGNOSTIC_RUNNERS))
def test_with_no_camera_a_diagnostic_runner_is_refused_naming_it(make_session, monkeypatch, runner):
    session = _camera_less(make_session, monkeypatch, 'LS850')

    _assert_names_the_camera(_refused(lambda: runner(session.scope.diagnostics)))


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
@pytest.mark.parametrize('off', list(SATISFIED.values()), ids=list(SATISFIED))
def test_with_no_camera_an_off_is_satisfied_with_its_own_answer(
    make_session, monkeypatch, centre_posts, model, off
):
    with_camera = off(make_session(model).scope.imaging)
    session = _camera_less(make_session, monkeypatch, model)
    centre_posts.clear()

    assert off(session.scope.imaging) == with_camera
    assert _camera_posts(centre_posts) == []


def test_with_no_camera_a_restore_of_a_cameraless_snapshot_is_satisfied_and_any_other_refused(
    make_session, monkeypatch
):
    im = _camera_less(make_session, monkeypatch, 'LS850').scope.imaging

    assert im.restore_camera_state(im.save_camera_state('cameraless')) is None
    _assert_names_the_camera(
        _refused(
            lambda: im.restore_camera_state(
                {
                    'tag': 'from a camera',
                    'gain_db': 5.0,
                    'exposure_ms': 20.0,
                    'frame_size': {'width': 1024, 'height': 1024},
                    'auto_gain_arm': None,
                }
            )
        )
    )


def test_with_no_camera_a_frame_listener_is_refused(make_session, monkeypatch):
    im = _camera_less(make_session, monkeypatch, 'LS850').scope.imaging

    def listener(image, timestamp, chunks):
        pass

    _assert_names_the_camera(_refused(lambda: im.add_frame_listener(listener)))
    im.remove_frame_listener(listener)


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_the_value_reads_answer_none(make_session, monkeypatch, model):
    scope = _camera_less(make_session, monkeypatch, model).scope
    im = scope.imaging

    assert (im.get_gain_db(), im.get_exposure_ms(), im.get_width(), im.get_height()) == (
        None,
        None,
        None,
        None,
    )
    assert scope.diagnostics.get_camera_temperatures_degc() is None
    assert scope.diagnostics.get_camera_link_info() is None


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_go_to_step_is_refused_before_anything_moves(
    make_session, monkeypatch, model
):
    session = _camera_less(make_session, monkeypatch, model)
    scope = session.scope
    position = {axis: scope.motion.get_current_position(axis) for axis in scope.capabilities.axes}
    before = _camera_store(session)

    _assert_names_the_camera(_refused(lambda: session.start_go_to_step(_protocol(), 0)))

    assert {
        axis: scope.motion.get_current_position(axis) for axis in scope.capabilities.axes
    } == position
    assert _camera_store(session) == before


def test_with_no_camera_the_session_binning_is_refused_naming_it(make_session, monkeypatch):
    session = _camera_less(make_session, monkeypatch, 'LS850')
    before = _camera_store(session)

    _assert_names_the_camera(_refused(lambda: session.set_binning_size(1)))

    assert _camera_store(session) == before


@pytest.mark.parametrize('member', ('set_high_conversion_gain', 'set_line_noise_reduction'))
def test_with_no_camera_a_session_camera_toggle_is_refused_naming_it(
    make_session, monkeypatch, member
):
    session = _camera_less(make_session, monkeypatch, 'LS850')

    _assert_names_the_camera(_refused(lambda: getattr(session, member)(False)))


def test_with_no_camera_a_layer_apply_from_the_gui_sends_no_camera_command(monkeypatch):
    import ui.layer_control as layer_control

    submits = []
    monkeypatch.setattr(
        layer_control,
        'submit_reported',
        lambda call, redraw, label, **kwargs: submits.append(label),
    )
    context = MagicMock()
    context.settings = {'Green': {'exposure_ms': 2.0, 'gain_db': 1.0}, 'protocol_led_on': True}
    context.session.run_lockout = False
    context.scope.camera_connected = False
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    widget = MagicMock()
    widget.layer = 'Green'
    widget._initializing = False
    widget.effective_auto_gain.return_value = False

    layer_control.LayerControl.apply_settings(widget, update_led=False)

    assert 'CAMERA_SETTINGS_Green' not in submits
    context.session.apply_layer_camera.assert_not_called()


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_a_run_an_autofocus_and_a_recording_are_refused_at_the_start(
    make_session, monkeypatch, tmp_path, model
):
    session = _camera_less(make_session, monkeypatch, model)
    runner = session.create_protocol_runner()

    with pytest.raises(ProtocolRunRefusedError) as run:
        runner.run_single_scan(
            protocol=_protocol(),
            sequence_name='s',
            parent_dir=str(tmp_path),
        )
    assert run.value.reason == 'hardware_disconnected'
    with pytest.raises(ProtocolRunRefusedError):
        runner.run_autofocus(layer='BF')
    with pytest.raises(RecordingRefusedError):
        session.manual_recording.start(layer='BF')


# --- After disconnect() ----------------------------------------------------------


def _disconnected(make_session, model: str):
    session = make_session(model)
    session.scope.disconnect()
    return session


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_a_command_says_the_scope_is_disconnected(make_session, model):
    im = _disconnected(make_session, model).scope.imaging

    for command in (*REFUSED.values(), *SATISFIED.values()):
        assert _refused(lambda command=command: command(im)).reason == 'scope_disconnected'
    assert _refused(lambda: im.set_gain_db(1000.0)).reason == 'scope_disconnected'
    assert (
        _refused(lambda: im.add_frame_listener(lambda *args: None)).reason == 'scope_disconnected'
    )


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_no_read_describes_the_camera_that_left(make_session, model):
    scope = _disconnected(make_session, model).scope
    im = scope.imaging

    assert (im.get_gain_db(), im.get_exposure_ms(), im.get_width(), im.get_height()) == (
        None,
        None,
        None,
        None,
    )
    assert im.frame_size_cached == {'width': 0, 'height': 0}
    assert im.gain_db_cached == -1.0
    assert scope.diagnostics.get_camera_temperatures_degc() is None
    assert scope.diagnostics.get_camera_link_info() is None


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_a_frame_at_the_departed_size_is_refused_and_not_stored(
    make_session, model
):
    session = _disconnected(make_session, model)
    frame = dict(session.settings['frame'])
    before = _camera_store(session)

    refusal = _refused(lambda: session.set_frame_size(int(frame['width']), int(frame['height'])))

    assert refusal.reason == 'scope_disconnected'
    assert _camera_store(session) == before


# --- A camera removed mid-run ----------------------------------------------------


def test_a_camera_removed_mid_run_ends_the_run_hardware_disconnected(
    make_session, monkeypatch, tmp_path
):
    session = make_session('LS850')
    scope = session.scope
    driver = scope._camera_driver
    real_capture = scope.imaging._capture_and_wait_impl

    def removed_after_this(*args, **kwargs):
        image = real_capture(*args, **kwargs)
        driver._mark_disconnected()
        return image

    monkeypatch.setattr(scope.imaging, '_capture_and_wait_impl', removed_after_this)
    first = {**_make_single_step_protocol().step(idx=0), 'Z': 3000.0}
    second = {**first, 'Name': 'A1_second', 'Label': 'A1_second', 'X': 12.0}
    runner = session.create_protocol_runner()

    run = runner.run_single_scan(
        protocol=_build_real_protocol([first, second]),
        sequence_name='removed',
        parent_dir=str(tmp_path),
    )

    assert _wait(run) == ('failed', 'hardware_disconnected')


@pytest.mark.parametrize('connected', (True, False))
def test_the_gui_camera_controls_follow_whether_a_camera_is_connected(monkeypatch, connected):
    from types import SimpleNamespace

    from modules import common_utils
    from ui.image_settings import ImageSettings

    layers = {layer: SimpleNamespace(camera_connected=None) for layer in common_utils.get_layers()}
    microscope = SimpleNamespace(camera_connected=None)
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=SimpleNamespace(camera_connected=connected),
            motion_settings=SimpleNamespace(ids={'microscope_settings_id': microscope}),
        ),
    )

    ImageSettings.set_camera_controls_support(
        SimpleNamespace(layer_lookup=lambda layer: layers[layer])
    )

    assert {layer.camera_connected for layer in layers.values()} == {connected}
    assert microscope.camera_connected is connected


def test_a_grab_is_sized_by_its_floors_when_the_camera_never_reported_an_exposure():
    from tests.test_camera_getter_sentinel_containment import (
        _build_imaging,
        all_reads_fail_driver,
        steady_good_driver,
    )

    assert _build_imaging(all_reads_fail_driver())._grab_sizing_exposure_s() == 0.0
    assert _build_imaging(steady_good_driver())._grab_sizing_exposure_s() == pytest.approx(0.05)

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

import pytest

import modules.app_context as _app_ctx
from modules.exceptions import (
    HardwareCommandRefusedError,
    MissingPart,
    ProtocolRunRefusedError,
    RecordingRefusedError,
)
from modules.lumascope_api import _lumascope
from modules.scope_session import ScopeSession
from modules.lumascope_api.camera import CameraAutoMode
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

# Every camera command an outside caller sends that needs a camera, by name,
# each run on the scope.
REFUSED = {
    'set_gain_db': lambda s: s.camera.set_gain_db(5.0),
    'set_exposure_ms': lambda s: s.camera.set_exposure_ms(20.0),
    'set_black_level': lambda s: s.camera.set_black_level(10.0),
    'set_auto_gain_on': lambda s: s.camera.set_auto_gain(True, _AUTO_GAIN),
    'set_auto_exposure_on': lambda s: s.camera.set_auto_exposure(CameraAutoMode.CONTINUOUS),
    'update_auto_gain_target_brightness': lambda s: s.camera.update_auto_gain_target_brightness(
        0.5
    ),
    'auto_gain_once': lambda s: s.camera.auto_gain_once(True, 0.5, 0.0, 20.0),
    'enable_streaming': lambda s: s.imaging.enable_streaming(),
    'set_frame_size': lambda s: s.camera.set_frame_size(1024, 1024),
    'set_binning': lambda s: s.camera.set_binning(1),
    'set_high_conversion_gain': lambda s: s.camera.set_high_conversion_gain(False),
    'set_line_noise_reduction': lambda s: s.camera.set_line_noise_reduction(False),
    'apply': lambda s: s.camera.apply(5.0, 20.0, layer='BF'),
    'capture_and_wait': lambda s: s.imaging.capture_and_wait(),
}

# The image mode's apply: with no camera it has no format to apply, and the
# mode is saved for bring-up to apply, so it is not refused then; after
# disconnect, and with a camera that leaves during the write, it is.
IMAGE_MODE = {'set_image_mode': lambda s: s.camera.set_image_mode('8bit')}

# The offs: satisfied with no camera, answering what each answers with one.
SATISFIED = {
    'disable_streaming': lambda s: s.imaging.disable_streaming(),
    'set_auto_gain_off': lambda s: s.camera.set_auto_gain(False, _AUTO_GAIN),
    'set_auto_exposure_off': lambda s: s.camera.set_auto_exposure(CameraAutoMode.OFF),
    'lock_auto_gain': lambda s: s.camera.lock_auto_gain(),
}

DIAGNOSTIC_RUNNERS = {
    'run_grab_lifecycle_benchmark': lambda d: d.run_grab_lifecycle_benchmark(2),
    'run_pylon_diagnostic_probe': lambda d: d.run_pylon_diagnostic_probe(0.1),
    'run_camera_bandwidth_test': lambda d: d.run_camera_bandwidth_test(2),
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
    camera, im = scope.camera, scope.imaging

    assert camera.set_gain_db(5.0) == pytest.approx(5.0, abs=0.5)
    assert camera.set_exposure_ms(20.0) == pytest.approx(20.0, rel=0.1)
    assert camera.read_gain_db() == pytest.approx(5.0, abs=0.5)
    assert camera.read_exposure_ms() == pytest.approx(20.0, rel=0.1)
    size = camera.read_frame_size()
    assert size.width > 0 and size.height > 0
    assert im.capture_and_wait() is not None
    im.disable_streaming()
    im.enable_streaming()
    camera.restore_camera_state(camera.save_camera_state('present'))


# --- No camera -------------------------------------------------------------------


@pytest.mark.parametrize('model', MODELS)
def test_a_scope_with_no_camera_comes_up_and_says_so_once(
    make_session, monkeypatch, centre_posts, model
):
    _camera_less(make_session, monkeypatch, model)

    reports = _camera_posts(centre_posts)
    assert len(reports) == 1, [(n.title, n.message) for n in reports]


def _not_named_as_the_camera(commands: dict, target) -> dict:
    """Each command of ``commands`` run on ``target`` that is not refused
    naming the camera, with what it answered instead."""
    wrong = {}
    for name, command in commands.items():
        try:
            answer = command(target)
        except HardwareCommandRefusedError as refusal:
            words = (refusal.reason, refusal.missing, str(refusal))
            if words != ('not_connected', MissingPart.CAMERA, 'The camera is not connected.'):
                wrong[name] = words
        else:
            wrong[name] = f'answered {answer!r}'
    return wrong


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_every_command_is_refused_naming_it(
    make_session, monkeypatch, centre_posts, model
):
    session = _camera_less(make_session, monkeypatch, model)
    before = _camera_store(session)
    centre_posts.clear()

    assert _not_named_as_the_camera(REFUSED, session.scope) == {}
    assert _camera_store(session) == before
    assert _camera_posts(centre_posts) == []


def test_with_no_camera_every_diagnostic_runner_is_refused_naming_it(make_session, monkeypatch):
    session = _camera_less(make_session, monkeypatch, 'LS850')

    assert _not_named_as_the_camera(DIAGNOSTIC_RUNNERS, session.scope.diagnostics) == {}


def test_with_no_camera_a_binning_preview_is_refused_naming_it(make_session, monkeypatch):
    # The preview asks for the camera before the offered sizes, as the apply
    # does: a scope with no camera offers none, so "does not support" would
    # name the wrong cause, and the frame it answered before was a guess.
    session = _camera_less(make_session, monkeypatch, 'LS850')

    assert (
        _not_named_as_the_camera(
            {'frame_at_binning': lambda s: s.camera.frame_at_binning(2)}, session
        )
        == {}
    )


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_every_off_is_satisfied_with_its_own_answer(
    make_session, monkeypatch, centre_posts, model
):
    scope = make_session(model).scope
    with_camera = {name: off(scope) for name, off in SATISFIED.items()}
    session = _camera_less(make_session, monkeypatch, model)
    centre_posts.clear()

    assert {name: off(session.scope) for name, off in SATISFIED.items()} == with_camera
    assert _camera_posts(centre_posts) == []


def test_with_no_camera_a_restore_of_a_cameraless_snapshot_is_satisfied_and_any_other_refused(
    make_session, monkeypatch
):
    camera = _camera_less(make_session, monkeypatch, 'LS850').scope.camera

    assert camera.restore_camera_state(camera.save_camera_state('cameraless')) is None
    _assert_names_the_camera(
        _refused(
            lambda: camera.restore_camera_state(
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

    def listener(frame):
        pass

    _assert_names_the_camera(_refused(lambda: im.add_frame_listener(listener)))
    im.remove_frame_listener(listener)


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_the_value_reads_answer_none(make_session, monkeypatch, model):
    scope = _camera_less(make_session, monkeypatch, model).scope
    camera = scope.camera

    assert (
        camera.read_gain_db(),
        camera.read_exposure_ms(),
        camera.read_frame_size(),
    ) == (
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

    _assert_names_the_camera(_refused(lambda: session.camera.set_binning(1)))

    assert _camera_store(session) == before


@pytest.mark.parametrize('member', ('set_high_conversion_gain', 'set_line_noise_reduction'))
def test_with_no_camera_a_session_camera_toggle_is_refused_naming_it(
    make_session, monkeypatch, member
):
    session = _camera_less(make_session, monkeypatch, 'LS850')

    _assert_names_the_camera(_refused(lambda: getattr(session.camera, member)(False)))


def test_with_no_camera_showing_a_channel_is_refused_naming_it(make_session, monkeypatch):
    """The GUI's drawer and settings apply ask show_channel, which shows the
    LEDs and then refuses the camera's half, naming the camera."""
    session = _camera_less(make_session, monkeypatch, 'LS850')
    before = _camera_store(session)

    _assert_names_the_camera(_refused(lambda: session.show_channel('Green')))

    assert _camera_store(session) == before


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_with_no_camera_a_run_an_autofocus_and_a_recording_are_refused_at_the_start(
    make_session, monkeypatch, tmp_path, model
):
    session = _camera_less(make_session, monkeypatch, model)
    runner = session.create_protocol_runner()

    with pytest.raises(ProtocolRunRefusedError) as run:
        runner.start_single_scan(
            protocol=_protocol(),
            sequence_name='s',
            parent_dir=str(tmp_path),
        )
    assert run.value.reason == 'hardware_disconnected'
    with pytest.raises(ProtocolRunRefusedError):
        runner.start_autofocus(layer='BF')
    with pytest.raises(RecordingRefusedError):
        session.manual_recording.start()


# --- After disconnect() ----------------------------------------------------------


def _disconnected(make_session, model: str):
    session = make_session(model)
    session.scope.disconnect()
    return session


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_a_command_says_the_scope_is_disconnected(make_session, model):
    scope = _disconnected(make_session, model).scope

    for command in (*REFUSED.values(), *IMAGE_MODE.values(), *SATISFIED.values()):
        assert _refused(lambda command=command: command(scope)).reason == 'scope_disconnected'
    assert _refused(lambda: scope.camera.set_gain_db(1000.0)).reason == 'scope_disconnected'
    assert (
        _refused(lambda: scope.imaging.add_frame_listener(lambda *args: None)).reason
        == 'scope_disconnected'
    )


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_no_read_describes_the_camera_that_left(make_session, model):
    scope = _disconnected(make_session, model).scope
    camera = scope.camera

    assert (
        camera.read_gain_db(),
        camera.read_exposure_ms(),
        camera.read_frame_size(),
    ) == (
        None,
        None,
        None,
    )
    assert camera.frame_size is None
    assert camera.gain_db is None
    assert scope.diagnostics.get_camera_temperatures_degc() is None
    assert scope.diagnostics.get_camera_link_info() is None


@pytest.mark.parametrize('model', ('LS850', 'LS620'))
def test_after_disconnect_a_frame_at_the_departed_size_is_refused_and_not_stored(
    make_session, model
):
    session = _disconnected(make_session, model)
    frame = dict(session.settings['frame'])
    before = _camera_store(session)

    refusal = _refused(
        lambda: session.camera.set_frame_size(int(frame['width']), int(frame['height']))
    )

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

    run = runner.start_single_scan(
        protocol=_build_real_protocol([first, second]),
        sequence_name='removed',
        parent_dir=str(tmp_path),
    )

    assert _wait(run) == ('failed', 'hardware_disconnected')


# --- A camera removed during its write -------------------------------------------

# The commands of REFUSED and IMAGE_MODE the simulator can take the camera
# away during, by the node each one writes.
REMOVED_DURING = {
    'gain': 'set_gain_db',
    'exposure': 'set_exposure_ms',
    'black_level': 'set_black_level',
    'auto_exposure': 'set_auto_exposure_on',
    'frame_size': 'set_frame_size',
    'binning': 'set_binning',
    'pixel_format': 'set_image_mode',
    'auto_gain': 'set_auto_gain_on',
    'auto_gain_once': 'auto_gain_once',
    'auto_gain_target_brightness': 'update_auto_gain_target_brightness',
}


def test_a_camera_removed_during_the_write_is_refused_as_an_absent_one(make_session, centre_posts):
    """A camera that leaves the bus during the write is refused as one gone
    before it: naming the camera, nothing stored, nothing posted by the API.
    Not a rejection of the value, and not a None a caller reads as applied."""
    answered = {}
    for node, name in REMOVED_DURING.items():
        session = make_session('LS850')
        before = _camera_store(session)
        session.scope.simulation.refuse_camera_write(node, 'removed')
        centre_posts.clear()
        try:
            {**REFUSED, **IMAGE_MODE}[name](session.scope)
        except HardwareCommandRefusedError as refusal:
            if (refusal.reason, refusal.missing) != ('not_connected', MissingPart.CAMERA):
                answered[name] = f'refused {refusal.reason} {refusal.missing}'
        except Exception as other:  # the defect under test: any other answer
            answered[name] = repr(other)
        else:
            answered[name] = 'no raise'
        if _camera_store(session) != before:
            answered[f'{name} store'] = 'changed'
        if _camera_posts(centre_posts):
            answered[f'{name} posts'] = [n.title for n in _camera_posts(centre_posts)]
    assert not answered, answered


def test_an_off_is_satisfied_when_the_camera_leaves_during_it(make_session):
    answers = {}
    for node, name in (
        ('auto_gain', 'set_auto_gain_off'),
        ('auto_exposure', 'set_auto_exposure_off'),
    ):
        session = make_session('LS850')
        session.scope.simulation.refuse_camera_write(node, 'removed')
        answers[name] = SATISFIED[name](session.scope)

    assert answers == {'set_auto_gain_off': True, 'set_auto_exposure_off': True}


def test_a_write_the_camera_left_during_records_nothing(make_session):
    session = make_session('LS850')
    camera = session.scope.camera
    camera.set_gain_db(3.0)
    before = (camera.gain_db, camera.frame_validity.target('gain'))
    session.scope.simulation.refuse_camera_write('gain', 'removed')

    _refused(lambda: camera.set_gain_db(9.0))

    assert (camera.gain_db, camera.frame_validity.target('gain')) == before


def test_a_write_that_raises_as_the_camera_leaves_is_refused_as_an_absent_one(
    make_session, monkeypatch
):
    """The driver raised and the camera was declared removed: the absence is
    the answer, chained to the driver's error, not a rejection of the value."""
    session = make_session('LS850')
    driver = session.scope._camera_driver

    # A stand-in by design: the simulator's 'removed' write answers False; a
    # write that raises as its camera leaves is a fault it does not offer.
    def leaves_and_raises(*args, **kwargs):
        session.scope.simulation.remove_camera()
        raise RuntimeError('simulated node write lost its device')

    monkeypatch.setattr(driver, 'gain', leaves_and_raises)

    refusal = _refused(lambda: session.scope.camera.set_gain_db(9.0))

    assert (refusal.reason, refusal.missing) == ('not_connected', MissingPart.CAMERA)
    assert isinstance(refusal.__cause__, RuntimeError)


def test_turning_auto_gain_off_as_the_camera_leaves_stores_it_off(make_session):
    """The lock's disarm finds the camera gone: no arm can stand, so the lock
    answers as with no camera, and the layer stores auto-gain off."""
    session = make_session('LS850')
    session.set_layer_auto_gain('BF', True)
    session.scope.camera.set_auto_gain(True, _AUTO_GAIN)
    session.scope.simulation.refuse_camera_write('auto_gain', 'removed')

    lock = session.set_layer_auto_gain('BF', False)

    assert lock.state is None
    assert session.settings['BF']['auto_gain'] is False


def test_a_camera_removed_as_the_run_takes_the_auto_gain_arm_ends_the_run_hardware_disconnected(
    make_session, tmp_path
):
    """The run's start turns a live-view arm off; the camera leaving then is
    the run's disconnect, not a crash of the run loop."""
    session = make_session('LS850')
    session.scope.camera.set_auto_gain(True, _AUTO_GAIN)
    session.scope.simulation.refuse_camera_write('auto_gain', 'removed')
    step = {**_make_single_step_protocol().step(idx=0), 'Z': 3000.0}

    run = session.create_protocol_runner().start_single_scan(
        protocol=_build_real_protocol([step]),
        sequence_name='removed',
        parent_dir=str(tmp_path),
    )

    assert _wait(run) == ('failed', 'hardware_disconnected')


def test_a_camera_removed_as_live_view_re_arms_auto_gain_keeps_the_frame(
    make_session, monkeypatch, centre_posts
):
    """The frame was grabbed before the camera left; the re-arm after it is
    the camera's absence, reported once, and the frame is still returned."""
    session = make_session('LS850')
    im = session.scope.imaging
    driver = session.scope._camera_driver
    session.scope.camera.set_auto_gain(True, _AUTO_GAIN)
    real_auto_gain = driver.auto_gain

    # A stand-in by design: the simulator refuses a node's next write, and the
    # re-arm is the capture's second auto-gain write (the lock's disarm is the
    # first).
    def leaves_on_the_re_arm(state=True, **kwargs):
        if state:
            session.scope.simulation.remove_camera()
            return False
        return real_auto_gain(state, **kwargs)

    monkeypatch.setattr(driver, 'auto_gain', leaves_on_the_re_arm)
    centre_posts.clear()

    image = im.capture_and_wait()

    assert image is not None
    assert [n.message for n in _camera_posts(centre_posts)] == ['The camera is not connected.']


def test_a_camera_removed_during_bring_up_is_reported_once_and_the_scope_comes_up(
    make_session, monkeypatch, centre_posts
):
    """Bring-up's first camera write finds the camera gone: the camera's
    settings stop there, reported once, and the session comes up."""

    configure = ScopeSession._configure_scope

    # A stand-in by design: the simulator's faults are injected on a session
    # that is already up; this arms one on the scope as bring-up configures it.
    def removed_at_bring_up(session, replaced):
        session.scope.simulation.refuse_camera_write('binning', 'removed')
        return configure(session, replaced)

    monkeypatch.setattr(ScopeSession, '_configure_scope', removed_at_bring_up)

    session = make_session('LS850')

    assert session.scope.camera.camera_removed
    assert [n.message for n in _camera_posts(centre_posts)] == ['The camera is not connected.']


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
        _build_camera,
        all_reads_fail_driver,
        steady_good_driver,
    )

    assert _build_camera(all_reads_fail_driver())._grab_sizing_exposure_s() == 0.0
    assert _build_camera(steady_good_driver())._grab_sizing_exposure_s() == pytest.approx(0.05)

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A held camera read answers what the camera last reported or took, and None whenever that is unknown.

``scope.camera.gain_db`` and its siblings read no camera. Each answers the
value the camera last reported (a fresh read, bring-up's populate) or took
(a write the camera accepted), and None whenever that is not known: before
the camera's first report, with no camera, and after an auto-gain cycle
whose read-back failed. No held read answers a stand-in such as -1, 0 or
1x. The auto modes answer a ``CameraAutoMode``, a one-shot ``ONCE`` until a
fresh read finds it ended. High conversion gain and line-noise reduction
have no driver read, so theirs is the write the camera accepted.

Run on the simulated LS850: a Basler body with auto modes and no high
conversion gain or line-noise reduction.
"""

from __future__ import annotations

import threading

import pytest

from modules.exceptions import ArgumentRefusedError, CameraSettingRejected
from modules.lumascope_api.camera import CameraAPI, CameraAutoMode
from modules.scope_capabilities import FrameSize
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# Every held read that describes the camera; image_mode is the saved one.
_HELD = (
    'gain_db',
    'exposure_ms',
    'black_level',
    'binning',
    'frame_size',
    'pixel_format',
    'auto_gain',
    'auto_exposure',
    'high_conversion_gain',
    'line_noise_reduction',
)

_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 24.0}


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def test_every_held_read_is_a_published_property():
    unpublished = [
        name
        for name in (*_HELD, 'image_mode', 'active')
        if not isinstance(vars(CameraAPI).get(name), property)
    ]
    assert unpublished == []


def test_with_no_camera_every_held_read_is_none(diagnostic_scope):
    held = {name: getattr(diagnostic_scope.camera, name) for name in _HELD}
    assert held == dict.fromkeys(_HELD)
    assert diagnostic_scope.camera.active is False


def test_after_bring_up_every_held_read_is_known(session):
    camera = session.camera
    unknown = [name for name in _HELD if getattr(camera, name) is None]
    assert unknown == [], f'{unknown} unknown after bring-up'
    assert isinstance(camera.frame_size, FrameSize)
    # The simulated LS850 has neither feature, so it is at its standard setting.
    assert (camera.high_conversion_gain, camera.line_noise_reduction) == (False, False)
    assert camera.image_mode == session.settings['image_mode']


def test_a_write_the_camera_took_is_held(session):
    camera = session.camera
    camera.set_gain_db(3.0)
    camera.set_exposure_ms(12.0)
    camera.set_black_level(4.0)
    assert (camera.gain_db, camera.exposure_ms, camera.black_level) == (3.0, 12.0, 4.0)


def test_an_auto_gain_arm_holds_auto_gain_and_auto_exposure_and_its_lock_holds_both_off(session):
    camera = session.camera
    camera.set_auto_gain(True, _AUTO_GAIN)
    assert (camera.auto_gain, camera.auto_exposure) == (
        CameraAutoMode.CONTINUOUS,
        CameraAutoMode.CONTINUOUS,
    )
    camera.lock_auto_gain()
    assert (camera.auto_gain, camera.auto_exposure) == (CameraAutoMode.OFF, CameraAutoMode.OFF)


def test_a_one_shot_is_held_running_until_a_read_finds_it_ended(session):
    camera = session.camera
    camera.auto_gain_once(state=True, target_brightness=0.5, min_gain_db=0.0, max_gain_db=24.0)
    assert (camera.auto_gain, camera.auto_exposure) == (CameraAutoMode.ONCE, CameraAutoMode.ONCE)
    # The simulated one-shot converges at once, so the camera already reads Off.
    assert (camera.read_auto_gain(), camera.read_auto_exposure()) == (
        CameraAutoMode.OFF,
        CameraAutoMode.OFF,
    )
    assert (camera.auto_gain, camera.auto_exposure) == (CameraAutoMode.OFF, CameraAutoMode.OFF)


def test_auto_exposure_is_set_by_its_mode_and_a_one_shot_is_refused(session):
    camera = session.camera
    camera.set_auto_exposure(CameraAutoMode.CONTINUOUS)
    assert camera.read_auto_exposure() is CameraAutoMode.CONTINUOUS

    with pytest.raises(ArgumentRefusedError) as refused:
        camera.set_auto_exposure(CameraAutoMode.ONCE)

    assert refused.value.reason == 'auto_mode_unknown'
    assert camera.read_auto_exposure() is CameraAutoMode.CONTINUOUS


def test_a_failed_read_back_after_auto_gain_leaves_the_gain_unknown_and_the_camera_usable(session):
    camera = session.camera
    camera.set_auto_gain(True, _AUTO_GAIN)
    session.scope.simulation.refuse_camera_read('gain', 'raise')
    camera.set_auto_gain(False, _AUTO_GAIN)
    assert camera.gain_db is None

    assert camera.set_gain_db(3.0) == 3.0
    assert camera.gain_db == 3.0
    assert session.scope.imaging.capture_and_wait(timeout_s=5.0) is not None


def test_a_fresh_read_is_held(session):
    camera = session.camera
    camera.set_gain_db(5.0)
    assert camera.read_gain_db() == camera.gain_db == 5.0


# Each write the camera holds a value from: the simulator node it writes, the
# write, and the held reads it records.
_HELD_WRITES = (
    ('gain', lambda camera: camera.set_gain_db(7.0), ('gain_db',)),
    ('exposure', lambda camera: camera.set_exposure_ms(30.0), ('exposure_ms',)),
    ('black_level', lambda camera: camera.set_black_level(4.0), ('black_level',)),
    ('frame_size', lambda camera: camera.set_frame_size(800, 600), ('frame_size',)),
    ('binning', lambda camera: camera.set_binning(2), ('binning',)),
    (
        'auto_exposure',
        lambda camera: camera.set_auto_exposure(CameraAutoMode.CONTINUOUS),
        ('auto_exposure',),
    ),
    (
        'auto_gain',
        lambda camera: camera.set_auto_gain(True, _AUTO_GAIN),
        ('auto_gain', 'auto_exposure'),
    ),
)


def test_a_write_the_camera_never_answered_leaves_its_value_unknown(session):
    camera = session.camera
    still_known = {}
    for node, write, held in _HELD_WRITES:
        # Unanswered twice: the write and the API's one resend.
        session.scope.simulation.refuse_camera_write(node, 'no_answer', count=2)
        with pytest.raises(CameraSettingRejected) as refused:
            write(camera)
        assert refused.value.reason == 'camera_no_answer', node
        known = {name: getattr(camera, name) for name in held if getattr(camera, name) is not None}
        if known:
            still_known[node] = known
    assert still_known == {}


@pytest.mark.parametrize(
    ('read', 'write', 'getter', 'held'),
    [
        ('read_black_level', 'set_black_level', 'get_black_level', 'black_level'),
        ('read_gain_db', 'set_gain_db', 'get_gain', 'gain_db'),
    ],
)
def test_a_fresh_read_that_a_write_overtakes_holds_the_write(
    session, monkeypatch, read, write, getter, held
):
    camera = session.camera
    driver = session.scope._camera_driver
    getattr(camera, write)(2.0)
    answered = getattr(driver, getter)
    in_flight, release = threading.Event(), threading.Event()

    def slow_read():
        value = answered()
        in_flight.set()
        release.wait(5.0)
        return value

    # a stand-in by design: the simulator has no door that holds a read open
    monkeypatch.setattr(driver, getter, slow_read)
    reader = threading.Thread(target=getattr(camera, read))
    reader.start()
    assert in_flight.wait(5.0), 'the read never reached the camera'
    monkeypatch.setattr(driver, getter, answered)
    getattr(camera, write)(4.0)
    release.set()
    reader.join(5.0)

    assert getattr(camera, held) == 4.0

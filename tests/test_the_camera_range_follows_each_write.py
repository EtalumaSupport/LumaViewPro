# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera's gain and exposure range is the one in force: read again
after each change of the camera's readout, so the API refuses what the
camera would refuse (``CameraAPI.setting_limits``).

Seen on the bench 2026-11-11: an a2A3536 read 12 us as its floor at
connect, in Mono8; in Mono12 the camera refused 12 us ("must be greater
than or equal 14"), the API having passed it against the connect copy, and
the characterization lost its whole dynamic-range stage at the first point.
The simulated camera is a daA3840: 14 us in Mono8, 29 us in Mono12, as both
bench bodies read it.
"""

import pytest

from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.modules.exceptions import CameraSettingRejected, ProtocolRunRefusedError
from _lumascope.modules.image_mode import ImageMode
from tests.test_a_protocol_needs_its_objectives_on_the_turret import _prepare
from tests.test_a_run_holds_its_steps_to_the_hardware import _steps
from _lumascope.modules.lumascope_api.camera import AutoGainSettings, CameraAutoMode
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    executor,
    executors,
    scope,
)

_RANGE_WORDS = "The camera's gain and exposure range could not be read"
_READ_AGAIN = 'read again at the next change of binning, frame size or pixel format'

_AUTO_GAIN = AutoGainSettings(
    target_brightness=0.5,
    min_gain_db=0.0,
    max_gain_db=20.0,
    max_exposure_ms=100.0,
    min_exposure_ms=0.1,
)


def _leave_the_range_unread(camera):
    """A readout change whose range read fails, as the simulated camera fails it."""
    camera._scope.simulation.refuse_camera_read('minimum_exposure', 'failed', count=1)
    camera.set_image_mode(ImageMode.SCIENTIFIC)


def test_an_exposure_under_the_twelve_bit_floor_is_raised_to_it_by_the_api(sim_camera, monkeypatch):
    """After Mono8 to Mono12, 14 us is sent as the 29 us floor the camera
    now declares; before the change it was taken as asked."""
    camera, cam = sim_camera
    camera.set_image_mode(ImageMode.EIGHT_BIT)
    assert camera.set_exposure_ms(0.014) == pytest.approx(0.014)
    camera.set_image_mode(ImageMode.SCIENTIFIC)
    sent = []
    taken = cam.exposure_t
    monkeypatch.setattr(cam, 'exposure_t', lambda ms: sent.append(ms) or taken(ms))

    assert camera.set_exposure_ms(0.014) == pytest.approx(0.029)

    assert sent == [pytest.approx(0.029)], (
        'the API held it to the range; the camera was asked the floor'
    )


def test_the_range_is_read_again_after_a_readout_change(sim_camera):
    camera, _cam = sim_camera
    camera.set_image_mode(ImageMode.EIGHT_BIT)
    assert camera.setting_limits.exposure_min_ms == pytest.approx(0.014)

    camera.set_image_mode(ImageMode.SCIENTIFIC)

    assert camera.setting_limits.exposure_min_ms == pytest.approx(0.029)


def test_a_failed_range_read_is_raised_until_a_read_succeeds(sim_camera):
    """The write before the failed read landed and returns as it did; every
    reader then raises the failure, and nothing is sent, until the next
    readout change reads the range again."""
    camera, _cam = sim_camera
    camera.set_image_mode(ImageMode.EIGHT_BIT)
    camera._scope.simulation.refuse_camera_read('minimum_exposure', 'failed', count=1)

    camera.set_image_mode(ImageMode.SCIENTIFIC)

    assert camera.pixel_format == 'Mono12', 'the format write landed'
    with pytest.raises(CameraReadFailedError) as unread:
        _ = camera.setting_limits
    assert unread.value.reason == 'camera_read_failed'
    assert _RANGE_WORDS in str(unread.value) and _READ_AGAIN in str(unread.value)
    exposure = camera.exposure_ms
    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_exposure_ms(1.0)
    assert refused.value.reason == 'camera_read_failed'
    assert isinstance(refused.value.__cause__, CameraReadFailedError)
    assert camera.exposure_ms == exposure, 'nothing was sent'

    camera.set_image_mode(ImageMode.EIGHT_BIT)

    assert camera.setting_limits.exposure_min_ms == pytest.approx(0.014)


def test_a_maximum_the_camera_reads_as_zero_is_no_maximum(sim_camera, monkeypatch):
    # Zero is the drivers' answer for an inactive camera, never a real cap.
    camera, cam = sim_camera
    monkeypatch.setattr(cam, 'get_max_exposure', lambda: 0.0)

    camera.set_image_mode(ImageMode.SCIENTIFIC)

    assert camera.setting_limits.exposure_max_ms is None


def test_a_whole_number_maximum_is_a_float(sim_camera, monkeypatch):
    camera, cam = sim_camera
    monkeypatch.setattr(cam, 'get_max_exposure', lambda: 500)

    camera.set_image_mode(ImageMode.SCIENTIFIC)

    maximum = camera.setting_limits.exposure_max_ms
    assert maximum == 500.0 and isinstance(maximum, float)


def test_every_write_checked_against_an_unread_range_is_refused_unsent_naming_what_it_asked(
    sim_session,
):
    """R22: a failed read before a write is a write not sent,
    ``CameraSettingRejected('camera_read_failed')`` chained to the read. A
    write that asks for several settings names every one, the auto-gain arm
    among them, so a run's auto-gain step reads the arm as not taken and
    does not capture."""
    camera = sim_session.scope.camera
    _leave_the_range_unread(camera)
    before = (camera.read_gain_db(), camera.read_exposure_ms(), camera.read_auto_gain())
    writes = {
        'set_gain_db': (lambda: camera.set_gain_db(5.0), 'gain_db'),
        'set_exposure_ms': (lambda: camera.set_exposure_ms(5.0), 'exposure_ms'),
        'apply': (lambda: camera.apply(5.0, 5.0), 'gain_db,exposure_ms'),
        'apply armed': (
            lambda: camera.apply(
                5.0, 5.0, auto_gain=CameraAutoMode.CONTINUOUS, auto_gain_settings=_AUTO_GAIN
            ),
            'gain_db,exposure_ms,auto_gain',
        ),
        'set_auto_gain': (
            lambda: camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN),
            'auto_gain',
        ),
        'apply_channel_camera': (
            lambda: sim_session.apply_channel_camera('BF'),
            # A channel's hold asks for its auto-gain mode too, the shipped
            # BF's off, and sends none of the three.
            'gain_db,exposure_ms,auto_gain',
        ),
    }
    wrong = {}
    for name, (write, named) in writes.items():
        try:
            write()
        except CameraSettingRejected as refused:
            said = (
                refused.reason,
                refused.setting,
                isinstance(refused.__cause__, CameraReadFailedError),
                _RANGE_WORDS in str(refused) and _READ_AGAIN in str(refused),
            )
            if said != ('camera_read_failed', named, True, True):
                wrong[name] = said
        else:
            wrong[name] = 'not refused'
    assert not wrong, wrong
    after = (camera.read_gain_db(), camera.read_exposure_ms(), camera.read_auto_gain())
    assert after == before, 'nothing was sent'


def test_a_run_is_refused_while_the_range_is_unread(executor, scope, tmp_path):
    _leave_the_range_unread(scope.camera)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        _prepare(executor, _steps(gain_db=1.0), tmp_path)

    assert refused.value.reason == 'camera_range_unread'
    assert _RANGE_WORDS in str(refused.value) and 'The run was not started.' in str(refused.value)

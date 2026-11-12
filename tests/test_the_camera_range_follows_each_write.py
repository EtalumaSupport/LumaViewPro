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
from _lumascope.modules.exceptions import CameraSettingOutOfRangeError
from _lumascope.modules.image_mode import ImageMode


def test_an_exposure_under_the_twelve_bit_floor_is_refused_by_the_api(sim_camera, monkeypatch):
    """After Mono8 to Mono12, 14 us is refused by the API naming 29 us, and
    nothing reaches the camera; before the change it was taken."""
    camera, cam = sim_camera
    camera.set_image_mode(ImageMode.EIGHT_BIT)
    assert camera.set_exposure_ms(0.014) == pytest.approx(0.014)
    camera.set_image_mode(ImageMode.SCIENTIFIC)
    sent = []
    taken = cam.exposure_t
    monkeypatch.setattr(cam, 'exposure_t', lambda ms: sent.append(ms) or taken(ms))

    with pytest.raises(CameraSettingOutOfRangeError) as refused:
        camera.set_exposure_ms(0.014)

    assert refused.value.reason == 'exposure_ms_out_of_range'
    assert refused.value.limits[0] == pytest.approx(0.029)
    assert '0.029 to' in str(refused.value)
    assert sent == [], 'the API refused it; the camera was never asked'


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
    exposure = camera.exposure_ms
    with pytest.raises(CameraReadFailedError):
        camera.set_exposure_ms(1.0)
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

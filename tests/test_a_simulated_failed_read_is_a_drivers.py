# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failed read on the simulated camera answers what a real driver answers.

The simulator stands in for the bench, so a read fault injected on a node
fails that node's read the way the real driver that reads it fails, and
never in a way no driver does. The getters below are the real drivers',
written here from the drivers, not from the simulator's table: every one
raises ``CameraReadFailedError`` on Pylon and IDS alike. ``'removed'`` is
the same raise with the camera gone from the bus.
"""

from __future__ import annotations

from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera

# node: the getter a real driver reads it through
_DRIVER_GETTERS = {
    'gain': 'get_gain',
    'exposure': 'get_exposure_t',
    'binning': 'get_binning_size',
    'frame_size': '_hardware_frame_size',
    'frame_grid': '_frame_grid',
    'pixel_format': 'get_pixel_format',
    'pixel_formats': 'get_supported_pixel_formats',
    'black_level': 'get_black_level',
    'frame_rate': 'get_resulting_frame_rate',
    'minimum_exposure': 'get_min_exposure',
    'auto_gain': 'get_auto_gain_mode',
    'auto_exposure': 'get_auto_exposure_mode',
}


def _offered(node: str, answer: str) -> SimulatedCamera | None:
    """A camera with ``answer`` armed on ``node``'s next read, or None when not offered."""
    cam = SimulatedCamera()
    try:
        cam.refuse_read(node, answer)
    except ValueError:
        return None
    return cam


def test_every_offered_failed_read_answers_as_the_driver_does():
    wrong = []
    for node, getter in _DRIVER_GETTERS.items():
        for answer in SimulatedCamera.FAILED_READ_ANSWERS:
            cam = _offered(node, answer)
            if cam is None:
                wrong.append(f'{node}/{answer}: not offered, though a driver fails this read')
                continue
            try:
                got = getattr(cam, getter)()
            except Exception as e:
                got = type(e)
            if got is not CameraReadFailedError:
                wrong.append(f'{node}/{answer}: {got!r}, a driver raises CameraReadFailedError')
            if (answer == 'removed') == cam.is_connected():
                wrong.append(f'{node}/{answer}: connected is {cam.is_connected()}')
    assert not wrong, '\n'.join(wrong)


def test_a_failed_binning_read_stops_a_binning_write():
    """The binning write reads the binning first (base ``set_binning_size``),
    so a failed read sends nothing: the read's raise, the binning unchanged."""
    cam = _offered('binning', 'failed')
    try:
        cam.set_binning_size(2)
    except CameraReadFailedError:
        pass
    else:
        raise AssertionError('a binning write whose read failed was sent')
    assert cam.get_binning_size() == 1

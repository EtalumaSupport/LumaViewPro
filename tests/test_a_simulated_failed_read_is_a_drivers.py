# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A failed read on the simulated camera answers what a real driver answers.

The simulator stands in for the bench, so a read fault injected on a node
fails that node's read the way the real driver that reads it fails, and
never in a way no driver does. The answers below are the real getters',
written here from the drivers, not from the simulator's table: the gain,
exposure and binning getters catch every exception and answer -1, the
frame-size and pixel-format getters answer None, and the black-level,
frame-rate, minimum-exposure (IDS) and auto-mode (Pylon) getters raise
``CameraReadFailedError``. ``'removed'`` is the same answer with the camera
gone from the bus.
"""

from __future__ import annotations

from drivers.exceptions import CameraReadFailedError
from drivers.simulated_camera import SimulatedCamera

_RAISES = CameraReadFailedError

# node: (the getter that reads it, what a real driver answers when the read fails)
_DRIVER_ANSWERS = {
    'gain': ('get_gain', -1),
    'exposure': ('get_exposure_t', -1),
    'binning': ('get_binning_size', -1),
    'frame_size': ('get_frame_size', None),
    'pixel_format': ('get_pixel_format', None),
    'black_level': ('get_black_level', _RAISES),
    'frame_rate': ('get_resulting_frame_rate', _RAISES),
    'minimum_exposure': ('get_min_exposure', _RAISES),
    'auto_gain': ('get_auto_gain_mode', _RAISES),
    'auto_exposure': ('get_auto_exposure_mode', _RAISES),
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
    for node, (getter, expected) in _DRIVER_ANSWERS.items():
        for answer in SimulatedCamera.FAILED_READ_ANSWERS:
            cam = _offered(node, answer)
            if cam is None:
                wrong.append(f'{node}/{answer}: not offered, though a driver fails this read')
                continue
            try:
                got = getattr(cam, getter)()
            except Exception as e:
                got = type(e)
            if got != expected:
                wrong.append(f'{node}/{answer}: {got!r}, a driver answers {expected!r}')
            if (answer == 'removed') == cam.is_connected():
                wrong.append(f'{node}/{answer}: connected is {cam.is_connected()}')
    assert not wrong, '\n'.join(wrong)


def test_no_offered_binning_read_fault_stops_a_binning_write():
    stopped = []
    met = 0
    for answer in SimulatedCamera.FAILED_READ_ANSWERS:
        if answer == 'removed':
            continue
        cam = _offered('binning', answer)
        if cam is None:
            continue
        met += 1
        try:
            applied = cam.set_binning_size(2)
        except Exception as e:
            applied = type(e)
        if applied is not True or cam.get_binning_size() != 2:
            stopped.append(f'{answer}: {applied!r}, binning {cam.get_binning_size()}')
    assert met
    assert not stopped, '\n'.join(stopped)

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler USB camera starts with its frame rate capped under what its link carries.

A daA3840 on firmware 1.1.0, stopped under ``CompleteExposure`` while it
streams at its link limit, stalls the stop 11 s, and every stop after it
until a power cycle; held under the link for the whole grab, it does not.
So every start enables ``AcquisitionFrameRate`` at 0.92 x
``DeviceLinkThroughputLimit`` / ``PayloadSize``, clamped to the node's
range; a limit switched off caps under the limit's maximum. A camera not on
USB, or without the node, starts as it did before the cap. A cap the camera
does not take fails the start, so no camera runs uncapped.

The driver's start runs over the suite's SDK stand-ins (``bare_pylon_camera``).
"""

from __future__ import annotations

import pytest
from pypylon import genicam

from tests.camera_fakes import bare_pylon_camera
from tests.pypylon_stub import IBoolean, IEnumeration, PlaceholderParameter

# 1900x1900 Mono8 with its 72 B chunk, and the daA3840's Default limit.
_PAYLOAD = 1900 * 1900 + 72
_LIMIT = 160_000_000
_LIMIT_MAX = 419_430_400


class _RangedNode:
    """A numeric node as the camera hands it: a value, its range, and a write."""

    def __init__(self, value, low, high, refuses=None):
        self.value = value
        self.Min = low
        self.Max = high
        self.refuses = refuses
        self.readable = True

    def GetValue(self):
        return self.value

    def GetUnit(self):
        return ''

    def SetValue(self, value):
        if self.refuses is not None:
            raise self.refuses
        self.value = value


def _camera(*, device_class='BaslerUsb', mode='On', rate_node=True, payload=_PAYLOAD, refuses=None):
    """A Basler camera, not grabbing, with the nodes the cap reads and writes."""
    # a stand-in by design: no simulated camera runs pyloncamera.py, and the
    # subject is what the driver writes to the nodes the SDK hands it.
    cam = bare_pylon_camera()
    # Set by __init__, which the fake's __new__ skips: the shipped strategy.
    cam._grab_strategy_name = 'LatestImageOnly'
    camera = cam.active
    camera.GetDeviceInfo.return_value.GetDeviceClass.return_value = device_class
    camera.PayloadSize = _RangedNode(payload, 0, 2**32)
    camera.DeviceLinkThroughputLimitMode = IEnumeration(mode)
    camera.DeviceLinkThroughputLimit = _RangedNode(_LIMIT, 1_000_000, _LIMIT_MAX)
    camera.AcquisitionFrameRate = (
        _RangedNode(30.0, 1.0, 1000.0, refuses)
        if rate_node
        else PlaceholderParameter('AcquisitionFrameRate')
    )
    camera.AcquisitionFrameRateEnable = IBoolean(False)
    return cam


def test_a_usb_camera_starts_capped_at_the_share_of_its_link():
    cam = _camera()

    cam.start_grabbing()

    camera = cam.active
    assert camera.AcquisitionFrameRate.value == pytest.approx(0.92 * _LIMIT / _PAYLOAD)
    assert camera.AcquisitionFrameRateEnable.GetValue() is True
    camera.StartGrabbing.assert_called_once()


def test_the_cap_is_clamped_to_the_rate_nodes_range():
    # A 4x4 frame: the link would carry far more than the node takes.
    cam = _camera(payload=4 * 4 + 72)

    cam.start_grabbing()

    assert cam.active.AcquisitionFrameRate.value == 1000.0


def test_a_limit_switched_off_caps_under_the_limits_maximum():
    cam = _camera(mode='Off')

    cam.start_grabbing()

    assert cam.active.AcquisitionFrameRate.value == pytest.approx(0.92 * _LIMIT_MAX / _PAYLOAD)
    assert cam.active.AcquisitionFrameRateEnable.GetValue() is True


# parametrize-justified: each camera is its own body, built per case.
@pytest.mark.parametrize(
    'body',
    [{'device_class': 'BaslerGigE'}, {'rate_node': False}],
    ids=['not_on_usb', 'no_rate_node'],
)
def test_a_camera_the_cap_does_not_apply_to_starts_as_before(body):
    cam = _camera(**body)

    cam.start_grabbing()

    assert cam.active.AcquisitionFrameRateEnable.GetValue() is False
    cam.active.StartGrabbing.assert_called_once()


def test_a_cap_the_camera_does_not_take_fails_the_start():
    cam = _camera(refuses=genicam.GenericException('out of range'))

    cam.start_grabbing()

    assert cam.active.AcquisitionFrameRateEnable.GetValue() is False
    cam.active.StartGrabbing.assert_not_called()

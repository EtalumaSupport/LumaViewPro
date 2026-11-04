# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pylon's auto-mode read that fails raises ``CameraReadFailedError``.

``Camera.get_auto_gain_mode`` / ``get_auto_exposure_mode`` raise a
``HardwareError`` when the camera does not answer. The SDK can fail at the
node's lookup (``LogicalErrorException`` before any read) or at its read;
either is the driver's ``CameraReadFailedError``, naming the node, and the
camera is kept, as every other raising Pylon getter keeps it.

A stand-in by design: the SDK's faults on a node are what is under test,
and only a stub camera produces them.
"""

from __future__ import annotations

from unittest.mock import PropertyMock

import pytest
from pypylon import genicam

from drivers.exceptions import CameraReadFailedError
from tests.camera_fakes import bare_pylon_camera

_GETTERS = (('get_auto_gain_mode', 'GainAuto'), ('get_auto_exposure_mode', 'ExposureAuto'))


def test_a_failed_node_lookup_raises_naming_the_node():
    for getter, node in _GETTERS:
        cam = bare_pylon_camera()
        setattr(
            type(cam.active),
            node,
            PropertyMock(side_effect=genicam.LogicalErrorException('Node not existing')),
        )
        with pytest.raises(CameraReadFailedError, match=f'{node} read failed'):
            getattr(cam, getter)()
        cam._mark_disconnected.assert_not_called()


def test_a_failed_node_read_raises_naming_the_node():
    for getter, node in _GETTERS:
        cam = bare_pylon_camera()
        getattr(cam.active, node).GetCurrentEntry.side_effect = genicam.RuntimeException(
            'transient'
        )
        with pytest.raises(CameraReadFailedError, match=f'{node} read failed'):
            getattr(cam, getter)()
        cam._mark_disconnected.assert_not_called()


def test_an_answered_read_is_the_mode():
    for getter, node in _GETTERS:
        cam = bare_pylon_camera()
        getattr(
            cam.active, node
        ).GetCurrentEntry.return_value.GetSymbolic.return_value = 'Continuous'
        assert getattr(cam, getter)() == 'Continuous'

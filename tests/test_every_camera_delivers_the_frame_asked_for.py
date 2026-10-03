# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every camera delivers exactly the frame size asked for.

Before, only IDS did: it acquired the next legal window up and cropped back.
The FX2, Pylon and simulated cameras floored the request to their grids
without a word (an FX2 asked for 1898 delivered 1896, the simulator asked
for 1900 delivered 1872). Now the camera base plans the acquisition for
every driver and crops every stored frame back to the request; a frame of
any other size, from a window no longer set, is not stored.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from drivers.camera import ImageHandlerBase
from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import bare_pylon_camera, grab_a_frame_made_after_now


def _store(handler, w, h):
    handler._store_frame(
        np.zeros((h, w), dtype=np.uint8), datetime.datetime.now(), significant_bits=8
    )


@pytest.fixture
def sim_camera():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    yield cam
    cam.disconnect()


def test_the_simulated_camera_delivers_a_size_off_its_grid(sim_camera):
    # The simulator's grid is 48 wide and 4 tall: neither side is on it.
    assert sim_camera.set_frame_size(1897, 1101) == {'width': 1897, 'height': 1101}
    assert sim_camera.get_frame_size() == {'width': 1897, 'height': 1101}
    ok, *_ = grab_a_frame_made_after_now(sim_camera)
    assert ok
    assert sim_camera.array.shape[:2] == (1101, 1897)


def test_the_fx2_delivers_a_size_off_its_grid():
    from drivers.fx2driver import REG_COL_SIZE, REG_ROW_SIZE, FX2Camera
    from drivers.simulated_fx2 import SimulatedFX2

    sim = SimulatedFX2()
    cam = FX2Camera(connection=sim.connection)
    try:
        assert cam.set_frame_size(1897, 1898) == {'width': 1897, 'height': 1898}
        assert cam.get_frame_size() == {'width': 1897, 'height': 1898}
        # The sensor acquires the next window up on the FX2's grid of 4.
        registers = sim.device.sensor.registers
        assert registers[REG_COL_SIZE] == 1900 + 3
        assert registers[REG_ROW_SIZE] == 1900 + 1
        _store(cam.cam_image_handler, 1900, 1900)
        ok, image, *_ = cam.cam_image_handler.get_last_image()
        assert ok
        assert image.shape == (1898, 1897)
    finally:
        cam.disconnect()


def _pylon_camera(max_size=(1920, 1200), step=(4, 2)):
    cam = bare_pylon_camera()
    sdk = cam.active
    for axis, maximum, inc in (('Width', max_size[0], step[0]), ('Height', max_size[1], step[1])):
        node = getattr(sdk, axis)
        node.Max = maximum
        node.GetMax.return_value = maximum
        node.Inc = inc
        node.GetInc.return_value = inc
        node.Min = inc
        node.GetMin.return_value = inc
        node.GetValue.return_value = maximum
    sdk.BslCenterX = MagicMock()
    sdk.BslCenterY = MagicMock()
    cam.cam_image_handler = ImageHandlerBase()
    return cam


def test_a_pylon_camera_delivers_a_size_off_its_grid():
    cam = _pylon_camera()
    assert cam.set_frame_size(1897, 1101) == {'width': 1897, 'height': 1101}
    cam.active.Width.SetValue.assert_called_with(1900)
    cam.active.Height.SetValue.assert_called_with(1102)
    _store(cam.cam_image_handler, 1900, 1102)
    ok, image, *_ = cam.cam_image_handler.get_last_image()
    assert ok
    assert image.shape == (1101, 1897)


def test_a_frame_from_a_window_no_longer_set_is_not_stored():
    cam = _pylon_camera()
    cam.set_frame_size(1897, 1101)
    handler = cam.cam_image_handler
    _store(handler, 1920, 1200)
    assert handler.frames_delivered == 0
    _store(handler, 1900, 1102)
    assert handler.frames_delivered == 1


def test_a_binning_change_lets_the_new_frame_through_until_a_window_is_set(sim_camera):
    sim_camera.set_frame_size(1897, 1101)
    assert sim_camera.set_binning_size(2)
    ok, *_ = grab_a_frame_made_after_now(sim_camera)
    assert ok
    # The window set at 1x no longer fits; frames pass as the camera makes them.
    assert sim_camera.array.shape[:2] != (1101, 1897)
    sim_camera.set_frame_size(901, 501)
    grab_a_frame_made_after_now(sim_camera)
    assert sim_camera.array.shape[:2] == (501, 901)


def test_a_window_survives_a_rebuilt_frame_handler():
    cam = _pylon_camera()
    cam.set_frame_size(1897, 1101)
    cam.cam_image_handler = ImageHandlerBase()
    cam._reapply_frame_callbacks()
    _store(cam.cam_image_handler, 1900, 1102)
    _, image, *_ = cam.cam_image_handler.get_last_image()
    assert image.shape == (1101, 1897)


@pytest.mark.parametrize('member', ['set_frame_size', 'get_frame_size', 'set_binning_size'])
def test_no_driver_frames_its_own_frames(member):
    # The window lives once, in the camera base; a driver that answered one of
    # these itself would deliver frames the base never cropped.
    from drivers.camera import Camera
    from drivers.fx2driver import FX2Camera
    from drivers.idscamera import IDSCamera
    from drivers.pyloncamera import PylonCamera

    for driver in (FX2Camera, IDSCamera, PylonCamera, SimulatedCamera):
        assert getattr(driver, member) is getattr(Camera, member), (driver.__name__, member)

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bare camera-driver instances for behavioral driver tests.

These build REAL driver objects (via __new__, no SDK connect) with a
controllable fake camera attached, so tests drive production methods
and assert observable behavior -- return values, raises, disconnect
marks, enqueue decisions -- instead of grepping driver source text.
Pairs with tests/pypylon_stub.py, which makes the handler classes
constructible in the first place.
"""

from __future__ import annotations

import contextlib
import queue as _queue_mod
import threading
from unittest.mock import MagicMock


def bare_pylon_camera():
    """PylonCamera with a fake SDK camera attached.

    `cam.active` is a MagicMock standing in for the pylon
    InstantCamera, so tests set side_effects on its node accessors.
    update_camera_config is replaced with a no-op context manager so
    the grab-loop bounce stays out of unit scope.
    """
    from drivers import pyloncamera

    cam = pyloncamera.PylonCamera.__new__(pyloncamera.PylonCamera)
    cam._state_lock = threading.Lock()
    cam.active = MagicMock()
    cam._mark_disconnected = MagicMock()
    cam.update_camera_config = lambda: contextlib.nullcontext()
    return cam


def disconnectable_pylon_camera():
    """bare_pylon_camera prepared so the REAL disconnect() can run.

    Grab-loop, idle-wait, and Stage B worker internals are stubbed so
    tests can drive disconnect() end-to-end and assert the SDK teardown
    calls (Close / DetachDevice / DestroyDevice) and state transitions
    on the fake handle.
    """
    cam = bare_pylon_camera()
    cam._device_removed = False
    cam.is_grabbing = lambda: False
    cam.stop_grabbing = MagicMock()
    cam._wait_for_acquisition_idle = MagicMock(return_value=True)
    cam._stop_image_grab_worker = MagicMock()
    return cam


def bare_ids_camera():
    """IDSCamera analog of bare_pylon_camera: fake remote_nodemap."""
    from drivers import idscamera

    cam = idscamera.IDSCamera.__new__(idscamera.IDSCamera)
    cam._state_lock = threading.Lock()
    cam.active = True
    cam.remote_nodemap = MagicMock()
    cam._mark_disconnected = MagicMock()
    cam.update_camera_config = lambda: contextlib.nullcontext()
    return cam


def bare_image_handler():
    """ImageHandler wired to a bare PylonCamera parent, Stage B mocked.

    Drives the REAL Stage A callback (OnImageGrabbed / OnImagesSkipped)
    with controllable fake grab results; handler._worker is a MagicMock
    so enqueue decisions are observable and Stage B stays out of scope.
    """
    from drivers import pyloncamera

    parent = bare_pylon_camera()
    parent._device_removed = False
    parent._schedule_async_teardown = MagicMock()
    handler = pyloncamera.ImageHandler(parent)
    handler._worker = MagicMock()
    return handler, parent


def bare_grab_worker():
    """_PylonImageGrabWorker with a bare parent and a spied failure
    counter, for driving Stage B classification directly."""
    from drivers import pyloncamera
    from drivers.camera import ImageHandlerBase

    parent = bare_pylon_camera()
    parent._device_removed = False
    base = ImageHandlerBase()
    base._record_failure = MagicMock(return_value=False)
    worker = pyloncamera._PylonImageGrabWorker(parent, base, _queue_mod.Queue(maxsize=1))
    return worker, base

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


def init_configurable_pylon_camera():
    """bare_pylon_camera prepared so the REAL init_camera_config() can
    run.

    The sub-setters it fans out to (chunks, pixel format, gain,
    exposure, frame size, AG init) are stubbed, leaving the UserSet /
    acquisition-mode / trigger SDK writes observable on the fake
    camera handle.
    """
    cam = bare_pylon_camera()
    cam.is_grabbing = lambda: False
    cam._use_camera_emulation = True
    for name in (
        '_enable_validity_chunks',
        '_read_timestamp_tick_frequency',
        'set_pixel_format',
        'auto_gain',
        'gain',
        'exposure_t',
        'set_frame_size',
        'init_auto_gain_focus',
    ):
        setattr(cam, name, MagicMock())
    return cam


def fake_trigger_entry(symbolic: str, available: bool = True):
    """One TriggerSelector.GetEntries() element for init-config tests."""
    entry = MagicMock()
    entry.IsAvailable.return_value = available
    entry.GetSymbolic.return_value = symbolic
    return entry


class FakeDiagNode:
    """Nodemap node returning a queued sequence of values (last repeats)."""

    def __init__(self, *values):
        self._values = list(values)

    def GetValue(self):
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class RecordingNodeMap:
    """GetNode-style nodemap that records every requested node name.

    Missing names return None (the camera-nodemap missing-node shape);
    present names return a FakeDiagNode. Pass FakeDiagNode instances as
    values for multi-read sequences (e.g. pre/post counter reads).
    """

    def __init__(self, values=None):
        self.requested = []
        self._values = dict(values or {})

    def GetNode(self, name):
        self.requested.append(name)
        value = self._values.get(name)
        if value is None:
            return None
        return value if isinstance(value, FakeDiagNode) else FakeDiagNode(value)


def diag_snapshot_pylon_camera(camera_values=None, grabber_values=None):
    """bare_pylon_camera wired so the REAL read_diagnostic_snapshot()
    can run: recording nodemaps on the camera and stream-grabber sides
    so tests can assert which nodes were probed and what the snapshot
    reported for them."""
    cam = bare_pylon_camera()
    nodemap = RecordingNodeMap(camera_values)
    grabber_nodemap = RecordingNodeMap(grabber_values)
    cam.active.GetNodeMap.return_value = nodemap
    cam.active.GetStreamGrabberNodeMap.return_value = grabber_nodemap
    return cam, nodemap, grabber_nodemap


def stats_poll_pylon_camera():
    """bare_pylon_camera prepared so one REAL _stats_poller_loop cycle
    runs deterministically: the one-shot validation walk is skipped and
    the live fps / temperature reads return plain values (tests override
    per-node mocks as needed)."""
    cam = bare_pylon_camera()
    cam._pylon_self_validation_done = True
    cam.active.BslResultingAcquisitionFrameRate.GetValue.return_value = 30.0
    cam.active.TemperatureState.GetValue.return_value = 'Ok'
    return cam


def run_one_stats_poll(cam):
    """Drive exactly one iteration of the real _stats_poller_loop:
    the stop event reports not-set once, then set."""
    ev = MagicMock()
    ev.wait.side_effect = [False, True]
    cam._stats_poller_stop = ev
    cam._stats_poller_loop()


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

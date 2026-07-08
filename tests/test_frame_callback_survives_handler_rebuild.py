# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: per-frame callbacks survive a camera handler rebuild.

Callbacks registered on a Camera live in the durable
``Camera._registered_frame_callbacks`` registry, NOT on the ephemeral image
handler. A driver that rebuilds its handler (connect / recovery) must re-apply
the registry via ``_reapply_frame_callbacks()`` so manual recording and
per-frame plugin listeners keep receiving frames after a reconnect.

Before this fix the callbacks lived only on the handler's dispatch list, so a
handler rebuild silently dropped every listener with no error surfaced. These
tests exercise the mechanism at the driver level with SimulatedCamera (no
hardware) plus a bare ImageHandlerBase standing in for a freshly-built handler.
"""

from drivers.camera import ImageHandlerBase
from drivers.simulated_camera import SimulatedCamera


def _cb(image, ts, chunks):
    """A no-op per-frame callback with the (image, timestamp, chunks) signature."""


def test_registered_callback_stored_in_durable_registry():
    """register_frame_callback records the callback in the Camera-owned durable
    registry (not just the handler), so it can outlive a handler rebuild."""
    cam = SimulatedCamera()
    try:
        cam.register_frame_callback(_cb)
        assert _cb in cam._registered_frame_callbacks
    finally:
        cam.disconnect()


def test_reapply_pushes_registry_onto_freshly_built_handler():
    """A rebuilt handler starts with an empty dispatch list; after
    _reapply_frame_callbacks() it carries every durably-registered callback --
    the reconnect guarantee for recording / plugins."""
    cam = SimulatedCamera()
    try:
        cam.register_frame_callback(_cb)

        # Simulate a driver handler rebuild (connect / recovery): a fresh handler
        # with an empty dispatch list replaces whatever was there.
        fresh = ImageHandlerBase()
        assert _cb not in fresh._frame_callbacks  # empty before re-apply

        cam.cam_image_handler = fresh
        cam._reapply_frame_callbacks()

        assert _cb in fresh._frame_callbacks  # re-applied from the durable registry
    finally:
        cam.cam_image_handler = None
        cam.disconnect()


def test_unregister_removes_from_durable_registry():
    """unregister_frame_callback drops the callback from the durable registry so
    a later rebuild does not resurrect a listener the caller removed."""
    cam = SimulatedCamera()
    try:
        cam.register_frame_callback(_cb)
        cam.unregister_frame_callback(_cb)
        assert _cb not in cam._registered_frame_callbacks
    finally:
        cam.disconnect()

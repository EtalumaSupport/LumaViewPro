# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: disconnect() returns per-instance lifecycle state to baseline.

The base was written under "disconnect = fatal," so reconnect-relevant state
persisted across a disconnect: the start gate stayed OPEN (so a same-instance
reconnect's open_and_start() never restarted grabbing) and the last-frame buffer
kept the pre-disconnect image (so the live view briefly showed a stale frame).
`found` was a once-in-__init__ snapshot that never tracked a later disconnect.

L1-c makes disconnect() call Camera._reset_lifecycle_state() (start gate +
frame buffer) and turns `found` into a property derived from `active`, so a
disconnect / same-instance reconnect starts clean. Exercised with
SimulatedCamera (no hardware).
"""

import numpy as np

from drivers.simulated_camera import SimulatedCamera


def test_found_is_derived_from_active():
    """`found` tracks `active`, not a stale __init__ snapshot: True while
    connected, False once the driver nulls active on disconnect."""
    cam = SimulatedCamera()
    try:
        assert cam.found is True  # connected after construct
        cam.active = None
        assert cam.found is False  # derived live from active, no refresh needed
    finally:
        cam.disconnect()


def test_disconnect_resets_start_gate_and_frame_buffer():
    """disconnect() closes the start gate and clears the last-frame buffer so a
    same-instance reconnect re-grabs and never serves the pre-disconnect image."""
    cam = SimulatedCamera()
    # Simulate a live session: gate opened + a frame buffered.
    cam._grab_gate_open = True
    cam.array = np.array([[1, 2], [3, 4]])

    assert cam.disconnect() is True
    assert cam._grab_gate_open is False  # gate CLOSED for the next open_and_start
    assert cam.array.size == 0  # stale frame cleared
    assert cam.found is False  # active nulled -> found derives False

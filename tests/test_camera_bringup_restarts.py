# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Invariant tests for the camera bring-up start gate.

These encode the contract from CAMERA_START_GATE_PLAN.md. The gate is the
camera-lifecycle split: connect() returns the camera configured but NOT
grabbing (a per-instance latch starts CLOSED), and open_and_start() is the
single configure-complete -> start transition that opens the latch and
fires the one start. Enforcement is structural -- nothing starts the grab
before open_and_start(), so start_grabbing() stays a pure restartable
primitive (no runtime gate guard).

The restart-accounting tests pin the invariant that configuration applied
while NOT grabbing causes zero stop/start churn (what the closed gate
guarantees). The IDS packed-format 12-bit recognition lands with the
Mono8-push fix in a later phase, so its test carries ``xfail(strict=True)``
-- it FAILS today and FLIPS GREEN when that fix lands; ``strict=True`` then
turns red so the stale marker is removed.

Built on tests/camera_fakes.py: real driver objects (via __new__) with a
fake SDK attached, so production methods run and observable behavior is
asserted rather than driver source text grepped.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.camera_fakes import bare_fx2_camera, bare_ids_camera, bare_pylon_camera

# The three real production camera drivers, each as a bare fake. Every
# bring-up invariant must hold identically across all three (the bug was
# the same structural conflation in each connect()).
DRIVERS = [
    pytest.param(bare_pylon_camera, id='pylon'),
    pytest.param(bare_ids_camera, id='ids'),
    pytest.param(bare_fx2_camera, id='fx2'),
]

# -- Restart accounting: the closed-window invariant (passes today) --------


@pytest.mark.parametrize('make_cam', DRIVERS)
def test_gate_closed_at_construction(make_cam):
    """A freshly built camera has its start gate CLOSED."""
    cam = make_cam()
    assert cam._grab_gate_open is False


@pytest.mark.parametrize('make_cam', DRIVERS)
def test_config_while_not_grabbing_causes_zero_restart(make_cam):
    """Applying a saved-settings batch while not grabbing never bounces
    the grab loop.

    Every real setter wraps its SDK writes in ``update_camera_config()``;
    a saved-settings batch is modelled as repeated entries. While the gate
    is closed the camera is not grabbing, so each entry must be a no-op --
    zero stop, zero start. This is the churn the gate eliminates.
    """
    cam = make_cam()
    cam.start_grabbing = MagicMock()
    cam.stop_grabbing = MagicMock()

    for _ in range(5):
        with cam.update_camera_config():
            pass

    cam.stop_grabbing.assert_not_called()
    cam.start_grabbing.assert_not_called()


@pytest.mark.parametrize('make_cam', DRIVERS)
def test_config_while_grabbing_bounces_once(make_cam):
    """Contrast case: a single config applied while grabbing stops then
    starts exactly once.

    Confirms the zero-restart result above is caused by the not-grabbing
    state (the closed gate), not by an inert wrapper.
    """
    cam = make_cam()
    cam.is_grabbing = lambda: True
    cam.start_grabbing = MagicMock()
    cam.stop_grabbing = MagicMock()

    with cam.update_camera_config():
        pass

    cam.stop_grabbing.assert_called_once()
    cam.start_grabbing.assert_called_once()


# -- Gate contract: the single configure-complete -> start transition ------


@pytest.mark.parametrize('make_cam', DRIVERS)
def test_full_bringup_lifecycle_fires_exactly_one_start(make_cam):
    """The canonical invariant: gate closed -> apply settings batch ->
    zero restart -> open gate -> exactly one start.

    open_and_start() is the single release that fires the one start; a
    second call (the two release sites fire ~0.3 s apart) is a no-op.
    """
    cam = make_cam()
    cam.start_grabbing = MagicMock()
    cam.stop_grabbing = MagicMock()

    assert cam._grab_gate_open is False
    for _ in range(5):
        with cam.update_camera_config():
            pass
    cam.start_grabbing.assert_not_called()
    cam.stop_grabbing.assert_not_called()

    cam.open_and_start()
    assert cam._grab_gate_open is True
    cam.start_grabbing.assert_called_once()

    cam.open_and_start()
    cam.start_grabbing.assert_called_once()


def test_connect_does_not_eager_start_and_open_and_start_releases():
    """connect() returns configured but NOT grabbing; open_and_start()
    is the single release that begins streaming."""
    from drivers.simulated_camera import SimulatedCamera

    cam = SimulatedCamera()
    assert cam._grab_gate_open is False
    assert cam.is_grabbing() is False
    cam.open_and_start()
    assert cam._grab_gate_open is True
    assert cam.is_grabbing() is True


# -- Pixel-format capability: exact match, additive (pure logic) -----------


def test_pylon_mono12_still_recognized_as_12bit():
    """The 12-bit capability check stays additive: a Pylon camera
    advertising Mono12 still reports 12-bit, and Mono8 never does."""
    from modules.image_mode import camera_supports_12bit

    assert camera_supports_12bit(('Mono12',)) is True
    assert camera_supports_12bit(('Mono8',)) is False


def test_ids_packed_format_12bit_recognition():
    """The IDS sensor advertises the packed IDS-specific format, not bare
    Mono12. Mono12g24IDS is 12-bit; Mono10g40IDS is not -- the match must
    be format-exact so Mono10 is never offered the 12-bit modes."""
    from modules.image_mode import camera_supports_12bit

    assert camera_supports_12bit(('Mono12g24IDS',)) is True
    assert camera_supports_12bit(('Mono10g40IDS',)) is False

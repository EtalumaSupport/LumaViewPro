# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issues #671 (defect B) and #721: dark-floor frame
rejection, symmetric with the saturation guard.

Bug shape: frame acceptance in the capture path judged content on one
tail only -- a near-saturated frame was rejected, but a near-black frame
(stale pre-LED integration, or an external camera consumer starving the
feed) was accepted and silently saved. The dark floor closes the gap.

The expectation is DERIVED, not declared: capture_and_wait reads the
illumination API's own commanded-lit state (a channel counts as lit only
at strictly positive current), so a lit capture that delivers a black
frame is retried until timeout then rejected loudly (None + warning),
while a nothing-commanded capture accepts its dark frame as by-design.
``accept_dark=True`` is the one caller-intent override (autofocus sweeps,
benchmark probes). Public ``get_image`` is the ungated primitive and
never dark-rejects.

The metric is lit-pixel COUNT against the frame's payload depth -- sparse
fluorescence (a few bright cells on a black background) must pass, and
the depth rule must match ``_saturated_fraction`` (12-bit-in-uint16
measures against 4095, not 65535).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modules.lumascope_api.imaging as imaging_module
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


@pytest.fixture
def dark_scope():
    """A full simulated scope: real IlluminationAPI over SimulatedLEDBoard,
    streaming camera. The derivation must be exercised against live LED
    state -- an illumination stub would let an always-False derivation
    pass every test here (the defect shape that killed two plan drafts).
    """
    scope = Lumascope(simulate=True)
    scope._led_driver.set_timing_mode('fast')
    scope._motion_driver.set_timing_mode('fast')
    scope._camera_driver.set_timing_mode('fast')
    scope._camera_driver.load_cycle_images()
    scope.imaging.start_streaming()
    yield scope
    scope.imaging.stop_streaming()
    scope.disconnect()


_DARK = np.full((8, 8), 6, dtype=np.uint8)  # max 2.4% of full scale -- no signal
_LIT = np.full((8, 8), 120, dtype=np.uint8)


class TestDarkFloorRejection:
    def test_stale_dark_frame_healed_by_retry(self, dark_scope, monkeypatch):
        """The #671-B symptom: the first frame integrated before the LED
        lit; the very next frame is good. Retry must heal the capture."""
        frames = [_DARK, _LIT, _LIT]
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: frames.pop(0))

        out = dark_scope.imaging._get_image_impl(dark_floor_check=True, timeout_s=2.0)
        assert out is not None, 'retry must heal a transient dark frame'
        assert out.max() >= 120, 'the LIT retry frame must be returned, not the dark one'

    def test_persistent_dark_frames_rejected_loudly(self, dark_scope, monkeypatch):
        """The #721 symptom under the derived contract: a channel is
        commanded lit at real current, the camera keeps delivering black
        frames. The capture must fail as None with a warning naming the
        dark rejection -- never a silently saved black file. Illumination
        is REAL here: the derivation reads commanded state end to end."""
        dark_scope.illumination.led_on('BF', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        with patch.object(imaging_module, 'logger') as mock_logger:
            out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)

        assert out is None, 'a persistently dark frame under a lit channel must be rejected'
        warned = ' '.join(str(c).lower() for c in mock_logger.warning.call_args_list)
        assert 'dark' in warned and 'rejected' in warned, (
            f'rejection must be named in a warning; saw: {warned!r}'
        )

    def test_sparse_fluorescence_accepted(self, dark_scope, monkeypatch):
        """False-positive guard: a sparse fluorescence field (a handful of
        bright pixels on a black background) carries real signal and must
        pass under the derived check while lit -- the metric is lit-pixel
        count, not mean, and this floor must not move."""
        sparse = np.zeros((100, 100), dtype=np.uint8)
        sparse.flat[:4] = 200  # 4e-4 lit fraction, just above the minimum
        dark_scope.illumination.led_on('Blue', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: sparse)

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.5)
        assert out is not None, 'sparse fluorescence must not be rejected as dark'

    def test_dark_by_design_accepted_when_nothing_commanded(self, dark_scope, monkeypatch):
        """Nothing commanded -> the derivation reads not-lit and a dark
        frame is what the caller asked for: accepted unchanged."""
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.5)
        assert out is not None
        assert out.max() == 6, 'the dark frame itself must be returned untouched'

    def test_zero_current_channel_is_dark_by_design(self, dark_scope, monkeypatch):
        """A channel commanded ON at 0 mA lights nothing: the derivation
        must read it as dark, not lit. An enabled-flag derivation reads it
        as lit and rejects the by-design black frame -- the defect that
        killed the first draft of this fold."""
        dark_scope.illumination.led_on('BF', 0)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.5)
        assert out is not None, 'a 0 mA channel must derive as dark by design'

    def test_accept_dark_overrides_a_lit_rejection(self, dark_scope, monkeypatch):
        """The one caller-intent override: an autofocus sweep runs with
        LEDs ON yet must accept a dark frame (an out-of-focus fluorescence
        plane can carry no signal). Same state without the override must
        reject -- proving the override, not a broken derivation, is what
        accepted the frame."""
        dark_scope.illumination.led_on('Blue', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        rejected = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)
        assert rejected is None, 'without the override a lit-black frame must reject'

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.5, accept_dark=True)
        assert out is not None, 'accept_dark must admit the dark frame while lit'

    def test_lit_peer_rejects_dark_luminescence_capture(self, dark_scope, monkeypatch):
        """A luminescence capture taken while a peer channel is lit is
        rejected when the frame is black: the lit peer contaminates the
        capture, so the loud failure is the correct outcome (a deliberate
        product decision, not an accident of the derivation)."""
        dark_scope.illumination.led_on('Red', 150)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)
        assert out is None, 'a lit peer must make a black frame a loud failure'

    def test_public_get_image_never_dark_rejects(self, dark_scope, monkeypatch):
        """Public get_image is the ungated primitive: liveness probes and
        diagnostics must see what the camera sees, dark or not, even while
        a channel is lit. (The recording prologue counts arrivals through
        it; a dark-rejection there reads as feed-dead.)"""
        dark_scope.illumination.led_on('BF', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        out = dark_scope.imaging.get_image(timeout_s=0.5)
        assert out is not None, 'the ungated primitive must return the dark frame'


class TestLitFractionDepthRule:
    def test_payload_depth_not_container_depth(self):
        """12-bit payload in a uint16 container: the floor is 3% of 4095,
        not 3% of 65535. A 200-count pixel is lit at 12-bit depth and
        would be wrongly dark if measured against the container."""
        arr = np.full((8, 8), 200, dtype=np.uint16)
        assert ImagingAPI._lit_fraction(arr, 12) == 1.0
        assert ImagingAPI._lit_fraction(arr, 16) == 0.0

    def test_empty_and_none_are_unlit(self):
        assert ImagingAPI._lit_fraction(None, 8) == 0.0
        assert ImagingAPI._lit_fraction(np.empty((0,), dtype=np.uint8), 8) == 0.0


class TestProtocolWriterWiring:
    """The protocol writer no longer owns illumination knowledge: it
    commands the LED per step and the capture path derives the dark-floor
    expectation itself. A writer that re-derives and posts the fact back
    is a mirror needing manual sync -- the shape this fold retired."""

    def _run_capture(self, illumination_ma):
        from tests.test_audit_fixes import _bare_protocol_writer, _protocol_step

        writer = _bare_protocol_writer()
        scope = writer._scope
        scope.capabilities.has_turret = False
        scope.led_connected = False
        protocol = MagicMock()
        protocol.capture_root.return_value = ''
        writer.capture(
            save_folder='/tmp',
            step=_protocol_step(Illumination=illumination_ma),
            output_format='TIFF',
            protocol=protocol,
            enable_image_saving=True,
        )
        return scope.imaging._capture_and_wait_impl.call_args.kwargs

    def test_writer_posts_no_dark_floor_fact(self):
        for illumination_ma in (350.0, 0.0):
            kwargs = self._run_capture(illumination_ma)
            assert 'dark_floor_check' not in kwargs, (
                'the writer must not re-derive the dark-floor expectation; '
                f'it posted one at Illumination={illumination_ma}'
            )


class TestLiveCaptureConfigSeam:
    """Every key ui/composite_capture.py reads off a get_layer_configs()
    entry must exist in the dict get_layer_configs() actually emits.

    The emitted keys carry unit suffixes (illumination_ma, exposure_ms,
    gain_db) that are easy to drop when writing a new read site; a stale
    key is a KeyError on the capture worker, surfaced only as a generic
    task-failure popup. The schema below is taken from a real
    get_layer_configs() call, so this guard tracks the producer instead
    of pinning a copy of its key list.
    """

    def test_layer_config_keys_read_are_emitted(self):
        import ast
        from pathlib import Path

        from modules import config_helpers

        layer_settings = {
            'acquire': None,
            'video_config': {},
            'autofocus': False,
            'false_color': False,
            'ill_ma': 100.0,
            'sum': 1,
            'gain_db': 0.0,
            'auto_gain': False,
            'exp_ms': 33.0,
            'focus': 0.0,
        }
        schema = set(
            config_helpers.get_layer_configs({'BF': layer_settings}, specific_layers=['BF'])['BF']
        )

        src = (Path(__file__).resolve().parent.parent / 'ui' / 'composite_capture.py').read_text()
        read_keys = []
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'layer_configs'
            ):
                read_keys.append((node.slice.value, node.lineno))

        assert read_keys, (
            'expected layer_configs[...][key] reads in composite_capture.py; '
            'if they moved, retarget this guard to the new reader'
        )
        stale = [(key, line) for key, line in read_keys if key not in schema]
        assert stale == [], (
            f'keys read but never emitted by get_layer_configs: {stale}; '
            f'emitted keys: {sorted(schema)}'
        )

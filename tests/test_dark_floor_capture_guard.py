# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the capture path does with a dark frame: measures it, records it,
and DELIVERS it.

A dark frame is never refused. Pixel content cannot distinguish an LED
that failed to light from a genuinely dark sample or a deliberately dim
transmitted setting, so darkness decides nothing about the hardware --
capture readiness is owned by tracked illumination state
(``modules/frame_validity.py``), never inferred from pixels. Refusing a
dark frame destroyed real observations; the operator can see a dark
image perfectly well.

What survives is the measurement. A capture whose illumination is
commanded lit (a channel counts as lit only at strictly positive
current) retries until its budget expires -- which still heals the
stale pre-LED frame that issue #671 defect B reported -- and then
returns the frame it has, with a warning and a ``dark_saved`` fact on
``last_capture_info`` so a writer, an L2 caller or REST can tell a dark
capture from a lit one without re-measuring pixels. A nothing-commanded
capture is dark by design and is not measured at all.
``accept_dark=True`` skips the measurement for callers that expect dark
frames (autofocus sweeps, benchmark probes). Public ``get_image`` is the
ungated primitive.

The metric is lit-pixel COUNT against the frame's payload depth -- sparse
fluorescence (a few bright cells on a black background) must pass, and
the depth rule must match ``saturated_fraction`` (12-bit-in-uint16
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


class TestDarkFrameDelivery:
    def test_stale_dark_frame_healed_by_retry(self, dark_scope, monkeypatch):
        """The #671-B symptom: the first frame integrated before the LED
        lit; the very next frame is good. Retry must heal the capture."""
        frames = [_DARK, _LIT, _LIT]
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: frames.pop(0))

        out = dark_scope.imaging._get_image_impl(dark_floor_check=True, timeout_s=2.0)
        assert out is not None, 'retry must heal a transient dark frame'
        assert out.max() >= 120, 'the LIT retry frame must be returned, not the dark one'

    def test_persistent_dark_frames_are_saved_and_recorded(self, dark_scope, monkeypatch):
        """A channel commanded lit at real current, the camera delivering
        black frames throughout: the frame is DELIVERED, not refused.

        Destroying it would destroy a real observation -- a dim
        transmitted setting or a genuinely dark sample looks exactly like
        this, and the operator can see the dark image for themselves. The
        darkness is not silent: it is warned in the log and carried on
        last_capture_info so a writer, an L2 caller or REST can tell a
        dark frame from a lit one without re-measuring pixels."""
        dark_scope.illumination.led_on('BF', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        with patch.object(imaging_module, 'logger') as mock_logger:
            out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)

        assert out is not None, 'a dark frame under a lit channel must still be delivered'
        assert out.max() == _DARK.max(), 'the dark frame itself must come back untouched'
        warned = ' '.join(str(c).lower() for c in mock_logger.warning.call_args_list)
        assert 'dark' in warned, f'the darkness must be named in a warning; saw: {warned!r}'
        assert 'rejected' not in warned, (
            f'the capture was not rejected; the warning must not say so: {warned!r}'
        )
        assert dark_scope.imaging.last_capture_info.get('dark_saved') is True, (
            'a dark frame must be recorded as dark_saved, or no caller can tell'
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

    def test_accept_dark_suppresses_the_darkness_measurement(self, dark_scope, monkeypatch):
        """``accept_dark`` no longer overrides a rejection -- there is no
        rejection to override. What it still does is skip the measurement
        entirely, so an autofocus sweep (which expects dark planes by
        construction) neither pays for the scan nor files a dark_saved
        fact on every frame of the sweep.

        Both calls return the frame; only the recorded fact differs, and
        that difference is the whole remaining meaning of the argument."""
        dark_scope.illumination.led_on('Blue', 100)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        measured = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)
        assert measured is not None, 'a dark frame is delivered with or without the override'
        assert dark_scope.imaging.last_capture_info.get('dark_saved') is True, (
            'without the override the darkness must be measured and recorded'
        )

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.5, accept_dark=True)
        assert out is not None, 'accept_dark must still admit the dark frame'
        assert dark_scope.imaging.last_capture_info.get('dark_saved') is None, (
            'accept_dark skips the measurement, so no dark_saved fact is filed'
        )

    def test_lit_peer_records_a_dark_luminescence_capture(self, dark_scope, monkeypatch):
        """A luminescence capture taken while a peer channel is lit still
        arrives. The lit peer is why the darkness is worth recording --
        something was commanded to emit and the frame carries no signal --
        but that is a fact about the frame, not grounds to destroy it."""
        dark_scope.illumination.led_on('Red', 150)
        monkeypatch.setattr(dark_scope._camera_driver, 'get_array', lambda: _DARK)

        out = dark_scope.imaging._capture_and_wait_impl(timeout_s=0.3)
        assert out is not None, 'a black frame under a lit peer is delivered, not refused'
        assert dark_scope.imaging.last_capture_info.get('dark_saved') is True, (
            'a lit peer makes the darkness worth recording on the capture'
        )

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
        # The objective the frame is taken with, read at capture.
        scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
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
            'illumination_ma': 100.0,
            'sum': 1,
            'gain_db': 0.0,
            'auto_gain': False,
            'exposure_ms': 33.0,
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

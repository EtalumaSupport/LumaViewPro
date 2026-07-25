# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issues #671 (defect B) and #721: dark-floor frame
rejection, symmetric with the saturation guard.

Bug shape: frame acceptance in ImagingAPI.get_image judged content on one
tail only -- a near-saturated frame was rejected, but a near-black frame
(stale pre-LED integration, or an external camera consumer starving the
feed) was accepted and silently saved. The dark floor closes the gap:
with ``dark_floor_check=True`` a frame with essentially no pixel above
the floor is retried until timeout, then rejected loudly (None + warning).

The metric is lit-pixel COUNT against the frame's payload depth -- sparse
fluorescence (a few bright cells on a black background) must pass, and
the depth rule must match ``_saturated_fraction`` (12-bit-in-uint16
measures against 4095, not 65535).
"""

from unittest.mock import MagicMock, patch

import numpy as np

import modules.lumascope_api.imaging as imaging_module
from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.runtime_state import RuntimeState


def _sim_backed_imaging():
    """ImagingAPI on a connected SimulatedCamera with a minimal scope stub
    (same idiom as the saturation-guard tests)."""
    cam = SimulatedCamera()
    cam.connect()
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope.runtime_state = RuntimeState(scope)
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


_DARK = np.full((8, 8), 6, dtype=np.uint8)  # max 2.4% of full scale -- no signal
_LIT = np.full((8, 8), 120, dtype=np.uint8)


class TestDarkFloorRejection:
    def test_stale_dark_frame_healed_by_retry(self, monkeypatch):
        """The #671-B symptom: the first frame integrated before the LED
        lit; the very next frame is good. Retry must heal the capture."""
        imaging, cam = _sim_backed_imaging()
        frames = [_DARK, _LIT, _LIT]
        monkeypatch.setattr(cam, 'get_array', lambda: frames.pop(0))

        out = imaging.get_image(dark_floor_check=True, timeout_s=2.0)
        assert out is not None, 'retry must heal a transient dark frame'
        assert out.max() >= 120, 'the LIT retry frame must be returned, not the dark one'

    def test_persistent_dark_frames_rejected_loudly(self, monkeypatch):
        """The #721 symptom: the camera keeps delivering black frames.
        The capture must fail as None with a warning naming the dark
        rejection -- never a silently saved black file."""
        imaging, cam = _sim_backed_imaging()
        monkeypatch.setattr(cam, 'get_array', lambda: _DARK)

        with patch.object(imaging_module, 'logger') as mock_logger:
            out = imaging.get_image(dark_floor_check=True, timeout_s=0.3)

        assert out is None, 'a persistently dark frame must be rejected'
        warned = ' '.join(str(c).lower() for c in mock_logger.warning.call_args_list)
        assert 'dark' in warned and 'rejected' in warned, (
            f'rejection must be named in a warning; saw: {warned!r}'
        )

    def test_sparse_fluorescence_accepted(self, monkeypatch):
        """False-positive guard: a sparse fluorescence field (a handful of
        bright pixels on a black background) carries real signal and must
        pass -- the metric is lit-pixel count, not mean."""
        sparse = np.zeros((100, 100), dtype=np.uint8)
        sparse.flat[:4] = 200  # 4e-4 lit fraction, just above the minimum
        imaging, cam = _sim_backed_imaging()
        monkeypatch.setattr(cam, 'get_array', lambda: sparse)

        out = imaging.get_image(dark_floor_check=True, timeout_s=0.5)
        assert out is not None, 'sparse fluorescence must not be rejected as dark'

    def test_dark_by_design_exempt_when_check_off(self, monkeypatch):
        """A by-design-dark capture (brightfield at illumination 0) passes
        dark_floor_check=False and must be accepted unchanged."""
        imaging, cam = _sim_backed_imaging()
        monkeypatch.setattr(cam, 'get_array', lambda: _DARK)

        out = imaging.get_image(dark_floor_check=False, timeout_s=0.5)
        assert out is not None
        assert out.max() == 6, 'the dark frame itself must be returned untouched'


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
    """The protocol writer owns the illumination knowledge: a step driving
    its LED gates the dark floor ON; an illumination-0 step (BF dark by
    design) gates it OFF."""

    def _run_capture(self, illumination_ma):
        from tests.test_audit_fixes import _bare_protocol_writer, _protocol_step

        writer = _bare_protocol_writer()
        scope = writer._scope
        scope.motion.has_turret.return_value = False
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
        return scope.imaging.capture_and_wait.call_args.kwargs

    def test_lit_step_gates_dark_floor_on(self):
        assert self._run_capture(350.0)['dark_floor_check'] is True

    def test_illumination_zero_step_gates_dark_floor_off(self):
        assert self._run_capture(0.0)['dark_floor_check'] is False

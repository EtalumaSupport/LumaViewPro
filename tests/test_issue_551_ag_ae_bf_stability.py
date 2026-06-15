# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#551 regression: AG/AE BF stability -- ROI 50% centered + MinimizeGain.

Bug
---
Customer reported AutoGain/Exposure instability in BF -- the camera
controller was bouncing between gain and exposure values, producing
flickering brightness in the live preview. Two contributing factors
in drivers/pyloncamera.py::init_auto_gain_focus:

1. AutoFunction ROI was set to (Width.Max - 2*offsetX, Height.Max -
   2*offsetY) -- effectively full-frame. The controller sampled plate
   edges + dust + uneven illumination, driving oscillation.

2. AutoFunctionProfile was set to 'MinimizeExposureTime', which makes
   gain track noise; on BF where light is bright + consistent, this
   manifested as visible gain bouncing.

Fix
---
- ROI now 50%x50% centered (Width.Max/2, Height.Max/2 with centered
  offsets). 16-pixel aligned per Basler ace 2 / dart step granularity.
- AutoFunctionProfile switched to 'MinimizeGain' (pin gain low,
  adjust exposure first).

Test approach
-------------
Behavioral: the REAL init_auto_gain_focus runs against
camera_fakes.FakeAutoRoiCamera, a stateful simulator that enforces the
Basler node interdependency (Offset.Max = sensor extent - current ROI
size; Width/Height.Max shrink while an offset is applied; SetValue
outside [0, Max] raises like the SDK). The dart-family geometry that
originally rejected the centered offset is reproduced by construction,
so the offset-zero-before-sizing ordering and the per-node Max clamps
are load-bearing, not source-text pins.

Bench verification gates the actual stability claim; these tests are
the regression catch for the structural fix.
"""

from __future__ import annotations

from tests.camera_fakes import auto_roi_pylon_camera

# Geometry for the default simulator (ace-2-like 3536x2624 sensor):
# width 3536 // 2 = 1768 -> 16-px align-down 1760; height 2624 // 2 =
# 1312 (already aligned); centered offsets (3536-1760)//2 = 888 -> 880
# and (2624-1312)//2 = 656.
_EXPECTED_DEFAULT_ROI = (1760, 1312, 880, 656)


def test_init_auto_gain_focus_uses_minimize_gain_profile():
    """Every AutoFunctionProfile write must be 'MinimizeGain' -- the
    'MinimizeExposureTime' profile caused BF gain bouncing."""
    cam = auto_roi_pylon_camera()
    cam.init_auto_gain_focus()
    profile_writes = [value for node, value in cam.active.calls if node == 'AutoFunctionProfile']
    assert profile_writes, 'init_auto_gain_focus must set AutoFunctionProfile. (#551)'
    assert profile_writes == ['MinimizeGain'], (
        f'AutoFunctionProfile must be set to "MinimizeGain" for BF '
        f'stability; got {profile_writes}. (#551)'
    )


def test_init_auto_gain_focus_sets_half_centered_aligned_roi():
    """The committed ROI must be the 50% centered crop, 16-px aligned --
    not the old full-frame derivation that sampled plate edges."""
    cam = auto_roi_pylon_camera()
    cam.init_auto_gain_focus()
    roi = cam.active.roi
    assert roi == _EXPECTED_DEFAULT_ROI, (
        f'Expected 50% centered 16-px-aligned ROI {_EXPECTED_DEFAULT_ROI}; got {roi}. (#551)'
    )
    assert all(value % 16 == 0 for value in roi), (
        f'Every ROI dimension must be 16-px aligned (Basler ace 2 / dart '
        f'step granularity); got {roi}. (#551)'
    )


def test_init_auto_gain_focus_zeroes_offsets_before_sizing():
    """Pylon node interdependency: a pre-existing non-zero offset caps
    the achievable Width/Height (Width.Max = sensor - offset). Starting
    from a dart-shaped state with large committed offsets, only the
    zero-offsets-first ordering reaches the full 50% centered crop --
    skipping the zero step (or sizing first) either raises in the
    simulator or commits a smaller, off-center ROI."""
    cam = auto_roi_pylon_camera(initial_offset_x=1800, initial_offset_y=1400)
    cam.init_auto_gain_focus()
    roi = cam.active.roi
    assert roi == _EXPECTED_DEFAULT_ROI, (
        f'With pre-existing offsets the method must still commit the '
        f'full centered crop {_EXPECTED_DEFAULT_ROI} (zero offsets '
        f'first, then size, then re-center); got {roi}.'
    )
    profile_writes = [value for node, value in cam.active.calls if node == 'AutoFunctionProfile']
    assert profile_writes == ['MinimizeGain'], (
        'The configuration pass must complete (a mid-sequence '
        'OutOfRange raise would skip the profile write).'
    )


def test_init_auto_gain_focus_clamps_to_autofunction_roi_node_max():
    """Defensive clamp against the AutoFunctionROI* node's own Max --
    the dart family reports tighter bounds on these nodes than on the
    sensor's Width / Height proper. Without the clamp, the 50% crop
    derived from Width.Max would exceed AutoFunctionROIWidth.Max and
    raise OutOfRangeException."""
    cam = auto_roi_pylon_camera(roi_w_cap=1000, roi_h_cap=800)
    cam.init_auto_gain_focus()
    roi_w, roi_h, off_x, off_y = cam.active.roi
    assert roi_w == 992 and roi_h == 800, (
        f'ROI must clamp to the AutoFunctionROI node caps (992x800 '
        f'after 16-px alignment of caps 1000x800); got {roi_w}x{roi_h}.'
    )
    assert (off_x, off_y) == (1264, 912), (
        f'Offsets must re-center the clamped ROI (and 16-px align); got ({off_x}, {off_y}).'
    )
    profile_writes = [value for node, value in cam.active.calls if node == 'AutoFunctionProfile']
    assert profile_writes == ['MinimizeGain']

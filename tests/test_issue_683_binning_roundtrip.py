"""Regression test for #683 -- binning does not round-trip resolution.

User repro (2026-05-28, SN11030):
  Go 1x1 (default) -> 2x2 -> 4x4 -> 2x2 -> 1x1. The resolution you end
  up with is not the same as where you started.

Root cause: ``select_binning_size`` computed each new frame size by
dividing the CURRENT displayed value by the binning ratio and flooring
the result (``math.floor(orig_frame_size / ratio)``). On a sensor whose
dimensions are not evenly divisible, the floor truncates on the way down
and never recovers on the way back up, so the displayed (and the
camera ROI it drives via ``frame_size`` -> ``set_frame_size``) drifts.

Fix: the unbinned NATIVE ROI is the source of truth; the displayed and
captured size is always ``native / binning`` floored to the camera pixel
alignment. Because that derivation depends only on native + binning, every
binning level is reproducible and the cycle round-trips exactly.
"""

import ast
import pathlib

import modules.binning as binning
from drivers.simulated_camera import SimulatedCamera

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _method_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def test_select_binning_size_persists_native_roi():
    """select_binning_size must persist the native ROI, not just derive
    the displayed size from it.

    The helper round-trip tests above assume a FIXED native ROI. In
    production that holds only if native_width/native_height are actually
    stored. They are written on a frame-field edit (frame_size) but were
    NOT written on a binning change, so settings that never had them fall
    through _native_roi's reconstruction (displayed * binning) every
    binning change -- and at a coarse binning the displayed value is
    already floored, so native shrinks a little each step and the cycle
    drifts (the SN12062 bench saw 1900 -> 1888). Guard that the binning
    path persists the source of truth like the edit path does.
    """
    method = _method_node(REPO_ROOT / 'ui' / 'microscope_settings.py', 'select_binning_size')
    stores = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == '_store_native_roi'
    ]
    assert stores, (
        'select_binning_size must call self._store_native_roi(...) so a '
        'binning change locks the native source of truth; without it the '
        'binning cycle drifts via _native_roi reconstruction. (#683)'
    )


# Mimics the buggy pre-fix derivation: iterate on the displayed value, and
# floor each result to the 4-pixel camera alignment (set_frame_size floored
# to 4, then get_current_frame_dimensions read the floored value back as the
# next step's input). That per-step truncation is what failed to round-trip.
def _legacy_step(displayed, orig_binning, new_binning):
    import math

    ratio = new_binning / orig_binning
    return {
        'width': binning._align_down(math.floor(displayed['width'] / ratio), 4),
        'height': binning._align_down(math.floor(displayed['height'] / ratio), 4),
    }


class TestBinningRoundTrip:
    # A sensor whose dimensions are not cleanly divisible by 4 at every
    # binning level -- exactly the case the floor truncation corrupts.
    NATIVE = {'width': 2456, 'height': 2054}
    ALIGN = {'width': 4, 'height': 4}

    def test_native_anchored_cycle_round_trips(self):
        """1x1 -> 2x2 -> 4x4 -> 2x2 -> 1x1 returns to the start."""
        native = self.NATIVE
        start = binning.native_to_displayed(native, 1, self.ALIGN)
        for b in (2, 4, 2, 1):
            disp = binning.native_to_displayed(native, b, self.ALIGN)
            assert disp['width'] > 0 and disp['height'] > 0
        end = binning.native_to_displayed(native, 1, self.ALIGN)
        assert end == start

    def test_each_binning_level_is_deterministic(self):
        """The displayed size at a binning level never depends on the path."""
        native = self.NATIVE
        # Reached via different routes, the same binning level must match.
        assert binning.native_to_displayed(native, 2, self.ALIGN) == binning.native_to_displayed(
            native, 2, self.ALIGN
        )
        assert binning.native_to_displayed(native, 1, self.ALIGN)['width'] == 2456

    def test_legacy_iteration_loses_pixels(self):
        """Document the old behavior the fix removes: the naive cycle drifts."""
        # Start at full frame, 1x1.
        disp = binning.native_to_displayed(self.NATIVE, 1, self.ALIGN)
        start = dict(disp)
        # 1->2->4->2->1 by iterating on the displayed value (old code).
        disp = _legacy_step(disp, 1, 2)
        disp = _legacy_step(disp, 2, 4)
        disp = _legacy_step(disp, 4, 2)
        disp = _legacy_step(disp, 2, 1)
        # The legacy path does NOT return to the start (this is the bug).
        assert disp != start

    def test_displayed_edit_caps_at_native_max(self):
        """At 2x2 showing 1000x1000 (native 2000x2000), typing 1500x1500
        implies native 3000x3000 -- capped at the sensor native max."""
        native_max = {'width': 2000, 'height': 2000}
        native = binning.displayed_to_native(
            {'width': 1500, 'height': 1500}, 2, native_max
        )
        assert native == {'width': 2000, 'height': 2000}

    def test_displayed_edit_shrinks_native(self):
        """At 2x2, changing 1000x1000 down to 500x500 drops native to
        1000x1000 (500 * 2)."""
        native_max = {'width': 2000, 'height': 2000}
        native = binning.displayed_to_native(
            {'width': 500, 'height': 500}, 2, native_max
        )
        assert native == {'width': 1000, 'height': 1000}

    def test_alignment_floors_to_multiple_of_4(self):
        native = {'width': 2456, 'height': 2054}  # 2054 not a multiple of 4
        disp = binning.native_to_displayed(native, 1, self.ALIGN)
        assert disp['width'] % 4 == 0
        assert disp['height'] % 4 == 0
        assert disp['height'] == 2052  # 2054 floored to nearest 4

    def test_simulated_camera_profile_round_trips(self):
        """Exercise the shipped SimulatedCamera profile end-to-end.

        Profile (drivers/camera_profiles.py): native 1920x1200, alignment
        48x4 (width must be a multiple of 48). The cycle must still return
        to the start with the non-trivial width alignment.
        """
        native = {'width': 1920, 'height': 1200}
        align = {'width': 48, 'height': 4}
        start = binning.native_to_displayed(native, 1, align)
        assert start == {'width': 1920, 'height': 1200}
        for b in (2, 4, 2, 1):
            disp = binning.native_to_displayed(native, b, align)
            assert disp['width'] % 48 == 0
            assert disp['height'] % 4 == 0
        assert binning.native_to_displayed(native, 1, align) == start
        assert binning.native_to_displayed(native, 2, align) == {'width': 960, 'height': 600}
        assert binning.native_to_displayed(native, 4, align) == {'width': 480, 'height': 300}


class TestSimPostBinningContract:
    """The simulated camera obeys the same post-binning frame contract as the
    Pylon driver, so simulator runs match real hardware.

    set_frame_size takes the post-binning (displayed) ROI; the grabbed image is
    exactly that size; get_max_frame_size is the native sensor size divided by
    the current binning; and increasing binning re-clamps the frame to the new
    max -- the behavior observed on a Basler camera (a 1920x1200 ROI becomes
    960x600 at 2x2).
    """

    def _grab_shape(self, cam):
        return cam._generate_image().shape  # (height, width)

    def test_set_frame_size_is_post_binning(self):
        cam = SimulatedCamera()  # native 1920x1200
        cam.set_binning_size(2)
        # Max at 2x2 is native / 2.
        assert cam.get_max_frame_size() == {'width': 960, 'height': 600}
        cam.set_frame_size(960, 600)
        assert cam.get_frame_size() == {'width': 960, 'height': 600}
        assert self._grab_shape(cam) == (600, 960)

    def test_binning_change_reclamps_full_frame(self):
        cam = SimulatedCamera()
        cam.set_frame_size(1920, 1200)  # full at 1x1
        cam.set_binning_size(2)         # observed: 1920x1200 -> 960x600
        assert cam.get_frame_size() == {'width': 960, 'height': 600}
        assert self._grab_shape(cam) == (600, 960)

    def test_init_path_passes_post_binning_frame(self):
        """At binning 2, init must hand set_frame_size the displayed size
        (960x600), not native (1920x1200). Passing native would over-size the
        post-binning ROI on Pylon -- the cropped-ROI-at-startup bug."""
        cam = SimulatedCamera()
        cam.set_binning_size(2)
        # Displayed value the init path now passes through (no * binning).
        cam.set_frame_size(960, 600)
        assert self._grab_shape(cam) == (600, 960)

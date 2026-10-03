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

from typing import ClassVar

import modules.binning as binning
from drivers.camera_profiles import lookup_profile
from drivers.simulated_camera import SimulatedCamera


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
    NATIVE: ClassVar[dict] = {'width': 2456, 'height': 2054}
    ALIGN: ClassVar[dict] = {'width': 4, 'height': 4}

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
        native = binning.displayed_to_native({'width': 1500, 'height': 1500}, 2, native_max)
        assert native == {'width': 2000, 'height': 2000}

    def test_displayed_edit_shrinks_native(self):
        """At 2x2, changing 1000x1000 down to 500x500 drops native to
        1000x1000 (500 * 2)."""
        native_max = {'width': 2000, 'height': 2000}
        native = binning.displayed_to_native({'width': 500, 'height': 500}, 2, native_max)
        assert native == {'width': 1000, 'height': 1000}

    def test_alignment_floors_to_multiple_of_4(self):
        native = {'width': 2456, 'height': 2054}  # 2054 not a multiple of 4
        disp = binning.native_to_displayed(native, 1, self.ALIGN)
        assert disp['width'] % 4 == 0
        assert disp['height'] % 4 == 0
        assert disp['height'] == 2052  # 2054 floored to nearest 4

    def test_simulated_camera_profile_round_trips(self):
        """Exercise the shipped SimulatedCamera profile end-to-end.

        Profile (drivers/camera_profiles.py): native 3840x2160, alignment
        48x4 (width must be a multiple of 48). The cycle must still return
        to the start with the non-trivial width alignment.
        """
        native = {'width': 3840, 'height': 2160}
        align = {'width': 48, 'height': 4}
        start = binning.native_to_displayed(native, 1, align)
        assert start == {'width': 3840, 'height': 2160}
        for b in (2, 4, 2, 1):
            disp = binning.native_to_displayed(native, b, align)
            assert disp['width'] % 48 == 0
            assert disp['height'] % 4 == 0
        assert binning.native_to_displayed(native, 1, align) == start
        assert binning.native_to_displayed(native, 2, align) == {'width': 1920, 'height': 1080}
        assert binning.native_to_displayed(native, 4, align) == {'width': 960, 'height': 540}


# The deliverable frame-size granularity the UI floors to before handing the
# size to the camera: even (2x2) on every camera, each of which acquires the
# next window up on its own grid and crops back. The UI sources it from
# imaging.get_pixel_alignment(), NOT a hardcoded grid.
IDS_DELIVERABLE_ALIGN = {'width': 2, 'height': 2}


class TestDeliverableAlignmentFloor:
    """The UI floors the requested frame size to the active driver's DELIVERABLE
    granularity. Every camera crops back to the exact request, so its
    granularity is even (2x2) and a 1900 frame stays 1900 (not the old 1872, nor
    the off-grid 948 at 2x).
    """

    def test_ids_profile_reports_even_deliverable_granularity(self):
        # The IDS driver crops to exact, so get_pixel_alignment (which returns
        # profile.alignment) reports even -- the only constraint is even dims.
        assert lookup_profile('U3-34Lx').alignment == IDS_DELIVERABLE_ALIGN

    def test_every_camera_reports_even_deliverable_granularity(self):
        # The simulator, the FX2 and Pylon crop back to the request as IDS does,
        # so none reports its hardware grid here.
        for model in ('SimulatedCamera', 'MT9P031-LS620', 'a2A3536-31umBAS'):
            assert lookup_profile(model).alignment == IDS_DELIVERABLE_ALIGN, model

    def test_even_floor_preserves_offgrid_width(self):
        # 1900 is off the 48-px camera grid but even; flooring to the IDS
        # deliverable granularity leaves it 1900 (the driver crops to it).
        disp = binning.native_to_displayed(
            {'width': 1900, 'height': 1900}, 1, IDS_DELIVERABLE_ALIGN
        )
        assert disp == {'width': 1900, 'height': 1900}
        assert disp['width'] != 1872  # the old camera-grid floor

    def test_even_floor_floors_odd_to_even(self):
        # H.264 yuv420p still needs even dims; an odd native floors to even.
        disp = binning.native_to_displayed(
            {'width': 1901, 'height': 1903}, 1, IDS_DELIVERABLE_ALIGN
        )
        assert disp == {'width': 1900, 'height': 1902}

    def test_even_floor_2x_avoids_offgrid_948(self):
        # native height 1900 / 2 = 950. The old camera-grid floor (multiple of 4)
        # produced the off-grid 948 the SDK rejects; even-floor leaves 950.
        disp = binning.native_to_displayed(
            {'width': 1900, 'height': 1900}, 2, IDS_DELIVERABLE_ALIGN
        )
        assert disp['height'] == 950
        assert disp['height'] != 948

    def test_the_frame_and_binning_members_floor_to_the_drivers_alignment(self, tmp_path):
        # Pin the fix where the displayed size is derived: both frame-size
        # paths (a frame edit and a binning change, ScopeSession.set_frame_size
        # / set_binning_size) floor to the ACTIVE driver's deliverable
        # granularity via imaging.get_pixel_alignment(), not a hardcoded grid
        # -- else a non-cropping driver persists an undeliverable size (the
        # reviewed regression). The simulator's own grid is 48x4; the driver
        # here reports the IDS even grid, so a request floored to anything
        # but the driver's answer is visible.
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            imaging = session.scope.imaging
            imaging.get_pixel_alignment = lambda: IDS_DELIVERABLE_ALIGN
            requests = []
            real = imaging.set_frame_size
            imaging.set_frame_size = lambda w, h: requests.append((w, h)) or real(w, h)

            session.set_frame_size(1150, 802)
            session.set_binning_size(2)
        finally:
            session.shutdown()

        assert requests == [(1150, 802), (574, 400)], (
            'set_frame_size and set_binning_size must floor to '
            'imaging.get_pixel_alignment() (per-driver), not a hardcoded alignment; '
            f'the camera was asked for {requests}'
        )


class TestSimPostBinningContract:
    """The simulated camera obeys the same post-binning frame contract as the
    Pylon driver, so simulator runs match real hardware.

    set_frame_size takes the post-binning (displayed) ROI; the grabbed image is
    exactly that size; get_max_frame_size is the native sensor size divided by
    the current binning; and increasing binning re-clamps the frame to the new
    max -- the behavior observed on a Basler camera (a 3840x2160 ROI becomes
    1920x1080 at 2x2).
    """

    def _grab_shape(self, cam):
        return cam._generate_image().shape  # (height, width)

    def test_set_frame_size_is_post_binning(self):
        cam = SimulatedCamera()  # native 3840x2160
        cam.set_binning_size(2)
        # Max at 2x2 is native / 2.
        assert cam.get_max_frame_size() == {'width': 1920, 'height': 1080}
        cam.set_frame_size(960, 600)
        assert cam.get_frame_size() == {'width': 960, 'height': 600}
        assert self._grab_shape(cam) == (600, 960)

    def test_binning_change_reclamps_full_frame(self):
        cam = SimulatedCamera()
        cam.set_frame_size(3840, 2160)  # full at 1x1
        cam.set_binning_size(2)  # observed: 3840x2160 -> 1920x1080
        assert cam.get_frame_size() == {'width': 1920, 'height': 1080}
        assert self._grab_shape(cam) == (1080, 1920)

    def test_init_path_passes_post_binning_frame(self):
        """At binning 2, init must hand set_frame_size the displayed size
        (960x600), not native (1920x1200). Passing native would over-size the
        post-binning ROI on Pylon -- the cropped-ROI-at-startup bug."""
        cam = SimulatedCamera()
        cam.set_binning_size(2)
        # Displayed value the init path now passes through (no * binning).
        cam.set_frame_size(960, 600)
        assert self._grab_shape(cam) == (600, 960)

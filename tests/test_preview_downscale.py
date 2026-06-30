"""Tests for ``modules.image_utils.decimate_for_preview``.

The live-view FPS ceiling was the full-resolution Kivy ``blit_buffer`` on the
main thread, serialized against capture/convert on the GIL. The preview now
downscales each frame to its on-screen (contain-fit) size before the blit.
These tests pin the contract: identity when no downscale applies, and an
aspect-preserving shrink to the displayed size at any ratio otherwise (so a
frame only slightly larger than the widget still downscales, unlike an
integer-step factor that rounds to 1 for any frame under 2x the widget).
"""

from __future__ import annotations

import numpy as np

from modules.image_utils import (
    click_offset_to_um,
    decimate_for_preview,
    scaled_preview_target,
)


class TestDecimateForPreview:
    def test_none_target_returns_input_unchanged(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, None)
        assert out is img

    def test_none_image_returns_none(self):
        assert decimate_for_preview(None, (700, 700)) is None

    def test_color_frame_skipped(self):
        # 3-D arrays go through the RGB/bullseye path, not this luminance blit.
        img = np.zeros((1900, 1900, 3), dtype=np.uint8)
        out = decimate_for_preview(img, (300, 300))
        assert out is img

    def test_zero_target_returns_input_unchanged(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        assert decimate_for_preview(img, (0, 500)) is img
        assert decimate_for_preview(img, (500, 0)) is img

    def test_frame_smaller_than_target_unchanged(self):
        # step would be 0/1 -> no downscale (never upscale a small frame).
        img = np.zeros((480, 640), dtype=np.uint8)
        out = decimate_for_preview(img, (1280, 1024))
        assert out is img

    def test_square_frame_downscales_to_widget(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (700, 700))  # factor 700/1900 -> 700
        assert out.shape == (700, 700)
        assert out.dtype == np.uint8

    def test_square_frame_smaller_widget(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (300, 300))  # factor 300/1900 -> 300
        assert out.shape == (300, 300)

    def test_near_widget_frame_still_downscales(self):
        # The integer-step contract left any frame under 2x the widget at full
        # resolution (1900 // 1000 == 1); contain-fit downscales it.
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (1000, 1000))  # factor 1000/1900 -> 1000
        assert out.shape == (1000, 1000)

    def test_result_matches_target_on_limiting_axis(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        for box in (200, 300, 450, 700, 950, 1000, 1700):
            out = decimate_for_preview(img, (box, box))
            assert out.shape == (box, box)

    def test_non_square_widget_fits_to_short_axis(self):
        # A wide-but-short widget shows the square frame letterboxed to its
        # height, so the frame downscales to that displayed (height) size.
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (1800, 200))  # factor min(1800,200)/1900
        assert out.shape == (200, 200)

    def test_non_square_frame_preserves_aspect(self):
        img = np.zeros((1000, 2000), dtype=np.uint8)  # H=1000, W=2000
        out = decimate_for_preview(img, (500, 500))  # factor min(500/2000,500/1000)=0.25
        assert out.shape == (250, 500)  # aspect 2:1 preserved

    def test_output_is_contiguous_and_blittable(self):
        # blit_buffer needs len(bytes) == w*h for the declared texture size.
        img = (np.arange(1900 * 1900, dtype=np.uint32) % 256).astype(np.uint8).reshape(1900, 1900)
        out = decimate_for_preview(img, (600, 600))
        assert out.flags['C_CONTIGUOUS']
        assert len(out.tobytes()) == out.shape[0] * out.shape[1]

    def test_area_interpolation_preserves_uniform_value(self):
        # A flat field must downscale to the same flat value (area-average).
        img = np.full((1900, 1900), 137, dtype=np.uint8)
        out = decimate_for_preview(img, (600, 600))
        assert np.all(out == 137)


class TestScaledPreviewTarget:
    def test_none_base_returns_none(self):
        assert scaled_preview_target(None, 2.5) is None

    def test_scale_one_is_widget_box(self):
        assert scaled_preview_target((700, 700), 1.0) == (700, 700)

    def test_zoom_in_scales_up(self):
        assert scaled_preview_target((700, 600), 2.0) == (1400, 1200)

    def test_zoom_out_clamped_to_box(self):
        # A zoomed-out view is already small on screen; never over-shrink it.
        assert scaled_preview_target((700, 700), 0.5) == (700, 700)

    def test_bad_scale_falls_back_to_one(self):
        assert scaled_preview_target((700, 700), None) == (700, 700)
        assert scaled_preview_target((700, 700), 0.0) == (700, 700)


class TestZoomPreservesFullResolution:
    """The load-bearing property: at 1:1 zoom (Scatter scale = sensor/widget),
    the target reaches sensor size and decimation collapses to identity, so the
    1:1 button shows true full-resolution pixels -- not a downscaled blur."""

    def test_one_to_one_zoom_yields_full_resolution(self):
        sensor = 1900
        widget = 700
        img = np.zeros((sensor, sensor), dtype=np.uint8)
        # one2one_image sets scale = max(sensor/widget) on the enclosing Scatter.
        scale = sensor / widget
        target = scaled_preview_target((widget, widget), scale)
        out = decimate_for_preview(img, target)
        assert out is img  # identity -> full-resolution blit

    def test_one_to_one_full_resolution_survives_float_floor(self):
        # widget*sensor/widget lands a sub-LSB below sensor for ~6% of pairs;
        # scaled_preview_target rounds so the 1:1 view stays exactly full-res
        # instead of doing a pointless 1-pixel contain-fit shrink.
        for widget, sensor in ((850, 975), (218, 4064), (1447, 1470)):
            img = np.zeros((sensor, sensor), dtype=np.uint8)
            target = scaled_preview_target((widget, widget), sensor / widget)
            assert decimate_for_preview(img, target) is img

    def test_default_fit_view_downscales(self):
        # Default (scale 1.0, fit-to-window) is where the FPS win must apply.
        img = np.zeros((1900, 1900), dtype=np.uint8)
        target = scaled_preview_target((700, 700), 1.0)  # (700, 700)
        out = decimate_for_preview(img, target)
        assert out.shape == (700, 700)


class TestClickOffsetToUm:
    """Click-to-center / cursor-readout geometry must scale by the full-resolution
    sensor frame, never the (preview-downscaled) display texture -- using the
    downscaled size under-moves the stage by the downscale factor."""

    def test_center_click_is_zero_offset(self):
        assert click_offset_to_um(950, 1900, 1900, 0.5) == 0.0

    def test_right_edge_is_half_sensor_width(self):
        # Clicking the displayed image's right edge moves the stage half the
        # field of view: (frame_extent / 2) * pixel_size_um, in SENSOR microns.
        assert click_offset_to_um(1900, 1900, 1900, 0.5) == 950 * 0.5

    def test_left_edge_is_negative_half_sensor_width(self):
        assert click_offset_to_um(0, 1900, 1900, 0.5) == -950 * 0.5

    def test_invariant_to_display_size(self):
        # Same physical click fraction (0.75) on the same 1900-px frame must give
        # the same micron distance whether the widget is large or small -- the
        # result depends on the full-resolution frame, not the on-screen size.
        big = click_offset_to_um(0.75 * 1000, 1000, 1900, 0.5)
        small = click_offset_to_um(0.75 * 500, 500, 1900, 0.5)
        assert big == small

    def test_downscaled_extent_would_under_report(self):
        # The regression guard: feeding the downscaled texture extent (700)
        # instead of the full-resolution frame (1900) shrinks the distance.
        full = click_offset_to_um(700, 700, 1900, 0.5)  # right edge, full-res
        downscaled = click_offset_to_um(700, 700, 700, 0.5)  # the old bug
        assert abs(downscaled) < abs(full)

    def test_degenerate_extent_returns_zero(self):
        assert click_offset_to_um(0, 0, 1900, 0.5) == 0.0
        assert click_offset_to_um(10, -5, 1900, 0.5) == 0.0

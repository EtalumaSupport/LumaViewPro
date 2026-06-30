"""Tests for ``modules.image_utils.decimate_for_preview``.

The live-view FPS ceiling was the full-resolution Kivy ``blit_buffer`` on the
main thread, serialized against capture/convert on the GIL. The preview now
downscales each frame to ~widget size before the blit. These tests pin the
decimation contract: identity when no downscale applies, aspect-preserving
integer-step shrink otherwise, and never below the target on the limiting axis.
"""

from __future__ import annotations

import numpy as np

from modules.image_utils import decimate_for_preview, scaled_preview_target


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

    def test_square_frame_step_two(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (700, 700))  # 1900 // 700 == 2
        assert out.shape == (950, 950)
        assert out.dtype == np.uint8

    def test_square_frame_larger_factor(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (300, 300))  # 1900 // 300 == 6
        assert out.shape == (316, 316)

    def test_result_never_below_target_on_limiting_axis(self):
        img = np.zeros((1900, 1900), dtype=np.uint8)
        for box in (200, 300, 450, 700, 950):
            out = decimate_for_preview(img, (box, box))
            assert out.shape[0] >= box and out.shape[1] >= box

    def test_non_square_widget_uses_limiting_axis(self):
        # A wide-but-short widget must not shrink below its width.
        img = np.zeros((1900, 1900), dtype=np.uint8)
        out = decimate_for_preview(img, (1800, 200))  # min(1900//200, 1900//1800)=1
        assert out is img

    def test_non_square_frame_preserves_aspect(self):
        img = np.zeros((1000, 2000), dtype=np.uint8)  # H=1000, W=2000
        out = decimate_for_preview(img, (500, 500))  # step=min(1000//500,2000//500)=2
        assert out.shape == (500, 1000)

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

    def test_default_fit_view_downscales(self):
        # Default (scale 1.0, fit-to-window) is where the FPS win must apply.
        img = np.zeros((1900, 1900), dtype=np.uint8)
        target = scaled_preview_target((700, 700), 1.0)
        out = decimate_for_preview(img, target)
        assert out.shape == (950, 950)

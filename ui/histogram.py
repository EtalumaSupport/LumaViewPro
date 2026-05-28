# Copyright Etaluma, Inc.
import logging

import numpy as np

from kivy.graphics import Color, Mesh
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget

import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.histogram')


class Histogram(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] Histogram.__init__()')
        if self.bg_color is None:
            self.bg_color = (1, 1, 1, 1)

        self.hist_range_set = False
        self.edges = [0, 255]
        self.stablize = 0.3
        self._mesh = None
        self._mesh_color = None

    def _is_displayed(self, ctx) -> bool:
        """Whether this histogram is actually visible on screen.

        True only when the settings drawer is open, this layer's
        accordion is expanded, and the layer's camera controls (which
        contain the histogram) are shown. Any failure to resolve the
        widgets is treated as not-displayed so we err toward skipping
        the work, never toward computing for an off-screen widget.
        """
        image_settings = getattr(ctx, 'image_settings', None)
        if image_settings is None or self.layer is None:
            return False
        try:
            # Settings drawer collapsed: the whole panel is off-screen.
            if image_settings.ids['toggle_imagesettings'].state == 'normal':
                return False
            # This layer's accordion collapsed: another layer is showing.
            item = image_settings.accordion_item_lookup(layer=self.layer)
            if item is None or item.collapse:
                return False
            # Camera controls (which host the histogram) hidden for this layer.
            layer_obj = image_settings.layer_lookup(layer=self.layer)
            if layer_obj is not None and not layer_obj.show_camera_controls:
                return False
        except (KeyError, AttributeError):
            return False
        return True

    def histogram(self, *args):
        ctx = _app_ctx.ctx

        # Skip when live preview is paused. cam_toggle in main_display
        # sets scope_display.play=False and pauses the display thread,
        # but the 0.5 s histogram Clock keeps ticking; without this
        # guard each tick fetches a frame from the camera buffer, builds
        # a 128-bin mesh, and uploads to the GPU for nothing.
        scope_display = getattr(ctx, 'scope_display', None)
        if scope_display is not None and not scope_display.play:
            return

        # Skip during protocol acquisition. The histogram contends with
        # the capture / protocol pipeline for get_image_from_buffer and
        # the texture isn't user-visible during a run anyway.
        protocol_running = getattr(ctx, 'protocol_running', None)
        if protocol_running is not None and protocol_running.is_set():
            return

        # The histogram is a live-image tool: compute only when it is
        # actually on screen. The 0.5 s Clock keeps ticking in states
        # where the widget is not displayed, so guard the expensive work
        # (camera read + 128-bin mesh + GPU upload) on the real display
        # conditions rather than trusting Clock scheduling alone.
        if not self._is_displayed(ctx):
            return

        bins = 128

        if ctx.scope.imaging.camera_active:
            image, _ = ctx.scope.imaging.get_image_from_buffer(force_to_8bit=True)
            if image is None:
                return

            # Subsample image for faster histogram (~16x fewer pixels)
            sampled = image[::4, ::4]
            counts, _ = np.histogram(sampled, bins=bins, range=(0, 256))

            ctx.viewer.black = 0.0
            ctx.viewer.white = 1.0

            # Compute bar heights with vectorized numpy
            layer_obj = ctx.image_settings.layer_lookup(layer=self.layer)
            use_log = layer_obj.ids['logHistogram_id'].active

            if use_log:
                heights = np.log(counts.astype(np.float64) + 1)
            else:
                heights = counts.astype(np.float64)

            max_height = heights.max()
            if max_height <= 0:
                self.canvas.clear()
                self._mesh = None
                self._mesh_color = None
                return

            x = self.x
            y = self.y
            w = self.width
            h = self.height
            bin_size = w / bins
            scale = h / max_height
            heights = heights * scale

            # Build triangle strip vertices: 2 triangles per bar (bottom-left, top-left, bottom-right, top-right)
            # Each vertex: (x, y, u, v) where u,v are texture coords (unused but required by Mesh)
            vertices = []
            for i in range(bins):
                bx = x + i * bin_size
                bar_h = heights[i]
                vertices.extend(
                    [
                        bx,
                        y,
                        0,
                        0,
                        bx,
                        y + bar_h,
                        0,
                        0,
                        bx + bin_size,
                        y,
                        0,
                        0,
                        bx + bin_size,
                        y + bar_h,
                        0,
                        0,
                    ]
                )

            # Build indices for individual triangle strips per bar
            indices = []
            for i in range(bins):
                base = i * 4
                indices.extend([base, base + 1, base + 2, base + 1, base + 2, base + 3])

            r, b, g, a = self.bg_color
            self.hist = (counts, None)
            if self._mesh is None:
                self.canvas.clear()
                with self.canvas:
                    self._mesh_color = Color(r, b, g, a / 2)
                    self._mesh = Mesh(vertices=vertices, indices=indices, mode='triangles')
            else:
                self._mesh_color.rgba = (r, b, g, a / 2)
                self._mesh.vertices = vertices
                self._mesh.indices = indices

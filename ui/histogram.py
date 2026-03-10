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
        self.edges = [0,255]
        self.stablize = 0.3


    def histogram(self, *args):
        ctx = _app_ctx.ctx
        bins = 128

        if ctx.scope.camera is not None and ctx.scope.camera.active is not None:
            image = ctx.scope.get_image_from_buffer(force_to_8bit=True)
            if image is None or image is False:
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
                vertices.extend([bx, y, 0, 0, bx, y + bar_h, 0, 0,
                                 bx + bin_size, y, 0, 0, bx + bin_size, y + bar_h, 0, 0])

            # Build indices for individual triangle strips per bar
            indices = []
            for i in range(bins):
                base = i * 4
                indices.extend([base, base + 1, base + 2, base + 1, base + 2, base + 3])

            self.canvas.clear()
            r, b, g, a = self.bg_color
            self.hist = (counts, None)
            with self.canvas:
                Color(r, b, g, a / 2)
                Mesh(vertices=vertices, indices=indices, mode='triangles')

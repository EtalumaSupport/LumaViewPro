# Copyright Etaluma, Inc.
"""
Matplotlib-to-Kivy bridge widget.

Replaces the deprecated kivy-garden.matplotlib package with a simple
Agg backend renderer that blits into a Kivy Image texture.
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from kivy.graphics.texture import Texture as KivyTexture
from kivy.uix.image import Image as KivyImage


class FigureCanvasKivyAgg(KivyImage):
    """Render a matplotlib figure as a Kivy Image widget using the Agg backend."""

    def __init__(self, figure, **kwargs):
        super().__init__(**kwargs)
        self.figure = figure
        self._canvas_agg = FigureCanvasAgg(figure)
        self.draw()

    def draw(self):
        self._canvas_agg.draw()
        w, h = self._canvas_agg.get_width_height()
        buf = self._canvas_agg.buffer_rgba()
        texture = KivyTexture.create(size=(w, h), colorfmt='rgba')
        texture.blit_buffer(bytes(buf), colorfmt='rgba', bufferfmt='ubyte')
        texture.flip_vertical()
        self.texture = texture

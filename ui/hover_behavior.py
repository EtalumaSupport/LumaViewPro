# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
HoverBehavior mixin for Kivy widgets.

Adds ``hovered`` BooleanProperty that is True when the mouse cursor is
over the widget.  Use in KV rules to change appearance on hover::

    <RoundedButton>:
        canvas.before:
            Color:
                rgba: (0.45, 0.48, 0.52, 1) if self.hovered else \\
                      (0.35, 0.38, 0.42, 1) if self.state == 'normal' else \\
                      (0.25, 0.28, 0.32, 1)
"""
from kivy.core.window import Window
from kivy.properties import BooleanProperty


class HoverBehavior:
    """Mixin that sets ``self.hovered = True`` while the cursor is inside."""

    hovered = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self._on_mouse_pos_hover)

    def _on_mouse_pos_hover(self, window, pos):
        # Don't update if widget isn't visible or window doesn't have focus
        if not self.get_root_window():
            self.hovered = False
            return
        self.hovered = self.collide_point(*self.to_widget(*pos))

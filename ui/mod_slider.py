# Copyright Etaluma, Inc.
from kivy.uix.slider import Slider


class ModSlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(ModSlider, self).__init__(**kwargs)
        self.user_interacting = False
        self.step = 5

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super(ModSlider, self).on_touch_up(touch)
        self.user_interacting = False
        if touch.grab_current == self:
            self.dispatch('on_release')
            return True

    def on_touch_down(self, touch):
        # Mouse-wheel scroll over the slider track adjusts value by
        # step (5x with shift). Default Kivy Slider ignores scroll
        # events; users had to click+drag to adjust illumination /
        # exposure / gain / Z. on_release fires per tick so wired
        # hardware updates on each scroll click without buffering.
        if (
            'button' in touch.profile
            and touch.button in ('scrollup', 'scrolldown')
            and self.collide_point(*touch.pos)
        ):
            from kivy.core.window import Window
            modifiers = set(Window.modifiers)
            multiplier = 5 if (modifiers & {'shift', 'rshift'}) else 1
            delta = self.step * multiplier
            if touch.button == 'scrollup':
                self.value = min(self.max, self.value + delta)
            else:
                self.value = max(self.min, self.value - delta)
            self.dispatch('on_release')
            return True
        out = super().on_touch_down(touch)
        # If the slider accepted the touch, it will grab it.
        if touch.grab_current == self:
            self.user_interacting = True
        return out

    def on_touch_move(self, touch):
        super(ModSlider, self).on_touch_move(touch)
        out = super().on_touch_move(touch)
        if touch.grab_current == self:
            self.user_interacting = True
        return out

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
        super(ModSlider, self).on_touch_down(touch)
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

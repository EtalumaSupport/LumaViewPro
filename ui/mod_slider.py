# Copyright Etaluma, Inc.
from kivy.factory import Factory
from kivy.uix.scrollview import ScrollView
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


class ModSliderAwareScrollView(ScrollView):
    # ScrollView yields scroll-wheel events to ModSlider descendants
    # before consuming them for content scrolling. Kivy v2.3.1's
    # ScrollView.on_scroll_start dispatches via dispatch_children,
    # which only walks IMMEDIATE children -- a ModSlider nested
    # inside a BoxLayout never sees the event, and the ScrollView
    # then consumes the wheel for content scroll. Result: ModSlider's
    # per-step scroll handler silently fails at every slider site
    # inside a scrollable panel.
    #
    # Here we intercept wheel events at on_touch_down: if the touch
    # lands inside us, transform to content-space and dispatch via
    # the standard Widget on_touch_down chain (recursive). If a
    # ModSlider claims it, return without scrolling; otherwise fall
    # through to ScrollView's normal path.

    def on_touch_down(self, touch):
        if (
            'button' in touch.profile
            and touch.button in ('scrollup', 'scrolldown')
            and self.collide_point(*touch.pos)
        ):
            touch.push()
            touch.apply_transform_2d(self.to_local)
            try:
                for child in self.children[:]:
                    if child.dispatch('on_touch_down', touch):
                        return True
            finally:
                touch.pop()
        return super().on_touch_down(touch)


Factory.register('ModSliderAwareScrollView', cls=ModSliderAwareScrollView)

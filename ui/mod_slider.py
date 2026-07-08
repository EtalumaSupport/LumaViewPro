# Copyright Etaluma, Inc.
import weakref

from kivy.factory import Factory
from kivy.graphics import Color, Line
from kivy.properties import BooleanProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider


class ModSlider(Slider):
    # A slider is "armed" for mouse-wheel adjust only after you click it, and
    # only one slider is armed at a time. An armed slider draws a highlight
    # border so it is always visible which slider the wheel will move. It
    # disarms (and un-highlights) as soon as the cursor leaves its own bounds,
    # so scrolling anywhere else scrolls the enclosing menu instead of nudging
    # a slider you are no longer pointing at. Re-click to re-arm. The slider's
    # own bounds are the disarm boundary -- no arbitrary distance, and a small
    # twitch while still on the slider will not drop the arm.
    armed = BooleanProperty(False)

    # Weakref to the currently-armed slider so arming a new one disarms the
    # previous. Weakref so an unmounted slider is not kept alive past teardown.
    _armed_ref: 'weakref.ReferenceType | None' = None

    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super().__init__(**kwargs)
        self.user_interacting = False
        self.step = 5
        with self.canvas.after:
            self._armed_color = Color(rgba=(0, 0, 0, 0))
            self._armed_border = Line(width=1.5)
        self.bind(
            pos=self._refresh_armed_visual,
            size=self._refresh_armed_visual,
            armed=self._refresh_armed_visual,
        )
        self._refresh_armed_visual()

    def _refresh_armed_visual(self, *args):
        self._armed_border.rectangle = (self.x, self.y, self.width, self.height)
        # Cyan accent while armed; fully transparent otherwise. Colour/width
        # are cosmetic -- tune to taste; only the on/off visibility is contract.
        self._armed_color.rgba = (0.1, 0.8, 1.0, 1.0) if self.armed else (0, 0, 0, 0)

    def _arm(self):
        from kivy.core.window import Window

        prev = ModSlider._armed_ref() if ModSlider._armed_ref is not None else None
        if prev is not None and prev is not self:
            prev._disarm()
        ModSlider._armed_ref = weakref.ref(self)
        if not self.armed:
            self.armed = True
            Window.bind(mouse_pos=self._disarm_if_cursor_left)

    def _disarm(self):
        from kivy.core.window import Window

        if self.armed:
            self.armed = False
            Window.unbind(mouse_pos=self._disarm_if_cursor_left)
        if ModSlider._armed_ref is not None and ModSlider._armed_ref() is self:
            ModSlider._armed_ref = None

    def _disarm_if_cursor_left(self, window, pos):
        # Window.mouse_pos is in window coordinates; to_widget maps it through
        # the parent transforms (including the enclosing scroll-view) into this
        # slider's coordinate space so collide_point tests real containment.
        # Leaving the slider's bounds disarms it, so the wheel goes back to
        # scrolling the menu.
        if not self.collide_point(*self.to_widget(*pos)):
            self._disarm()

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super().on_touch_up(touch)
        self.user_interacting = False
        if touch.grab_current == self:
            self.dispatch('on_release')
            return True

    def on_touch_down(self, touch):
        from kivy.core.window import Window

        # Mouse-wheel over the slider adjusts value by step (5x with shift),
        # but ONLY while this slider is armed (most-recently-clicked, cursor
        # still on it). An unarmed slider lets the wheel fall through so the
        # parent scroll-view scrolls the menu. The touch.ud marker tells the
        # ModSliderAwareScrollView owner that an armed slider consumed the
        # wheel, so a text box or range slider claiming the same touch cannot
        # block the menu scroll. on_release fires per tick so wired hardware
        # updates on each scroll click.
        if (
            'button' in touch.profile
            and touch.button in ('scrollup', 'scrolldown')
            and self.collide_point(*touch.pos)
        ):
            if not self.armed:
                return False
            modifiers = set(Window.modifiers)
            multiplier = 5 if (modifiers & {'shift', 'rshift'}) else 1
            delta = self.step * multiplier
            if touch.button == 'scrollup':
                self.value = min(self.max, self.value + delta)
            else:
                self.value = max(self.min, self.value - delta)
            self.dispatch('on_release')
            touch.ud['modslider_scroll_consumed'] = True
            return True
        # A non-scroll click that lands on this slider arms it (and disarms
        # whatever was armed before). Clicking a non-slider widget leaves the
        # prior arm in place, but that slider disarms the moment the cursor
        # leaves it, so no stray adjustment results.
        if self.collide_point(*touch.pos):
            self._arm()
        out = super().on_touch_down(touch)
        # If the slider accepted the touch, it will grab it.
        if touch.grab_current == self:
            self.user_interacting = True
        return out

    def on_touch_move(self, touch):
        super().on_touch_move(touch)
        out = super().on_touch_move(touch)
        if touch.grab_current == self:
            self.user_interacting = True
        return out


class ModSliderAwareScrollView(ScrollView):
    # Single owner of mouse-wheel routing for a sidebar panel. Kivy v2.3.1's
    # ScrollView.on_scroll_start dispatches via dispatch_children, which only
    # walks IMMEDIATE children -- a ModSlider nested inside a BoxLayout never
    # sees the wheel and the ScrollView consumes it for content scroll.
    #
    # We intercept the wheel at on_touch_down and dispatch it down the normal
    # (recursive) touch chain so a nested ModSlider can adjust -- but ONLY an
    # armed slider may consume it. It marks the touch when it does. Any other
    # widget under the pointer (a text box, a range slider, an unarmed slider)
    # may claim the touch, but without that marker we ignore the claim and fall
    # through to normal content scroll, so the menu still scrolls.

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
                        # A widget claimed the wheel. Honor it only if it was an
                        # armed slider adjusting; otherwise ignore the claim so
                        # the menu still scrolls.
                        if touch.ud.get('modslider_scroll_consumed'):
                            return True
                        break
            finally:
                touch.pop()
        return super().on_touch_down(touch)


Factory.register('ModSliderAwareScrollView', cls=ModSliderAwareScrollView)

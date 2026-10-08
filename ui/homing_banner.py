# Copyright Etaluma, Inc.
"""The banner over the middle of the window while a home holds the scope."""

from kivy.animation import Animation
from kivy.properties import BooleanProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout

# One turn of the spinner's arc.
_TURN_S = 1.0


class HomingBanner(BoxLayout):
    """'System Homing...' beside a turning arc, shown while ``active``.

    The arc turns only while the banner is shown, so a hidden banner costs
    no frames. ``active`` is bound to the app's ``homing`` mirror, written on
    the run-state edge that greys and frees the controls.
    """

    active = BooleanProperty(False)
    # The arc's rotation in degrees, animated while active.
    angle = NumericProperty(0)

    def on_active(self, _banner, active: bool) -> None:
        Animation.cancel_all(self, 'angle')
        self.angle = 0
        if active:
            turn = Animation(angle=360, duration=_TURN_S) + Animation(angle=0, duration=0)
            turn.repeat = True
            turn.start(self)

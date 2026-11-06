# Copyright Etaluma, Inc.
"""What the window shows while the scope is brought up: the splash, a progress bar and the step.

The window's first content, drawn from the first frame, before any session
exists. It holds no control and reads no session, scope or settings, so
nothing on it can be pressed into a crash; it is written only from
bring-up's progress report (``ScopeSession.create(on_progress=)``).
"""

from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.widget import Widget

# The logo, the bar and the step, as one small panel in the middle of the window.
_PANEL_WIDTH_DP = 540
_LOGO_HEIGHT_DP = 135
_BAR_HEIGHT_DP = 8
_STEP_HEIGHT_DP = 32
_SPACING_DP = 12


class BringUpView(BoxLayout):
    """A small splash image over a progress bar and the step bring-up is on, centred in the window."""

    def __init__(self, splash: str, **kwargs):
        """A view showing ``splash``, the image file's path, before any step is told."""
        super().__init__(orientation='vertical', **kwargs)
        panel = BoxLayout(
            orientation='vertical',
            spacing=dp(_SPACING_DP),
            size_hint=(None, None),
            width=dp(_PANEL_WIDTH_DP),
            height=dp(_LOGO_HEIGHT_DP + _BAR_HEIGHT_DP + _STEP_HEIGHT_DP + 2 * _SPACING_DP),
            pos_hint={'center_x': 0.5},
        )
        panel.add_widget(
            Image(source=splash, fit_mode='contain', size_hint_y=None, height=dp(_LOGO_HEIGHT_DP))
        )
        self._bar = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(_BAR_HEIGHT_DP))
        panel.add_widget(self._bar)
        self._step = Label(
            text='Starting LumaViewPro', size_hint_y=None, height=dp(_STEP_HEIGHT_DP)
        )
        panel.add_widget(self._step)
        # Room above and below, so the panel sits in the middle.
        self.add_widget(Widget())
        self.add_widget(panel)
        self.add_widget(Widget())

    def show_step(self, percent: float, step: str | None) -> None:
        """Show how far bring-up has got and the step it is on."""
        self._bar.value = percent
        if step is not None:
            self._step.text = step

    @property
    def step(self) -> str:
        """The step the view shows."""
        return self._step.text

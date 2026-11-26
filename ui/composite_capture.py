# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import logging

from kivy.properties import BooleanProperty
from kivy.uix.floatlayout import FloatLayout

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules import gui_logger, image_utils
from _lumascope.modules.ui_queue import schedule_ui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _lumascope.modules.operation import Operation
    from _lumascope.modules.run_outcome import RunOutcome
from _lumascope.modules.run_events import RunEvents
from ui.ui_helpers import (
    run_reported,
    run_unasked,
    show_captured_frame,
    show_captured_still,
    start_pressed_off_a_refused_edit,
    submit_reported,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):
    # The handle this button's last start returned: what its Stop names.
    # The engine answers whether it is still the live run.
    _composite_run: 'Operation | None' = None
    # True while this button's own request is on its way to the engine; the
    # button is disabled until that request's redraw.
    composite_pending = BooleanProperty(False)
    # True while anything but this button's own run holds the scope -- another
    # run, a recording, a diagnostic -- as the Session answers; greys the
    # button, and its own run leaves it live as that run's Stop.
    composite_held = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def live_capture(self):
        """Capture a still through the session, and show how it went.

        Every decision about the still -- channel, folder, name, summing,
        format, depth, overlay copy, refusals -- is the session's
        ``manual_capture``; the button supplies only the overlay copies the
        user asked for, read here on the main thread. A second press while a
        still is in flight, and a press while a run holds the camera, are
        the member's refusals; the reporter shows them, and whatever the
        still's operation settles with, like any other outcome. The saved
        still is shown for the user's time, then live view comes back.
        """
        gui_logger.button('LIVE_CAPTURE')
        if start_pressed_off_a_refused_edit('Still', None):
            return
        ctx = _app_ctx.ctx
        bullseye = ctx.scope_display.use_bullseye
        crosshairs = ctx.scope_display.use_crosshairs
        run_reported(
            lambda: ctx.session.manual_capture.start_save_still(
                bullseye=bullseye,
                crosshairs=crosshairs,
                events=RunEvents(frame_captured=show_captured_still),
            ).add_done_callback(lambda done: schedule_ui(lambda _dt: _still_settled(done))),
            None,
            'LIVE_CAPTURE',
        )

    # capture and save a composite image using the current settings
    def composite_capture(self):
        """Start a composite run, or stop the one this button started.

        A composite is a sequenced run like a scan or a z-stack, so this
        is a run starter and nothing more. It states no run parameters,
        assembles no config and pre-checks nothing: everything the run
        needs is settings the engine already reads, and every refusal --
        a rival run, files still draining, too few channels, a camera
        that is absent, an unknown objective -- is the engine's to raise
        and the boundary's to show, once. A still mid-capture is not a
        refusal at all: the run waits for it.

        Whether the press means Stop is the engine's answer -- is the run
        this button started still live -- never the toggle's, which Kivy
        has already flipped. The button changes nothing ahead of the
        engine's answer; draw_composite_button shows it. The merged
        composite is shown for the user's time once the run has ended.
        """
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx
        runner = ctx.session.protocol_runner()
        run = self._composite_run
        # The button is disabled until this request's own redraw, so a
        # second press cannot race the first one to the pool.
        self.composite_pending = True

        if run is not None and run.progress().is_live:
            submit_reported(
                run.stop,
                self._composite_request_done,
                'COMPOSITE_CAPTURE',
                stop=True,
            )
            return
        if start_pressed_off_a_refused_edit('Composite', self._composite_request_done):
            return

        events = RunEvents(frame_captured=show_captured_frame)

        def _start():
            started = runner.start_composite(
                sequence_name='composite',
                events=events,
            )
            self._composite_run = started
            started.add_done_callback(show_merged_composite)

        submit_reported(_start, self._composite_request_done, 'COMPOSITE_CAPTURE')

    def _composite_request_done(self):
        self.composite_pending = False
        self.draw_composite_button()

    def draw_composite_button(self):
        """Show the composite this button started, as the engine reports it.

        The only code that styles the button: after each of its own
        requests, and on every run-state edge -- including the run's return
        to idle -- so a run that ends on its own and a start that was
        refused land on the same drawing. It draws this button and its own
        run's title only: the same edge fires when another run takes the
        scope, and what every run shares is draw_shared_run_displays'.
        """
        ctx = _app_ctx.ctx
        self.composite_held = ctx.session.held_by_other(self._composite_run)
        run = self._composite_run
        live = run is not None and run.progress().is_live
        self.ids['composite_btn'].state = 'down' if live else 'normal'
        gui_logger.display(
            'COMPOSITE_BUTTON', f'{"Stop" if live else "Composite"} held={self.composite_held}'
        )


def _still_settled(done) -> None:
    """Read the settled still on the Kivy thread: its failure goes to the reporter."""
    run_reported(done.result, None, 'LIVE_CAPTURE')


def show_merged_composite(run: 'Operation[RunOutcome]') -> None:
    """Hold the composite a run merged on screen, for the user's ``capture_shown_s``.

    The composite button's done callback, on the operations' callback thread,
    where the merged file is read, so the Kivy thread only draws it; then live
    view comes back by itself. A run that merged nothing shows nothing here:
    its outcome is the engine's to report. The merge is a lossless 8-bit RGB
    TIFF, so the file is the image, read as the color image it is: a
    composite whose other channels were dark is not collapsed to its one
    populated plane.
    """
    outcome = run.result()
    if outcome.artifact_path is None:
        return
    image, significant_bits, _white = image_utils.load_pixels(
        outcome.artifact_path, collapse_legacy_false_color=False
    )
    ctx = _app_ctx.ctx
    held_s = ctx.session.setting('capture_shown_s')
    display = ctx.scope_display
    display.post_held_frame(
        lambda: run_unasked(
            lambda: display.hold_derived_image(
                image, significant_bits, held_s=held_s, label='the composite'
            ),
            'COMPOSITE_SHOWN',
        )
    )

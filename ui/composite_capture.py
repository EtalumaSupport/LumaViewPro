# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import logging

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.exceptions import CaptureError, HardwareCommandRefusedError, ObjectiveUnknownError
from modules.run_outcome import PendingRunOutcome
from ui.ui_helpers import (
    live_display_callbacks,
    set_last_save_folder,
    set_title_event_text,
    submit_reported,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):
    # The handle this button's last start returned: what its Stop names.
    # The engine answers whether it is still the live run.
    _composite_run: PendingRunOutcome | None = None
    # True while this button's own request is on its way to the engine; the
    # button is disabled until that request's redraw.
    composite_pending = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def live_capture(self):
        """Capture a still through the session, and show how it went.

        Every decision about the still -- channel, folder, name, summing,
        format, depth, overlay copy, refusals -- is the session's
        ``manual_capture``; the button supplies only what the user is
        looking at, read here on the main thread. A second press while a
        still is in flight, and a press while a run holds the camera, are
        the member's refusals, shown below like any other.
        """
        gui_logger.button('LIVE_CAPTURE')
        ctx = _app_ctx.ctx
        layer = common_utils.get_opened_layer(ctx.image_settings)
        false_color_on = (
            ctx.image_settings.layer_lookup(layer=layer).ids['false_color'].active
            if layer is not None
            else False
        )
        try:
            future = ctx.session.manual_capture.capture(
                layer=layer,
                false_color_on=false_color_on,
                bullseye=ctx.scope_display.use_bullseye,
                crosshairs=ctx.scope_display.use_crosshairs,
                engineering_mode=ctx.engineering_mode,
            )
        except HardwareCommandRefusedError as refused:
            _show_capture_failure(refused)
            return
        future.add_done_callback(
            lambda done: Clock.schedule_once(lambda _dt: _show_capture_outcome(done))
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
        engine's answer; draw_composite_button shows it.
        """
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx
        runner = ctx.session.create_protocol_runner()
        run = self._composite_run
        # The button is disabled until this request's own redraw, so a
        # second press cannot race the first one to the pool.
        self.composite_pending = True

        if ctx.sequenced_capture_runner.is_live_run(run):
            submit_reported(
                lambda: ctx.sequenced_capture_runner.reset(run),
                self._composite_request_done,
                'COMPOSITE_CAPTURE',
                stop=True,
            )
            return

        engineering_mode = ctx.engineering_mode
        callbacks = {**live_display_callbacks()}

        def _start():
            self._composite_run = runner.start_composite(
                sequence_name='composite',
                callbacks=callbacks,
                run_trigger_source='composite',
                engineering_mode=engineering_mode,
            )
            # Only reachable once the run is committed, so the saved
            # folder can only ever name THIS run's directory.
            set_last_save_folder(dir=runner.run_dir())

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
        live = ctx.sequenced_capture_runner.is_live_run(self._composite_run)
        self.ids['composite_btn'].state = 'down' if live else 'normal'
        if live:
            set_title_event_text('Compositing...')


def _show_capture_outcome(future) -> None:
    exc = future.exception()
    if exc is None:
        set_last_save_folder(dir=future.result()[0].parent)
        return
    _show_capture_failure(exc)


def _show_capture_failure(exc: BaseException) -> None:
    from modules.notification_center import notifications

    logger.error(f'[LVP Main  ] manual capture saved nothing: {exc!r}')
    if isinstance(exc, HardwareCommandRefusedError):
        notifications.warning('Capture', exc.title, str(exc))
    elif isinstance(exc, (CaptureError, ObjectiveUnknownError)):
        # Both are written for the user: the capture engine's cause, or what
        # to do about the objective in the light path.
        notifications.error('Capture', 'No image was saved', str(exc))
    else:
        notifications.error(
            'Capture', 'Capture failed', 'No image was saved. Check the main log for details.'
        )

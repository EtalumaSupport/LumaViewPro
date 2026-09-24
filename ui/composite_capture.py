# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import functools
import logging

from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.exceptions import CaptureError, HardwareCommandRefusedError, ObjectiveUnknownError
from modules.run_outcome import PendingRunOutcome
from modules.sequential_io_executor import IOTask, PRIORITY_HIGH
from ui.ui_helpers import (
    live_display_callbacks,
    live_histo_off,
    live_histo_reverse,
    reset_title,
    reset_with_refusal_boundary,
    run_with_refusal_boundary,
    set_last_save_folder,
    set_title_event_text,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):
    # The handle this button's last start returned: what its Stop names.
    # The engine answers whether it is still the live run.
    _composite_run: PendingRunOutcome | None = None

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
        """Start a composite run, or stop the one already running.

        A composite is a sequenced run like a scan or a z-stack, so this
        is a run starter and nothing more. It states no run parameters,
        assembles no config and pre-checks nothing: everything the run
        needs is settings the engine already reads, and every refusal --
        a rival run, files still draining, too few channels, a camera
        that is absent, an unknown objective -- is the engine's to raise
        and this button's to display, once. A still mid-capture is not a
        refusal at all: the run waits for it. The one thing decided here
        is what only a toggle can know, that this click is the second of
        a pair.
        """
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx
        composite_btn = self.ids['composite_btn']
        runner = ctx.session.create_protocol_runner()

        # The button is its own stop control, so a second click means stop.
        # It reads as one of two things: a toggle already back to 'normal',
        # or a click arriving while this starter's own run is live. The
        # handle this button's start returned is what separates the second
        # case from a click during someone ELSE's run, which the engine
        # refuses rather than aborting a run this button never started.
        #
        # Reset goes onto the worker pool at high priority because the pool
        # runs exactly one worker: a stop that queued behind ordinary work
        # would not arrive until that work finished, which is the thing the
        # user is trying to interrupt.
        if composite_btn.state == 'normal' or runner.is_live_run(self._composite_run):
            ctx.worker_pool.put(
                IOTask(
                    # Through the boundary, like every run control's Stop: a
                    # Stop after the run ended is nothing left to do, and a
                    # stale one while another run is live is the engine's
                    # refusal, already notified once.
                    action=functools.partial(
                        reset_with_refusal_boundary,
                        ctx.sequenced_capture_runner,
                        self._composite_run,
                    ),
                    # The executor's generic failure popup would be a second
                    # notification titled with this partial's repr.
                    silent_on_failure=True,
                    priority=PRIORITY_HIGH,
                )
            )
            return

        # Every path that does not reach a live run hands the UI back in
        # the finally below, toggle included: left 'down', the NEXT click
        # reads as the second click of a pair and is swallowed as an abort
        # of a run that was never started. The finally rather than each
        # exit because the refusal boundary catches only the typed
        # refusal, and a programming error at the call site raises
        # straight past it.
        started = False
        try:
            live_histo_off()
            set_title_event_text('Compositing...')

            def _start():
                nonlocal started
                self._composite_run = runner.start_composite(
                    sequence_name='composite',
                    callbacks={
                        **live_display_callbacks(),
                        'run_complete': self._composite_finished,
                    },
                    run_trigger_source='composite',
                    engineering_mode=ctx.engineering_mode,
                )
                started = True
                # Only reachable once the run is committed, so the saved
                # folder can only ever name THIS run's directory.
                set_last_save_folder(dir=runner.run_dir())

            # A refusal is logged and shown once by the engine's funnel
            # (solicited, so it reaches the user during a run of any
            # kind); the finally below undoes the cosmetics.
            run_with_refusal_boundary(_start, on_refused=lambda: None)
        except Exception as e:
            logger.error(f'[LVP Main  ] composite_capture failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
        finally:
            if not started:
                self._composite_finished()

    def _composite_finished(self, **kwargs):
        """Hand the UI back after a composite ends or never starts.

        One handler for both, because every step is level-based rather
        than a guess about what the run did: the reconcile reads the LED
        driver instead of assuming, and the histogram and title helpers
        are idempotent. A second handler for the not-started path would
        be the same five lines with one omitted.

        This fires at RUN end, not merge end. The merged file lands about
        a second later, exactly as it does for every other run kind; a
        button that waited for it would be the only one in the app that
        did.
        """
        self.ids['composite_btn'].state = 'normal'
        reset_title()
        live_histo_reverse()
        # The run's LED restore has settled by now, so reconcile every
        # enable toggle to what the driver actually reports: a restore that
        # emits no LED events leaves the buttons stale otherwise.
        _app_ctx.ctx.ui_listener_bridge.reconcile_led_buttons()


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

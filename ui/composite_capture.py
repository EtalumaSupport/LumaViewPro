# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import functools
import logging
import pathlib
import threading


from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.exceptions import CaptureError, HardwareCommandRefusedError, ObjectiveUnknownError
from modules.sequential_io_executor import IOTask, PRIORITY_HIGH
from ui.ui_helpers import (
    live_display_callbacks,
    live_histo_off,
    live_histo_reverse,
    reset_title,
    run_with_refusal_boundary,
    set_last_save_folder,
    set_title_event_text,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):
    _capturing = threading.Event()  # Thread-safe guard against rapid double-clicks

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def live_capture(self):
        """Capture a still through the session, and show how it went.

        Every decision about the still -- channel, folder, name, summing,
        format, depth, overlay copy, refusals -- is the session's
        ``manual_capture``; the button supplies only what the user is
        looking at, read here on the main thread.
        """
        gui_logger.button('LIVE_CAPTURE')
        # The Composite button reads this too: a still and a composite never
        # overlap from this window. A second still is the member's refusal.
        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Capture already in progress, ignoring')
            return
        ctx = _app_ctx.ctx
        layer = common_utils.get_opened_layer(ctx.image_settings)
        false_color_on = (
            ctx.image_settings.layer_lookup(layer=layer).ids['false_color'].active
            if layer is not None
            else False
        )
        CompositeCapture._capturing.set()
        try:
            future = ctx.session.manual_capture.capture(
                layer=layer,
                false_color_on=false_color_on,
                bullseye=ctx.scope_display.use_bullseye,
                crosshairs=ctx.scope_display.use_crosshairs,
                engineering_mode=ctx.engineering_mode,
            )
        except HardwareCommandRefusedError as refused:
            CompositeCapture._capturing.clear()
            _show_capture_failure(refused)
            return
        except BaseException:
            CompositeCapture._capturing.clear()
            raise
        future.add_done_callback(
            lambda done: Clock.schedule_once(lambda _dt: _show_capture_outcome(done))
        )

    # capture and save a composite image using the current settings
    def composite_capture(self):
        """Start a composite run, or stop the one already running.

        A composite is a sequenced run like a scan or a z-stack, so this
        is a run starter and nothing more. It states no run parameters and
        assembles no config: everything the run needs is settings the
        engine already reads, and duplicating that assembly here is what
        put a second composite implementation in the GUI to begin with.

        Only the concerns the engine cannot own stay here. It cannot know
        the toggle was clicked a second time, it does not share the guard
        that makes the two capture buttons mutually exclusive, and it has
        no refusal for a camera that is connected but not yet streaming.
        Everything else -- a rival run, files still draining, too few
        channels -- is the engine's refusal to raise, not this starter's
        to pre-check.
        """
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx
        composite_btn = self.ids['composite_btn']
        runner = ctx.session.create_protocol_runner()

        # The button is its own stop control, so a second click means stop.
        # It reads as one of two things: a toggle already back to 'normal',
        # or a click arriving while this starter's own run is live. The
        # trigger source is what separates the second case from a click
        # during someone ELSE's run, which must fall through to the engine
        # and be refused rather than aborting a run this button never
        # started.
        #
        # Reset goes onto the worker pool at high priority because the pool
        # runs exactly one worker: a stop that queued behind ordinary work
        # would not arrive until that work finished, which is the thing the
        # user is trying to interrupt.
        if composite_btn.state == 'normal' or (
            runner.is_running() and runner.run_trigger_source() == 'composite'
        ):
            ctx.worker_pool.put(
                IOTask(
                    action=functools.partial(runner.reset, requester='composite'),
                    # The engine's refusal notifies once on its own; the
                    # executor's generic failure popup would be a second
                    # notification titled with this partial's repr.
                    silent_on_failure=True,
                    priority=PRIORITY_HIGH,
                )
            )
            return

        # Every gate below puts the toggle back before returning. Left
        # 'down', it makes the NEXT click read as the second click of a
        # pair, and that click is swallowed as an abort of a run that was
        # never started.
        if CompositeCapture._capturing.is_set():
            # Names "a capture", not "a composite": a composite's own second
            # click is taken by the stop branch above, so the only way to
            # arrive here is a live capture still holding the guard. Saying
            # "composite" reported the wrong subsystem to the user.
            logger.warning('[LVP Main  ] A capture is already running, ignoring composite press')
            composite_btn.state = 'normal'
            from ui.notification_popup import show_notification_popup

            show_notification_popup(
                title='Capture In Progress',
                message=(
                    'A capture is still running.\n\n'
                    'Wait for it to finish before starting a composite.'
                ),
            )
            return

        from modules.notification_center import notifications

        if not getattr(ctx.scope, 'camera_connected', True):
            notifications.warning(
                'Camera',
                'Camera not connected',
                'Cannot capture composite -- camera is not connected. '
                'Check USB and reconnect, then try again.',
            )
            composite_btn.state = 'normal'
            return

        if not ctx.scope.imaging.active_cached:
            notifications.warning(
                'Camera',
                'Camera not active',
                'Cannot capture composite -- the camera is not streaming. '
                'Wait for the camera to start, then try again.',
            )
            composite_btn.state = 'normal'
            return

        # Set only once every gate above has passed, and cleared in exactly
        # one place per outcome: the finally below for anything that does
        # not reach a live run, and the run's own completion for anything
        # that does. A guard left set is permanent -- both capture entry
        # points return at their is_set() check before enqueuing the work
        # whose completion would clear it -- so a path with no clearer
        # disables both capture buttons for the life of the process. That
        # is why the clear sits in a finally rather than at each exit: the
        # refusal boundary only catches the typed refusal, and a
        # programming error at the call site raises straight past it.
        CompositeCapture._capturing.set()
        started = False
        try:
            live_histo_off()
            set_title_event_text('Compositing...')

            settings = ctx.settings
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'Manual' / 'Composites'

            def _start():
                nonlocal started
                runner.start_composite(
                    sequence_name='composite',
                    parent_dir=parent_dir,
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

            # A refusal is always LOGGED by the engine's funnel, and the
            # finally below undoes the cosmetics. It is not necessarily
            # SHOWN: the centre drops non-fatal notifications for the whole
            # of a run nobody is watching, and a rival run owning the scope
            # is the only way this starter is refused, since it carries no
            # rival gate of its own. So that refusal currently reaches the
            # user nowhere: a known hole, recorded rather than closed.
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
        CompositeCapture._capturing.clear()


# A refusal carries a reason code for callers that branch on it, and no
# words: the caller that provoked it writes them.
_REFUSAL_TEXT = {
    'exclusive_activity_running': 'A run is using the camera. Capture again when it ends.',
    'capture_in_flight': 'A capture is still being saved. Try again in a moment.',
}


def _show_capture_outcome(future) -> None:
    CompositeCapture._capturing.clear()
    exc = future.exception()
    if exc is None:
        set_last_save_folder(dir=future.result()[0].parent)
        return
    _show_capture_failure(exc)


def _show_capture_failure(exc: BaseException) -> None:
    from modules.notification_center import notifications

    logger.error(f'[LVP Main  ] manual capture saved nothing: {exc!r}')
    if isinstance(exc, HardwareCommandRefusedError):
        notifications.warning('Capture', 'Capture refused', _REFUSAL_TEXT[exc.reason])
    elif isinstance(exc, (CaptureError, ObjectiveUnknownError)):
        # Both are written for the user: the capture engine's cause, or what
        # to do about the objective in the light path.
        notifications.error('Capture', 'No image was saved', str(exc))
    else:
        notifications.error(
            'Capture', 'Capture failed', 'No image was saved. Check the main log for details.'
        )

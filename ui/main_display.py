# Copyright Etaluma, Inc.
"""
MainDisplay -- primary application display (recording, camera, fit/zoom)
extracted from lumaviewpro.py.
"""

import logging

from kivy.clock import Clock

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from ui.ui_helpers import run_reported, set_last_save_folder, submit_reported
from ui.composite_capture import CompositeCapture

logger = logging.getLogger('LVP.ui.main_display')


class MainDisplay(CompositeCapture):  # i.e. global lumaview
    def __init__(self, scope, **kwargs):
        # The scope is the session's, handed in. It goes on the widget
        # BEFORE the widget's own init builds the kv tree, because every
        # reader of `lumaview.scope` expects it there from construction.
        self.scope = scope
        super().__init__(**kwargs)
        # Manual recording lives in the session's ManualRecordingController;
        # this widget keeps only button wiring, the status poll, and titles.
        self._recording_poll = None
        self._pause_led_snapshot = None  # save/restore via API

    def cam_toggle(self):
        logger.info('[LVP Main  ] MainDisplay.cam_toggle()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if not self.scope.imaging.active_cached:
            gui_logger.button('CAM_TOGGLE', 'no-op (camera inactive)')
            return
        gui_logger.toggle('CAM_PLAY', not scope_display.play)
        run_reported(lambda: self._toggle_play(scope_display), None, 'CAM_PLAY')

    def _toggle_play(self, scope_display):
        io_executor = _app_ctx.ctx.io_executor
        illumination = self.scope.illumination
        if scope_display.play:
            scope_display.play = False
            # pause() instead of stop()+start() so the
            # display thread stays alive across pause-resume; no
            # Thread spawn/join overhead; generation does NOT bump
            # so the texture stays on the last rendered frame.
            scope_display.pause()
            if self.scope.led_connected:
                self._pause_led_snapshot = illumination.save_led_state('camera_pause')
                # LED observer handles UI button sync
                submit_reported(illumination.leds_off, None, 'CAM_PAUSE_LEDS', lane=io_executor)
        else:
            if self._pause_led_snapshot:
                snapshot = self._pause_led_snapshot
                self._pause_led_snapshot = None
                # LED observer handles UI button sync
                submit_reported(
                    lambda: illumination.restore_led_state(snapshot),
                    None,
                    'CAM_RESUME_LEDS',
                    lane=io_executor,
                )

            scope_display.play = True
            scope_display.resume()

    def record_button(self):
        """Start a manual recording, or stop the one that is recording.

        Whether the press means Stop is the API's answer -- is a recording
        open -- never the toggle's, which Kivy has already flipped. Start and
        Stop go through the boundary, which shows a refusal once; the button
        changes nothing ahead of the answer, and draw_record_button shows it.
        """
        gui_logger.button('RECORD')
        ctx = _app_ctx.ctx
        controller = ctx.session.manual_recording

        if controller.is_recording:
            run_reported(controller.stop, self.draw_record_button, 'RECORD')
            return

        # The open layer names the channel; its toggle governs display
        # only. Reading one without the other is what left a brightfield
        # recording with no channel name at all. Both are read here, on the
        # GUI thread, since .ids access is not thread-safe.
        layer = common_utils.get_opened_layer(ctx.image_settings)
        false_color_on = (
            ctx.image_settings.layer_lookup(layer=layer).ids['false_color'].active
            if layer is not None
            else False
        )

        def _start():
            controller.start(
                layer=layer,
                false_color_on=false_color_on,
                on_complete=self._on_recording_complete,
            )

        # On the worker pool: the controller's start opens the encoder and
        # probes disk, which must not stall the GUI thread.
        submit_reported(_start, self._recording_request_done, 'RECORD')

    def _recording_request_done(self):
        self.draw_record_button()

    def draw_record_button(self):
        """Show the manual recording as the API reports it.

        The only code that styles the Record toggle: after each of its own
        requests, from the status poll, at the recording's finish and on
        every run-state edge. The first time it sees a recording open it
        starts the status poll, which owns the titles from there.
        """
        controller = _app_ctx.ctx.session.manual_recording
        button = self.ids['record_btn']
        if not controller.is_recording:
            button.state = 'normal'
            return
        button.state = 'down'
        if self._recording_poll is None:
            # Set immediately so "Open Last Save Folder" works during the
            # recording, not only after cleanup lands.
            if controller.save_folder is not None:
                set_last_save_folder(controller.save_folder)
            self._recording_poll = Clock.schedule_interval(self._poll_recording_state, 0.1)

    def _poll_recording_state(self, dt=None):
        """Main-thread poll: titles and button state only.

        The controller owns the recording AND its health bounds (the
        duration cap and camera-death detection arm themselves on the
        session's scheduler); this poll purely reflects that state into
        the display.
        """
        from ui.ui_helpers import set_title_event_text

        controller = _app_ctx.ctx.session.manual_recording
        if controller.is_recording:
            set_title_event_text(f'Recording Manual Video: {controller.elapsed_s:.1f}s')
            return
        # Selection closed (Stop, duration cap, or budget full); the
        # recording announced that edge itself, so this tick only keeps the
        # title counting down the drain.
        self.draw_record_button()
        if controller.is_draining:
            set_title_event_text(
                f'Writing Manual Video: {controller.pending_writes} frames remaining'
            )

    def _on_recording_complete(self):
        """Finish-thread callback from the controller; dispatch to GUI."""
        Clock.schedule_once(lambda dt: self._finish_recording_ui(), 0)

    def _finish_recording_ui(self):
        """Main thread: drain + finish done -- clear poll, title, button."""
        from ui.ui_helpers import set_title_event_text

        if self._recording_poll is not None:
            Clock.unschedule(self._recording_poll)
            self._recording_poll = None
        controller = _app_ctx.ctx.session.manual_recording
        if controller.save_folder is not None:
            set_last_save_folder(controller.save_folder)
        set_title_event_text(None)
        self.draw_record_button()
        logger.info('[LVP Main  ] Manual recording UI cleanup complete')

    def open_save_folder_button(self):
        gui_logger.button('OPEN_SAVE_FOLDER')
        from ui.post_processing import open_last_save_folder

        open_last_save_folder()

    def fit_image(self):
        gui_logger.button('FIT_IMAGE')
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if not self.scope.imaging.active_cached:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0, 0)

    def one2one_image(self):
        gui_logger.button('ONE_TO_ONE_IMAGE')
        logger.info('[LVP Main  ] MainDisplay.one2one_image()')
        if not self.scope.imaging.active_cached:
            return
        scope = _app_ctx.ctx.scope
        w = self.width
        h = self.height
        scale_hor = float(scope.imaging.get_width()) / float(w)
        scale_ver = float(scope.imaging.get_height()) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (int((w - scale * w) / 2), int((h - scale * h) / 2))

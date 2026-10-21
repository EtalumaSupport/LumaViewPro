# Copyright Etaluma, Inc.
"""
MainDisplay -- primary application display (recording, camera, fit/zoom)
extracted from lumaviewpro.py.
"""

import logging

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty

import modules.app_context as _app_ctx
from modules import gui_logger
from ui.ui_helpers import run_reported, set_last_save_folder, submit_reported
from ui.composite_capture import CompositeCapture

logger = logging.getLogger('LVP.ui.main_display')


class MainDisplay(CompositeCapture):  # i.e. global lumaview
    # The status line's height, along the bottom of the window: the live view,
    # the panels and the camera buttons all sit on top of it.
    status_line_height = NumericProperty(dp(22))
    # Whether the scope has a motorised axis, which the red STOP is shown for:
    # the capability the motion controls are shown by, set with them.
    stage_has_motors = BooleanProperty(False)

    def __init__(self, scope, **kwargs):
        # The scope is the session's, handed in. It goes on the widget
        # BEFORE the widget's own init builds the kv tree, because every
        # reader of `lumaview.scope` expects it there from construction.
        self.scope = scope
        super().__init__(**kwargs)
        # Manual recording lives in the session's ManualRecordingController;
        # this widget keeps only button wiring, the status poll, and titles.
        self._recording_poll = None

    def cam_toggle(self):
        """Freeze the live picture, or unfreeze it.

        A freeze is the display's alone: the LEDs, captures, recordings and
        runs carry on as they were, and no camera is needed. The display
        thread's pause is the one store; the button is drawn from it.
        """
        logger.info('[LVP Main  ] MainDisplay.cam_toggle()')
        thread = _app_ctx.ctx.scope_display_thread
        gui_logger.toggle('CAM_PLAY', thread.is_paused)
        if thread.is_paused:
            thread.resume()
        else:
            thread.pause()
        self.draw_live_button()

    def draw_live_button(self):
        """Show the live picture's pause as the display thread holds it.

        The only code that sets the Freeze toggle's state: Kivy flips a
        ToggleButton on touch-down, and a touch ending off the button flips
        it with no handler run, so the state is redrawn from the store.
        """
        paused = _app_ctx.ctx.scope_display_thread.is_paused
        self.ids['live_btn'].state = 'down' if paused else 'normal'

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

        # The shown layer's false-colour toggle governs display only; the
        # recording names its channel itself. Read here, on the GUI thread,
        # since .ids access is not thread-safe.
        layer = ctx.session.shown_channel.layer
        false_color_on = ctx.image_settings.layer_lookup(layer=layer).ids['false_color'].active

        def _start():
            recording = controller.start(false_color_on=false_color_on)
            recording.add_done_callback(self._on_recording_complete)

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
        # Greyed by the run lockout (kv: disabled: app.lock.run_lockout).
        gui_logger.display(
            'RECORD_BUTTON',
            f'{"Stop" if controller.is_recording else "Record"} '
            f'held={_app_ctx.ctx.session.controls_lock.named}',
        )
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
        from ui.ui_helpers import fixed_number, set_title_event_text

        controller = _app_ctx.ctx.session.manual_recording
        if controller.is_recording:
            # Padded so a new digit does not jump the title; the title's own
            # font still draws each digit a little differently wide.
            elapsed = fixed_number(controller.elapsed_s, whole_digits=4, decimals=1)
            set_title_event_text(
                f'Recording Manual Video: {elapsed}s', stage='Recording Manual Video'
            )
            return
        # Selection closed (Stop, duration cap, or budget full); the
        # recording announced that edge itself, so this tick only keeps the
        # title counting down the drain.
        self.draw_record_button()
        if controller.is_draining:
            set_title_event_text(
                f'Writing Manual Video: '
                f'{fixed_number(controller.pending_writes, whole_digits=5, decimals=0)} frames remaining',
                stage='Writing Manual Video',
            )

    def _on_recording_complete(self, _recording):
        """The recording's done-callback, on the callback thread; dispatch to GUI."""
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

    def stop_the_stage(self):
        gui_logger.button('STOP')
        from ui.ui_helpers import stop_the_stage

        stop_the_stage()

    def take_back_control(self):
        gui_logger.button('TAKE_BACK_CONTROL')
        from ui.ui_helpers import take_back_control

        take_back_control()

    def open_save_folder_button(self):
        gui_logger.button('OPEN_SAVE_FOLDER')
        from ui.post_processing import open_last_save_folder

        open_last_save_folder()

    def fit_image(self):
        gui_logger.button('FIT_IMAGE')
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if not self.scope.camera.active:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0, self.status_line_height)

    def one2one_image(self):
        gui_logger.button('ONE_TO_ONE_IMAGE')
        logger.info('[LVP Main  ] MainDisplay.one2one_image()')
        frame = self.scope.camera.frame_size
        if not self.scope.camera.active or frame is None:
            return
        # The viewer's own size: the status line below it takes a line of the window.
        w, h = self.ids['viewer_id'].size
        # The size the camera delivers, held, not a live read (read above):
        # a live read waits on the camera, which a grab stop can hold for
        # seconds.
        scale_hor = float(frame.width) / float(w)
        scale_ver = float(frame.height) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (
            int((w - scale * w) / 2),
            self.status_line_height + int((h - scale * h) / 2),
        )

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
from modules.exceptions import RecordingRefusedError
from modules.sequential_io_executor import IOTask
from ui.ui_helpers import set_last_save_folder
from ui.composite_capture import CompositeCapture

logger = logging.getLogger('LVP.ui.main_display')


class MainDisplay(CompositeCapture):  # i.e. global lumaview
    def __init__(self, camera_type='ids', simulate=False, **kwargs):
        import modules.lumascope_api as lumascope_api

        super().__init__(**kwargs)
        # The GUI ships in the same commit as the API it calls, so the
        # PRE-RELEASE FutureWarning has nothing to tell it and only reaches
        # the user's console. It still fires for out-of-tree consumers.
        from modules.settings_init import settings

        self.scope = lumascope_api.Lumascope(
            camera_type=camera_type,
            simulate=simulate,
            warn_pre_release=False,
            configured_model=settings.get('microscope'),
        )
        # LVP-A-2: camera_temps_event moved to Lumascope.start_camera_temp_logging.
        # Manual recording lives in the session's ManualRecordingController;
        # this widget keeps only button wiring, the status poll, and titles.
        self._recording_poll = None
        self._pause_led_snapshot = None  # save/restore via API

    def cam_toggle(self):
        try:
            logger.info('[LVP Main  ] MainDisplay.cam_toggle()')

            scope_display = self.ids['viewer_id'].ids['scope_display_id']
            if not self.scope.imaging.active_cached:
                gui_logger.button('CAM_TOGGLE', 'no-op (camera inactive)')
                return

            gui_logger.toggle('CAM_PLAY', not scope_display.play)
            if scope_display.play:
                scope_display.play = False
                # Stage B1: pause() instead of stop()+start() so the
                # display thread stays alive across pause-resume; no
                # Thread spawn/join overhead; generation does NOT bump
                # so the texture stays on the last rendered frame.
                scope_display.pause()
                if self.scope.led_connected:
                    self._pause_led_snapshot = self.scope.illumination.save_led_state(
                        'camera_pause'
                    )
                    self.scope.illumination.leds_off_async()
                    # LED observer handles UI button sync
            else:
                if self._pause_led_snapshot:
                    self.scope.illumination.restore_led_state(self._pause_led_snapshot)
                    self._pause_led_snapshot = None
                    # LED observer handles UI button sync

                scope_display.play = True
                scope_display.resume()
        except Exception as e:
            logger.error(f'[UI] cam_toggle failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def record_button(self):
        gui_logger.button('RECORD')
        ctx = _app_ctx.ctx
        controller = ctx.session.manual_recording

        # Record and Stop are the same ToggleButton: a press that leaves
        # the button 'normal' is the user stopping a live recording.
        if self.ids['record_btn'].state == 'normal':
            controller.stop()
            return

        if controller.is_recording:
            return

        # H-3 fix: snapshot widget values on main thread before submitting
        # to camera executor, since .ids access is not thread-safe.
        # The open layer names the channel; its toggle governs display
        # only. Reading one without the other is what left a brightfield
        # recording with no channel name at all.
        layer = common_utils.get_opened_layer(ctx.image_settings)
        false_color_on = (
            ctx.image_settings.layer_lookup(layer=layer).ids['false_color'].active
            if layer is not None
            else False
        )

        # Start on the camera executor: the controller's start opens the
        # encoder and probes disk, which must not stall the GUI thread.
        ctx.camera_executor.put(
            IOTask(
                self._start_recording_task,
                kwargs={'layer': layer, 'false_color_on': false_color_on},
            )
        )

    def _start_recording_task(self, layer=None, false_color_on=False, dt=None):
        """Camera-executor task: start the controller, surface refusals."""
        controller = _app_ctx.ctx.session.manual_recording
        try:
            controller.start(
                layer=layer,
                false_color_on=false_color_on,
                on_complete=self._on_recording_complete,
            )
        except RecordingRefusedError as e:
            logger.warning(f'[LVP Main  ] Recording refused ({e.reason}): {e.message}')
            from modules.notification_center import notifications

            notifications.error('Recording', e.title, e.message)
            Clock.schedule_once(lambda dt: self._reset_record_button(), 0)
            return
        except Exception:
            # Only refusals were handled, so every other failure left the
            # toggle 'down' with nothing recording -- and a 'down' toggle
            # sends the next press to the stop branch, which returns
            # silently when no recording exists. The user pressed Record
            # and nothing happened, twice. The executor still reports the
            # failure itself, so this re-raises rather than swallowing.
            Clock.schedule_once(lambda dt: self._reset_record_button(), 0)
            raise
        Clock.schedule_once(lambda dt: self._begin_recording_ui(), 0)

    def _begin_recording_ui(self):
        """Main thread: recording is live -- start the status poll."""
        controller = _app_ctx.ctx.session.manual_recording
        # The claim grant already notified the run-state listener, but
        # the engine may not have been live yet at that instant; this
        # level republish lands the recording mirror now that it is.
        _app_ctx.ctx.session.notify_run_state()
        # Set immediately so "Open Last Save Folder" works during the
        # recording, not only after cleanup lands (issue #603's shape).
        if controller.save_folder is not None:
            set_last_save_folder(controller.save_folder)
        if self._recording_poll is None:
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
        # Selection closed (Stop, duration cap, or budget full). The
        # live->drain flip has no claim transition of its own, so this
        # poll is the edge that republishes the mirrors (unlocking the
        # controls); the claim stays held until the drain ends, so a
        # protocol start attempted during the drain still gets its loud
        # refusal.
        _app_ctx.ctx.session.notify_run_state()
        self._reset_record_button()
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
        _app_ctx.ctx.session.notify_run_state()
        controller = _app_ctx.ctx.session.manual_recording
        if controller.save_folder is not None:
            set_last_save_folder(controller.save_folder)
        set_title_event_text(None)
        self._reset_record_button()
        logger.info('[LVP Main  ] Manual recording UI cleanup complete')

    def _reset_record_button(self):
        try:
            if self.ids['record_btn'].state != 'normal':
                self.ids['record_btn'].state = 'normal'
        except Exception as e:
            logger.warning(f'[LVP Main  ] Failed to reset record button state: {e}')

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
        try:
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
        except Exception as e:
            logger.error(f'[UI] one2one_image failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

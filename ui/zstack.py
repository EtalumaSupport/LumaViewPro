# Copyright Etaluma, Inc.
import logging

from kivy.clock import Clock
from kivy.properties import BooleanProperty

from kivy.uix.floatlayout import FloatLayout

import modules.common_utils as common_utils
import modules.app_context as _app_ctx
from modules import gui_logger
from modules.config_ui_getters import is_image_saving_enabled
from modules.run_outcome import PendingRunOutcome
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    live_display_callbacks,
    reset_title,
    set_last_save_folder,
    set_recording_title,
    set_writing_title,
    submit_reported,
    sync_layer_widgets_from_settings,
)
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.ui.zstack')


class ZStack(FloatLayout):
    # The handle this button's last start returned: what its Stop names.
    # The engine answers whether it is still the live run.
    _zstack_run: PendingRunOutcome | None = None
    # True while this button's own request is on its way to the engine; the
    # button is disabled until that request's redraw.
    zstack_pending = BooleanProperty(False)

    def set_steps(self):
        logger.info('[LVP Main  ] ZStack.set_steps()')
        settings = _app_ctx.ctx.settings

        # This handler rewrites the widget only when it coerces a bad entry, so
        # comparing the text before and after IS the coercion signal. The typed
        # value itself is recorded by log_step_field, which the kv binds ahead
        # of this handler on the same events so it reads the box first.
        typed = {wid: self.ids[wid].text for wid in ('zstack_stepsize_id', 'zstack_range_id')}

        try:
            step_size = float(self.ids['zstack_stepsize_id'].text)
            if step_size < 0:
                step_size = 0
                self.ids['zstack_stepsize_id'].text = str(step_size)
        except Exception:
            step_size = 0
            self.ids['zstack_stepsize_id'].text = str(step_size)
        finally:
            with _app_ctx.ctx.settings_lock:
                settings['zstack']['step_size'] = step_size

        try:
            step_range = float(self.ids['zstack_range_id'].text)
            if step_range < 0:
                step_range = 0
                self.ids['zstack_range_id'].text = str(step_range)
        except Exception:
            step_range = 0
            self.ids['zstack_range_id'].text = str(step_range)
        finally:
            with _app_ctx.ctx.settings_lock:
                settings['zstack']['range'] = step_range

        for wid, name in (
            ('zstack_stepsize_id', 'ZSTACK_STEP_SIZE'),
            ('zstack_range_id', 'ZSTACK_RANGE'),
        ):
            if self.ids[wid].text != typed[wid]:
                gui_logger.text_input(f'{name}_APPLIED', self.ids[wid].text)

        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=self.ids['zstack_spinner'].text
        )

        zstack_config = ZStackConfig(
            range=settings['zstack']['range'],
            step_size=settings['zstack']['step_size'],
            current_z_reference=z_reference,
            current_z_value=None,
        )

        self.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

    def log_step_field(self, name: str, widget_id: str) -> None:
        """Record a typed commit in one of the two z-stack extent fields.

        Both fields share ``set_steps`` as their handler, so a log call placed
        inside it could not say which field the user edited and would fire for
        both whenever either committed. Logging from the binding keeps each
        field's record its own.

        The value recorded is what the user typed. ``set_steps`` coerces a bad
        entry to 0; that coercion is the system reacting, which belongs in the
        main log, while this file records what the user did.
        """
        gui_logger.text_input(name, self.ids[widget_id].text)

    def set_position(self) -> None:
        gui_logger.select('ZSTACK_REFERENCE_POSITION', self.ids['zstack_spinner'].text)
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            ctx.settings['zstack']['position'] = self.ids['zstack_spinner'].text

    def run_zstack_acquire_from_ui(self):
        """Start a z-stack, or stop the one this button started.

        Everything about the run -- the objective, the stack, the capture,
        and every refusal of it -- is the member's, and the boundary shows
        a refusal once. This button states only what a running GUI knows:
        the open drawer, its own token, the live engineering flag and the
        engineering panel's saving switch, all read here on the GUI thread.

        Whether the press means Stop is the engine's answer -- is the run
        this button started still live -- never the toggle's, which Kivy has
        already flipped. The button changes nothing ahead of the engine's
        answer; draw_zstack_button shows it.
        """
        gui_logger.button('ZSTACK')
        logger.info('[LVP Main  ] ZStack.run_zstack_acquire_from_ui()')
        ctx = _app_ctx.ctx
        run = self._zstack_run
        # The button is disabled until this request's own redraw, so a
        # second press cannot race the first one to the pool.
        self.zstack_pending = True

        if ctx.sequenced_capture_runner.is_live_run(run):
            submit_reported(
                lambda: ctx.sequenced_capture_runner.reset(run),
                self._zstack_request_done,
                'ZSTACK',
                stop=True,
            )
            return

        runner = ctx.session.create_protocol_runner()
        layer = common_utils.get_opened_layer(ctx.image_settings)
        engineering_mode = ctx.engineering_mode
        enable_image_saving = is_image_saving_enabled()
        callbacks = {
            **live_display_callbacks(),
            'move_position': _handle_ui_update_for_axis,
            # Each slice redraws the button, which reads the step from the
            # engine: a redraw from any other edge draws the same thing.
            'update_step_number': lambda step_num: self.draw_zstack_button(),
            # LED observer handles UI sync -- no manual callbacks needed
            'sync_layer_widgets': sync_layer_widgets_from_settings,
            'set_recording_title': set_recording_title,
            'set_writing_title': set_writing_title,
            'reset_title': reset_title,
            'pause_live_ui': lambda: (
                ctx.scope_display.stop(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1),
            ),
        }

        def _start():
            self._zstack_run = runner.run_zstack(
                layer=layer,
                callbacks=callbacks,
                run_trigger_source='zstack',
                engineering_mode=engineering_mode,
                enable_image_saving=enable_image_saving,
            )
            # A refusal raises out of run_zstack before this line, so the
            # save folder can only ever point at THIS run's directory,
            # never a previous run's stale data.
            set_last_save_folder(dir=runner.run_dir())

        submit_reported(_start, self._zstack_request_done, 'ZSTACK')

    def _zstack_request_done(self):
        self.zstack_pending = False
        self.draw_zstack_button()

    def draw_zstack_button(self):
        """Show the z-stack this button started, as the engine reports it.

        The only code that styles the button: after each of its own
        requests, on every slice, and on every run-state edge -- including
        the run's return to idle. "Z n/total" is the run's own step and
        count, asked of the engine, because the member built the protocol
        and this widget never holds it.
        """
        engine = _app_ctx.ctx.sequenced_capture_runner
        button = self.ids['zstack_aqr_btn']
        if not engine.is_live_run(self._zstack_run):
            button.state = 'normal'
            button.text = 'Acquire'
            return
        button.state = 'down'
        if engine.is_stopping(self._zstack_run):
            button.text = 'Stopping...'
            return
        step, total = engine.run_step_number(), engine.run_num_steps()
        # Both are None only once the run has ended, between the live read
        # above and these; the run-state edge that follows draws it idle.
        button.text = 'Running Z-Stack' if step is None or total is None else f'Z {step}/{total}'

# Copyright Etaluma, Inc.
import logging

from kivy.properties import BooleanProperty

from kivy.uix.floatlayout import FloatLayout

import modules.common_utils as common_utils
import modules.app_context as _app_ctx
from modules import gui_logger
from modules.config_ui_getters import is_image_saving_enabled
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import RunHandle
from modules.run_events import RunEvents
from ui.ui_helpers import (
    set_last_save_folder,
    show_captured_frame,
    show_video_progress,
    submit_reported,
    sync_layer_widgets_from_settings,
    typed_number,
)
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.ui.zstack')


class ZStack(FloatLayout):
    # The handle this button's last start returned: what its Stop names.
    # The engine answers whether it is still the live run.
    _zstack_run: 'RunHandle | None' = None
    # True while this button's own request is on its way to the engine; the
    # button is disabled until that request's redraw.
    zstack_pending = BooleanProperty(False)
    # True while anything but this button's own run holds the scope -- another
    # run, a recording, a diagnostic -- as the Session answers; greys the
    # button, and its own run leaves it live as that run's Stop.
    zstack_held = BooleanProperty(False)

    def set_steps(self):
        logger.info('[LVP Main  ] ZStack.set_steps()')
        settings = _app_ctx.ctx.settings

        # This handler rewrites the widget only when it puts back an entry
        # that is not a number, so comparing the text before and after IS the
        # put-back signal. The typed value itself is recorded by
        # log_step_field, which the kv binds ahead of this handler on the same
        # events so it reads the box first.
        typed = {wid: self.ids[wid].text for wid in ('zstack_stepsize_id', 'zstack_range_id')}

        # A number is stored as typed, whatever its sign: a stack whose step
        # or range is not above zero is refused when it is built, naming the
        # values the person entered, and the Steps field reads 0 meanwhile.
        for wid, key in (('zstack_stepsize_id', 'step_size'), ('zstack_range_id', 'range')):
            box = self.ids[wid]

            def put_back(box=box, key=key):
                box.text = str(settings['zstack'][key])

            value = typed_number(box.text, float, put_back)
            if value is not None:
                _app_ctx.ctx.update_settings(f'zstack.{key}', value)

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

        The value recorded is what the user typed. ``set_steps`` puts back an
        entry that is not a number and records that under ``<NAME>_APPLIED``;
        this records what the user did.
        """
        gui_logger.text_input(name, self.ids[widget_id].text)

    def set_position(self) -> None:
        gui_logger.select('ZSTACK_REFERENCE_POSITION', self.ids['zstack_spinner'].text)
        _app_ctx.ctx.update_settings('zstack.position', self.ids['zstack_spinner'].text)

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

        if run is not None and run.is_live:
            submit_reported(
                run.stop,
                self._zstack_request_done,
                'ZSTACK',
                stop=True,
            )
            return

        runner = ctx.session.create_protocol_runner()
        layer = common_utils.get_opened_layer(ctx.image_settings)
        engineering_mode = ctx.engineering_mode
        enable_image_saving = is_image_saving_enabled()
        events = RunEvents(
            frame_captured=show_captured_frame,
            # Each slice redraws the button, which reads the step from the
            # engine: a redraw from any other edge draws the same thing.
            step_started=lambda step_idx: self.draw_zstack_button(),
            video_progress=show_video_progress,
            run_ended=lambda *ended: sync_layer_widgets_from_settings(),
        )

        def _start():
            started = runner.run_zstack(
                layer=layer,
                events=events,
                run_trigger_source='zstack',
                engineering_mode=engineering_mode,
                enable_image_saving=enable_image_saving,
            )
            self._zstack_run = started
            # A refusal raises out of run_zstack before this line, so the
            # save folder can only ever point at THIS run's directory,
            # never a previous run's stale data.
            set_last_save_folder(dir=started.run_dir)

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
        ctx = _app_ctx.ctx
        self.zstack_held = ctx.session.held_by_other(self._zstack_run)
        button = self.ids['zstack_aqr_btn']
        label = _running_label(self._zstack_run)
        if label is None:
            button.state = 'normal'
            button.text = 'Acquire'
            return
        button.state = 'down'
        button.text = label


def _running_label(run: 'RunHandle | None') -> str | None:
    """What the button says while its run is live; None when it is not.

    None also when the run ended between the live read and the progress
    reads -- its progress is None then -- so the button draws what it read
    rather than a running label for a run that has ended.
    """
    if run is None or not run.is_live:
        return None
    if run.is_stopping:
        return 'Stopping...'
    step, total = run.step_number, run.num_steps
    if step is None or total is None:
        return None
    return f'Z {step}/{total}'

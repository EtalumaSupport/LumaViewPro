# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import time
import typing

from kivy.clock import Clock
from kivy.properties import BooleanProperty

from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from modules.config_ui_getters import (
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_image_capture_config_from_ui,
    get_layer_configs,
    get_protocol_time_params,
    get_selected_labware,
    get_zstack_params,
    is_image_saving_enabled,
)
from modules.protocol import Protocol
from modules.run_outcome import PendingRunOutcome
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from ui.step_navigation import go_to_step
from modules.timedelta_formatter import strfdelta
from modules import exceptions, gui_logger
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    _update_step_number_callback,
    live_display_callbacks,
    reset_acquire_ui,
    reset_stim_ui,
    reset_title,
    run_reported,
    set_last_save_folder,
    set_recording_title,
    set_writing_title,
    submit_reported,
    sync_layer_widgets_from_settings,
)
from ui.progress_popup import show_popup

logger = logging.getLogger('LVP.ui.protocol_settings')


def _offer_wedged_writer_recovery():
    """Modal offering discard-and-unlock recovery for a stalled file writer.

    Names the stuck write and the cost of recovery; declining leaves the
    queue untouched, and a run start's refusal carries the same recovery."""
    from ui.notification_popup import show_confirmation_popup

    session = _app_ctx.ctx.session
    pending = session.protocol_files_pending
    stuck = session.protocol_files_stuck_write

    def _recover():
        logger.warning('[LVP Main  ] User confirmed wedged-writer recovery')
        # Through the Session, which refuses while a run or a diagnostic
        # holds the scope; the executor's own recovery would not.
        run_reported(session.recover_file_writer, None, 'RECOVER_FILE_WRITER')

    show_confirmation_popup(
        title='File Writer Stalled',
        message=(
            f'File saving has stopped making progress ({stuck}). '
            f'Discarding will unlock the app; {pending} unsaved image(s) '
            f'from the last run will be lost. A partial file from the stuck '
            f'write may remain on disk and stay locked until the stuck '
            f'write releases it.'
        ),
        confirm_text=f'Discard {pending} unsaved and unlock',
        cancel_text='Keep waiting',
        on_confirm=_recover,
    )


_ABORT_BACKGROUND = './data/icons/abort_protocol_background.png'


class _PanelRunButton(typing.NamedTuple):
    button_id: str
    held_flag: str  # the panel property that greys the button while another holds the scope
    label: str  # the boundary's category for this button's requests
    idle_text: str
    running_text: str
    running_background: str | None
    idle_background: str | None


# The protocol panel's three run buttons, by the trigger each starts its run
# with -- the key its run's handle is kept under.
_PANEL_RUN_BUTTONS = {
    'scan': _PanelRunButton(
        'run_scan_btn',
        'scan_held',
        'PROTOCOL_SCAN',
        'Run One Scan',
        'Abort One Scan',
        _ABORT_BACKGROUND,
        None,
    ),
    'protocol': _PanelRunButton(
        'run_protocol_btn',
        'protocol_held',
        'PROTOCOL_RUN',
        'Run Full Protocol',
        '',
        _ABORT_BACKGROUND,
        'atlas://data/images/defaulttheme/button_pressed',
    ),
    'autofocus_scan': _PanelRunButton(
        'run_autofocus_btn',
        'autofocus_scan_held',
        'PROTOCOL_AF_SCAN',
        'Autofocus All Steps',
        'Running Autofocus Scan',
        None,
        None,
    ),
}


def _protocol_remaining_text(engine) -> str:
    """The Full Protocol button's running label, from the live run's own count."""
    remaining_scans = engine.remaining_scans()
    remaining_duration_str = strfdelta(
        tdelta=remaining_scans * engine.protocol_interval(),
        fmt='{H}h {M}m',
        inputtype='timedelta',
    )
    scan_word = 'scan' if remaining_scans == 1 else 'scans'
    return f'{remaining_scans} {scan_word} ({remaining_duration_str}) remaining.\nPress to ABORT'


class ProtocolSettings(FloatLayout):
    done = BooleanProperty(False)
    # Drives the advisory label's height and opacity in the kv. Display state
    # only -- whether a protocol IS large is the session's answer, not this
    # flag's.
    protocol_size_advisory_active = BooleanProperty(False)
    # True while that run button's own request is on its way to the engine;
    # the button is disabled until the request's redraw.
    scan_pending = BooleanProperty(False)
    protocol_pending = BooleanProperty(False)
    autofocus_scan_pending = BooleanProperty(False)
    # Each True while anything but that button's own run holds the scope --
    # another run, a recording, a diagnostic -- as the Session answers; greys
    # the button, and its own run leaves it live as that run's Stop.
    scan_held = BooleanProperty(False)
    protocol_held = BooleanProperty(False)
    autofocus_scan_held = BooleanProperty(False)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        logger.info('[LVP Main  ] ProtocolSettings.__init__()')

        # Create trigger for debounced UI updates to prevent memory leaks
        self._update_step_ui_trigger = Clock.create_trigger(self._do_update_step_ui, 0.05)

        # The handle each of this panel's run buttons' last start returned,
        # by trigger: what that button's Stop names. The engine answers
        # whether it is still the live run.
        self._runs_started_here: dict[str, PendingRunOutcome] = {}
        # The drain display's tick: one pending at a time, however many
        # redraws ask for it.
        self._drain_tick_trigger = Clock.create_trigger(self._drain_tick, 0.5)

        self.curr_step = -1

        from modules.common_utils import DEFAULT_STAGE_TRAVEL_UM

        self.tiling_min = {
            'x': int(DEFAULT_STAGE_TRAVEL_UM['x']),
            'y': int(DEFAULT_STAGE_TRAVEL_UM['y']),
        }
        self.tiling_max = {'x': 0, 'y': 0}

        # Protocol is owned by AppContext, not this widget.
        # Property delegation below ensures all existing self._protocol
        # references keep working while the canonical owner is ctx.
        self._protocol = None  # bootstraps before ctx exists

        self.exposures = 1  # 1 indexed
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)

    def _do_update_step_ui(self, *args):
        """Actual UI update method, called by trigger."""
        self.update_step_ui_immediate()

    def update_step_ui(self):
        """Triggered version - debounces rapid calls."""
        self._update_step_ui_trigger()

    def update_step_ui_immediate(self):
        """Non-triggered version for immediate updates."""
        num_steps = self._protocol.num_steps()

        # Only update if values changed to prevent unnecessary layout recalculation
        new_step_num = str(self.curr_step + 1)
        if self.ids['step_number_input'].text != new_step_num:
            self.ids['step_number_input'].text = new_step_num

        new_total = str(num_steps)
        if self.ids['step_total_input'].text != new_total:
            self.ids['step_total_input'].text = new_total

        self.generate_step_name_input()
        self._update_step_focus_readout(num_steps=num_steps)
        self._update_protocol_size_advisory()

    def _update_step_focus_readout(self, num_steps: int):
        """Show the selected step's Z in the step editor."""
        label = self.ids.get('step_focus_z_label')
        if label is None:
            return
        if num_steps <= 0 or self.curr_step < 0:
            label.text = ''
            return
        try:
            step = self._protocol.step(idx=self.curr_step)
            label.text = f'{float(step["Z"]):.0f} um'
        except Exception:
            label.text = ''

    def _update_protocol_size_advisory(self):
        """Show the session's size advisory for this protocol, if it has one.

        Renders only. Whether a protocol is large enough to warn about, what
        the sentence says and which settings the estimate needs are all the
        session's answer -- this asks and displays what comes back.
        """
        label = self.ids.get('protocol_size_advisory_label')
        if label is None:
            return

        ctx = _app_ctx.ctx
        # A run cannot change the protocol: the editing surface is locked for
        # its duration and the run mutates its own copy, not this one. Without
        # this the estimate would recompute at the step-navigation refresh rate
        # for the whole length of every run. The label keeps its last text,
        # which stays correct.
        if ctx.session.run_lockout:
            return

        advisory = ctx.session.protocol_size_advisory(self._protocol)
        self.protocol_size_advisory_active = advisory is not None
        label.text = advisory.message if advisory is not None else ''

    @property
    def tiling_config(self):
        """The installation's tiling grids, asked of the scope at each use.

        The panel is built before the session exists, so it cannot hold its
        own copy from construction; it never reads tiling.json itself.
        """
        return _app_ctx.ctx.session.scope.protocols.tiling_config()

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error(
                    '[LVP Main  ] ProtocolSettings._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        settings = ctx.settings

        tiling_config = self.tiling_config
        self.ids['tiling_size_spinner'].values = tiling_config.available_configs()
        self.ids['tiling_size_spinner'].text = tiling_config.default_config()

        # The persisted protocol is NOT loaded here. Whether the scope can
        # perform it depends on the turret, and on a turreted scope the
        # objective at the current slot is not known until the startup
        # question has been answered -- which happens later, in on_start.
        # Loading first meant judging a protocol against a turret
        # configuration that was about to change. The startup sequence
        # calls load_persisted_protocol() once the question resolves.
        #
        # The panel still needs A protocol so nothing downstream reads
        # None.
        self._protocol = ctx.session.create_empty_protocol()

        # The panel applying the plate it already shows, so the scope is on it
        # even when no protocol loaded; not a user pick.
        gui_logger.note_write_back('LABWARE', self.ids['labware_spinner'].text)
        self.select_labware()
        self.update_step_ui()

        # DISABLED: BF AF for fluorescence -- not yet tested, hidden for 4.0.0.
        # Force off regardless of saved settings to prevent untested code path.
        if 'protocol' in settings:
            settings['protocol']['bf_af_for_fluorescence'] = False
        self.ids['bf_af_for_fluorescence_btn'].state = 'normal'

    # Update Protocol Period
    def commit_period(self) -> float | None:
        """Store the typed capture period as soon as it is a number.

        Bound to the field's ``on_text``, so the store tracks the field on
        every keystroke rather than waiting for enter or focus loss. Kivy
        runs a button's handler BEFORE the focus-loss commit, so without
        this a user who types a period and clicks Run, Save or New Protocol
        is read from a store still holding the previous value -- while the
        screen shows the new one.

        Silent and tolerant by design: a half-typed value is not an error,
        it is just not a value yet, and the enter / focus-loss path still
        reports one that never parses. Reporting here instead would consume
        the notification bus's dedup slot for this category and swallow the
        legitimate sub-second clamp warning that follows it.

        The store write is deliberately OUTSIDE the parse guard: an absent
        ``protocol`` container is a broken configuration, not a typing
        error, and the template ships the key.
        """
        try:
            raw_period = float(self.ids['capture_period'].text)
        except ValueError:
            return None
        _app_ctx.ctx.settings['protocol']['period'] = raw_period
        return raw_period

    def commit_duration(self) -> float | None:
        """Store the typed capture duration as soon as it is a number.

        The period twin above carries the reasoning; this is the same
        contract for the duration field.
        """
        try:
            raw_duration = float(self.ids['capture_dur'].text)
        except ValueError:
            return None
        _app_ctx.ctx.settings['protocol']['duration'] = raw_duration
        return raw_duration

    def update_period(self):
        # One import for the three messages below, deferred to call time the
        # way every notification site in this file is.
        from modules.notification_center import notifications

        logger.info('[LVP Main  ] ProtocolSettings.update_period()')
        try:
            raw_period = self.commit_period()
            if raw_period is None:
                raise ValueError(self.ids['capture_period'].text)
            # Warn once, at the edit, when a sub-1s period is raised to the 1s
            # minimum -- so the user is told why the field shows 0.016667 min
            # instead of their typed value. The getter stays silent so save /
            # run-start do not re-warn.
            if config_helpers.protocol_time_clamped(raw_period, 'minutes'):
                notifications.warning(
                    'Protocol',
                    'Capture Timing',
                    'The capture period was below the 1-second minimum and was '
                    'raised to 1 second (shown as 0.016667 min). Enter a period '
                    'of at least 1 second.',
                )
        except Exception:
            logger.exception('[LVP Main  ] Update Period is not an acceptable value')
            # Say so where the value was typed. The store keeps its previous
            # period, so without this the edit would look like it was taken
            # while the protocol still ran on the old schedule.
            notifications.warning(
                'Protocol',
                'Capture Timing',
                'The capture period was not a number, so it was not changed. '
                'Enter a period in minutes.',
            )

        gui_logger.text_input('PROTOCOL_PERIOD', self.ids['capture_period'].text)

        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        try:
            time_params = get_protocol_time_params()
        except exceptions.ConfigError as e:
            # The stored schedule itself is unusable -- a hand-edited settings
            # file reaches here, because the load compares container shape and
            # never scalar values. Render what the store refused and leave the
            # protocol on its current timing rather than crashing the handler.
            logger.error(f'[LVP Main  ] Stored protocol timing is unusable: {e}')
            notifications.warning('Protocol', 'Capture Timing', str(e))
            return
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    # Update Protocol Duration
    def update_duration(self):
        from modules.notification_center import notifications

        logger.info('[LVP Main  ] ProtocolSettings.update_duration()')
        try:
            raw_duration = self.commit_duration()
            if raw_duration is None:
                raise ValueError(self.ids['capture_dur'].text)
            # Duration is in HOURS, so a sub-1s value shows as 0.000278 hr (not
            # 0.016667 min). Warn once, at the edit, with the hour value.
            if config_helpers.protocol_time_clamped(raw_duration, 'hours'):
                notifications.warning(
                    'Protocol',
                    'Capture Timing',
                    'The capture duration was below the 1-second minimum and was '
                    'raised to 1 second (shown as 0.000278 hr). Enter a duration '
                    'of at least 1 second.',
                )
        except Exception:
            logger.warning('[LVP Main  ] Update Duration is not an acceptable value')
            # Same reason as the period field: the store keeps its previous
            # duration, so a silent return would look like the edit was taken.
            notifications.warning(
                'Protocol',
                'Capture Timing',
                'The capture duration was not a number, so it was not changed. '
                'Enter a duration in hours.',
            )

        gui_logger.text_input('PROTOCOL_DURATION', self.ids['capture_dur'].text)

        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        try:
            time_params = get_protocol_time_params()
        except exceptions.ConfigError as e:
            logger.error(f'[LVP Main  ] Stored protocol timing is unusable: {e}')
            notifications.warning('Protocol', 'Capture Timing', str(e))
            return
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    def step_name_validation(self, text: str):
        # What the user typed, before the sanitiser and the rename decide what
        # to make of it. RENAME_STEP below reports the name that took effect;
        # without this line a name the sanitiser changed, or a blank entry that
        # kept the old name, leaves nothing saying what was actually entered.
        gui_logger.text_input('STEP_NAME', text)
        if (
            hasattr(self, '_protocol')
            and (self._protocol is not None)
            and (self._protocol.num_steps() > 0 and self.curr_step >= 0)
        ):
            new_name = common_utils.resolve_step_rename(text, Protocol.sanitize_step_name)
            if new_name is None:
                # Blank field = keep the existing name; leave the field
                # empty so the auto-name hint shows.
                self.ids['step_name_input'].text = ''
                return
            self._protocol.modify_name(step_idx=self.curr_step, step_name=new_name)
            gui_logger.protocol_action('RENAME_STEP', f'step={self.curr_step} name={new_name!r}')
            self.ids['step_name_input'].text = new_name
        else:
            self.ids['step_name_input'].text = ''

    def update_capture_root(self, text: str):
        # Sanitize and store capture root on protocol to avoid invalid path chars
        sanitized = Protocol.sanitize_step_name(text)
        # What the user typed, then what sanitizing made of it. Recording only
        # the sanitized string asserts the user typed something they did not.
        gui_logger.text_input('CAPTURE_ROOT', text)
        if sanitized != text:
            gui_logger.text_input('CAPTURE_ROOT_APPLIED', sanitized)
        self.ids['capture_root'].text = sanitized
        if hasattr(self, '_protocol') and (self._protocol is not None):
            self._protocol.modify_capture_root(capture_root=sanitized)

    # Labware Selection
    def select_labware(self, labware: str | None = None):
        ctx = _app_ctx.ctx
        wellplate_loader = ctx.wellplate_loader

        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        if labware is None:
            spinner = self.ids['labware_spinner']
            spinner.values = wellplate_loader.get_plate_list()
            gui_logger.select('LABWARE', spinner.text)
            # An empty spinner (not yet populated at startup) names no
            # plate, so there is nothing to select; bring-up already put
            # the stored plate in place and it stays there.
            selected = spinner.text
        else:
            center_plate_str = 'Center Plate'
            spinner = self.ids['labware_spinner']
            spinner.values = [center_plate_str]
            # Forcing the spinner re-enters this method through its text event;
            # the app falling back to Center Plate is not the user choosing it.
            gui_logger.note_write_back('LABWARE', center_plate_str)
            spinner.text = center_plate_str
            selected = labware

        if selected:
            # A refusal -- a plate change while a recording still holds the
            # scope, or a stored name the catalogue no longer has -- leaves
            # both stores on the plate they had, and the tail below renders
            # that plate, so the panel is never left describing a selection
            # the scope did not take.
            run_reported(lambda: ctx.session.select_labware(selected), None, 'LABWARE')

        labware_id, _labware_obj = get_selected_labware()

        if self._protocol is not None:
            self._protocol.modify_labware(labware_id=labware_id)

        ctx.stage.full_redraw()

    def set_focus_control_visibility(self, visible: bool) -> None:
        for focus_id in (
            'step_focus_row_id',
            'protocol_zstacking_box_layout_id',
            'protocol_acquire_zstack_box_id',
            'run_autofocus_btn',
        ):
            self.ids[focus_id].visible = visible

    def set_labware_selection_visibility(self, visible):
        labware_spinner = self.ids['labware_spinner']
        labware_spinner.visible = visible
        labware_spinner.size_hint_y = None if visible else 0
        labware_spinner.height = '30dp' if visible else 0
        labware_spinner.opacity = 1 if visible else 0
        labware_spinner.disabled = not visible

        if not visible:
            # The app hides the choice and parks the spinner; not a user pick.
            gui_logger.note_write_back('LABWARE', 'Center Plate')
            labware_spinner.text = 'Center Plate'
        else:
            # UI-1 follow-up (plate-spinner): when re-enabling labware
            # selection after a scope switch (e.g., LS620 -> LS850), the
            # spinner widget is re-enabled but the dropdown values are
            # still locked to ['Center Plate'] from the prior
            # select_labware('Center Plate') call. User can click the
            # spinner but has no other choices. Restore the full plate
            # list and the saved labware.
            ctx = _app_ctx.ctx
            saved_labware = ctx.settings.get('protocol', {}).get('labware')
            wellplate_loader = ctx.wellplate_loader
            try:
                labware_spinner.values = wellplate_loader.get_plate_list()
                if saved_labware and saved_labware in labware_spinner.values:
                    gui_logger.note_write_back('LABWARE', saved_labware)
                    labware_spinner.text = saved_labware
            except Exception as e:
                logger.warning(f'[LVP Main  ] Failed to restore labware list on scope switch: {e}')

    def apply_tiling(self) -> None:
        # At entry, not on success: the protocol can refuse the grid, and a
        # record conditional on success would make that refusal
        # indistinguishable from the user never pressing the button.
        gui_logger.button('APPLY_TILING')
        run_reported(self._apply_tiling, None, 'APPLY_TILING')

    def _apply_tiling(self) -> None:
        """Ask the protocol for the chosen grid, then show the steps it built.

        Every refusal (an unknown grid, an already-tiled protocol, an unknown
        objective, a scope with no X/Y motor, a tile outside the stage's
        travel) is the protocol's, raised before any step changes; the
        boundary reports it, so nothing here decides it.
        """
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        logger.info('[LVP Main  ] Apply tiling to protocol')

        axes_config = ctx.lumaview.scope.motion.get_axes_config()
        _, labware = get_selected_labware()
        stage_offset = settings['stage_offset']
        overlap_percent = self.get_tiling_overlap_percent()

        self._protocol.apply_tiling(
            tiling=self.ids['tiling_size_spinner'].text,
            frame_dimensions=config_helpers.get_frame_dimensions_from_settings(settings),
            binning_size=get_binning_from_ui(),
            curr_step_idx=self.curr_step,
            axes_config=axes_config,
            labware=labware,
            stage_offset=stage_offset,
            overlap_percent=overlap_percent,
            capabilities=ctx.lumaview.scope.capabilities,
            objective_helper=ctx.lumaview.scope.objective_helper,
        )

        self._protocol.optimize_step_ordering()
        ctx.stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step(step_idx=self.curr_step)

    def get_tiling_overlap_percent(self) -> float:
        """Tile overlap percentage, read from the persisted system setting.

        The single accessor for tile overlap at scan/apply time; the editor
        (the spinner in Advanced Settings) only ever writes the setting.
        """
        return _app_ctx.ctx.settings['tiling_overlap_percent']

    def apply_zstacking(self) -> None:
        # At entry, not on success: the protocol can refuse the stack, and a
        # record conditional on success would make that refusal
        # indistinguishable from the user never pressing the button.
        gui_logger.button('APPLY_ZSTACKING')
        run_reported(self._apply_zstacking, None, 'APPLY_ZSTACKING')

    def _apply_zstacking(self) -> None:
        """Ask the protocol for a z-stack of the panel's values, then show the steps.

        Every refusal (a range or step size not greater than zero, a scope
        with no Z motor, a slice outside the Z travel) is the protocol's,
        raised before any step changes; the boundary reports it, so nothing
        here decides it.
        """
        ctx = _app_ctx.ctx

        logger.info('[LVP Main  ] Apply Z-Stacking to protocol')

        self._protocol.apply_zstacking(
            zstack_params=get_zstack_params(),
            axes_config=ctx.lumaview.scope.motion.get_axes_config(),
        )

        self._protocol.optimize_step_ordering()
        ctx.stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step(step_idx=self.curr_step)

    def generate_step_name_input(self):
        num_steps = self._protocol.num_steps()
        if num_steps > 0:
            step = self.get_curr_step()
            if step['Auto_Named'] or step['Label'] == '':
                # A step still on its auto-generated name shows the rendered
                # default as a placeholder hint and leaves the field blank,
                # so the user can type over it; blank means "keep".
                new_text = ''
                new_hint = self.get_default_name_for_curr_step()
            else:
                # A user-labeled step shows its label -- the user's own text,
                # not the rendered name it decorates.
                new_text = step['Label']
                new_hint = self.ids['step_name_input'].hint_text  # Keep existing hint

        else:
            new_text = ''
            new_hint = 'Step Name'

        # Only update if changed to prevent unnecessary ScrollView layout recalculation
        if self.ids['step_name_input'].text != new_text:
            self.ids['step_name_input'].text = new_text
        if self.ids['step_name_input'].hint_text != new_hint:
            self.ids['step_name_input'].hint_text = new_hint

    def new_protocol(self):
        ctx = _app_ctx.ctx

        logger.info('[LVP Main  ] ProtocolSettings.new_protocol()')

        # The click, before any of the ways this returns without building
        # anything. Every refusal below does notify, but the notification text
        # is shared -- the builder's refusal is shared by nine callers -- so
        # without this line the bundle shows a refusal and no way to tell
        # which button provoked it. Recorded once at the top rather than at
        # each return: one line gives the attribution, and the reason arrives
        # in the notification that follows.
        gui_logger.button('NEW_PROTOCOL')

        # New Protocol resets each step to its channel's saved focus baseline.
        # A per-(well, channel) Z carry-over from the prior in-memory protocol
        # used to run here, but it harvested autofocus-refined Z along with
        # user-tuned Z and so overrode a freshly-saved focus; it is no longer
        # the default. Per-well focus is re-established on demand via
        # "Autofocus All Steps". Protocol.from_config still honors an explicit
        # previous_well_z map (left dormant) so this can return as an opt-in
        # setting without re-plumbing.

        # The two authoring choices live only in this panel's widgets, so the
        # panel states them; the Session assembles everything else. Left to
        # the member's defaults they read 1x1 and no z-stack, and the built
        # protocol silently loses the user's choice.
        tiling = self.ids['tiling_size_spinner'].text
        use_zstacking = self.ids['acquire_zstack_id'].active
        built = []

        def _build():
            # The config carries the active objective, so it raises while
            # that is unknown (a turret move in flight, an unassigned slot);
            # the boundary shows the API's reason as a refusal.
            config = ctx.session.get_sequenced_capture_config(
                tiling=tiling, use_zstacking=use_zstacking
            )
            built.append(ctx.scope.protocols.create_protocol(input_config=config))

        # Inline, so the lines below see what the build produced; a refused
        # or failed build has been shown by the boundary and built nothing.
        run_reported(_build, self.update_step_ui, 'NEW_PROTOCOL')
        if not built:
            return
        protocol = built[0]

        if protocol.num_steps() == 0:
            # Zero steps has two distinct causes: no channel is enabled for
            # acquisition, or the labware has no wells (e.g. Blank, a 0x0
            # plate). Only the first is a channel problem -- attribute it by
            # checking the same channel predicate Add uses. A no-well labware
            # with channels enabled creates an empty protocol the user builds
            # up with Add at the current stage position.
            layer_configs = get_layer_configs()
            any_channel_enabled = any(lc['acquire'] is not None for lc in layer_configs.values())
            if not any_channel_enabled:
                logger.warning('[LVP Main  ] new_protocol: no channels enabled for acquisition')
                from ui.notification_popup import show_notification_popup

                show_notification_popup(
                    title='No Channels Selected',
                    message=(
                        'No channels are enabled for acquisition. Please enable '
                        'at least one channel for image or video capture in the '
                        'layer settings on the right, then create the protocol '
                        'again.'
                    ),
                )
                return
            logger.info(
                '[LVP Main  ] new_protocol: labware has no wells; created '
                'empty protocol (use Add to insert steps)'
            )

        # Recorded like LOAD and SAVE: creating a protocol replaces the whole
        # step table, so it is one of the few actions that changes what every
        # later step value means. Without it the interaction log shows a run
        # starting over steps that appear from nowhere, and reconstructing a
        # field report costs an investigation.
        gui_logger.protocol_action('NEW', f'steps={protocol.num_steps()}')

        def _redraw():
            # A new protocol has no file. Once the panel holds the one this
            # press built, its file name and capture root are cleared; a
            # refused adoption leaves both as they were.
            if self._protocol is protocol:
                self.ids['protocol_filename'].text = ''
                self.ids['capture_root'].text = ''
            self._draw_protocol_steps()

        run_reported(lambda: self.new_protocol_ex(protocol), _redraw, 'NEW_PROTOCOL')

    def new_protocol_ex(self, protocol):
        """Adopt *protocol* once the API accepts the objectives it names.

        The API owns the rule and raises its own refusal, so there is nothing
        to decide or announce here: a protocol built from a selection the
        scope cannot address is simply not adopted. The move to the first
        step comes last, reached only once the protocol is the panel's.
        """
        ctx = _app_ctx.ctx
        ctx.scope.protocols.refuse_unaddressable_objectives(protocol.steps()['Objective'].to_list())
        self._protocol = protocol
        ctx.settings['protocol']['filepath'] = ''
        self.curr_step = 0
        self.go_to_step(step_idx=0)

    def _draw_protocol_steps(self) -> None:
        """Show the panel's protocol: its steps on the stage and in the step editor."""
        _app_ctx.ctx.stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()

    def _validate_labware(self, labware: str):
        ctx = _app_ctx.ctx

        # Asked of the drivers rather than the selected scope model, which a
        # user can change mid-session -- see set_ui_features_for_scope.
        # If XY motion is available, any type of labware is acceptable
        if ctx.lumaview.scope.capabilities.has_xy_stage:
            return True, labware

        # If XY motion is not available, only Center Plate
        if labware == 'Center Plate':
            return True, labware
        else:
            return False, 'Center Plate'

    @show_popup
    def _show_popup_message(self, popup, title, message, delay_sec):
        popup.title = title
        popup.text = message
        time.sleep(delay_sec)
        # `done` is a Kivy BooleanProperty whose `done=True` write triggers
        # the bound `popup.dismiss` dispatch. The decorator runs this method
        # on a daemon Thread; writing the property here would dispatch
        # `popup.dismiss` on the worker thread and can corrupt the Kivy
        # property graph mid-dispatch. Marshal to the UI thread.
        Clock.schedule_once(lambda dt: setattr(self, 'done', True), 0)

    def load_persisted_protocol(self) -> None:
        """Load the protocol the last session left behind, once, at startup.

        Called by the startup sequence after the objective question has
        been answered or found not to be owed, because the answer decides
        what the turret carries and therefore whether the saved protocol
        can be performed at all.

        A refusal KEEPS the remembered path. The file is real and the user
        chose it; what is wrong is the scope's turret, which they can fix
        and then reload. Clearing it would answer "your protocol is gone"
        to a problem that is not about the protocol. Only a path with no
        file behind it is forgotten, because there is nothing left to
        remember.

        Non-navigating, as the startup load has always been: it adopts the
        protocol and fills the panel without driving the stage.
        """
        ctx = _app_ctx.ctx
        settings = ctx.settings
        filepath = settings['protocol']['filepath']

        try:
            loaded = self.load_protocol(filepath=filepath, suppress_popup=True, navigate=False)
        except Exception:
            logger.exception('[LVP Main  ] Error loading protocol at startup')
            loaded = False

        if loaded:
            return

        if not filepath or not pathlib.Path(filepath).exists():
            logger.info('[LVP Main  ] No saved protocol loaded at startup -- using empty protocol')
            settings['protocol']['filepath'] = ''
        else:
            # The file is still there and something about this scope
            # refused it. Keep the name on screen as well as in settings:
            # it is the only thing telling the user which protocol they
            # need to come back to, and load_protocol only writes the
            # label on the path where it succeeds.
            self.ids['protocol_filename'].text = os.path.basename(filepath)
            logger.info(
                f'[LVP Main  ] Saved protocol {filepath} was not adopted at startup; '
                'its path is kept so it can be reloaded once the scope can perform it'
            )

        self._protocol = ctx.session.create_empty_protocol()
        self.update_step_ui()

    # Load Protocol from File
    def load_protocol(
        self, filepath='./data/new_default_protocol.tsv', suppress_popup=False, *, navigate
    ):
        """Adopt a protocol from disk and fill the panel.

        ``navigate`` says whether a person asked for this load, and so
        whether the stage may drive to the current step. Required rather
        than defaulted: a default is what let the startup adoption inherit
        an answer nobody chose for it.
        """
        gui_logger.protocol_action('LOAD', filepath)
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

        if not pathlib.Path(filepath).exists():
            if suppress_popup:
                return False
            raise FileNotFoundError(f'Protocol not found at {filepath}')

        try:
            protocol = ctx.scope.protocols.load_protocol(file_path=filepath)
        except OSError:
            return False

        except exceptions.ProtocolRunRefusedError:
            # Already logged and shown to the user by the API's funnel, in
            # its own words. The blanket handler below would render str(e),
            # which is the joined `reason: message` debugging form.
            return False

        except Exception as e:
            logger.warning(f'[LVP Main  ] Protocol load failed: {e}')
            if not suppress_popup:
                error_title = 'Protocol Loading Error'
                error_msg = f'Cannot load protocol from file: {e}'
                from ui.notification_popup import show_notification_popup

                show_notification_popup(title=error_title, message=error_msg)
            return False

        self._protocol = protocol

        settings['protocol']['filepath'] = filepath
        self.ids['protocol_filename'].text = os.path.basename(filepath)

        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            self.curr_step = -1
        else:
            self.curr_step = 0

        # 6 decimals (matching the TSV write side) so a short period/duration
        # doesn't collapse to 0.0 on reload -- a 1s duration is ~0.000278 h
        # and rounding to 2 decimals showed 0.0 (#568). Decimal units stay
        # awkward for short values; H:M:S entry is the tracked follow-up.
        period = round(self._protocol.period().total_seconds() / 60, 6)
        duration = round(self._protocol.duration().total_seconds() / 3600, 6)
        labware = self._protocol.labware()

        # If the scope has no XY stage, then don't allow the protocol to modify
        # the labware. The drivers answer that, not the selected scope model.
        if not ctx.lumaview.scope.capabilities.has_xy_stage:
            labware = 'Center Plate'

        self.ids['capture_period'].text = str(period)
        self.ids['capture_dur'].text = str(duration)

        settings['protocol']['period'] = period
        settings['protocol']['duration'] = duration
        # The plate takes the route start-up uses: the spinner shows it, and
        # the explicit call below hands it to the Session, the one writer of
        # the labware key for every host. The spinner's own event cannot be
        # relied on for that: an assignment equal to the current text does
        # not dispatch, and the text can already show a plate the scope
        # never took (a refused pick leaves it where the user put it).
        # Declared twice because only one declaration is pending per name
        # and either emission may be the one that consumes it.
        gui_logger.note_write_back('LABWARE', labware)
        self.ids['labware_spinner'].text = labware
        gui_logger.note_write_back('LABWARE', labware)
        self.select_labware()
        self.ids['capture_root'].text = self._protocol.capture_root()

        # Restore per-layer UI state from the protocol's Layer Settings
        # block (v6) or, for v5 files, from the inferred per-layer state
        # built from the steps Color column. Layers that aren't named in
        # the protocol fall back to disabled (acquire=None) so the UI
        # shows them as not-part-of-this-protocol; their other slider
        # values (illumination/gain/exposure/etc.) are untouched, since
        # the user's prior choices for those layers shouldn't be lost
        # just because the loaded protocol didn't reference them.
        layer_settings_from_protocol = self._protocol.layer_settings()
        for layer in common_utils.get_layers():
            settings[layer]['acquire'] = None
            if 'stim_config' in settings[layer] and settings[layer]['stim_config'] is not None:
                settings[layer]['stim_config']['enabled'] = False
        for layer_name, vals in (layer_settings_from_protocol or {}).items():
            if layer_name not in common_utils.get_layers():
                logger.warning(
                    f'[LVP Main  ] Protocol carries settings for unknown layer '
                    f'{layer_name!r}; that layer is dropped on load.'
                )
                continue
            self._apply_layer_settings_row(settings, layer_name, vals)

        reset_acquire_ui()
        reset_stim_ui()

        # Make steps available for drawing locations
        ctx.stage.set_protocol_steps(df=self._protocol.steps())

        # Restore the tiling selection. Tiling is baked into the steps as
        # expanded tile positions (one row per tile), not stored as a
        # scalar, so the spinner otherwise stays at its 1x1 default and
        # misrepresents an already-tiled protocol. Infer the NxN label back
        # from the steps' Tile column; fall back to no-tiling when the
        # protocol isn't tiled (or the layout isn't square).
        try:
            inferred_tiling = self.tiling_config.determine_tiling_label_from_tiles(
                self._protocol.steps()['Tile'].tolist()
            )
        except Exception as e:
            logger.warning(f'[LVP Main  ] Could not infer tiling from protocol: {e}')
            inferred_tiling = None
        self.ids['tiling_size_spinner'].text = (
            inferred_tiling or self.tiling_config.no_tiling_label()
        )

        self.update_step_ui()
        # Only a load a person asked for may drive the stage. The startup
        # adoption happens before anyone has asked for anything, so it fills
        # the panel and stops there -- no stage move, no LED change until
        # their first explicit navigation. This used to read "am I still
        # booting?", which answered the same way only while the load ran
        # inside the constructor; once it moved behind the objective
        # question it was answering a question it could no longer see.
        if navigate:
            self.go_to_step(step_idx=self.curr_step)

        return True

    @staticmethod
    def _apply_layer_settings_row(settings: dict, layer_name: str, vals: dict) -> None:
        """Apply a single Layer Settings row to settings[layer_name][*].

        Handles the column-name <-> settings-key mapping plus the string
        -> bool/float/int casting required when the row was parsed off
        disk. Missing or blank values are skipped so an explicit empty
        cell doesn't clobber a sensible default.
        """

        def _as_bool(s):
            if isinstance(s, bool):
                return s
            return str(s).strip().lower() == 'true'

        def _as_float(s, default=None):
            try:
                return float(s)
            except (TypeError, ValueError):
                return default

        def _as_int(s, default=None):
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return default

        layer = settings.setdefault(layer_name, {})

        acquire = vals.get('Acquire', '')
        if acquire in ('image', 'video'):
            layer['acquire'] = acquire

        for col, key, caster in (
            ('Illumination', 'illumination_ma', _as_float),
            ('Gain', 'gain_db', _as_float),
            ('Exposure', 'exposure_ms', _as_float),
            ('Sum', 'sum', _as_int),
        ):
            raw = vals.get(col, '')
            if raw == '' or raw is None:
                continue
            cast = caster(raw)
            if cast is not None:
                layer[key] = cast

        for col, key in (('Auto_Gain', 'auto_gain'), ('False_Color', 'false_color')):
            raw = vals.get(col, '')
            if raw == '' or raw is None:
                continue
            layer[key] = _as_bool(raw)

        # Stim_Enabled is the per-layer stim master switch (the rest of
        # the stim_config sub-dict is preserved). Blank means "leave the
        # current stim_config alone"; explicit True/False sets the flag.
        stim_raw = vals.get('Stim_Enabled', '')
        if stim_raw not in ('', None):
            stim_cfg = layer.get('stim_config')
            if isinstance(stim_cfg, dict):
                stim_cfg['enabled'] = _as_bool(stim_raw)

    def _gather_layer_settings_for_save(self) -> dict:
        """Collect current per-layer UI settings for inclusion in to_file().

        Only layers with acquire in ('image', 'video') are returned --
        these are the layers the user has marked as part of the
        protocol. Disabled layers are omitted so reload doesn't
        resurrect them.
        """
        settings = _app_ctx.ctx.settings
        out = {}
        for layer_name in common_utils.get_layers():
            layer = settings.get(layer_name)
            if not isinstance(layer, dict):
                continue
            acquire = layer.get('acquire')
            if acquire not in ('image', 'video'):
                continue
            row = {
                'Layer': layer_name,
                'Acquire': acquire,
                'Illumination': layer.get('illumination_ma', ''),
                'Gain': layer.get('gain_db', ''),
                'Auto_Gain': layer.get('auto_gain', ''),
                'Exposure': layer.get('exposure_ms', ''),
                'False_Color': layer.get('false_color', ''),
                'Sum': layer.get('sum', ''),
                'Stim_Enabled': '',
            }
            stim_cfg = layer.get('stim_config')
            if isinstance(stim_cfg, dict) and 'enabled' in stim_cfg:
                row['Stim_Enabled'] = stim_cfg['enabled']
            out[layer_name] = row
        return out

    def get_default_name_for_curr_step(self):
        step = self.get_curr_step()
        return common_utils.build_step_name(common_utils.step_components(step))

    # Save Protocol to File
    def save_protocol(self, filepath='', update_protocol_filepath: bool = True):
        gui_logger.protocol_action('SAVE', filepath)
        logger.info('[LVP Main  ] ProtocolSettings.save_protocol()')

        def _save():
            nonlocal filepath
            settings = _app_ctx.ctx.settings

            time_params = get_protocol_time_params()
            self._protocol.modify_time_params(
                period=time_params['period'], duration=time_params['duration']
            )

            if (isinstance(filepath, str)) and len(filepath) == 0:
                # If there is no current file path, "save" button will act as "save as"
                if len(settings['protocol']['filepath']) == 0:
                    from ui.file_dialogs import FileSaveBTN

                    FileSaveBTN_instance = FileSaveBTN()
                    FileSaveBTN_instance.choose('saveas_protocol')
                    return
                filepath = settings['protocol']['filepath']

            if (isinstance(filepath, str)) and (filepath[-4:].lower() != '.tsv'):
                filepath = filepath + '.tsv'

            # v6: include the per-layer UI state in the saved TSV header
            # so reload restores acquire mode + illumination/gain/exp
            # without needing inference from step rows. Existing
            # downstream callers (REST save, headless tests) keep their
            # signature -- to_file() with no kwarg still works and falls
            # back to inference on reload.
            self._protocol.to_file(
                file_path=filepath,
                layer_settings=self._gather_layer_settings_for_save(),
            )

            # Reached only once the file is written: a failed save leaves the
            # panel naming the file it had, which is still the one on disk.
            if update_protocol_filepath:
                settings['protocol']['filepath'] = filepath
            self.ids['protocol_filename'].text = os.path.basename(filepath)

        run_reported(_save, None, 'SAVE_PROTOCOL')

    #
    # Multiple Exposures
    # ------------------------------
    #
    # # increase exposure count
    # def exposures_down_button(self):
    #     logger.info('[LVP Main  ] ProtocolSettings.exposures_up_button()')
    #     self.exposures = max(self.exposures-1,1)
    #     self.ids['exposures_number_input'].text = str(self.exposures)

    # # increase exposure count
    # def exposures_up_button(self):
    #     logger.info('[LVP Main  ] ProtocolSettings.exposures_up_button()')
    #     self.exposures = self.exposures+1
    #     self.ids['exposures_number_input'].text = str(self.exposures)

    #
    # Edit steps
    # ------------------------------
    #
    def handle_step_ui_input_change(self) -> None:
        obj = self.ids['step_number_input']
        # Captured before either path below rewrites the box.
        typed = obj.text
        gui_logger.text_input('STEP_NUMBER', typed)
        try:
            val = int(obj.text)
        except Exception:
            num_steps = self._protocol.num_steps()
            if num_steps < 1:
                val = 0
            else:
                val = 1

            obj.text = f'{val}'
            gui_logger.text_input('STEP_NUMBER_APPLIED', obj.text)
            return

        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            val = 0
            obj.text = f'{val}'
        elif val < 1:
            val = 1
            obj.text = f'{val}'
        elif val > num_steps:
            val = num_steps
            obj.text = f'{val}'

        if obj.text != typed:
            gui_logger.text_input('STEP_NUMBER_APPLIED', obj.text)

        self.go_to_step(step_idx=val - 1)

    def go_to_step(self, step_idx: int):
        # step_idx is required so every caller states its target instead of
        # pre-writing curr_step: the navigation module detects a real step
        # change by comparing the target against curr_step, and a caller
        # that writes the store first makes that comparison read itself --
        # the LED preview then never fires. List-bookkeeping writes (load /
        # new / insert / delete keeping the pointer valid) stay with the
        # callers and legitimately compare equal here: protocol edits do
        # not drive the LEDs, user navigation does.
        go_to_step(
            protocol=self._protocol,
            step_idx=step_idx,
            include_move=True,
        )

    # Goto to Previous Step
    def prev_step(self) -> None:
        gui_logger.button('PREV_STEP')
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            self.curr_step = -1
            self.update_step_ui()
            return

        self.update_step_ui()
        self.go_to_step(step_idx=max(self.curr_step - 1, 0))

    # Go to Next Step
    def next_step(self) -> None:
        gui_logger.button('NEXT_STEP')
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            return

        self.update_step_ui()
        self.go_to_step(step_idx=min(self.curr_step + 1, num_steps - 1))

    # Delete Current Step of Protocol
    def delete_step(self):
        try:
            ctx = _app_ctx.ctx

            gui_logger.protocol_action('DELETE_STEP', f'curr_step={self.curr_step}')
            logger.info('[LVP Main  ] ProtocolSettings.delete_step()')

            if self._protocol.num_steps() <= 0:
                return

            self._protocol.delete_step(step_idx=self.curr_step)

            ctx.stage.set_protocol_steps(df=self._protocol.steps())

            if self._protocol.num_steps() <= 0:
                self.curr_step = -1
            else:
                self.curr_step = max(self.curr_step - 1, 0)

            self.update_step_ui()
            self.go_to_step(step_idx=self.curr_step)
        except Exception as e:
            logger.error(f'[UI] delete_step failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def modify_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if self._protocol.num_steps() < 1:
            return

        gui_logger.protocol_action('MODIFY_STEP', f'curr_step={self.curr_step}')
        ctx = _app_ctx.ctx
        active_layer, _ = get_active_layer_config(common_utils.get_opened_layer(ctx.image_settings))
        name_field = self.ids['step_name_input'].text
        run_reported(
            lambda: self.modify_step_ex(active_layer, name_field),
            self._draw_protocol_steps,
            'MODIFY_STEP',
        )

    def modify_step_ex(self, active_layer: str, name_field: str) -> None:
        """Change the current step to the open layer, renamed as the name field says."""
        # A non-blank name field is a user rename; blank keeps the step's
        # existing label and auto/user flag. The rendered Name re-derives
        # from the updated columns inside modify_step, so an auto-named
        # step's channel token tracks a channel change and a user label
        # rides along untouched -- no name branching needed here.
        label = common_utils.resolve_step_rename(name_field, Protocol.sanitize_step_name)
        name = _app_ctx.ctx.session.update_step(
            self._protocol, self.curr_step, layer=active_layer, label=label
        )
        logger.info(
            "[LVP Main  ] modify_step_ex: channel -> %s; step name -> '%s'",
            self._protocol.step(idx=self.curr_step)['Color'],
            name,
        )
        self._warn_if_steps_invalid('Step modified')

    # add_step
    def insert_step(self, after_current_step: bool = True):
        gui_logger.protocol_action(
            'INSERT_STEP', f'after_current={after_current_step} curr_step={self.curr_step}'
        )
        logger.info('[LVP Main  ] ProtocolSettings.insert_step()')
        run_reported(
            lambda: self.insert_step_ex(after_current_step),
            self._draw_protocol_steps,
            'INSERT_STEP',
        )

    def insert_step_ex(self, after_current_step: bool = True) -> None:
        """Add a step at the current stage position, beside the current step, and go to it.

        The move to the new step is a navigation that belongs only to a step
        the API accepted, so it is the last call, never reached on a refusal.
        """
        if after_current_step:
            after_step = self.curr_step
            before_step = None
        else:
            after_step = None
            before_step = self.curr_step

        names = _app_ctx.ctx.session.add_step(
            self._protocol, before_step=before_step, after_step=after_step
        )

        if after_current_step:
            self.curr_step += len(names)
        elif self.curr_step < 0:
            self.curr_step += 1

        self._warn_if_steps_invalid('Step added')
        self.go_to_step(step_idx=self.curr_step)

    def _warn_if_steps_invalid(self, what: str) -> None:
        errors = self._protocol.validate_steps(_app_ctx.ctx.lumaview.scope.objective_helper)
        if errors:
            from ui.notification_popup import show_notification_popup

            msg = '\n'.join(errors)
            show_notification_popup(
                title='Protocol Validation Warning',
                message=f'{what} with validation issues:\n\n{msg}',
            )

    def update_acquire_zstack(self):
        gui_logger.toggle('ACQUIRE_ZSTACK', bool(self.ids['acquire_zstack_id'].active))

    def log_disable_image_saving(self) -> None:
        """Record the disable-image-saving checkbox.

        The control is collapsed to zero height unless a caller opens it, so
        it is reachable only in that configuration -- which is the reason a
        gesture on it is worth a line: a bundle from a run that saved nothing
        otherwise gives no sign the box was ever touched.
        """
        gui_logger.toggle(
            'PROTOCOL_DISABLE_IMAGE_SAVING',
            bool(self.ids['protocol_disable_image_saving_id'].active),
        )

    def update_tiling_selection(self):
        gui_logger.select('TILING', self.ids['tiling_size_spinner'].text)

    def get_curr_step(self):
        if self._protocol.num_steps() == 0:
            return None

        return self._protocol.step(idx=self.curr_step)

    def draw_protocol_buttons(self) -> None:
        """Show each of the panel's three runs as the engine reports it.

        The only code that styles the Scan, Protocol and Autofocus Scan
        buttons: after each of their own requests, on every run-state edge
        -- including a run's return to idle and the end of its file drain --
        and, while a finished run's files drain, on a half-second tick for
        the pending count. Four states, each read from the API: running;
        stopping (a Stop accepted, the teardown still going); writing the
        finished run's files; idle.
        """
        ctx = _app_ctx.ctx
        engine = ctx.sequenced_capture_runner
        session = ctx.session
        # A finished run's writes still going. A start pressed now is the
        # engine's to refuse; the button that started the run shows the count.
        draining = session.protocol_files_draining

        for trigger, look in _PANEL_RUN_BUTTONS.items():
            button = self.ids[look.button_id]
            run = self._runs_started_here.get(trigger)
            setattr(self, look.held_flag, session.held_by_other(run))
            if engine.is_live_run(run):
                button.state = 'down'
                if engine.is_stopping(run):
                    button.text = 'Stopping...'
                elif trigger == 'protocol':
                    button.text = _protocol_remaining_text(engine)
                else:
                    button.text = look.running_text
                if look.running_background is not None:
                    button.background_down = look.running_background
                continue

            button.state = 'normal'
            if draining and run is not None and run is engine.run_outcome():
                button.text = (
                    'File writer stalled'
                    if session.protocol_files_stalled
                    else f'Writing Files... ({session.protocol_files_pending})'
                )
            else:
                button.text = look.idle_text
            if look.idle_background is not None:
                button.background_down = look.idle_background

        if draining:
            self._drain_tick_trigger()
        else:
            self._wedge_recovery_offered = False

    _wedge_recovery_offered = False  # One recovery offer per drain

    def _drain_tick(self, dt) -> None:
        """While a finished run's files drain: the count, and a stalled writer's offer.

        A stalled writer would otherwise hold "Writing Files..." forever --
        the surface stuck-run reports were stuck on -- so the first tick
        that finds it stalled offers the recovery, once per drain; the run
        starts re-offer on any later attempt if the person declines.
        """
        session = _app_ctx.ctx.session
        if (
            session.protocol_files_stalled
            and not session.is_protocol_running
            and not self._wedge_recovery_offered
        ):
            self._wedge_recovery_offered = True
            _offer_wedged_writer_recovery()
        self.draw_protocol_buttons()

    def _press_panel_run(
        self,
        trigger: str,
        log_stop: typing.Callable[[], None] | None,
        build_start: typing.Callable[[], typing.Callable[[], None]],
    ) -> None:
        """Start this button's run, or stop the one it started.

        Whether the press means Stop is the engine's answer -- is the run
        this button started still live -- never the toggle's, which Kivy
        has already flipped. The button changes nothing ahead of the
        engine's answer; draw_protocol_buttons shows it. Every refusal is
        the engine's to raise and the boundary's to show, once.
        """
        runner = _app_ctx.ctx.sequenced_capture_runner
        run = self._runs_started_here.get(trigger)
        if runner.is_live_run(run):
            if log_stop is not None:
                log_stop()
            self._submit_panel_request(trigger, lambda: runner.reset(run), stop=True)
            return

        self._submit_panel_request(trigger, build_start())

    def _submit_panel_request(
        self, trigger: str, call: typing.Callable[[], None], stop: bool = False
    ) -> None:
        # The button is disabled until this request's own redraw, so a
        # second press cannot race the first one to the pool.
        setattr(self, f'{trigger}_pending', True)
        submit_reported(
            call,
            lambda: self._panel_request_done(trigger),
            _PANEL_RUN_BUTTONS[trigger].label,
            stop=stop,
        )

    def _panel_request_done(self, trigger: str) -> None:
        setattr(self, f'{trigger}_pending', False)
        self.draw_protocol_buttons()

    def _autofocus_run_complete_callback(self, **kwargs):
        # Copy the Z-heights from the autofocus scan into the protocol --
        # but only from a scan that actually finished. An aborted or
        # failed scan focused some prefix of its steps and left the rest
        # at their pre-scan values, so copying that column back
        # overwrites the user's protocol with the steps that never ran.
        # The run's own terminal status is the only thing that can tell
        # the two apart; where the stage ended cannot.
        focused_protocol = kwargs['protocol']
        if kwargs.get('status') == 'completed':
            self._protocol.steps()['Z'] = focused_protocol.steps()['Z']
        self.reset_autofocus_ui()

    def debug_func(self):
        pass

    def update_bf_af_for_fluorescence(self):
        """Toggle: use BF autofocus result for all fluorescence channels."""
        ctx = _app_ctx.ctx
        enabled = self.ids['bf_af_for_fluorescence_btn'].state == 'down'
        gui_logger.toggle('BF_AF_FOR_FLUORESCENCE', enabled)
        with ctx.settings_lock:
            ctx.settings['protocol']['bf_af_for_fluorescence'] = enabled
        logger.info(f'[Protocol  ] BF AF for fluorescence: {enabled}')

    def run_autofocus_scan_from_ui(self):
        gui_logger.protocol_action('AF_SCAN')
        self._press_panel_run(
            'autofocus_scan',
            lambda: gui_logger.protocol_action('ABORT_AF_SCAN'),
            self._autofocus_scan_start,
        )

    def _autofocus_scan_start(self) -> typing.Callable[[], None]:
        """Read the autofocus scan's inputs from the panel; return the call that starts it."""
        ctx = _app_ctx.ctx
        settings = ctx.settings
        engine = ctx.sequenced_capture_runner
        trigger_source = 'autofocus_scan'

        callbacks = {
            **live_display_callbacks(),
            'move_position': _handle_ui_update_for_axis,
            # Pause live UI during recording-heavy runs for throughput
            'pause_live_ui': lambda: (
                ctx.scope_display.stop(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1),
            ),
            'run_scan_pre': self._run_scan_pre_callback,
            'scan_iterate_post': self.draw_protocol_buttons,
            'update_step_number': _update_step_number_callback,
            'go_to_step': go_to_step,
            'run_complete': self._autofocus_run_complete_callback,
            # LED observer handles UI sync -- no manual callbacks needed
            'sync_layer_widgets': sync_layer_widgets_from_settings,
            'set_recording_title': set_recording_title,
            'set_writing_title': set_writing_title,
            'reset_title': reset_title,
        }

        sequence = self._protocol.copy_for_execution()
        sequence.modify_autofocus_all_steps(enabled=True)
        image_capture_config = get_image_capture_config_from_ui()
        autogain_settings = get_auto_gain_settings()
        engineering_mode = ctx.engineering_mode
        autofocus_snapshot = config_helpers.autofocus_snapshot_from_settings(
            settings, ctx.settings_lock
        )

        def _start():
            plan = engine.prepare(
                protocol=sequence,
                run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                run_trigger_source=trigger_source,
                max_scans=1,
                sequence_name='af_scan',
                parent_dir=None,
                image_capture_config=image_capture_config,
                enable_image_saving=False,
                autogain_settings=autogain_settings,
                callbacks=callbacks,
                update_z_pos_from_autofocus=True,
                leds_state_at_end='off',
                engineering_mode=engineering_mode,
                autofocus_snapshot=autofocus_snapshot,
                # The autofocus scan must NOT hold the excitation LED across
                # focus moves (photobleaching) and saves nothing; the
                # helper's autofocus-scan branch forces both off.
                **config_helpers.get_sequenced_run_settings(
                    settings, run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
                ),
            )
            self._runs_started_here[trigger_source] = engine.start(plan)

        return _start

    def _scan_run_complete(self, **kwargs):
        self.reset_autofocus_ui()

    def run_scan_from_ui(self):
        gui_logger.protocol_action('SCAN')
        logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui()')
        self._press_panel_run(
            'scan',
            lambda: gui_logger.protocol_action('ABORT_SCAN'),
            self._scan_start,
        )

    def _scan_start(self) -> typing.Callable[[], None]:
        ctx = _app_ctx.ctx
        callbacks = {
            'run_scan_pre': self._run_scan_pre_callback,
            'scan_iterate_post': self.draw_protocol_buttons,
            'run_complete': self._scan_run_complete,
            # LED observer handles UI sync -- no manual callbacks needed
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
        return self._sequenced_capture_start(
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            run_trigger_source='scan',
            max_scans=1,
            protocol=self._protocol.copy_for_execution(),
            callbacks=callbacks,
        )

    def _protocol_run_complete(self, **kwargs):
        self.reset_autofocus_ui()

    def _protocol_files_complete(self, **kwargs):
        """Called once per run, after its run_complete, when its files are done."""
        self._dispatch_post_processing_auto_run(_app_ctx.ctx, **kwargs)

    def _dispatch_post_processing_auto_run(self, ctx, **kwargs):
        """Fire post_processing plugins opted into
        PluginSpec.auto_run_on_protocol_complete=True. UI-trigger only
        today; REST-triggered runs gain this when the dispatch moves
        down to the orchestration layer.
        """
        from modules.plugins import run_protocol_complete_processors

        # The finished run hands its directory over in the callback. Read
        # back off the runner it would be whatever run holds the scope
        # NOW: this dispatch can reach the user's next run, because the
        # file-drain wait re-enables the z-stack, composite and autofocus
        # starters while it is still pending.
        run_dir = kwargs.get('run_dir')
        if run_dir is None:
            return
        run_dir_str = str(run_dir)
        protocol = kwargs.get('protocol')
        manifest = {
            'protocol_name': getattr(protocol, 'name', '') if protocol else '',
            'run_dir': run_dir_str,
            'trigger_source': 'ui_protocol_button',
        }
        run_protocol_complete_processors(
            ctx,
            input_dir=run_dir_str,
            manifest=manifest,
            output_dir=run_dir_str,
            files=kwargs['files'],
        )

    def run_protocol_from_ui(self):
        gui_logger.protocol_action('RUN')
        logger.info('[LVP Main  ] ProtocolSettings.run_protocol_from_ui()')
        self._press_panel_run(
            'protocol',
            lambda: gui_logger.protocol_action('ABORT_PROTOCOL'),
            self._protocol_start,
        )

    def _protocol_start(self) -> typing.Callable[[], None]:
        ctx = _app_ctx.ctx
        callbacks = {
            'protocol_iterate_pre': lambda **kwargs: self.draw_protocol_buttons(),
            'run_scan_pre': self._run_scan_pre_callback,
            'run_complete': self._protocol_run_complete,
            'files_complete': self._protocol_files_complete,
            # LED observer handles UI sync -- no manual callbacks needed
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
        # The run takes the panel's timing as it reads now, on its own
        # copy: the panel's protocol is the person's, and is not the run's
        # to change.
        protocol = self._protocol.copy_for_execution()
        time_params = get_protocol_time_params()
        protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )
        return self._sequenced_capture_start(
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            run_trigger_source='protocol',
            max_scans=None,
            protocol=protocol,
            callbacks=callbacks,
        )

    def reset_autofocus_ui(self, **kwargs):
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            layer_obj._initializing = True
            try:
                layer_obj.ids['autofocus'].state = (
                    'down' if settings[layer]['autofocus'] else 'normal'
                )
            finally:
                layer_obj._initializing = False

    def _run_scan_pre_callback(self):
        Clock.schedule_once(lambda dt: self.update_step_ui(), 0)

    def _sequenced_capture_start(
        self,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        protocol: Protocol,
        callbacks: dict[str, typing.Callable],
    ) -> typing.Callable[[], None]:
        """Read a Scan or Protocol run's inputs from the panel; return the call that starts it.

        Runs on the GUI thread, so every value a widget or the settings
        store holds is read here and closed over. The call it returns is
        what the worker pool runs -- prepare, start, the handle this
        button's Stop names, and the save folder -- and touches no widget.
        """
        logger.info('[LVP Main  ] ProtocolSettings._sequenced_capture_start()')

        ctx = _app_ctx.ctx
        settings = ctx.settings
        engine = ctx.sequenced_capture_runner

        def restore_layer_shader_for_open_accordion():
            """Re-apply the shader for the currently-open accordion's
            layer. Called by protocol_cleanup to undo per-step shader
            changes (Red tint for Red step, etc.) so the live preview
            returns to the user's visible-layer false-color setting.
            Runs on the UI thread via _schedule_ui in protocol_cleanup.
            """
            ctx_inner = _app_ctx.ctx
            layer_name = common_utils.get_opened_layer(ctx_inner.image_settings)
            if layer_name is not None:
                layer_obj = ctx_inner.image_settings.layer_lookup(layer=layer_name)
                layer_obj.update_shader(dt=0)
                return
            # No open accordion -- default to BF (no false-color tint)
            ctx_inner.viewer.update_shader(false_color='BF')

        callbacks.update(
            {
                **live_display_callbacks(),
                'move_position': _handle_ui_update_for_axis,
                # LED observer handles UI sync -- no manual callbacks needed
                'update_step_number': _update_step_number_callback,
                'go_to_step': go_to_step,
                'sync_layer_widgets': sync_layer_widgets_from_settings,
                'set_recording_title': set_recording_title,
                'set_writing_title': set_writing_title,
                'reset_title': reset_title,
                'restore_layer_shader': restore_layer_shader_for_open_accordion,
            }
        )

        parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'ProtocolData'
        sequence_name = self.ids['protocol_filename'].text
        image_capture_config = get_image_capture_config_from_ui()
        autogain_settings = get_auto_gain_settings()
        enable_image_saving = is_image_saving_enabled()
        engineering_mode = ctx.engineering_mode
        autofocus_snapshot = config_helpers.autofocus_snapshot_from_settings(
            settings, ctx.settings_lock
        )

        def _start():
            plan = engine.prepare(
                protocol=protocol,
                run_mode=run_mode,
                run_trigger_source=run_trigger_source,
                max_scans=max_scans,
                sequence_name=sequence_name,
                parent_dir=parent_dir,
                image_capture_config=image_capture_config,
                enable_image_saving=enable_image_saving,
                autogain_settings=autogain_settings,
                callbacks=callbacks,
                disable_saving_artifacts=False,
                return_to_position=None,
                leds_state_at_end='off',
                engineering_mode=engineering_mode,
                autofocus_snapshot=autofocus_snapshot,
                **config_helpers.get_sequenced_run_settings(settings, run_mode=run_mode),
            )
            self._runs_started_here[run_trigger_source] = engine.start(plan)
            # A start() that failed during setup unwound as a failed run: it
            # nulled run_dir (set_last_save_folder no-ops on None), so the
            # saved folder never names a run that did not happen.
            set_last_save_folder(dir=engine.run_dir())

        return _start

    def cancel_all_protocols(self):
        """Stop whatever run is live, at app close.

        No run's handle is held here and no person waits for an outcome, so
        it is the engine's shutdown override rather than a Stop, and a
        failure is logged for the shutdown to carry on past.
        """
        logger.info('[LVP Main  ] ProtocolSettings.cancel_all_protocols()')
        try:
            _app_ctx.ctx.sequenced_capture_runner.force_reset(reason='app shutdown')
        except Exception as e:
            logger.error(f'[Protocol] Cleanup error: {e}', exc_info=True)

# Copyright Etaluma, Inc.
import copy
import logging
import os
import pathlib
import threading
import time
import typing

import pandas as pd

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
from modules.config_ui_getters import (
    get_active_layer_config,
    get_auto_gain_settings,
    get_binning_from_ui,
    get_current_frame_dimensions,
    get_image_capture_config_from_ui,
    get_layer_configs,
    get_protocol_time_params,
    get_selected_labware,
    get_sequenced_capture_config_from_ui,
    get_stim_configs,
    get_zstack_params,
    is_image_saving_enabled,
)
from modules.path_utils import get_source_root
from modules.protocol import Protocol
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import IOTask, PRIORITY_MED
from ui.step_navigation import go_to_step
from modules.tiling_config import TilingConfig
from modules.timedelta_formatter import strfdelta
from modules import gui_logger
from ui.ui_helpers import (
    _handle_ui_update_for_axis,
    _update_step_number_callback,
    live_display_callbacks,
    live_histo_off,
    live_histo_reverse,
    reset_acquire_ui,
    reset_stim_ui,
    reset_title,
    run_with_refusal_boundary,
    set_last_save_folder,
    set_recording_title,
    set_title_event_text,
    set_writing_title,
    text_input_debounced,
    update_autofocus_selection_after_protocol,
)
from ui.progress_popup import show_popup

logger = logging.getLogger('LVP.ui.protocol_settings')


def _offer_wedged_writer_recovery():
    """Modal offering discard-and-unlock recovery for a stalled file writer.

    Names the stuck write and the cost of recovery; declining leaves the
    queue untouched (the gates keep refusing and will re-offer)."""
    from ui.notification_popup import show_confirmation_popup

    ctx = _app_ctx.ctx
    file_io_executor = ctx.file_io_executor
    pending = file_io_executor.protocol_queue_size()
    stuck = file_io_executor.describe_running_task()

    def _recover():
        logger.warning('[LVP Main  ] User confirmed wedged-writer recovery')
        file_io_executor.recover_wedged_protocol_queue()

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


def require_file_writes_idle(operation: str) -> bool:
    """One gate for operations that must wait for the protocol file queue.

    Returns True when the queue is idle so the operation may proceed.
    Healthy drain: refuse with the live pending count -- the writes will
    finish. Stalled drain (the in-flight write ran past the writer's fatal
    stall budget): offer discard-and-unlock recovery instead of an
    unfulfillable "please wait"; the operation is still refused this click
    and the user retries once unlocked.
    """
    from modules.protocol_image_writer import WRITE_STALL_FATAL_S

    ctx = _app_ctx.ctx
    file_io_executor = ctx.file_io_executor
    if not file_io_executor.is_protocol_queue_active():
        return True
    if file_io_executor.protocol_drain_stalled(WRITE_STALL_FATAL_S):
        logger.warning(
            f'[LVP Main  ] Cannot {operation} - file writer stalled on '
            f'{file_io_executor.describe_running_task()}; offering recovery'
        )
        _offer_wedged_writer_recovery()
    else:
        from ui.notification_popup import show_notification_popup

        pending = file_io_executor.protocol_queue_size()
        logger.warning(
            f'[LVP Main  ] Cannot {operation} - {pending} file(s) still being written to disk'
        )
        show_notification_popup(
            title='Operation Blocked',
            message=(
                f'Please wait - {pending} file(s) from the previous scan '
                f'are still being written to disk.'
            ),
        )
    return False


class ProtocolSettings(FloatLayout):
    done = BooleanProperty(False)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        logger.info('[LVP Main  ] ProtocolSettings.__init__()')

        # Create trigger for debounced UI updates to prevent memory leaks
        self._update_step_ui_trigger = Clock.create_trigger(self._do_update_step_ui, 0.05)

        # Thread-safe flag to prevent duplicate file completion handlers
        self._scan_files_completed_event = threading.Event()

        # source_path: use ctx if available, otherwise derive from install-aware defaults
        ctx = _app_ctx.ctx
        source_root = get_source_root(ctx.source_path if ctx is not None else None)

        self.curr_step = -1

        self.tiling_config = TilingConfig(
            tiling_configs_file_loc=source_root / 'data' / 'tiling.json'
        )

        from modules.common_utils import DEFAULT_STAGE_TRAVEL_UM

        self.tiling_min = {
            'x': int(DEFAULT_STAGE_TRAVEL_UM['x']),
            'y': int(DEFAULT_STAGE_TRAVEL_UM['y']),
        }
        self.tiling_max = {'x': 0, 'y': 0}

        self.tiling_count = self.tiling_config.get_mxn_size(self.tiling_config.default_config())

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

        self.ids['tiling_size_spinner'].values = self.tiling_config.available_configs()
        self.ids['tiling_size_spinner'].text = self.tiling_config.default_config()

        try:
            filepath = settings['protocol']['filepath']
            protocol_success = ctx.motion_settings.ids['protocol_settings_id'].load_protocol(
                filepath=filepath, suppress_popup=True
            )

            if not protocol_success:
                logger.info(
                    '[LVP Main  ] No saved protocol loaded at startup -- using empty protocol'
                )
                # If protocol file is missing or incomplete, file name and path are cleared from memory.
                filepath = ''
                settings['protocol']['filepath'] = ''

                protocol_config = get_sequenced_capture_config_from_ui()
                self._protocol = ctx.scope.protocols.create_protocol(
                    empty_config=protocol_config,
                )

        except Exception:
            logger.exception('[LVP Main  ] Error loading protocol at startup')
            filepath = ''
            settings['protocol']['filepath'] = ''
            protocol_config = get_sequenced_capture_config_from_ui()
            self._protocol = ctx.scope.protocols.create_protocol(
                empty_config=protocol_config,
            )

        self.select_labware()
        self.update_step_ui()

        # DISABLED: BF AF for fluorescence -- not yet tested, hidden for 4.0.0.
        # Force off regardless of saved settings to prevent untested code path.
        if 'protocol' in settings:
            settings['protocol']['bf_af_for_fluorescence'] = False
        self.ids['bf_af_for_fluorescence_btn'].state = 'normal'

    # Update Protocol Period
    def update_period(self):
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] ProtocolSettings.update_period()')
        try:
            raw_period = float(self.ids['capture_period'].text)
            settings['protocol']['period'] = raw_period
            # Warn once, at the edit, when a sub-1s period is raised to the 1s
            # minimum -- so the user is told why the field shows 0.016667 min
            # instead of their typed value. The getter stays silent so save /
            # run-start do not re-warn.
            if config_helpers.protocol_time_clamped(raw_period, 'minutes'):
                from modules.notification_center import notifications

                notifications.warning(
                    'Protocol',
                    'Capture Timing',
                    'The capture period was below the 1-second minimum and was '
                    'raised to 1 second (shown as 0.016667 min). Enter a period '
                    'of at least 1 second.',
                )
        except Exception:
            logger.exception('[LVP Main  ] Update Period is not an acceptable value')

        text_input_debounced('PROTOCOL_PERIOD', self.ids['capture_period'].text)

        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    # Update Protocol Duration
    def update_duration(self):
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] ProtocolSettings.update_duration()')
        try:
            raw_duration = float(self.ids['capture_dur'].text)
            settings['protocol']['duration'] = raw_duration
            # Duration is in HOURS, so a sub-1s value shows as 0.000278 hr (not
            # 0.016667 min). Warn once, at the edit, with the hour value.
            if config_helpers.protocol_time_clamped(raw_duration, 'hours'):
                from modules.notification_center import notifications

                notifications.warning(
                    'Protocol',
                    'Capture Timing',
                    'The capture duration was below the 1-second minimum and was '
                    'raised to 1 second (shown as 0.000278 hr). Enter a duration '
                    'of at least 1 second.',
                )
        except Exception:
            logger.warning('[LVP Main  ] Update Duration is not an acceptable value')

        text_input_debounced('PROTOCOL_DURATION', self.ids['capture_dur'].text)

        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    def step_name_validation(self, text: str):
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
        self.ids['capture_root'].text = sanitized
        if hasattr(self, '_protocol') and (self._protocol is not None):
            self._protocol.modify_capture_root(capture_root=sanitized)
        text_input_debounced('CAPTURE_ROOT', sanitized)

    # Labware Selection
    def select_labware(self, labware: str | None = None):
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx
        wellplate_loader = ctx.wellplate_loader

        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        if labware is None:
            spinner = self.ids['labware_spinner']
            spinner.values = wellplate_loader.get_plate_list()
            gui_logger.select('LABWARE', spinner.text)
            # Settings is the single labware store; an empty spinner
            # (not yet populated at startup) is not a selection and must
            # not clobber the stored choice.
            if spinner.text:
                settings['protocol']['labware'] = spinner.text
        else:
            center_plate_str = 'Center Plate'
            spinner = self.ids['labware_spinner']
            spinner.values = [center_plate_str]
            spinner.text = center_plate_str
            settings['protocol']['labware'] = labware

        labware_id, labware = get_selected_labware()

        if labware is None:
            logger.error('Labware could not be loaded')
            return

        ctx.lumaview.scope.runtime_state.set_labware(labware=labware)

        if self._protocol is not None:
            self._protocol.modify_labware(labware_id=labware_id)

        ctx.stage.full_redraw()

    def set_labware_selection_visibility(self, visible):
        labware_spinner = self.ids['labware_spinner']
        labware_spinner.visible = visible
        labware_spinner.size_hint_y = None if visible else 0
        labware_spinner.height = '30dp' if visible else 0
        labware_spinner.opacity = 1 if visible else 0
        labware_spinner.disabled = not visible

        if not visible:
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
                    labware_spinner.text = saved_labware
            except Exception as e:
                logger.warning(f'[LVP Main  ] Failed to restore labware list on scope switch: {e}')

    def apply_tiling(self):
        try:
            settings = _app_ctx.ctx.settings
            ctx = _app_ctx.ctx

            logger.info('[LVP Main  ] Apply tiling to protocol')

            # Guard against compounding. apply_tiling appends new tile groups
            # to the existing steps, and there is no un-tile path yet, so
            # applying tiling to an already-tiled protocol multiplies the tiles
            # (e.g. 2x2 on a 2x2 -> 16). Detect the current tiling from the
            # steps' Tile column; if already tiled, refuse and tell the user
            # to reload the untiled base first.
            no_tiling = self.tiling_config.no_tiling_label()
            current_tiling = self.tiling_config.determine_tiling_label_from_tiles(
                self._protocol.steps()['Tile'].tolist()
            )
            if current_tiling not in (None, no_tiling):
                from ui.notification_popup import show_notification_popup

                show_notification_popup(
                    title='Protocol Already Tiled',
                    message=(
                        f'This protocol is already tiled ({current_tiling}). '
                        f'Applying tiling again would compound it. Reload the '
                        f'original (untiled) protocol before changing the tiling.'
                    ),
                )
                return

            axes_config = ctx.lumaview.scope.motion.get_axes_config()
            _, labware = get_selected_labware()
            stage_offset = settings['stage_offset']
            overlap_percent = self.get_tiling_overlap_percent()

            tile_status = self._protocol.apply_tiling(
                tiling=self.ids['tiling_size_spinner'].text,
                frame_dimensions=get_current_frame_dimensions(),
                binning_size=get_binning_from_ui(),
                curr_step_idx=self.curr_step,
                axes_config=axes_config,
                labware=labware,
                stage_offset=stage_offset,
                overlap_percent=overlap_percent,
                capabilities=ctx.lumaview.scope.capabilities,
            )

            tiles_skipped = tile_status['tiles_skipped']

            if tiles_skipped > 0:
                error_msg = f'Tiling application skipped {tiles_skipped} new tiles due to bounds outside of labware.'
                from ui.notification_popup import show_notification_popup

                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Tiling Warning', message=error_msg
                    ),
                    0,
                )

            self._protocol.optimize_step_ordering()
            ctx.stage.set_protocol_steps(df=self._protocol.steps())
            self.update_step_ui()
            self.go_to_step(step_idx=self.curr_step, protocol=False)
        except Exception as e:
            logger.error(f'[UI] apply_tiling failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def get_tiling_overlap_percent(self) -> float:
        """Tile overlap percentage, read from the persisted system setting.

        The single accessor for tile overlap at scan/apply time; the editor
        (the spinner in Advanced Settings) only ever writes the setting.
        """
        return _app_ctx.ctx.settings['tiling_overlap_percent']

    def apply_zstacking(self):
        try:
            ctx = _app_ctx.ctx

            logger.info('[LVP Main  ] Apply Z-Stacking to protocol')
            zstack_params = get_zstack_params()

            if zstack_params['range'] < 0 or zstack_params['step_size'] < 0:
                error_msg = 'Z-Stacking parameters are not valid. Please ensure range and step size are positive values.'
                logger.warning(error_msg)
                from ui.notification_popup import show_notification_popup

                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Z-Stacking Warning', message=error_msg
                    ),
                    0,
                )
                return
            elif zstack_params['range'] == 0 or zstack_params['step_size'] == 0:
                logger.warning('Z-stacking parameters are zero. No changes applied.')
                return

            axes_config = ctx.lumaview.scope.motion.get_axes_config()
            zstack_status = self._protocol.apply_zstacking(
                zstack_params=zstack_params,
                axes_config=axes_config,
            )

            zslices_skipped = zstack_status['zslices_skipped']
            if zslices_skipped > 0:
                error_msg = (
                    f'Z-stacking skipped {zslices_skipped} slices that fall outside the '
                    f'Z travel range. Reduce the range or adjust the focus position.'
                )
                from ui.notification_popup import show_notification_popup

                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Z-Stacking Warning', message=error_msg
                    ),
                    0,
                )

            self._protocol.optimize_step_ordering()
            ctx.stage.set_protocol_steps(df=self._protocol.steps())
            self.update_step_ui()
            self.go_to_step(step_idx=self.curr_step, protocol=False)
        except Exception as e:
            logger.error(f'[UI] apply_zstacking failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

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

        if not require_file_writes_idle('create a new protocol'):
            return

        config = get_sequenced_capture_config_from_ui()

        # New Protocol resets each step to its channel's saved focus baseline.
        # A per-(well, channel) Z carry-over from the prior in-memory protocol
        # used to run here, but it harvested autofocus-refined Z along with
        # user-tuned Z and so overrode a freshly-saved focus; it is no longer
        # the default. Per-well focus is re-established on demand via
        # "Autofocus All Steps". Protocol.from_config still honors an explicit
        # previous_well_z map (left dormant) so this can return as an opt-in
        # setting without re-plumbing.

        try:
            protocol = ctx.scope.protocols.create_protocol(input_config=config)
        except Exception as e:
            logger.error(f'[LVP Main  ] Protocol creation failed: {e}')
            from ui.notification_popup import show_notification_popup

            show_notification_popup(
                title='Protocol Creation Error',
                message=str(e),
            )
            return

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

        # new_protocol_ex builds the step table from the labware + scan
        # parameters; bounded work, fits on worker_pool MED so the UI
        # remains responsive while it runs.
        _app_ctx.ctx.worker_pool.put(
            IOTask(
                action=self.new_protocol_ex,
                args=(protocol),
                callback=self.update_step_ui,
                priority=PRIORITY_MED,
            )
        )

    def new_protocol_ex(self, protocol):
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        if (ctx.lumaview.scope.capabilities.has_turret) and (
            not ctx.lumaview.scope.motion.is_current_turret_position_objective_set()
        ):
            error_msg = (
                'Cannot create new protocol. Please set objective for current turret position.'
            )
            logger.error(error_msg)

            from ui.notification_popup import show_notification_popup

            Clock.schedule_once(
                lambda dt: show_notification_popup(
                    title='Protocol Creation Error', message=error_msg
                ),
                0,
            )
            return

        if not self._validate_objectives_in_protocol(protocol_df=protocol.steps()):
            error_msg = 'Cannot create new protocol. Not all objectives are in turret config.'
            logger.error(error_msg)
            Clock.schedule_once(
                lambda dt: Popup(
                    title='Protocol Creation Error',
                    content=Label(text=error_msg),
                    size_hint=(0.85, 0.85),
                ),
                0,
            )

            return

        self._protocol = protocol
        ctx.protocol = protocol  # canonical owner is AppContext

        ctx.stage.set_protocol_steps(df=self._protocol.steps())

        def temp():
            self.ids['protocol_filename'].text = ''
            self.ids['capture_root'].text = ''

        settings['protocol']['filepath'] = ''
        Clock.schedule_once(lambda dt: temp(), 0)
        self.curr_step = 0
        self.go_to_step(step_idx=0, protocol=False)

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

    def _validate_objectives_in_protocol(self, protocol_df: pd.DataFrame) -> bool:
        ctx = _app_ctx.ctx

        # Validation for objectives with multi-objective protocol
        protocol_objective_ids = set(protocol_df['Objective'].to_list())

        # For single objective protocols, don't perform any objective validation (legacy)
        if len(protocol_objective_ids) == 1:
            return True

        # Otherwise, check all the objectives used in the protocol and confirm
        # they are all part of the current turret config
        turret_objective_ids = set(ctx.lumaview.scope.runtime_state.get_turret_config().values())
        return protocol_objective_ids.issubset(turret_objective_ids)

    # Load Protocol from File
    def load_protocol(self, filepath='./data/new_default_protocol.tsv', suppress_popup=False):
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

        except Exception as e:
            logger.warning(f'[LVP Main  ] Protocol load failed: {e}')
            if not suppress_popup:
                error_title = 'Protocol Loading Error'
                error_msg = f'Cannot load protocol from file: {e}'
                from ui.notification_popup import show_notification_popup

                show_notification_popup(title=error_title, message=error_msg)
            return False

        if protocol is False:
            error_title = 'Empty Protocol Steps'
            error_msg = 'Warning: Selected protocol had no steps. Empty protocol loaded.'
            protocol_config = get_sequenced_capture_config_from_ui()

            protocol = ctx.scope.protocols.create_protocol(empty_config=protocol_config)

        if protocol is None:
            logger.error(f'Unable to load protocol at {filepath}')
            return

        if not self._validate_objectives_in_protocol(protocol_df=protocol.steps()):
            error_msg = 'Cannot load protocol. Not all objectives are in turret config.'
            logger.error(error_msg)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Protocol Loading Error', message=error_msg)
            return False

        self._protocol = protocol
        ctx.protocol = protocol  # canonical owner is AppContext

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
        settings['protocol']['labware'] = labware
        self.ids['labware_spinner'].text = settings['protocol']['labware']
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
        # During startup the persisted protocol loads before the user has
        # asked for anything: no stage move, no LED change until their
        # first explicit navigation.
        if not ctx.initializing:
            self.go_to_step(step_idx=self.curr_step, protocol=False)

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
        try:
            gui_logger.protocol_action('SAVE', filepath)
            settings = _app_ctx.ctx.settings

            logger.info('[LVP Main  ] ProtocolSettings.save_protocol()')

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
            else:
                if (isinstance(filepath, str)) and (filepath[-4:].lower() != '.tsv'):
                    filepath = filepath + '.tsv'

                if update_protocol_filepath:
                    settings['protocol']['filepath'] = filepath

            if (isinstance(filepath, str)) and (filepath[-4:].lower() != '.tsv'):
                filepath = filepath + '.tsv'

            # v6: include the per-layer UI state in the saved TSV header
            # so reload restores acquire mode + illumination/gain/exp
            # without needing inference from step rows. Existing
            # downstream callers (REST save, headless tests) keep their
            # signature -- to_file() with no kwarg still works and falls
            # back to inference on reload.
            result = self._protocol.to_file(
                file_path=filepath,
                layer_settings=self._gather_layer_settings_for_save(),
            )

            if result:  # Had an error saving
                from ui.notification_popup import show_notification_popup

                show_notification_popup(title='Protocol Saving Error', message=result)

            self.ids['protocol_filename'].text = os.path.basename(filepath)
        except Exception as e:
            logger.error(f'[UI] save_protocol failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

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
    def handle_step_ui_input_change(self):
        obj = self.ids['step_number_input']
        try:
            val = int(obj.text)
        except Exception:
            num_steps = self._protocol.num_steps()
            if num_steps < 1:
                val = 0
            else:
                val = 1

            obj.text = f'{val}'
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

        self.go_to_step(step_idx=val - 1, protocol=False)

    def go_to_step(self, step_idx: int, protocol=True):
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
            ignore_auto_gain=False,
            include_move=True,
            called_from_protocol=protocol,
        )

    # Goto to Previous Step
    def prev_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            self.curr_step = -1
            self.update_step_ui()
            return

        self.update_step_ui()
        self.go_to_step(step_idx=max(self.curr_step - 1, 0), protocol=False)

    # Go to Next Step
    def next_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        if not (hasattr(self, '_protocol') and self._protocol is not None):
            return
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            return

        self.update_step_ui()
        self.go_to_step(step_idx=min(self.curr_step + 1, num_steps - 1), protocol=False)

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
            self.go_to_step(step_idx=self.curr_step, protocol=False)
        except Exception as e:
            logger.error(f'[UI] delete_step failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def modify_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if self._protocol.num_steps() < 1:
            return

        gui_logger.protocol_action('MODIFY_STEP', f'curr_step={self.curr_step}')
        io_executor = _app_ctx.ctx.io_executor
        io_executor.put(IOTask(action=self.modify_step_ex, callback=self.update_step_ui))

    def modify_step_ex(self):
        try:
            ctx = _app_ctx.ctx
            from ui.notification_popup import show_notification_popup

            active_layer, active_layer_config = get_active_layer_config()

            if (
                'stim_config' in active_layer_config
                and active_layer_config['stim_config'] is not None
                and active_layer_config['stim_config']['enabled']
            ):
                # We want to keep the same acquire channel when we are only modifying the stim config.
                true_step_layer = self._protocol.step(idx=self.curr_step)['Color']
                active_layer = true_step_layer
                active_layer_config = get_layer_configs()[active_layer]

            plate_position = ctx.session.get_current_plate_position()
            objective_id, _ = ctx.session.get_current_objective_info()

            # logger.error(f"CURRENT Z POSITION IN UM {plate_position['z']}")

            if (ctx.lumaview.scope.capabilities.has_turret) and (
                not ctx.lumaview.scope.motion.is_current_turret_position_objective_set()
            ):
                error_msg = (
                    'Cannot modify protocol step. Please set objective for current turret position.'
                )
                logger.error(error_msg)
                # Runs on the io_executor worker; Kivy widgets must be
                # built on the main thread, so marshal via Clock.
                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Step Modification Error', message=error_msg
                    ),
                    0,
                )
                return

            # A non-blank name field is a user rename; blank keeps the step's
            # existing label and auto/user flag. The rendered Name re-derives
            # from the updated columns inside modify_step, so an auto-named
            # step's channel token tracks a channel change and a user label
            # rides along untouched -- no name branching needed here.
            label = common_utils.resolve_step_rename(
                self.ids['step_name_input'].text, Protocol.sanitize_step_name
            )

            self._protocol.modify_step(
                step_idx=self.curr_step,
                label=label,
                layer=active_layer,
                layer_config=active_layer_config,
                stim_configs=get_stim_configs(),
                plate_position=plate_position,
                objective_id=objective_id,
            )
            logger.info(
                "[LVP Main  ] modify_step_ex: channel -> %s; step name -> '%s'",
                active_layer,
                self._protocol.step(idx=self.curr_step)['Name'],
            )

            # Validate the modified step and warn the user if there are errors.
            errors = self._protocol.validate_steps()
            if errors:
                msg = '\n'.join(errors)
                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Validation Warning',
                        message=f'Step modified with validation issues:\n\n{msg}',
                    ),
                    0,
                )

            ctx.stage.set_protocol_steps(df=self._protocol.steps())
        except Exception as e:
            logger.error(f'[UI] modify_step_ex failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            # Runs on the io_executor worker; marshal the popup to the
            # main thread. Bind str(e) now -- the exception variable is
            # unbound by the time the scheduled lambda runs.
            Clock.schedule_once(
                lambda dt, m=str(e): show_notification_popup(title='Error', message=m),
                0,
            )

    # add_step
    def insert_step(self, after_current_step: bool = True):
        gui_logger.protocol_action(
            'INSERT_STEP', f'after_current={after_current_step} curr_step={self.curr_step}'
        )
        logger.info('[LVP Main  ] ProtocolSettings.insert_step()')
        io_executor = _app_ctx.ctx.io_executor
        io_executor.put(
            IOTask(
                action=self.insert_step_ex, args=(after_current_step), callback=self.update_step_ui
            )
        )

    def insert_step_ex(self, after_current_step: bool = True):
        try:
            ctx = _app_ctx.ctx
            from ui.notification_popup import show_notification_popup

            plate_position = ctx.session.get_current_plate_position()
            objective_id, _ = ctx.session.get_current_objective_info()

            if (ctx.lumaview.scope.capabilities.has_turret) and (
                not ctx.lumaview.scope.motion.is_current_turret_position_objective_set()
            ):
                error_msg = (
                    'Cannot add step to protocol. Please set objective for current turret position.'
                )
                logger.error(error_msg)
                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Add Step Error', message=error_msg
                    ),
                    0,
                )
                return

            if after_current_step:
                after_step = self.curr_step
                before_step = None
            else:
                after_step = None
                before_step = self.curr_step

            layer_configs = get_layer_configs()

            # Early return if no channels have acquire enabled (#548)
            if not any(lc['acquire'] is not None for lc in layer_configs.values()):
                return

            # Use custom channel order from settings if configured,
            # otherwise fall back to default get_layers() order.
            # This controls the order channels are added as protocol steps,
            # which matters for composite imaging association.
            settings = ctx.settings
            channel_order = settings.get('step_channel_order', None)
            if channel_order:
                # Only include channels that are in layer_configs
                ordered_layers = [ch for ch in channel_order if ch in layer_configs]
                # Append any channels not in the custom order
                for ch in layer_configs:
                    if ch not in ordered_layers:
                        ordered_layers.append(ch)
            else:
                ordered_layers = list(layer_configs.keys())

            stim_configs = get_stim_configs()

            # H5: Warn about invalid stim configs at insert time
            for stim_color, sc in stim_configs.items():
                if not isinstance(sc, dict) or not sc.get('enabled', False):
                    continue
                freq = sc.get('frequency', 0)
                if not isinstance(freq, (int, float)) or freq <= 0:
                    logger.warning(
                        f'[UI] Stim channel {stim_color}: frequency {freq} Hz is invalid (must be > 0). Disabling channel.'
                    )
                    sc['enabled'] = False
                exp = sc.get('exposure', 0)
                if isinstance(exp, (int, float)) and exp == 0 and sc.get('enabled', False):
                    logger.warning(
                        f'[UI] Stim channel {stim_color}: exposure is 0. This may produce no visible pulses.'
                    )
                illum = sc.get('illumination_ma', 0)
                if isinstance(illum, (int, float)) and illum <= 0 and sc.get('enabled', False):
                    logger.warning(
                        f'[UI] Stim channel {stim_color}: illumination {illum} mA is invalid (must be > 0). Disabling channel.'
                    )
                    sc['enabled'] = False

            for layer in ordered_layers:
                layer_config = layer_configs[layer]
                if layer_config['acquire'] is None:
                    continue

                _ = self._protocol.insert_step(
                    step_name=None,
                    layer=layer,
                    layer_config=layer_config,
                    stim_configs=stim_configs,
                    plate_position=plate_position,
                    objective_id=objective_id,
                    before_step=before_step,
                    after_step=after_step,
                )

                if after_current_step or (self.curr_step < 0):
                    self.curr_step += 1

            # Validate after inserting and warn the user if there are errors
            errors = self._protocol.validate_steps()
            if errors:
                msg = '\n'.join(errors)
                Clock.schedule_once(
                    lambda dt: show_notification_popup(
                        title='Protocol Validation Warning',
                        message=f'Step added with validation issues:\n\n{msg}',
                    ),
                    0,
                )

            ctx.stage.set_protocol_steps(df=self._protocol.steps())
            self.go_to_step(step_idx=self.curr_step, protocol=False)
        except Exception as e:
            logger.error(f'[UI] insert_step_ex failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            # Runs on the io_executor worker; marshal the popup to the
            # main thread. Bind str(e) now -- the exception variable is
            # unbound by the time the scheduled lambda runs.
            Clock.schedule_once(
                lambda dt, m=str(e): show_notification_popup(title='Error', message=m),
                0,
            )

    def update_acquire_zstack(self):
        gui_logger.toggle('ACQUIRE_ZSTACK', bool(self.ids['acquire_zstack_id'].active))

    def update_tiling_selection(self):
        gui_logger.select('TILING', self.ids['tiling_size_spinner'].text)

    def determine_and_set_run_autofocus_scan_allow(self):
        tiling = self.ids['tiling_size_spinner'].text
        zstack = self.ids['acquire_zstack_id'].active
        if zstack and (tiling != '1x1'):
            self.set_run_autofocus_scan_allow(allow=False)
        else:
            self.set_run_autofocus_scan_allow(allow=True)

    def set_run_autofocus_scan_allow(self, allow: bool):
        if allow:
            self.ids['run_autofocus_btn'].disabled = False
        else:
            self.ids['run_autofocus_btn'].disabled = True

    def get_curr_step(self):
        if self._protocol.num_steps() == 0:
            return None

        return self._protocol.step(idx=self.curr_step)

    def _reset_run_autofocus_scan_button(self, **kwargs):
        self.ids['run_autofocus_btn'].state = 'normal'
        self.ids['run_autofocus_btn'].text = 'Autofocus All Steps'
        self.ids['run_autofocus_btn'].disabled = False

    def _reset_run_scan_button(self, **kwargs):
        self.ids['run_scan_btn'].state = 'normal'
        self.ids['run_scan_btn'].text = 'Run One Scan'
        self.ids['run_scan_btn'].disabled = False

    def _reset_run_protocol_button(self, **kwargs):
        self.ids['run_protocol_btn'].state = 'normal'
        self.ids['run_protocol_btn'].text = 'Run Full Protocol'
        self.ids['run_protocol_btn'].disabled = False
        self.ids[
            'run_protocol_btn'
        ].background_down = 'atlas://data/images/defaulttheme/button_pressed'

    def _commit_running_ui_state(
        self, button_id: str, text: str, background_down: str | None = None
    ):
        """Commit the shared "a run is now underway" BUTTON state.

        Run-state truth is the session claim, committed inside start()
        and mirrored to kv by the session's run-state listener; what
        remains caller-side is the starter button's cosmetics. Runs
        between prepare and start so a refusal never leaves a button
        mid-run.
        """
        self.ids[button_id].text = text
        if background_down is not None:
            self.ids[button_id].background_down = background_down

    def _reset_run_button_cosmetics(
        self, button_id: str, text: str, background_down: str | None = None
    ):
        """Undo a starter's pre-gate button cosmetics after a run did NOT start.

        Cosmetics only: run-state truth is the session claim, which a
        refused start never touched, and the kv lockout mirrors follow
        the session's run-state listener -- there is nothing else for a
        refusal to undo.
        """
        self.ids[button_id].state = 'normal'
        self.ids[button_id].text = text
        self.ids[button_id].disabled = False
        if background_down is not None:
            self.ids[button_id].background_down = background_down

    def _is_protocol_valid(self) -> bool:
        from ui.notification_popup import show_notification_popup

        if self._protocol.num_steps() == 0:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            show_notification_popup(
                title='Protocol Invalid',
                message='Protocol has no steps. Add at least one step before running.',
            )
            return False

        # Validate save folder is accessible
        settings = _app_ctx.ctx.settings
        live_folder = settings.get('live_folder')
        if live_folder:
            import pathlib

            parent_dir = pathlib.Path(live_folder).resolve() / 'ProtocolData'
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = parent_dir / '.write_test'
                test_file.touch()
                test_file.unlink()
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error(f'[LVP Main  ] Save folder not writable: {parent_dir}: {e}')
                show_notification_popup(
                    title='Save Path Error',
                    message=f'Cannot write to save folder:\n{parent_dir}\n\nError: {e}',
                )
                return False

        # If turret is present, validate all protocol objectives are assigned (#606)
        ctx = _app_ctx.ctx
        if ctx.lumaview.scope.capabilities.has_turret and not self._validate_objectives_in_protocol(
            protocol_df=self._protocol.steps()
        ):
            turret_objectives = settings.get('turret_objectives', {})
            assigned = [v for v in turret_objectives.values() if v is not None]
            show_notification_popup(
                title='Turret Configuration Required',
                message='Protocol uses objectives not assigned to turret positions.\n\n'
                f'Assigned: {assigned if assigned else "None"}\n\n'
                'Please assign objectives in Objective Control > Turret before running.',
            )
            return False

        return True

    def _autofocus_run_complete_callback(self, **kwargs):
        ctx = _app_ctx.ctx

        # Don't reset immediately - keep running until files complete

        # Reset completion event for this run (thread-safe)
        self._scan_files_completed_event.clear()

        # Copy the Z-heights from the autofocus scan into the protocol first
        focused_protocol = kwargs['protocol']
        self._protocol.steps()['Z'] = focused_protocol.steps()['Z']

        file_io_executor = ctx.file_io_executor

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._wedge_recovery_offered = False
            self._file_write_status_event = Clock.schedule_interval(
                self._update_autofocus_write_status,
                0.5,  # Update every 500ms
            )
            # Initial button state
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_autofocus_btn'].state = 'normal'
            self.ids['run_autofocus_btn'].text = f'Writing Files... ({queue_size})'
            self.ids['run_autofocus_btn'].disabled = True

            # Disable other buttons
            self.ids['run_scan_btn'].disabled = True
            self.ids['run_protocol_btn'].disabled = True

            # Update window title
            set_title_event_text('Writing protocol scan files to disk...')
        else:
            # No files pending - proceed with normal reset
            live_histo_reverse()
            reset_acquire_ui()
            self._reset_run_autofocus_scan_button()

    _wedge_recovery_offered = False  # One recovery offer per write-lockout episode

    def _update_write_lockout_button(self, button_id: str) -> None:
        """Poll-tick body shared by the three 'Writing Files...' lockouts.

        A healthy drain shows the live pending count. A stalled writer would
        otherwise freeze that label forever -- the surface the stuck-run
        reports were actually stuck on -- so a stall swaps in the recovery
        offer (once per episode; the run-start gates re-offer on any later
        attempt if the user declines)."""
        from modules.protocol_image_writer import WRITE_STALL_FATAL_S

        file_io_executor = _app_ctx.ctx.file_io_executor
        if file_io_executor.protocol_drain_stalled(WRITE_STALL_FATAL_S):
            self.ids[button_id].text = 'File writer stalled'
            if not self._wedge_recovery_offered:
                self._wedge_recovery_offered = True
                _offer_wedged_writer_recovery()
        else:
            queue_size = file_io_executor.protocol_queue_size()
            self.ids[button_id].text = f'Writing Files... ({queue_size})'

    def _update_autofocus_write_status(self, dt):
        """Update UI to show file writing progress for autofocus."""
        ctx = _app_ctx.ctx
        file_io_executor = ctx.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            self._update_write_lockout_button('run_autofocus_btn')
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._autofocus_files_complete()

    def _autofocus_files_complete(self, **kwargs):
        """Called when ALL files are written to disk for autofocus run."""

        # Guard against multiple calls using thread-safe event
        if self._scan_files_completed_event.is_set():
            return
        self._scan_files_completed_event.set()

        # Cancel status update if still scheduled
        if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
            Clock.unschedule(self._file_write_status_event)
            self._file_write_status_event = None

        # Reset the autofocus button
        self._reset_run_autofocus_scan_button()

        # Re-enable other buttons
        self.ids['run_scan_btn'].disabled = False
        self.ids['run_protocol_btn'].disabled = False

        # Complete remaining cleanup
        live_histo_reverse()
        reset_acquire_ui()
        Clock.schedule_once(lambda dt: reset_title(), 0)

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
        try:
            gui_logger.protocol_action('AF_SCAN_START')
            from ui.notification_popup import show_notification_popup

            logger.info('[LVP Main  ] ProtocolSettings.run_autofocus_scan_from_ui()')
            trigger_source = 'autofocus_scan'
            run_not_started_func = self._reset_run_autofocus_scan_button

            ctx = _app_ctx.ctx
            sequenced_capture_runner = ctx.sequenced_capture_runner

            run_trigger_source = sequenced_capture_runner.run_trigger_source()

            live_histo_off()

            # Not-started paths undo cosmetics only: run-state truth is
            # the session claim, which a refusal never touched.
            def run_refused_func():
                self._reset_run_button_cosmetics('run_autofocus_btn', 'Autofocus All Steps')
                live_histo_reverse()

            # Only block if starting NEW autofocus scan (button is 'down'), not if aborting (button is 'normal')
            if self.ids['run_autofocus_btn'].state == 'down' and not require_file_writes_idle(
                'start the autofocus scan'
            ):
                run_refused_func()
                return

            if self.ids['run_autofocus_btn'].state == 'normal' or (
                sequenced_capture_runner.run_in_progress() and run_trigger_source == trigger_source
            ):
                self._cleanup_at_end_of_protocol(autofocus_scan=True)
                return

            if sequenced_capture_runner.run_in_progress() and (
                run_trigger_source != trigger_source
            ):
                run_refused_func()
                logger.warning(
                    f'Cannot start autofocus scan. Run already in progress from {run_trigger_source}'
                )
                return

            if not self._is_protocol_valid():
                run_refused_func()
                return

            def commit_ui_state():
                # Button cosmetics only: the run-state commit is the
                # session claim inside start(), and the kv mirrors
                # follow from the session's run-state listener.
                self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

            settings = _app_ctx.ctx.settings

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
                'autofocus_in_progress': self._autofocus_in_progress_callback,
                'autofocus_complete': self._autofocus_complete_callback,
                'scan_iterate_post': run_not_started_func,
                'update_step_number': _update_step_number_callback,
                'go_to_step': go_to_step,
                'run_complete': self._autofocus_run_complete_callback,
                'files_complete': self._autofocus_files_complete,
                # LED observer handles UI sync -- no manual callbacks needed
                'reset_autofocus_btns': update_autofocus_selection_after_protocol,
                'set_recording_title': set_recording_title,
                'set_writing_title': set_writing_title,
                'reset_title': reset_title,
            }

            autogain_settings = get_auto_gain_settings()

            sequence = copy.deepcopy(self._protocol)
            sequence.modify_autofocus_all_steps(enabled=True)

            def prepare_and_start():
                plan = sequenced_capture_runner.prepare(
                    protocol=sequence,
                    run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                    run_trigger_source=trigger_source,
                    max_scans=1,
                    sequence_name='af_scan',
                    parent_dir=None,
                    image_capture_config=get_image_capture_config_from_ui(),
                    enable_image_saving=False,
                    autogain_settings=autogain_settings,
                    callbacks=callbacks,
                    update_z_pos_from_autofocus=True,
                    leds_state_at_end='off',
                    engineering_mode=ctx.engineering_mode,
                    autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                        settings, ctx.settings_lock
                    ),
                    # The autofocus scan must NOT hold the excitation LED
                    # across focus moves (photobleaching) and saves nothing;
                    # the helper's autofocus-scan branch forces both off.
                    **config_helpers.get_sequenced_run_settings(
                        settings, run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
                    ),
                )
                commit_ui_state()
                sequenced_capture_runner.start(plan)

            run_with_refusal_boundary(prepare_and_start, on_refused=run_refused_func)
        except Exception as e:
            logger.error(f'[UI] run_autofocus_scan_from_ui failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

    def _scan_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx

        # The run's LED restore has settled by the time this callback is
        # scheduled, so reconcile every enable toggle to driver truth: the
        # step-nav run indicator can be left stale by a Stop, and an
        # all-dark restore emits no LED events to correct it.
        ctx.ui_listener_bridge.reconcile_led_buttons()

        # Reset completion event for this scan (thread-safe)
        self._scan_files_completed_event.clear()

        file_io_executor = ctx.file_io_executor

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._wedge_recovery_offered = False
            self._file_write_status_event = Clock.schedule_interval(
                self._update_file_write_status,
                0.5,  # Update every 500ms
            )
            # Initial button state
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_scan_btn'].state = 'normal'  # Reset to normal state
            self.ids['run_scan_btn'].text = f'Writing Files... ({queue_size})'
            self.ids['run_scan_btn'].disabled = True

            # Disable other buttons to prevent any operations while writing
            self.ids['run_protocol_btn'].disabled = True
            self.ids['run_autofocus_btn'].disabled = True

            # Update window title with custom message
            set_title_event_text('Writing protocol scan files to disk...')
        else:
            # No files pending - proceed with normal reset
            self._reset_run_scan_button()
            live_histo_reverse()
            reset_acquire_ui()
            self.reset_autofocus_ui()

    def _update_file_write_status(self, dt):
        """Update UI to show file writing progress."""
        ctx = _app_ctx.ctx
        file_io_executor = ctx.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            self._update_write_lockout_button('run_scan_btn')
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._scan_files_complete()

    def _scan_files_complete(self, **kwargs):
        """Called when ALL files are written to disk (deferred callback)."""
        # Guard against multiple calls using thread-safe event
        if self._scan_files_completed_event.is_set():
            return
        self._scan_files_completed_event.set()

        # Cancel status update if still scheduled
        if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
            Clock.unschedule(self._file_write_status_event)
            self._file_write_status_event = None

        # Now actually reset the button
        self._reset_run_scan_button()

        # Re-enable other buttons that were disabled during file writing
        self.ids['run_protocol_btn'].disabled = False
        self.ids['run_autofocus_btn'].disabled = False

        # Complete remaining cleanup
        live_histo_reverse()
        reset_acquire_ui()
        self.reset_autofocus_ui()
        reset_title()

    _scan_starting = False  # Re-entry guard for double-click prevention

    def run_scan_from_ui(self):
        if ProtocolSettings._scan_starting:
            logger.warning('[LVP Main  ] run_scan_from_ui() ignored -- already starting')
            return
        ProtocolSettings._scan_starting = True
        try:
            self._run_scan_from_ui_inner()
        finally:
            ProtocolSettings._scan_starting = False

    def _run_scan_from_ui_inner(self):
        gui_logger.protocol_action('SCAN')
        logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui()')
        trigger_source = 'scan'
        run_complete_func = self._scan_run_complete
        run_not_started_func = self._reset_run_scan_button

        # Not-started paths undo cosmetics only: run-state truth is
        # the session claim, which a refusal never touched.
        def run_refused_func():
            self._reset_run_button_cosmetics('run_scan_btn', 'Run One Scan')

        ctx = _app_ctx.ctx
        sequenced_capture_runner = ctx.sequenced_capture_runner

        # Only block if starting NEW scan (button is 'down'), not if aborting (button is 'normal')
        if self.ids['run_scan_btn'].state == 'down' and not require_file_writes_idle(
            'start the scan'
        ):
            run_refused_func()
            return

        # State of button immediately changed upon press, so we are checking if the button was previously not pressed, and if autofocus is happening
        if self.ids['run_scan_btn'].state == 'down' and ctx.autofocus_thread.is_running:
            run_refused_func()
            logger.warning('Cannot start scan. Autofocus still in progress.')
            return

        run_trigger_source = sequenced_capture_runner.run_trigger_source()
        if sequenced_capture_runner.run_in_progress() and (run_trigger_source != trigger_source):
            run_refused_func()
            logger.warning(f'Cannot start scan. Run already in progress from {run_trigger_source}')
            return

        # Abort BEFORE validity: the abort click must never be refused by
        # a validation failure (a mid-run unwritable save folder would
        # otherwise block the user's own Stop).
        if self.ids['run_scan_btn'].state == 'normal':
            gui_logger.protocol_action('ABORT_SCAN')
            logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui() - User ending scan early')
            # Hardware teardown finishes on the protocol thread; the scan
            # run-complete callback resets this label when it ends.
            self.ids['run_scan_btn'].text = 'Stopping...'
            self._cleanup_at_end_of_protocol(autofocus_scan=False)
            return

        if not self._is_protocol_valid():
            run_refused_func()
            return

        callbacks = {
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'scan_iterate_post': run_not_started_func,
            'run_complete': run_complete_func,
            'files_complete': self._scan_files_complete,
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

        run_with_refusal_boundary(
            lambda: self.run_sequenced_capture(
                run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
                run_trigger_source=trigger_source,
                max_scans=1,
                callbacks=callbacks,
                commit_ui_state=lambda: self._commit_running_ui_state(
                    'run_scan_btn',
                    'Abort One Scan',
                    './data/icons/abort_protocol_background.png',
                ),
            ),
            on_refused=run_refused_func,
        )

    def _protocol_run_complete(self, **kwargs):
        ctx = _app_ctx.ctx

        # See _scan_run_complete: reconcile enable toggles to driver truth
        # now that the run's LED restore has settled.
        ctx.ui_listener_bridge.reconcile_led_buttons()

        # Reset completion event for this run (thread-safe)
        self._scan_files_completed_event.clear()

        file_io_executor = ctx.file_io_executor

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._wedge_recovery_offered = False
            self._file_write_status_event = Clock.schedule_interval(
                self._update_protocol_write_status,
                0.5,  # Update every 500ms
            )
            # Initial button state
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_protocol_btn'].state = 'normal'
            self.ids['run_protocol_btn'].text = f'Writing Files... ({queue_size})'
            self.ids['run_protocol_btn'].disabled = True

            # Disable other buttons
            self.ids['run_scan_btn'].disabled = True
            self.ids['run_autofocus_btn'].disabled = True

            # Update window title
            set_title_event_text('Writing protocol scan files to disk...')
        else:
            # No files pending - proceed with normal reset
            self._reset_run_protocol_button()
            live_histo_reverse()
            reset_acquire_ui()
            self.reset_autofocus_ui()
            # Auto-run opted-in post_processing plugins. Mirrors the
            # files-pending path's call from _protocol_files_complete.
            self._dispatch_post_processing_auto_run(ctx, **kwargs)

    def _update_protocol_write_status(self, dt):
        """Update UI to show file writing progress for protocol."""
        ctx = _app_ctx.ctx
        file_io_executor = ctx.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            self._update_write_lockout_button('run_protocol_btn')
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._protocol_files_complete()

    def _protocol_files_complete(self, **kwargs):
        """Called when ALL files are written to disk for protocol run."""
        ctx = _app_ctx.ctx

        # Guard against multiple calls using thread-safe event
        if self._scan_files_completed_event.is_set():
            return
        self._scan_files_completed_event.set()

        # Cancel status update if still scheduled
        if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
            Clock.unschedule(self._file_write_status_event)
            self._file_write_status_event = None

        # Reset the protocol button
        self._reset_run_protocol_button()

        # Re-enable other buttons
        self.ids['run_scan_btn'].disabled = False
        self.ids['run_autofocus_btn'].disabled = False

        # Complete remaining cleanup
        live_histo_reverse()
        reset_acquire_ui()
        self.reset_autofocus_ui()
        reset_title()

        # Auto-run post_processing plugins that opted in.
        self._dispatch_post_processing_auto_run(ctx, **kwargs)

    def _dispatch_post_processing_auto_run(self, ctx, **kwargs):
        """Fire post_processing plugins opted into
        PluginSpec.auto_run_on_protocol_complete=True. UI-trigger only
        today; REST-triggered runs gain this when the dispatch moves
        down to the orchestration layer.
        """
        from modules.plugins import run_protocol_complete_processors

        run_dir = ctx.sequenced_capture_runner.run_dir()
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
        )

    _protocol_starting = False  # Re-entry guard for double-click prevention

    def run_protocol_from_ui(self):
        # Prevent double-click: if we're already in the process of starting,
        # ignore the second click entirely.
        if ProtocolSettings._protocol_starting:
            logger.warning('[LVP Main  ] run_protocol_from_ui() ignored -- already starting')
            return
        ProtocolSettings._protocol_starting = True
        try:
            self._run_protocol_from_ui_inner()
        finally:
            ProtocolSettings._protocol_starting = False

    def _run_protocol_from_ui_inner(self):
        try:
            gui_logger.protocol_action('RUN')
            from ui.notification_popup import show_notification_popup

            logger.info('[LVP Main  ] ProtocolSettings.run_protocol_from_ui()')
            trigger_source = 'protocol'
            run_complete_func = self._protocol_run_complete

            ctx = _app_ctx.ctx
            sequenced_capture_runner = ctx.sequenced_capture_runner

            # Not-started paths undo cosmetics only: run-state truth is
            # the session claim, which a refusal never touched.
            def run_refused_func():
                self._reset_run_button_cosmetics(
                    'run_protocol_btn',
                    'Run Full Protocol',
                    'atlas://data/images/defaulttheme/button_pressed',
                )

            # Only block if starting NEW protocol run (button is 'down'), not if aborting (button is 'normal')
            if self.ids['run_protocol_btn'].state == 'down' and not require_file_writes_idle(
                'start the protocol run'
            ):
                run_refused_func()
                return

            run_trigger_source = sequenced_capture_runner.run_trigger_source()

            # State of button immediately changed upon press, so we are checking if the button was previously not pressed, and if autofocus is happening
            if self.ids['run_protocol_btn'].state == 'down' and ctx.autofocus_thread.is_running:
                run_refused_func()
                logger.warning('Cannot start protocol run. Autofocus still in progress.')
                return

            if sequenced_capture_runner.run_in_progress() and (
                run_trigger_source != trigger_source
            ):
                run_refused_func()
                logger.warning(
                    f'Cannot start protocol run. Run already in progress from {run_trigger_source}'
                )
                return

            # Abort BEFORE validity: the abort click must never be refused
            # by a validation failure (a mid-run unwritable save folder
            # would otherwise block the user's own Stop).
            if self.ids['run_protocol_btn'].state == 'normal':
                gui_logger.protocol_action('ABORT_PROTOCOL')
                # Hardware teardown finishes on the protocol thread; the
                # protocol run-complete callback resets this label.
                self.ids['run_protocol_btn'].text = 'Stopping...'
                self._cleanup_at_end_of_protocol(autofocus_scan=False)
                return

            if not self._is_protocol_valid():
                run_refused_func()
                return

            callbacks = {
                'protocol_iterate_pre': self._update_protocol_run_button_status,
                'run_scan_pre': self._run_scan_pre_callback,
                'autofocus_in_progress': self._autofocus_in_progress_callback,
                'autofocus_complete': self._autofocus_complete_callback,
                'run_complete': run_complete_func,
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

            time_params = get_protocol_time_params()
            self._protocol.modify_time_params(
                period=time_params['period'],
                duration=time_params['duration'],
            )

            run_with_refusal_boundary(
                lambda: self.run_sequenced_capture(
                    run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
                    run_trigger_source=trigger_source,
                    max_scans=None,
                    callbacks=callbacks,
                    # Text is quickly overwritten by the remaining-scans status
                    commit_ui_state=lambda: self._commit_running_ui_state(
                        'run_protocol_btn', 'Running Protocol'
                    ),
                ),
                on_refused=run_refused_func,
            )
        except Exception as e:
            logger.error(f'[UI] run_protocol_from_ui failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))

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

    def _update_protocol_run_button_status(
        self,
        **kwargs,
    ):
        # The session's protocol truth drops BOTH stale-callback shapes:
        # a callback landing after the run ended, and one landing in the
        # post-run drain window (owner already freed) that would clobber
        # the "Writing Files..." text.
        if not _app_ctx.ctx.session.is_protocol_running:
            return

        remaining_scans = kwargs['remaining_scans']
        scan_interval = kwargs['interval']
        remaining_duration = remaining_scans * scan_interval
        remaining_duration_str = strfdelta(
            tdelta=remaining_duration,
            fmt='{H}h {M}m',
            inputtype='timedelta',
        )
        scan_word = 'scan' if remaining_scans == 1 else 'scans'

        self.ids[
            'run_protocol_btn'
        ].text = (
            f'{remaining_scans} {scan_word} ({remaining_duration_str}) remaining.\nPress to ABORT'
        )
        self.ids['run_protocol_btn'].background_down = './data/icons/abort_protocol_background.png'

    def _run_scan_pre_callback(self):
        ctx = _app_ctx.ctx
        ctx.motion_settings.ids['verticalcontrol_id'].is_complete = False
        Clock.schedule_once(lambda dt: self.update_step_ui(), 0)

    def _autofocus_in_progress_callback(self):
        ctx = _app_ctx.ctx
        ctx.motion_settings.ids['verticalcontrol_id']._set_run_autofocus_button()

    def _autofocus_complete_callback(self):
        ctx = _app_ctx.ctx
        ctx.motion_settings.ids['verticalcontrol_id']._reset_run_autofocus_button()
        ctx.motion_settings.ids['verticalcontrol_id'].is_complete = False
        # LED observer handles UI button sync after AF -- no manual update needed

    def run_sequenced_capture(
        self,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        callbacks: dict[str, typing.Callable],
        disable_saving_artifacts: bool = False,
        return_to_position: dict | None = None,
        commit_ui_state: typing.Callable[[], None] | None = None,
    ):
        """Prepare, commit UI running-state, and start a sequenced run.

        commit_ui_state runs between a successful prepare() and start(),
        so callers commit their "a run is now underway" state (events,
        buttons, motion locks) only once the run can no longer be
        refused -- a refusal raises out of prepare() before it runs.

        Raises:
            ProtocolRunRefusedError: The runner refused the request; the
                user was already notified and commit_ui_state never ran.
        """
        live_histo_off()

        logger.info('[LVP Main  ] ProtocolSettings.run_sequenced_capture()')

        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx
        sequenced_capture_runner = ctx.sequenced_capture_runner

        def restore_layer_shader_for_open_accordion():
            """Re-apply the shader for the currently-open accordion's
            layer. Called by protocol_cleanup to undo per-step shader
            changes (Red tint for Red step, etc.) so the live preview
            returns to the user's visible-layer false-color setting.
            Runs on the UI thread via _schedule_ui in protocol_cleanup.
            """
            ctx_inner = _app_ctx.ctx
            for layer_name in common_utils.get_layers():
                accordion_item = ctx_inner.image_settings.accordion_item_lookup(layer=layer_name)
                if not accordion_item.collapse:
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
                'reset_autofocus_btns': update_autofocus_selection_after_protocol,
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

        plan = sequenced_capture_runner.prepare(
            protocol=self._protocol,
            run_mode=run_mode,
            run_trigger_source=run_trigger_source,
            max_scans=max_scans,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=is_image_saving_enabled(),
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            disable_saving_artifacts=disable_saving_artifacts,
            return_to_position=return_to_position,
            leds_state_at_end='off',
            engineering_mode=ctx.engineering_mode,
            autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
                settings, ctx.settings_lock
            ),
            **config_helpers.get_sequenced_run_settings(settings, run_mode=run_mode),
        )
        if commit_ui_state is not None:
            commit_ui_state()
        sequenced_capture_runner.start(plan)

        # A start() that failed during setup unwound as a failed run: it
        # nulled run_dir (set_last_save_folder no-ops on None) and cleared
        # run-in-progress, so neither follow-up acts on the dead run.
        set_last_save_folder(dir=sequenced_capture_runner.run_dir())

        if (
            run_mode == SequencedCaptureRunMode.FULL_PROTOCOL
            and sequenced_capture_runner.run_in_progress()
        ):
            self._update_protocol_run_button_status(
                remaining_scans=sequenced_capture_runner.remaining_scans(),
                interval=sequenced_capture_runner.protocol_interval(),
            )

    def _cleanup_at_end_of_protocol(self, autofocus_scan: bool):
        ctx = _app_ctx.ctx
        deferred_to_cleanup = False

        try:
            sequenced_capture_runner = ctx.sequenced_capture_runner
            # True only on the abort flavor of this call: a run is still
            # unwinding, so reset() returns immediately and the hardware
            # teardown (LED off, camera restore, return-to-position) runs
            # on the protocol thread. The post-completion flavor (run
            # already finished; reset() is a light no-op) keeps the
            # synchronous restore below.
            deferred_to_cleanup = sequenced_capture_runner.run_in_progress()
            sequenced_capture_runner.reset()
            live_histo_reverse()
            self.reset_autofocus_ui()
            self._autofocus_complete_callback()

        except Exception as e:
            logger.error(f'[Protocol] Cleanup error: {e}', exc_info=True)
        finally:
            if deferred_to_cleanup:
                # Cleanup is unwinding on the protocol thread; the
                # run-complete callbacks it fires perform the full restore
                # (buttons, motion capability, hyperstacks) when it ends.
                # Restoring here would hand the stage back to the user
                # while the return-to-position move is still queued, and
                # re-arm the run buttons while the old run is tearing
                # down. Until then the run-in-progress guards refuse new
                # runs and the protocol-running lockout keeps the rest of
                # the UI held -- responsive, not frozen.
                pass
            else:
                # ALWAYS restore UI state, even if cleanup above threw.
                # Without this, buttons stay disabled and motion stays locked.
                self._reset_run_protocol_button()
                self._reset_run_scan_button()
                self._reset_run_autofocus_scan_button()

            # LED observer handles UI button sync after protocol -- no manual refresh needed

    def cancel_all_protocols(self):
        logger.info('[LVP Main  ] ProtocolSettings.cancel_all_protocols()')
        self._cleanup_at_end_of_protocol(autofocus_scan=False)

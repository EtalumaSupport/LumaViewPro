# Copyright Etaluma, Inc.
import copy
import datetime
import json
import logging
import os
import pathlib
import threading
import time

from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.binning as binning
import modules.common_utils as common_utils
from modules import gui_logger
from modules.config_helpers import (
    camera_max_exposure_for_ui,
    camera_max_gain_for_ui,
)
from modules.config_ui_getters import (
    firmware_stim_supported,
    get_binning_from_ui,
    get_current_frame_dimensions,
    get_selected_labware,
)
from modules.common_utils import CustomJSONizer
from modules.path_utils import resolve_data_file
from modules.scope_init_config import ScopeInitConfig
from modules.memory_profiler import MemoryLeakProfiler
from modules.sequential_io_executor import IOTask
import modules.image_mode as image_mode
from ui.ui_helpers import scope_leds_off
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.ui.microscope_settings')


class _CoalescingApplier:
    """One-at-a-time worker that keeps only the LATEST pending value.

    Used by MicroscopeSettings.frame_size to prevent the camera_executor
    queue from stacking up slow Pylon set_frame_size calls (issue #624).
    On large frames each stop_grabbing/start_grabbing cycle blocks the
    CAMERA_WORKER for ~11s; naive queueing of rapid user edits
    (tabbing between width and height fields) produced multi-minute
    backlogs that made the UI feel frozen.

    Pattern:
      - submit(value) stashes value in a single pending slot and
        returns True only when the caller should enqueue the worker
        task (i.e. no task already in flight and the value is not a
        repeat of what the hardware already holds).
      - apply_pending(fn) drains the pending slot and calls fn(value)
        for each value. Loops until pending is empty so late-arriving
        updates during an apply() are picked up in the SAME task
        rather than spawning a new one.

    Exact repeats of the last successfully applied value are absorbed.
    One user edit fires the bound handler up to four times (each text
    field binds both on_text_validate and on_focus loss, and the
    handler reads BOTH fields every call, so all four calls compute
    the identical value). On a slow camera the in-flight gate folds
    them; on a fast camera (FX2 applies in milliseconds) the gate
    closes between events and every repeat became a real hardware
    apply. A failed apply does not update the last-applied record, so
    a retry with the same value still goes through -- and "failed"
    covers BOTH failure shapes: a raising fn and a falsy return (the
    camera-absent no-op, or any apply whose acceptance is signaled by
    returning the applied value). Recording is gated on a truthy
    return, so a rejection can never poison the dedupe record and
    absorb the user's retry.
    """

    def __init__(self, name='coalescing_applier'):
        self._name = name
        self._pending = None
        self._in_flight = False
        self._last_applied = None
        self._lock = threading.Lock()

    def submit(self, value):
        with self._lock:
            if not self._in_flight and self._pending is None and value == self._last_applied:
                return False
            self._pending = value
            if self._in_flight:
                return False
            self._in_flight = True
            return True

    def apply_pending(self, fn):
        while True:
            with self._lock:
                val = self._pending
                self._pending = None
                if val is None:
                    self._in_flight = False
                    return
                if val == self._last_applied:
                    # A repeat of what the hardware already holds arrived
                    # while an apply was in flight; nothing new to send.
                    continue
            try:
                result = fn(val)
            except Exception as e:
                # The typed rejection was already logged + notified at the
                # API layer; this line ties it to the coalescer's value.
                logger.error(f'[{self._name}] apply failed for {val!r}: {e}', exc_info=True)
            else:
                if result:
                    # The recorded key is what the hardware actually holds:
                    # an fn that returns the APPLIED value (e.g. a clamped
                    # delivered size) records that, so a user retyping the
                    # original request after seeing the clamp is not
                    # absorbed against a value the camera never took. A
                    # bare True records the request itself. An fn returning
                    # a value must return it in the SAME shape submit()
                    # receives (the frame push returns a (w, h) tuple) --
                    # a mismatched shape would never equal a submitted key
                    # and dedupe would silently stop absorbing.
                    with self._lock:
                        self._last_applied = val if result is True else result


class MicroscopeSettings(BoxLayout):
    # Current scope model name, shown read-only in the panel. The selector
    # that changes it lives in Advanced Settings; this reflects the settings
    # SSOT and is refreshed in set_ui_features_for_scope (the one place a
    # scope change reconfigures the UI).
    current_scope_model = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] MicroscopeSettings.__init__()')
        # Coalesce rapid set_frame_size requests. See
        # _CoalescingApplier + issue #624.
        self._frame_size_applier = _CoalescingApplier(name='frame_size')

        scopes_path = resolve_data_file('scopes.json')
        try:
            with open(scopes_path) as read_file:
                self.scopes = json.load(read_file)
        except FileNotFoundError as e:
            logger.error(f'[LVP Main  ] scopes.json not found at {scopes_path}')
            raise RuntimeError(
                f'Required file scopes.json not found at {scopes_path}. '
                'Please reinstall or restore from backup.'
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f'[LVP Main  ] scopes.json is corrupt: {e}')
            raise RuntimeError(
                f'scopes.json is corrupt ({e}). Please restore from backup or reinstall.'
            ) from e

        self._validate_scopes(scopes_path)

    def _validate_scopes(self, filepath):
        """Check scopes.json has required structure per scope entry."""
        if not isinstance(self.scopes, dict):
            raise ValueError(
                f'scopes.json at {filepath}: expected dict, got {type(self.scopes).__name__}'
            )
        _REQUIRED_SCOPE_FIELDS = {'Focus': bool, 'XYStage': bool, 'Turret': bool, 'Layers': dict}
        for scope_id, scope in self.scopes.items():
            if not isinstance(scope, dict):
                logger.warning(f"[Scopes    ] '{scope_id}' should be dict in {filepath}")
                continue
            for field, expected_type in _REQUIRED_SCOPE_FIELDS.items():
                if field not in scope:
                    logger.warning(f"[Scopes    ] '{scope_id}' missing '{field}' in {filepath}")
                elif not isinstance(scope[field], expected_type):
                    logger.warning(
                        f"[Scopes    ] '{scope_id}'.'{field}' should be "
                        f'{expected_type.__name__}, got {type(scope[field]).__name__} in {filepath}'
                    )

        # try:
        #     os.chdir(source_path)
        #     with open('./data/objectives.json', "r") as read_file:
        #         self.objectives = json.load(read_file)
        # except Exception:
        #     logger.exception('[LVP Main  ] Unable to open objectives.json.')
        #     raise

    # def get_objective_info(self, objective_id: str) -> dict:
    #     return self.objectives[objective_id]

    def reconnect(self):
        ctx = _app_ctx.ctx

        gui_logger.button('RECONNECT_MICROSCOPE')
        logger.info('[LVP Main  ] Reconnecting to microscope...')

        lumaview = ctx.lumaview
        settings = ctx.settings

        lumaview.scope.disconnect()
        lumaview.scope = None
        # The frame-size dedupe record describes the OLD camera; carried
        # across the swap it would absorb the first matching apply on the
        # new one (and its in-flight bookkeeping belongs to tasks queued
        # against the discarded scope).
        self._frame_size_applier = _CoalescingApplier(name='frame_size')
        # Reinitialize the scope object (connects motorboard, ledboard, camera)
        import modules.lumascope_api as lumascope_api

        lumaview.scope = lumascope_api.Lumascope(
            camera_type=settings['camera_type'], simulate=ctx.simulate_mode
        )
        _labware_id, labware = get_selected_labware()

        # Single hardware initialization call
        scope_config = self.scopes.get(settings.get('microscope'))
        config = ScopeInitConfig.from_settings(settings, labware, scope_config=scope_config)
        lumaview.scope.initialize(config)
        # Start gate release: configuration is applied, so open the gate and
        # fire the single grab (the camera-lifecycle split -- connect() left
        # it configured but not grabbing).
        lumaview.scope.imaging.start_streaming()

        ctx.sequenced_capture_runner.set_scope(lumaview.scope)
        ctx.autofocus_runner.set_scope(lumaview.scope)

        # Restart display

        ctx.scope_display.stop()
        ctx.scope_display.start()

        # LVP-A-5: ScopeSession owns the standard startup orchestration
        # (ALL-axis home + turret-positioning) -- same path the App's
        # on_start uses. Pre-LVP-A-5 this block was open-coded here and
        # had subtly drifted from the App's version.
        ctx.session.start_application_session(disable_homing=ctx.disable_homing)
        # Resync the whole per-camera UI surface from the NEW camera: refresh
        # the slider caps first (reconnect previously left the gain cap stale,
        # a blackout risk on a lower-cap camera), then the per-layer ranges +
        # gates through the single grouping.
        ctx.max_exposure = camera_max_exposure_for_ui(lumaview.scope.imaging)
        ctx.max_gain = camera_max_gain_for_ui(lumaview.scope.imaging)
        ctx.image_settings.sync_camera_capability_ranges()
        # Re-apply the VISIBLE layer (not a hardcoded channel) so its controls
        # reflect the new camera -- e.g. a non-BF open layer's gain/exposure
        # sliders get re-enabled when the new camera lacks hardware auto-gain.
        visible_layer = ctx.image_settings.open_or_default_layer()
        layer_obj = ctx.image_settings.layer_lookup(layer=visible_layer)
        layer_obj.apply_settings()

        scope_leds_off()

        # Refresh position display after reconnect (M22)
        ctx.motion_settings.update_xy_stage_control_gui(full_redraw=True)

        logger.info('[LVP Main  ] Reconnection complete.')

    # load settings from JSON file
    def load_settings(self, filename='./data/current.json'):
        logger.info('[LVP Main  ] MicroscopeSettings.load_settings()')
        ctx = _app_ctx.ctx

        lumaview = ctx.lumaview
        settings = ctx.settings

        try:
            # Settings are imported at the very beginning of file

            if settings['profiling']['enabled']:
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # noqa: F841 -- deferred
                # Joined to the data directory, never CWD-relative: an
                # installed build cannot write beside its executable.
                profiling_save_path = os.path.join(ctx.source_path, 'logs/profiling')
                MemoryLeakProfiler.start(root_log_dir=profiling_save_path)
                logger.info('[LVP Main  ] Memory Profiler started.')

            # Handle / object-type leak diagnostic. Same opt-in pattern as
            # the memory profiler above; settings-driven so customers and
            # bench operators can enable without rebuilding.
            if settings.get('profiling', {}).get('handle_trace_enabled', False):
                from lib import handle_trace as _handle_trace

                _handle_trace.enable(
                    obj_sample_every=int(
                        settings['profiling'].get('handle_trace_obj_sample_every', 1000)
                    )
                )

            if 'autogain' not in settings['protocol']:
                settings['protocol']['autogain'] = {
                    'max_duration_seconds': 1.0,
                    'target_brightness': 0.3,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                }

            try:
                live_folder = pathlib.Path(settings['live_folder'])
                # Resolve relative paths against Documents app folder when installed,
                # not CWD (which is Program Files and not writable).
                if not live_folder.is_absolute():
                    from lvp_logger import lvp_appdata

                    live_folder = pathlib.Path(lvp_appdata) / live_folder
                live_folder = live_folder.resolve()
                live_folder.mkdir(exist_ok=True, parents=True)

            except Exception as e:
                logger.warning(
                    f'[LVP Main  ] Unable to find/create live image folder at {settings["live_folder"]}: {e}'
                )
                try:
                    from lvp_logger import lvp_appdata

                    live_folder = pathlib.Path(lvp_appdata) / 'capture'
                except Exception:
                    live_folder = pathlib.Path.home() / 'Documents' / 'LumaViewPro' / 'capture'
                live_folder = live_folder.resolve()
                live_folder.mkdir(exist_ok=True, parents=True)
                logger.info(f'[LVP Main  ] Defaulting live image folder to {live_folder!s}')

            settings['live_folder'] = str(live_folder)

            # update GUI values from JSON data:

            # Scope auto-detection. The model selector lives in Advanced
            # Settings; write the detected (or saved) model to the settings
            # SSOT here, then reconfigure the UI for it (control visibility +
            # read-only model label + stage redraw, in that order).
            detected_model = lumaview.scope.diagnostics.get_microscope_model()
            if detected_model in self.scopes:
                logger.info(f'[LVP Main  ] Auto-detected scope as {detected_model}')
                settings['microscope'] = detected_model
            else:
                # Fires whether or not `filename` exists on disk, so naming it
                # here would be a guess -- report the value actually in effect.
                logger.info(
                    f'[LVP Main  ] No scope model reported by hardware; keeping '
                    f'stored scope selection {settings["microscope"]!r}'
                )
            self.reconfigure_for_scope()

            # Image mode selector: populate the options from the camera's
            # capability, then select the stored mode. A stored 12-bit mode on
            # an 8-bit-only camera falls back to 8-bit and tells the user.
            # Setting the spinner text fires select_image_mode (on_text), which
            # caches the mode and applies the pixel format.
            formats = self.load_image_modes()
            mode = image_mode.resolve_settings_image_mode(settings)
            # Only downgrade when the camera DEFINITIVELY lacks 12-bit (formats
            # known and without Mono12). An empty list here means the camera
            # is not up yet -- keep the stored mode; the options refresh when
            # the spinner is next opened.
            if formats and mode not in image_mode.available_modes(formats):
                from modules.notification_center import notifications

                notifications.warning(
                    'Camera',
                    'Image mode not supported',
                    'This camera supports 8-bit capture only; the saved 12-bit '
                    'image mode was changed to 8-bit.',
                )
                mode = image_mode.IMAGE_MODE_8BIT
                settings['image_mode'] = mode
            self.ids['image_mode_spinner'].text = image_mode.IMAGE_MODE_LABELS[mode]

            self.ids['live_image_output_format_spinner'].text = settings['image_output_format'][
                'live'
            ]
            # JPG quality slider reflects the saved preference; enable
            # state is set by select_live_image_output_format (JPG only).
            jpg_quality = int(settings.get('jpg_quality', 90))
            self.ids['jpg_quality_slider'].value = jpg_quality
            self.ids['jpg_quality_value_label'].text = str(jpg_quality)
            self.select_live_image_output_format()

            # Migrate legacy 'ImageJ Hyperstack' spinner value to the
            # honest 'OME-TIFF Hyperstack' label. The underlying file
            # format never changed (always OME-TIFF); only the label
            # was misleading. Migrating on load means existing user
            # settings.json files keep working without manual edits.
            sequenced_fmt = settings['image_output_format']['sequenced']
            if sequenced_fmt == 'ImageJ Hyperstack':
                sequenced_fmt = image_mode.OUTPUT_FORMAT_HYPERSTACK
                settings['image_output_format']['sequenced'] = sequenced_fmt
            self.ids['sequenced_image_output_format_spinner'].text = sequenced_fmt
            self.select_sequenced_image_output_format()

            # The exposure/gain slider caps from the live camera (the resolver
            # applies the documented no-camera fallback; #616). The gain cap
            # keeps the slider honest per-camera -- a universal 48 dB let LS620
            # users overdrive past the usable range and black out the image.
            max_exposure = camera_max_exposure_for_ui(lumaview.scope.imaging)
            ctx.max_exposure = max_exposure
            max_gain = camera_max_gain_for_ui(lumaview.scope.imaging)
            ctx.max_gain = max_gain

            if not settings['video_as_frames']:
                self.ids['video_recording_format_spinner'].text = 'mp4'
            else:
                self.ids['video_recording_format_spinner'].text = 'Frames'

            self.select_video_recording_format()

            if 'live_view_fps' in settings:
                ctx.live_view_fps = settings['live_view_fps']
            else:
                ctx.live_view_fps = 30

            fps_label = 'Max (uncapped)' if ctx.live_view_fps == 0 else str(ctx.live_view_fps)
            logger.info(f'[LVP Main  ] Live view FPS set to {fps_label}')

            # Set Frame Size UI
            binning_size_str = settings['binning']['size']
            binning_size = binning.binning_size_str_to_int(text=binning_size_str)

            self.ids['frame_width_id'].text = str(settings['frame']['width'] * binning_size)
            self.ids['frame_height_id'].text = str(settings['frame']['height'] * binning_size)

            # Pixel Binning -- UI recalculation only, scope.imaging.set_binning_size()
            # handled by scope.initialize() below
            self.ids['binning_spinner'].text = binning_size_str
            self.select_binning_size()

            objective_id = settings['objective_id']

            # Mutate turret config keys from str to int for cleaner handling
            settings['turret_objectives'] = {
                int(k): v for k, v in settings['turret_objectives'].items()
            }

            if lumaview.scope.capabilities.has_turret:
                turret_objectives = list(settings['turret_objectives'].values())
                assigned = [obj for obj in turret_objectives if obj is not None]
                if not assigned:
                    from modules.notification_center import notifications

                    notifications.warning(
                        'Turret',
                        'No Turret Objectives Assigned',
                        'Turret positions have no objectives assigned. '
                        'Please assign objectives in Objective Control > Turret before running protocols.',
                    )
                elif objective_id not in assigned:
                    logger.warning(
                        f'Startup objective {objective_id} not found in turret objectives ({turret_objectives}).'
                    )

            vertical_control_id = ctx.motion_settings.ids['verticalcontrol_id']
            v_control_objective_spinner = vertical_control_id.ids['objective_spinner2']
            v_control_objective_spinner.text = objective_id

            objective_helper = ctx.objective_helper
            objective = objective_helper.get_objective_info(objective_id=objective_id)

            # The objective already in place at launch never passes through
            # the selection handler, so without this a session that changed
            # nothing would have no record of the scale it was using.
            common_utils.log_resolved_optics(
                objective_id=objective_id,
                focal_length=objective['focal_length'],
                binning_size=binning_size,
            )

            # Populate FOV fields at startup; otherwise the fields stay blank
            # until the user clicks Frame Size or selects an objective (both
            # have their own FOV-recalc handlers).
            fov_size = common_utils.get_field_of_view(
                focal_length=objective['focal_length'],
                frame_size=settings['frame'],
                binning_size=binning_size,
            )
            fov_w_text, fov_h_text = common_utils.format_field_of_view(fov_size)
            self.ids['field_of_view_width_id'].text = fov_w_text
            self.ids['field_of_view_height_id'].text = fov_h_text

            # Load previous turret position objectives
            for turret_pos, objective_id in settings['turret_objectives'].items():
                if objective_id is None:
                    button_text = f'{turret_pos}'
                else:
                    magnification = objective_helper.get_objective_info(objective_id=objective_id)[
                        'magnification'
                    ]
                    button_text = f'{magnification}x'

                vertical_control_id.ids[f'turret_pos_{turret_pos}_btn'].text = button_text

            if settings['scale_bar']['enabled']:
                self.ids['enable_scale_bar_btn'].state = 'down'
            else:
                self.ids['enable_scale_bar_btn'].state = 'normal'

            # Single hardware initialization call -- replaces scattered
            # scope.imaging.set_frame_size / set_binning_size / set_stage_offset /
            # set_turret_config / set_objective / set_scale_bar / set_acceleration_limit
            _labware_id, labware = get_selected_labware()
            scope_config = self.scopes.get(settings.get('microscope'))
            config = ScopeInitConfig.from_settings(settings, labware, scope_config=scope_config)
            lumaview.scope.initialize(config)
            # Start gate release (primary startup site): configuration is
            # applied, so open the gate and fire the single grab.
            lumaview.scope.imaging.start_streaming()

            protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
            protocol_settings.ids['capture_period'].text = str(settings['protocol']['period'])
            protocol_settings.ids['capture_dur'].text = str(settings['protocol']['duration'])
            protocol_settings.ids['labware_spinner'].text = settings['protocol']['labware']
            protocol_settings.select_labware()
            # Apply the persisted step-location view at startup; the toggle
            # that edits this now lives in Advanced Settings.
            ctx.stage.show_protocol_steps(enable=settings['show_step_locations'])

            zstack_settings = ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id']
            zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']
            zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
            zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])

            z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
                text_label=settings['zstack']['position']
            )

            zstack_config = ZStackConfig(
                range=settings['zstack']['range'],
                step_size=settings['zstack']['step_size'],
                current_z_reference=z_reference,
                current_z_value=None,
            )

            zstack_settings.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

            if 'show_tooltips' in settings:
                if settings['show_tooltips']:
                    self.ids['show_tooltips_btn'].state = 'down'
                    ctx.show_tooltips = True
                else:
                    self.ids['show_tooltips_btn'].state = 'normal'
                    ctx.show_tooltips = False

            # Stimulation is firmware-gated. The enable toggle lives in
            # Advanced Settings now; startup just establishes the setting and
            # pushes the persisted state down to every layer via the single
            # owner (which forces it off on unsupported firmware).
            if 'stimulation_enabled' not in settings:
                settings['stimulation_enabled'] = False
            self.apply_stimulation_support()

            # Protocol accordions are permanently disabled (no longer a setting)
            settings.pop('disable_protocol_accordions', None)

            for layer in common_utils.get_layers():
                layer_obj = ctx.image_settings.layer_lookup(layer=layer)

                # Set initializing flag to prevent apply_settings during load
                layer_obj._initializing = True

                if layer in common_utils.get_fluorescence_layers():
                    layer_obj.ids['composite_threshold_slider'].value = settings[layer][
                        'composite_brightness_threshold'
                    ]

                if 'ill_ma' in settings[layer]:
                    layer_obj.ids['ill_slider'].value = settings[layer]['ill_ma']

                # Size the sliders to the camera caps BEFORE setting the value
                # (the Kivy slider clamps the displayed value to its max). The
                # over-cap STORED value is reconciled + persisted by the single
                # clamp_layer_settings_to_caps pass after the loop, not a
                # duplicate inline clamp here.
                layer_obj.ids['gain_slider'].max = max_gain
                layer_obj.ids['gain_slider'].value = settings[layer]['gain_db']
                layer_obj.ids['exp_slider'].max = max_exposure
                layer_obj.ids['exp_slider'].value = settings[layer]['exp_ms']

                layer_obj.ids['false_color'].active = settings[layer]['false_color']

                if 'sum' in settings[layer]:
                    layer_obj.ids['sum_slider'].value = settings[layer]['sum']
                else:
                    layer_obj.ids['sum_slider'].value = 1

                if settings[layer]['acquire'] == 'image':
                    layer_obj.ids['acquire_image'].active = True
                elif settings[layer]['acquire'] == 'video':
                    layer_obj.ids['acquire_video'].active = True
                else:
                    settings[layer]['acquire'] = None
                    layer_obj.ids['acquire_none'].active = True

                video_config = settings[layer]['video_config']
                DEFAULT_VIDEO_DURATION_SEC = 5
                DEFAULT_VIDEO_FPS = 30

                if video_config is None:
                    video_config = {}

                if 'duration' not in video_config:
                    video_config['duration'] = DEFAULT_VIDEO_DURATION_SEC

                if 'fps' not in video_config or video_config['fps'] <= 0:
                    video_config['fps'] = DEFAULT_VIDEO_FPS

                settings[layer]['video_config'] = video_config

                layer_obj.ids['video_duration_text'].text = str(video_config['duration'])
                layer_obj.ids['video_duration_slider'].value = video_config['duration']

                layer_obj.ids['autofocus'].active = settings[layer]['autofocus']

                # Clear initializing flag - settings are now loaded
                layer_obj._initializing = False

                if 'stim_config' in settings[layer]:
                    # Default to hidden until enabled
                    layer_obj.show_stim_controls = False

                    stim_config = settings[layer]['stim_config']
                    layer_obj.ids['stim_enable_btn'].active = stim_config['enabled']
                    layer_obj.ids['stim_disable_btn'].active = not stim_config['enabled']
                    layer_obj.ids['stim_ill_text'].text = str(stim_config.get('illumination', 100))
                    layer_obj.ids['stim_ill_slider'].value = float(
                        stim_config.get('illumination', 100)
                    )
                    layer_obj.ids['stim_freq_text'].text = str(stim_config['frequency'])
                    layer_obj.ids['stim_freq_slider'].value = float(stim_config['frequency'])
                    layer_obj.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                    layer_obj.ids['stim_pulse_width_slider'].value = float(
                        stim_config['pulse_width']
                    )
                    layer_obj.ids['stim_pulse_count_text'].text = str(stim_config['pulse_count'])
                    layer_obj.ids['stim_pulse_count_slider'].value = int(stim_config['pulse_count'])

                    # Force hide until enabled
                    layer_obj.ids['stim_ill_box'].visible = False
                    layer_obj.ids['stim_pulse_count_box'].visible = False
                    layer_obj.ids['stim_freq_box'].visible = False
                    layer_obj.ids['stim_pulse_width_box'].visible = False
                    layer_obj.ids['stim_ill_box'].opacity = 0
                    layer_obj.ids['stim_pulse_count_box'].opacity = 0
                    layer_obj.ids['stim_freq_box'].opacity = 0
                    layer_obj.ids['stim_pulse_width_box'].opacity = 0

                    layer_obj.update_stim_controls_visibility()

            # Reconcile any layer whose stored gain/exposure exceeds the new
            # camera's cap down to it -- the single clamp owner, shared with the
            # reconnect resync, instead of the per-layer inline clamp this loop
            # used to carry.
            ctx.image_settings.clamp_layer_settings_to_caps()

        except json.JSONDecodeError as e:
            # Real "incompatible JSON" -- file content can't be parsed.
            logger.error(f'[LVP Main  ] load_settings: JSON parse error in {filename}: {e}')
        except FileNotFoundError as e:
            logger.error(f'[LVP Main  ] load_settings: settings file missing: {e}')
        except Exception as e:
            # LOG-3 / UI-LOAD-1: this used to log "Incompatible JSON file
            # for Microscope Settings" for ANY exception during load. The
            # message must name the actual failure mode -- kivy widget
            # exceptions, attribute errors, etc. were being misattributed
            # to the JSON file. Bit us when the wrapped wording sent the
            # operator to the JSON file when the bug was in widget code,
            # AND the swallow let execution continue into a second crash
            # in set_ui_features_for_scope below.
            logger.exception(
                f'[LVP Main  ] load_settings failed in {filename}: {type(e).__name__}: {e}'
            )
            # Re-raise so the caller (LumaViewProApp.build) sees the
            # failure and we don't silently degrade through the rest of
            # the build path. Without this, a kivy WidgetException in the
            # accordion-widget tree was caught and swallowed, then the
            # next call to set_ui_features_for_scope hit the same bug
            # uncaught -- a misleading "double-crash with first one
            # hidden" pattern.
            raise

        self.set_ui_features_for_scope()

    def update_bullseye_state(self):
        gui_logger.toggle('BULLSEYE', self.ids['enable_bullseye_btn_id'].state == 'down')
        if self.ids['enable_bullseye_btn_id'].state == 'down':
            _app_ctx.ctx.viewer.update_shader(false_color='BF')
            _app_ctx.ctx.scope_display.use_bullseye = True
        else:
            for layer in common_utils.get_layers():
                layer_obj = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
                accordion_item = _app_ctx.ctx.image_settings.accordion_item_lookup(layer=layer)
                if not accordion_item.collapse:
                    if layer_obj.ids['false_color'].active:
                        _app_ctx.ctx.viewer.update_shader(false_color=layer)

                    break

            _app_ctx.ctx.scope_display.use_bullseye = False

    def _supported_pixel_formats(self):
        """The active camera's supported pixel formats, or [] if unavailable."""
        try:
            return _app_ctx.ctx.lumaview.scope.imaging.get_supported_pixel_formats() or []
        except Exception:
            logger.warning('[LVP Main  ] Could not read camera pixel formats; assuming 8-bit only.')
            return []

    def load_image_modes(self):
        """Populate the image-mode spinner with the modes this camera supports.

        A camera without Mono12/Mono12p offers 8-bit only, so the 12-bit
        options never appear where they cannot work. Returns the queried
        formats so the load-time sync can reuse them.
        """
        formats = self._supported_pixel_formats()
        self.ids['image_mode_spinner'].values = image_mode.available_mode_labels(formats)
        return formats

    # Drives the 8-bit binning depth-loss hint row; the row height follows the
    # label's wrapped texture so the multi-line warning is not clipped.
    binning_depth_hint_active = BooleanProperty(False)

    def _refresh_binning_depth_hint(self):
        """Show the depth-loss hint below the binning control only when binning
        is active in an 8-bit mode (the binned range is truncated on save).
        """
        if 'binning_depth_hint_row' not in self.ids:
            return
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            return
        binning_size = binning.binning_size_str_to_int(self.ids['binning_spinner'].text)
        self.binning_depth_hint_active = image_mode.depth_truncation_warning_active(
            binning_size, scope_display.image_mode
        )

    def select_image_mode(self):
        ctx = _app_ctx.ctx

        label = self.ids['image_mode_spinner'].text
        mode = image_mode.LABEL_TO_IMAGE_MODE.get(label)
        if mode is None:
            return  # 'Select' placeholder or an unknown label -- ignore
        gui_logger.select('IMAGE_MODE', mode)

        # The mode mirrors commit SYNCHRONOUSLY (display consumers read
        # scope_display.image_mode on the next frame; the depth hint reads
        # settings); a rejected format apply is corrected by the failure
        # callback below -- commit-then-revert, so a rejected depth cannot
        # STAY recorded with captures tagged at a depth the camera never
        # took. The prior mode is captured first for the revert.
        settings = ctx.settings
        prior_mode = settings.get('image_mode')
        ctx.scope_display.image_mode = mode
        settings['image_mode'] = mode
        self._refresh_binning_depth_hint()

        # Apply the capture depth to the camera. Resolve to a format the
        # sensor actually supports BEFORE pushing, so we never request a
        # format it lacks (e.g. Mono8 on an IDS sensor that exposes only
        # Mono10/12 -- that logs a spurious 'Unsupported' warning). Route
        # through the camera executor to avoid racing the live-view grab loop.
        capture_depth = image_mode.resolve_image_mode(mode)['capture_depth']

        def _set_pixel_format():
            imaging = ctx.lumaview.scope.imaging
            target = image_mode.select_capture_pixel_format(
                capture_depth, imaging.get_supported_pixel_formats()
            )
            if target is None:
                # No matching format is a display-mode-only change:
                # nothing to apply, the mode commit stands.
                return True
            # The absent-camera False propagates to the callback so the
            # mode commit is reverted -- a format that never reached the
            # hardware must not stay recorded as the capture depth.
            return imaging.set_pixel_format(target)

        ctx.camera_executor.put(
            IOTask(
                action=_set_pixel_format,
                callback=self._on_image_mode_outcome,
                cb_args=(mode, prior_mode),
                pass_result=True,
                # The rejection is already notified at the API layer; the
                # callback owns the UI revert.
                silent_on_failure=True,
            )
        )

    def _on_image_mode_outcome(self, mode, prior_mode, result=None, exception=None):
        """UI-thread landing for an image-mode apply: no-op on success (the
        mirrors committed synchronously at select time); on failure, revert
        spinner, settings, and the display mode to the captured prior state."""
        if exception is None and result:
            return
        ctx = _app_ctx.ctx
        settings = ctx.settings
        if prior_mode is not None:
            settings['image_mode'] = prior_mode
            ctx.scope_display.image_mode = prior_mode
            prior_label = image_mode.IMAGE_MODE_LABELS.get(prior_mode)
            if prior_label:
                self.ids['image_mode_spinner'].text = prior_label
        self._refresh_binning_depth_hint()
        logger.error(
            f'[LVP Main  ] image mode {mode} not applied '
            f'({exception or "no result"}); reverted to {prior_mode}'
        )

    def select_live_image_output_format(self):
        settings = _app_ctx.ctx.settings
        fmt = self.ids['live_image_output_format_spinner'].text
        gui_logger.select('LIVE_IMAGE_OUTPUT_FORMAT', fmt)
        settings['image_output_format']['live'] = fmt
        # The JPG-quality row's visibility (and disabled state) follows the
        # selected format declaratively in lumaviewpro.kv (jpg_quality_row binds
        # to live_image_output_format_spinner.text), so no toggle is needed here.

    def update_jpg_quality(self, value):
        settings = _app_ctx.ctx.settings
        quality = int(value)
        settings['jpg_quality'] = quality
        if 'jpg_quality_value_label' in self.ids:
            self.ids['jpg_quality_value_label'].text = str(quality)
        gui_logger.slider('JPG_QUALITY', quality)

    def select_sequenced_image_output_format(self):
        settings = _app_ctx.ctx.settings
        fmt = self.ids['sequenced_image_output_format_spinner'].text
        gui_logger.select('SEQUENCED_IMAGE_OUTPUT_FORMAT', fmt)
        settings['image_output_format']['sequenced'] = fmt

    def select_video_recording_format(self):
        settings = _app_ctx.ctx.settings
        if self.ids['video_recording_format_spinner'].text == 'mp4':
            settings['video_as_frames'] = False
        else:
            settings['video_as_frames'] = True

    def update_scale_bar_state(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings

        if self.ids['enable_scale_bar_btn'].state == 'down':
            enabled = True
        else:
            enabled = False
        gui_logger.toggle('SCALE_BAR', enabled)

        ctx.lumaview.scope.imaging.set_scale_bar(enabled=enabled)
        settings['scale_bar']['enabled'] = enabled

    def update_crosshairs_state(self):
        enabled = self.ids['enable_crosshairs_btn'].state == 'down'
        gui_logger.toggle('CROSSHAIRS', enabled)
        scope_display = _app_ctx.ctx.scope_display
        if self.ids['enable_crosshairs_btn'].state == 'down':
            scope_display.use_crosshairs = True
            scope_display.show_crosshairs(True)
        else:
            scope_display.use_crosshairs = False
            scope_display.show_crosshairs(False)

    def update_live_image_histogram_equalization(self):
        ctx = _app_ctx.ctx
        enabled = self.ids['enable_live_image_histogram_equalization_btn'].state == 'down'
        gui_logger.toggle('LIVE_HISTOGRAM_EQUALIZATION', enabled)
        ctx.scope_display.use_live_image_histogram_equalization = enabled
        ctx.live_histo_setting = enabled

    def update_show_tooltips(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        enabled = self.ids['show_tooltips_btn'].state == 'down'
        gui_logger.toggle('SHOW_TOOLTIPS', enabled)
        ctx.show_tooltips = enabled
        settings['show_tooltips'] = enabled

    def apply_stimulation_support(self):
        """Push the persisted global stimulation enable to every channel.

        Single owner of the per-layer stimulation sync. Reads
        ``settings['stimulation_enabled']`` (the source of truth, populated
        by the settings load) rather than a widget, so the startup load and
        the Advanced Settings toggle both drive the same path. Firmware
        without stim support can never enable it, even if a stale setting
        says otherwise.
        """
        settings = _app_ctx.ctx.settings
        stimulation_enabled = firmware_stim_supported() and settings['stimulation_enabled']
        settings['stimulation_enabled'] = stimulation_enabled

        # Update all layer controls
        for layer in common_utils.get_layers():
            if layer in common_utils.get_fluorescence_layers():
                layer_obj = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
                if layer_obj:
                    if stimulation_enabled:
                        # Enable stimulation features
                        layer_obj.stimulation_support = True
                        # Don't automatically show stim controls, just enable support
                    else:
                        # Disable stimulation features
                        layer_obj.stimulation_support = False
                        layer_obj.show_stim_controls = False
                        layer_obj.show_camera_controls = True
                        # Set stim to disabled
                        if 'stim_disable_btn' in layer_obj.ids:
                            layer_obj.ids['stim_disable_btn'].active = True
                        # Disable stim_config
                        if 'stim_config' in settings[layer]:
                            settings[layer]['stim_config']['enabled'] = False

    # Save settings to JSON file
    def save_settings(self, file='./data/current.json', *, force=False):
        """Save the current settings dict to disk as JSON.

        LVP-A-4: when ``force=False`` (default), the save is skipped if
        no hardware was connected during the session -- without hardware
        the slider defaults (0.01ms exposure, etc.) would overwrite the
        user's real per-channel settings in current.json. The gate
        previously lived inline in ``lumaviewpro.py:on_stop``; lifted
        here so every caller (engineering plugin save-on-quit, REST
        endpoint, scheduled save, future CLI tools) gets the same
        guard.

        Pass ``force=True`` only when the caller really does want the
        save regardless of hardware presence (rare; e.g. an explicit
        "save folder paths only" path that doesn't touch per-channel
        values).

        TODO 4.1: split by section so non-hardware values (folder
        paths, protocol config) are always saved while hardware values
        (gain, exposure) are gated. Until then, all-or-nothing on
        hardware presence.
        """
        logger.info('[LVP Main  ] MicroscopeSettings.save_settings()')
        ctx = _app_ctx.ctx
        settings = ctx.settings

        # LVP-A-4: hardware-presence gate.
        if not force:
            scope = ctx.lumaview.scope if ctx.lumaview else None
            had_hardware = bool(
                scope and (scope.camera_connected or scope.motor_connected or scope.led_connected)
            )
            if not had_hardware:
                logger.info(
                    '[LVP Main  ] save_settings: skipped -- no hardware was '
                    'connected this session (would overwrite real per-channel '
                    'values with slider defaults). Pass force=True to override.'
                )
                return

        if isinstance(file, str) and (file[-5:].lower() != '.json'):
            file = file + '.json'

        t0 = time.monotonic()
        with ctx.settings_lock:
            settings_snapshot = copy.deepcopy(settings)
        # Resolve relative paths against source_path instead of relying on CWD
        if not os.path.isabs(file):
            file = os.path.join(ctx.source_path, file)
        with open(file, 'w') as write_file:
            json.dump(settings_snapshot, write_file, indent=4, cls=CustomJSONizer)
        dt = time.monotonic() - t0
        if dt > 0.1:
            logger.warning(f'[LVP Main  ] save_settings took {dt * 1000:.0f}ms')

        # Dispatch on_settings_changed to plugins whose subscribes_to
        # prefix-matches the keys that diffed since the last save.
        # The first call after startup caches the baseline without
        # firing; subsequent saves fire only when actual values change.
        try:
            from modules.plugins import fire_settings_save_hooks

            fire_settings_save_hooks(ctx, settings_snapshot)
        except Exception:
            logger.exception('[LVP Main  ] save_settings: plugin notification failed')

    def load_binning_sizes(self):
        spinner = self.ids['binning_spinner']
        # Use Lumascope API to get available binning sizes
        try:
            sizes = _app_ctx.ctx.lumaview.scope.imaging.get_available_binning_sizes()
        except Exception:
            logger.warning('[LVP Main  ] Could not read camera binning sizes, using defaults.')
            sizes = [1, 2, 4]
        spinner.values = [f'{s}x{s}' for s in sizes]

    def _ui_binning_size(self) -> int:
        """The binning factor the UI currently shows (the settings SSOT).

        Native-ROI reconstruction multiplies the displayed frame size by the
        binning it was entered at, so it must read the SYNCHRONOUS UI binning
        (``settings['binning']['size']``), NOT ``imaging.get_binning_size()``.
        The hardware binning is applied asynchronously through the camera
        executor, so right after a binning change the driver still reports the
        previous factor; reconstructing displayed * that stale factor rebuilds
        a wrong (and, when only one axis was previously off-square, non-square)
        native ROI -- the 1056x950-instead-of-950x950 bench bug.
        """
        settings = _app_ctx.ctx.settings
        return binning.binning_size_str_to_int(settings['binning']['size'])

    def _native_roi(self) -> dict:
        """Return the unbinned ROI -- the source of truth for frame sizing.

        Persisted as ``settings['frame']['native_width']/['native_height']``.
        The stored pair is the unconditional source of truth (binning never
        changes it). Only when absent (older settings files that stored just
        the displayed size) is it reconstructed from the displayed frame size
        times the UI binning, capped at the sensor native resolution.
        """
        ctx = _app_ctx.ctx
        frame = ctx.settings['frame']
        imaging = ctx.lumaview.scope.imaging
        native_max = imaging.get_native_resolution()
        if 'native_width' in frame and 'native_height' in frame:
            native = {
                'width': int(frame['native_width']),
                'height': int(frame['native_height']),
            }
            src = 'stored'
        else:
            cur_binning = self._ui_binning_size()
            displayed = {'width': int(frame['width']), 'height': int(frame['height'])}
            cap = native_max or {
                'width': displayed['width'] * cur_binning,
                'height': displayed['height'] * cur_binning,
            }
            # displayed_to_native already caps the reconstruction at the cap
            # (native_max when known), so no separate clamp is needed here.
            native = binning.displayed_to_native(displayed, cur_binning, cap)
            src = (
                f'reconstructed displayed={displayed["width"]}x{displayed["height"]} '
                f'ui_binning={cur_binning}'
            )
        # The stored pair is returned verbatim -- the unconditional source of
        # truth. It is deliberately NOT re-capped against the live native_max: a
        # transient small reading (a camera reconnect / init race) would
        # otherwise shrink the persisted native_* permanently when a binning
        # toggle re-stores it. The driver's set_frame_size is the real clamp to
        # the current sensor max.
        # Forensic line: whether the native ROI came from the stored source of
        # truth or was rebuilt from displayed*binning, and at which binning.
        # A reconstruction against a stale binning is how the native size
        # silently drifted, so the src + inputs stay visible in the log.
        logger.info(f'[LVP Main  ] native_roi: src={src} -> {native["width"]}x{native["height"]}')
        return native

    def _store_native_roi(self, native: dict) -> None:
        """Persist the native ROI source of truth into settings['frame']."""
        frame = _app_ctx.ctx.settings['frame']
        frame['native_width'] = int(native['width'])
        frame['native_height'] = int(native['height'])

    def select_binning_size(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        lumaview = ctx.lumaview
        imaging = lumaview.scope.imaging

        new_binning_size_str = self.ids['binning_spinner'].text
        new_binning_size = binning.binning_size_str_to_int(new_binning_size_str)

        # Reject a binning level this camera does not support and restore the
        # spinner to the camera's actual binning.
        if new_binning_size not in imaging.get_available_binning_sizes():
            from modules.notification_center import notifications

            notifications.warning(
                'Camera',
                'Binning not supported',
                f'This camera does not support {new_binning_size_str} binning.',
            )
            self.ids['binning_spinner'].text = binning.binning_size_int_to_str(
                imaging.get_binning_size()
            )
            return

        # Capture the native ROI BEFORE overwriting the binning setting.
        # _native_roi reconstruction multiplies the displayed value by the UI
        # binning (settings['binning']['size']), so it must read the OLD binning
        # the current displayed value corresponds to; reading it after the
        # overwrite would rebuild native against the new factor and skew it (the
        # non-square frame at 2x). Storing it locks the source of truth so this
        # and every later binning change round-trips exactly -- without it,
        # settings that never had native_* fall through reconstruction
        # (displayed * binning) on every change, and at a coarse binning the
        # already-floored displayed value shrinks native a little each step so
        # the cycle drifts (1x1 -> 4x4 -> 1x1 came back smaller).
        native = self._native_roi()
        self._store_native_roi(native)

        gui_logger.select('BINNING', new_binning_size_str)

        # The displayed/captured size is native / binning, floored to the active
        # driver's DELIVERABLE granularity: get_pixel_alignment reports the
        # camera grid for floor-only drivers (Pylon/FX2/sim) and just 'even' for
        # the IDS driver, which crops back to the exact request -- so a 1900
        # frame stays 1900 on IDS but floors to the real grid elsewhere.
        new_frame = binning.native_to_displayed(
            native, new_binning_size, imaging.get_pixel_alignment()
        )

        # The binning settings value commits SYNCHRONOUSLY: _ui_binning_size
        # (native-ROI reconstruction) and the FOV math both document that
        # they read the synchronous UI binning, so deferring this write to
        # the apply's completion would let a frame edit made during the
        # (multi-second Pylon) apply reconstruct against the wrong epoch.
        # A rejected factor is corrected by the failure callback below --
        # commit-then-revert, not defer-and-diverge. The prior value is
        # captured first so the revert restores the exact pre-select state.
        prior_binning_size_str = settings['binning']['size']
        prior_frame = {
            'width': int(settings['frame']['width']),
            'height': int(settings['frame']['height']),
        }
        settings['binning']['size'] = new_binning_size_str
        self._refresh_binning_depth_hint()
        self.ids['frame_width_id'].text = str(new_frame['width'])
        self.ids['frame_height_id'].text = str(new_frame['height'])

        # During app init, scope.initialize() handles all hardware calls;
        # the mirrors just reflect the settings being loaded.
        if ctx.initializing:
            return

        # Route through camera executor to prevent race with live view grab
        # loop. The frame push is enqueued after, so it lands once the new
        # binning is applied and the driver can clamp to the right max. The
        # completion callback acts ONLY on failure, reverting every mirror
        # to the captured prior state -- a rejected factor must not stay
        # recorded (it feeds every native-ROI / FOV / stitch derivation).
        ctx.camera_executor.put(
            IOTask(
                action=imaging.set_binning_size,
                kwargs={'size': new_binning_size},
                callback=self._on_binning_apply_outcome,
                cb_args=(new_binning_size_str, prior_binning_size_str, prior_frame),
                pass_result=True,
                # The rejection is already notified at the API layer; the
                # callback owns the UI revert.
                silent_on_failure=True,
            )
        )
        self._apply_displayed_frame(new_frame)

    def _on_binning_apply_outcome(
        self, new_binning_size_str, prior_binning_size_str, prior_frame, result=None, exception=None
    ):
        """UI-thread landing for a binning apply: no-op on success (all
        mirrors committed synchronously at select time); on failure, revert
        settings, spinner, and the frame derivation to the captured prior
        state so a rejected factor cannot stay recorded."""
        if exception is None and result:
            return
        ctx = _app_ctx.ctx
        ctx.settings['binning']['size'] = prior_binning_size_str
        self.ids['binning_spinner'].text = prior_binning_size_str
        self._refresh_binning_depth_hint()
        self._apply_displayed_frame(prior_frame)
        logger.error(
            f'[LVP Main  ] binning {new_binning_size_str} not applied '
            f'({exception or "no result"}); reverted to {prior_binning_size_str}'
        )

    def reconfigure_for_scope(self) -> None:
        """Apply the current scope to the UI in the canonical order.

        set_ui_features_for_scope first (control visibility + the read-only
        model label), then a stage redraw for the new model's geometry. The
        single owner of the scope-change reconfigure sequence -- called at
        startup and when the Advanced Settings selector changes the scope, so
        the order is identical on both paths.
        """
        self.set_ui_features_for_scope()
        _app_ctx.ctx.stage.full_redraw()

    def set_ui_features_for_scope(self) -> None:
        ctx = _app_ctx.ctx
        settings = ctx.settings

        microscope_settings = ctx.motion_settings.ids['microscope_settings_id']
        scope_configs = microscope_settings.scopes
        selected_scope_config = scope_configs[settings['microscope']]

        microscope_settings.current_scope_model = settings['microscope']

        motion_settings = ctx.motion_settings
        motion_settings.set_turret_control_visibility(visible=selected_scope_config['Turret'])
        motion_settings.set_xystage_control_visibility(visible=selected_scope_config['XYStage'])
        motion_settings.set_tiling_control_visibility(visible=selected_scope_config['XYStage'])
        motion_settings.set_objective_control_visibility(visible=selected_scope_config['Focus'])

        image_settings = ctx.image_settings
        layers_config = selected_scope_config['Layers']
        image_settings.set_df_layer_control_visibility(visible=layers_config['Darkfield'])
        image_settings.set_lumi_layer_control_visibility(visible=layers_config['Lumi'])
        image_settings.set_fluoresence_layer_controls_visibility(
            visible=layers_config['Fluorescence']
        )
        image_settings.set_phasecontrast_layer_control_visibility(
            visible=layers_config['PhaseContrast']
        )

        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        protocol_settings.set_labware_selection_visibility(visible=selected_scope_config['XYStage'])

        ctx.motion_settings.ids['post_processing_id'].ids[
            'stitch_controls_id'
        ].set_button_enabled_state(state=selected_scope_config['XYStage'])

        if selected_scope_config['XYStage'] is False:
            # XYStage=False scopes (Lumi, LS820) keep a single-plate
            # ("Center Plate") graphic in the protocol tab so the crosshair
            # position is visible; only the XY motion capability is disabled
            # (set below). Stitch is hidden -- it needs tiling.
            protocol_settings.select_labware(labware='Center Plate')
            ctx.motion_settings.ids['post_processing_id'].hide_stitch()

        ctx.stage.set_motion_capability(enabled=selected_scope_config['XYStage'])
        ctx.stage.set_xy_stage_capability(enabled=selected_scope_config['XYStage'])

        # Size the protocol-tab stage holder to its width-based aspect for
        # every scope. The plate graphic now renders on XYStage=False scopes
        # too (single Center Plate), so the holder is no longer collapsed.
        # The kv-defined ``protocol_stage_holder_id`` FloatLayout has
        # ``height: self.width * 2 / 3``; set it explicitly and bind to width
        # so the holder follows resize (bind once, tracked on the widget so
        # repeated scope toggles don't stack handlers).
        protocol_stage_holder = protocol_settings.ids.get('protocol_stage_holder_id')
        if protocol_stage_holder is not None:
            protocol_stage_holder.size_hint_y = None
            protocol_stage_holder.height = max(1, int(protocol_stage_holder.width * 2 / 3))
            if not getattr(protocol_stage_holder, '_lvp_height_bound', False):
                protocol_stage_holder.bind(
                    width=lambda inst, w: setattr(inst, 'height', max(1, int(w * 2 / 3)))
                )
                protocol_stage_holder._lvp_height_bound = True

        # UI-1 follow-up (2026-05-03): cheap "reset on switch" -- explicit
        # resort of both accordions after any scope-config change so
        # successive LS850 <-> LS820 <-> LS620 transitions can't leave the
        # children list in a non-canonical state. Eric 2026-05-03:
        # "maybe it could do a fully reset when you switch" -- this is
        # that approach.
        try:
            ctx.motion_settings._resort_accordion()
        except Exception as e:
            logger.debug(f'[LVP Main  ] motion_settings._resort_accordion failed: {e}')
        try:
            image_settings._resort_accordion()
        except Exception as e:
            logger.debug(f'[LVP Main  ] image_settings._resort_accordion failed: {e}')

    def frame_size(self):
        """Apply a user edit of the frame width/height fields.

        The typed value is a displayed (post-binning) size, so the native ROI
        becomes ``displayed * binning`` capped at the sensor native resolution.
        The displayed size is then re-derived from that native ROI and applied,
        keeping the native source of truth and the camera in sync.
        """
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        ctx = _app_ctx.ctx
        lumaview = ctx.lumaview

        if not lumaview.scope.camera_connected:
            return

        imaging = lumaview.scope.imaging
        try:
            typed = get_current_frame_dimensions()
        except ValueError:
            frame = ctx.settings['frame']
            typed = {'width': frame['width'], 'height': frame['height']}

        # The typed value is a displayed size at the UI binning, so reconstruct
        # native against the synchronous UI binning, not the async hardware
        # binning (see _ui_binning_size).
        cur_binning = self._ui_binning_size()
        native_max = imaging.get_native_resolution() or {
            'width': int(typed['width']) * cur_binning,
            'height': int(typed['height']) * cur_binning,
        }
        native = binning.displayed_to_native(typed, cur_binning, native_max)
        self._store_native_roi(native)

        # Floor to the active driver's deliverable granularity (see
        # select_binning_size): the IDS driver crops to the exact request, so
        # get_pixel_alignment reports 'even' for it and the real grid elsewhere.
        displayed = binning.native_to_displayed(native, cur_binning, imaging.get_pixel_alignment())
        self._apply_displayed_frame(displayed)

    def _apply_displayed_frame(self, frame: dict) -> None:
        """Persist a displayed frame size, update the UI + FOV, push to camera.

        Does NOT change the native ROI -- callers that change native (a binning
        change or a frame-field edit) do so before calling this. The size is
        already native-anchored and aligned; the driver's set_frame_size does
        the final clamp to the camera max at the active binning, so no live
        max clamp is applied here (it would read a stale max during a binning
        change before the executor applies it).
        """
        ctx = _app_ctx.ctx
        lumaview = ctx.lumaview

        if not lumaview.scope.camera_connected:
            return

        width = int(frame['width'])
        height = int(frame['height'])
        try:
            min_frame_size = lumaview.scope.imaging.min_frame_size_cached
            width = max(width, min_frame_size['width'])
            height = max(height, min_frame_size['height'])
        except Exception:
            logger.warning('[LVP Main  ] Could not clamp frame size to camera minimum.')

        # The single framing chokepoint: both the frame-field edit and the
        # binning toggle reach the camera through here, so one log call records
        # every framing change the user makes (the prior gap that left the
        # frame-box resize invisible in the GUI log).
        gui_logger.frame_size(width, height, get_binning_from_ui())

        # Every mirror of "current geometry" (settings, text fields, FOV
        # labels) is written from the DELIVERED size in the completion
        # callback, never from this request: a rejected or clamped apply
        # once left the mirrors claiming a size the camera never held, so
        # tiling/FOV math disagreed with the frames on disk. The typed
        # text stays visible during the (up to ~11 s Pylon) apply, then
        # snaps to what the camera delivered.

        # Coalesce rapid frame_size() calls -- see _CoalescingApplier
        # + issue #624. The UI can fire this method several times in
        # quick succession when the user tabs between width and height
        # text fields (on_focus loss + on_text_validate both bound to
        # the same handler), and Pylon's stop_grabbing/start_grabbing
        # cycle takes ~11s on large frames, so naive queueing creates
        # minute-scale UI freezes.
        if self._frame_size_applier.submit((width, height)):
            ctx.camera_executor.put(
                IOTask(
                    action=self._frame_size_applier.apply_pending,
                    args=(self._push_frame_size,),
                )
            )
        # FOV is derived state (settings frame x binning), and both inputs
        # are current right here -- settings['frame'] is delivered-sourced
        # and the binning committed synchronously. Refreshing now covers
        # the dedupe-absorbed case (binning changed, same displayed size:
        # no push, no delivered callback, but the FOV still halves).
        self._refresh_fov_labels()

    def _push_frame_size(self, wh):
        """Camera-executor side of a frame-size apply: push to the camera
        and marshal the DELIVERED geometry back to the UI mirrors.

        Returns the DELIVERED (width, height) tuple so the coalescer
        records what the camera actually holds as its dedupe key -- a
        clamped delivery recorded under the request key would absorb the
        user's retype of the original size while the field showed the
        clamped one. A rejection raises out of here (contained by
        apply_pending, already notified at the API layer) and an
        absent-camera no-op returns None -- neither is recorded, so a
        retry of the same size still reaches the hardware. The scope slot
        is None for the whole reconnect window; that is the absent shape,
        not an error.
        """
        scope = getattr(_app_ctx.ctx.lumaview, 'scope', None)
        if scope is None:
            return None
        delivered = scope.imaging.set_frame_size(*wh)
        if not delivered:
            return None
        Clock.schedule_once(lambda dt: self._on_frame_size_applied(delivered), 0)
        return (int(delivered['width']), int(delivered['height']))

    def _on_frame_size_applied(self, delivered: dict) -> None:
        """UI-thread landing for an ACCEPTED frame-size apply: write every
        geometry mirror from the size the camera actually delivered."""
        settings = _app_ctx.ctx.settings

        width = int(delivered['width'])
        height = int(delivered['height'])
        settings['frame']['width'] = width
        settings['frame']['height'] = height
        self.ids['frame_width_id'].text = str(width)
        self.ids['frame_height_id'].text = str(height)
        self._refresh_fov_labels()

    def _refresh_fov_labels(self) -> None:
        """Recompute the FOV readout from the current delivered-sourced
        frame settings and the UI binning."""
        ctx = _app_ctx.ctx
        settings = ctx.settings
        objective = ctx.objective_helper.get_objective_info(objective_id=settings['objective_id'])
        fov_size = common_utils.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=settings['frame'],
            binning_size=get_binning_from_ui(),
        )
        fov_w_text, fov_h_text = common_utils.format_field_of_view(fov_size)
        self.ids['field_of_view_width_id'].text = fov_w_text
        self.ids['field_of_view_height_id'].text = fov_h_text

    def open_advanced_settings(self):
        """Open the Advanced Settings modal (power-user / rarely-touched rows)."""
        gui_logger.button('OPEN_ADVANCED_SETTINGS')
        from ui.advanced_settings import AdvancedSettings

        self._advanced_settings_popup = AdvancedSettings()
        self._advanced_settings_popup.open()

    def generate_support_report(self):
        """Show confirmation dialog, then generate a tech support report."""
        gui_logger.button('GENERATE_SUPPORT_REPORT')
        from ui.notification_popup import show_confirmation_popup

        show_confirmation_popup(
            title='Tech Support Report',
            message=(
                'This will create a diagnostic report to send to\n'
                'Etaluma Tech Support.\n\n'
                'The stage will be homed and moved during testing.\n'
                'Please remove any samples from the stage.\n\n'
                'This may take a few minutes.'
            ),
            confirm_text='Generate',
            cancel_text='Cancel',
            on_confirm=self._start_support_report,
        )

    def _start_support_report(self):
        from ui.progress_popup import CustomPopup
        from modules.tech_support_report import TechSupportReport
        import threading

        self._report_popup = CustomPopup(
            title='Generating Support Report...',
            auto_dismiss=False,
        )
        self._report_popup.open()

        def run():
            try:
                report = TechSupportReport(scope=_app_ctx.ctx.lumaview.scope)

                def progress(pct, msg):
                    Clock.schedule_once(lambda dt: self._update_report_progress(pct, msg), 0)

                path = report.generate(callback=progress, include_bandwidth_test=False)
                Clock.schedule_once(lambda dt: self._report_done(path), 0)
            except Exception as e:
                logger.error(f'Support report failed: {e}', exc_info=True)
                Clock.schedule_once(lambda dt: self._report_done(None), 0)

        threading.Thread(target=run, daemon=True).start()

    def _update_report_progress(self, pct, msg):
        if hasattr(self, '_report_popup') and self._report_popup:
            self._report_popup.progress = pct
            self._report_popup.text = msg

    def _report_done(self, zip_path):
        if hasattr(self, '_report_popup') and self._report_popup:
            self._report_popup.dismiss()
            self._report_popup = None

        from ui.notification_popup import show_notification_popup

        if zip_path:
            show_notification_popup(
                title='Report Complete',
                message=(
                    f'Saved to Desktop:\n{zip_path.name}\n\n'
                    f'Please email this file to:\n'
                    f'techsupport@etaluma.com'
                ),
            )
        else:
            show_notification_popup(
                title='Report Failed',
                message=(
                    'Could not generate the report.\n'
                    'Check the log file for details and contact\n'
                    'techsupport@etaluma.com directly.'
                ),
            )

    def zip_logs_only(self):
        """Quick zip of logs + data + recent protocols. No hardware tests."""
        gui_logger.button('ZIP_LOGS')
        from ui.progress_popup import CustomPopup
        from modules.tech_support_report import TechSupportReport
        import threading

        self._zip_logs_popup = CustomPopup(
            title='Zipping Logs...',
            auto_dismiss=False,
        )
        self._zip_logs_popup.open()

        def run():
            try:
                report = TechSupportReport(scope=_app_ctx.ctx.lumaview.scope)

                def progress(pct, msg):
                    Clock.schedule_once(lambda dt: self._update_zip_logs_progress(pct, msg), 0)

                path = report.generate_logs_only(callback=progress)
                Clock.schedule_once(lambda dt: self._zip_logs_done(path), 0)
            except Exception as e:
                logger.error(f'Zip-logs failed: {e}', exc_info=True)
                Clock.schedule_once(lambda dt: self._zip_logs_done(None), 0)

        threading.Thread(target=run, daemon=True).start()

    def _update_zip_logs_progress(self, pct, msg):
        if hasattr(self, '_zip_logs_popup') and self._zip_logs_popup:
            self._zip_logs_popup.progress = pct
            self._zip_logs_popup.text = msg

    def _zip_logs_done(self, zip_path):
        if hasattr(self, '_zip_logs_popup') and self._zip_logs_popup:
            self._zip_logs_popup.dismiss()
            self._zip_logs_popup = None

        from ui.notification_popup import show_notification_popup

        if zip_path:
            show_notification_popup(
                title='Logs zipped',
                message=(
                    f'Saved to Desktop:\n{zip_path.name}\n\n'
                    f'Email this file to:\n'
                    f'techsupport@etaluma.com'
                ),
            )
        else:
            show_notification_popup(
                title='Zip failed',
                message=(
                    'Could not create the logs zip.\n'
                    'Check the log file for details and contact\n'
                    'techsupport@etaluma.com directly.'
                ),
            )

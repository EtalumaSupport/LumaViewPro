# Copyright Etaluma, Inc.
import copy
import datetime
import json
import logging
import math
import os
import pathlib
import threading
import time

import numpy as np

from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.binning as binning
import modules.common_utils as common_utils
from modules import gui_logger
from modules.config_helpers import DEFAULT_MAX_EXPOSURE_MS, DEFAULT_MAX_GAIN_DB
from modules.config_ui_getters import get_binning_from_ui, get_current_frame_dimensions, get_selected_labware
from modules.common_utils import CustomJSONizer
from modules.path_utils import resolve_data_file
from modules.scope_init_config import ScopeInitConfig
from modules.memory_profiler import MemoryLeakProfiler
from modules.sequential_io_executor import IOTask
from ui.ui_helpers import move_absolute_position, move_home, scope_leds_off
from modules.zstack_config import ZStackConfig

logger = logging.getLogger('LVP.ui.microscope_settings')


class MicroscopeSettings(BoxLayout):

    def __init__(self, **kwargs):
        super(MicroscopeSettings, self).__init__(**kwargs)
        logger.debug('[LVP Main  ] MicroscopeSettings.__init__()')

        scopes_path = resolve_data_file("scopes.json")
        try:
            with open(scopes_path, "r") as read_file:
                self.scopes = json.load(read_file)
        except FileNotFoundError:
            logger.error(f'[LVP Main  ] scopes.json not found at {scopes_path}')
            raise RuntimeError(
                f"Required file scopes.json not found at {scopes_path}. "
                "Please reinstall or restore from backup."
            )
        except json.JSONDecodeError as e:
            logger.error(f'[LVP Main  ] scopes.json is corrupt: {e}')
            raise RuntimeError(
                f"scopes.json is corrupt ({e}). "
                "Please restore from backup or reinstall."
            )

        self._validate_scopes(scopes_path)

    def _validate_scopes(self, filepath):
        """Check scopes.json has required structure per scope entry."""
        if not isinstance(self.scopes, dict):
            raise ValueError(f"scopes.json at {filepath}: expected dict, got {type(self.scopes).__name__}")
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
                        f"{expected_type.__name__}, got {type(scope[field]).__name__} in {filepath}"
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

        logger.info("[LVP Main  ] Reconnecting to microscope...")

        lumaview = ctx.lumaview
        settings = ctx.settings

        lumaview.scope.disconnect()
        lumaview.scope = None
        # Reinitialize the scope object (connects motorboard, ledboard, camera)
        import modules.lumascope_api as lumascope_api
        lumaview.scope = lumascope_api.Lumascope(camera_type=settings['camera_type'], simulate=ctx.simulate_mode)
        labware_id, labware = get_selected_labware()

        # Single hardware initialization call
        scope_config = self.scopes.get(settings.get('microscope'))
        config = ScopeInitConfig.from_settings(settings, labware, scope_config=scope_config)
        lumaview.scope.initialize(config)

        ctx.sequenced_capture_executor.set_scope(lumaview.scope)
        ctx.autofocus_executor.set_scope(lumaview.scope)

        # Restart display

        ctx.scope_display.stop()
        ctx.scope_display.start()

        if not ctx.disable_homing:
            # Home everything the board has — firmware homes Z, T, X, Y
            # in the same routine; on Z-only boards the firmware homes
            # what it has and reports the missing axes.
            task = IOTask(
                move_home,
                args=('ALL',)
            )
            ctx.io_executor.put(task)


        if lumaview.scope.has_turret():
            objective_id = settings['objective_id']
            turret_position = lumaview.scope.get_turret_position_for_objective_id(objective_id=objective_id)

            if turret_position is None:
                DEFAULT_POSITION = 1
                logger.info(f"Turret position for set objective {objective_id} not in turret objectives configuration. Setting to position {DEFAULT_POSITION}")
                turret_position = DEFAULT_POSITION

            ctx.io_executor.put(IOTask(
                move_absolute_position,
                kwargs= {
                    "axis": 'T',
                    "pos": turret_position,
                    "wait_until_complete": True
                }
            ))
        ctx.image_settings.set_layer_exposure_ranges()
        layer_obj = ctx.image_settings.layer_lookup(layer='BF')
        layer_obj.apply_settings()

        scope_leds_off()

        # Refresh position display after reconnect (M22)
        ctx.motion_settings.update_xy_stage_control_gui(full_redraw=True)

        logger.info("[LVP Main  ] Reconnection complete.")


    def acceleration_pct_slider(self):
        settings = _app_ctx.ctx.settings
        scope_configs = self.scopes
        selected_scope_config = scope_configs[settings['microscope']]

        if not selected_scope_config['XYStage']:
            return

        logger.info('[LVP Main  ] MicroscopeSettings.acceleration_pct_slider()')
        acc_val = self.ids['acceleration_pct_slider'].value
        gui_logger.slider('ACCELERATION', acc_val)
        self.set_acceleration_limit(val_pct=acc_val)


    def acceleration_pct_text(self):
        logger.info('[LVP Main  ] MicroscopeSettings.acceleration_pct_text()')
        acc_min = self.ids['acceleration_pct_slider'].min
        acc_max = self.ids['acceleration_pct_slider'].max
        try:
            acc_val = int(self.ids['acceleration_pct_text'].text)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid acceleration input: {self.ids["acceleration_pct_text"].text!r}')
            return

        acc_val = int(np.clip(acc_val, acc_min, acc_max))

        self.ids['acceleration_pct_slider'].value = acc_val
        self.ids['acceleration_pct_text'].text = str(acc_val)
        self.set_acceleration_limit(val_pct=acc_val)


    def set_acceleration_limit(self, val_pct: int):
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            ctx.settings['motion']['acceleration_max_pct'] = val_pct
        ctx.lumaview.scope.set_acceleration_limit(val_pct=val_pct)


    def set_acceleration_control_visibility(self, visible):
        for acceleration_id in ('acceleration_control_box',):
            self.ids[acceleration_id].visible = visible


    def live_view_fps_slider(self):
        ctx = _app_ctx.ctx
        fps_val = int(self.ids['live_view_fps_slider'].value)
        gui_logger.slider('FPS', fps_val)
        # Values above 60 mean "uncapped" — store 0 as sentinel
        if fps_val > 60:
            fps_val = 0
        ctx.live_view_fps = fps_val
        with ctx.settings_lock:
            ctx.settings['live_view_fps'] = fps_val
        logger.info(f'[LVP Main  ] Live view FPS set to {"Max (uncapped)" if fps_val == 0 else fps_val}')

        # Restart scope display with new FPS
        scope_display = ctx.scope_display
        if scope_display is not None:
            scope_display.stop()
            scope_display.start(fps=fps_val)


    # load settings from JSON file
    def load_settings(self, filename="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.load_settings()')
        ctx = _app_ctx.ctx

        lumaview = ctx.lumaview
        settings = ctx.settings

        try:
            # Settings are imported at the very beginning of file

            if settings['profiling']['enabled']:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                profiling_save_path = os.path.join(ctx.source_path, f'./logs/profiling')
                MemoryLeakProfiler.start(root_log_dir=profiling_save_path)
                logger.info('[LVP Main  ] Memory Profiler started.')

            if 'autogain' not in settings['protocol']:
                settings['protocol']['autogain'] = {
                    'max_duration_seconds': 1.0,
                    'target_brightness': 0.3,
                    'min_gain': 0.0,
                    'max_gain': 20.0,
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
                logger.warning(f"[LVP Main  ] Unable to find/create live image folder at {settings['live_folder']}: {e}")
                try:
                    from lvp_logger import lvp_appdata
                    live_folder = pathlib.Path(lvp_appdata) / 'capture'
                except Exception:
                    live_folder = pathlib.Path.home() / 'Documents' / 'LumaViewPro' / 'capture'
                live_folder = live_folder.resolve()
                live_folder.mkdir(exist_ok=True, parents=True)
                logger.info(f"[LVP Main  ] Defaulting live image folder to {str(live_folder)}")

            settings['live_folder'] = str(live_folder)

            # update GUI values from JSON data:

            # Scope auto-detection
            detected_model = lumaview.scope.get_microscope_model()
            if detected_model in self.scopes.keys():
                logger.info(f'[LVP Main  ] Auto-detected scope as {detected_model}')
                self.ids['scope_spinner'].text = detected_model
            else:
                logger.info(f'[LVP Main  ] Using scope selection from {filename}')
                self.ids['scope_spinner'].text = settings['microscope']

            if settings['use_full_pixel_depth']:
                self.ids['enable_full_pixel_depth_btn'].state = 'down'
            else:
                self.ids['enable_full_pixel_depth_btn'].state = 'normal'
            self.update_full_pixel_depth_state()

            if settings.get('false_color_16bit', False):
                self.ids['false_color_16bit_btn'].state = 'down'
            else:
                self.ids['false_color_16bit_btn'].state = 'normal'

            if 'separate_folder_per_channel' in settings:
                if settings['separate_folder_per_channel']:
                    self.ids['separate_folder_per_channel_id'].state = 'down'
                else:
                    self.ids['separate_folder_per_channel_id'].state = 'normal'
            self.update_separate_folders_per_channel()

            self.ids['live_image_output_format_spinner'].text = settings['image_output_format']['live']
            self.select_live_image_output_format()

            self.ids['sequenced_image_output_format_spinner'].text = settings['image_output_format']['sequenced']
            self.select_sequenced_image_output_format()

            # camera_max_exposure returns None when no camera is connected;
            # fall back to the default slider upper bound. See #616.
            max_exposure = lumaview.scope.camera_max_exposure or DEFAULT_MAX_EXPOSURE_MS

            ctx.max_exposure = max_exposure

            # Parallel treatment for gain — see #gain-slider-clamp note.
            # Pre-fix, the gain slider was hardcoded 0-48 dB in the kv,
            # which let users overdrive LS620 past its usable range (the
            # image went black at high gain). Pulling the cap from the
            # camera profile keeps the slider honest per-camera.
            max_gain = lumaview.scope.camera_max_gain or DEFAULT_MAX_GAIN_DB
            ctx.max_gain = max_gain

            if not settings['video_as_frames']:
                self.ids['video_recording_format_spinner'].text = 'mp4'
            else:
                self.ids['video_recording_format_spinner'].text = 'Frames'

            self.select_video_recording_format()

            if "live_view_fps" in settings:
                ctx.live_view_fps = settings['live_view_fps']
            else:
                ctx.live_view_fps = 30

            fps_label = "Max (uncapped)" if ctx.live_view_fps == 0 else str(ctx.live_view_fps)
            logger.info(f"[LVP Main  ] Live view FPS set to {fps_label}")
            # fps=0 (uncapped) maps to slider position 65 ("Max")
            self.ids['live_view_fps_slider'].value = 65 if ctx.live_view_fps == 0 else ctx.live_view_fps

            acceleration_limit = settings['motion']['acceleration_max_pct']
            self.ids['acceleration_pct_slider'].value = acceleration_limit
            self.ids['acceleration_pct_text'].text = str(acceleration_limit)

            # Set Frame Size UI
            binning_size_str = settings['binning']['size']
            binning_size = binning.binning_size_str_to_int(text=binning_size_str)

            self.ids['frame_width_id'].text = str(settings['frame']['width'] * binning_size)
            self.ids['frame_height_id'].text = str(settings['frame']['height'] * binning_size)

            # Pixel Binning — UI recalculation only, scope.set_binning_size()
            # handled by scope.initialize() below
            self.ids['binning_spinner'].text = binning_size_str
            self.select_binning_size()

            objective_id = settings['objective_id']

            # Mutate turret config keys from str to int for cleaner handling
            settings['turret_objectives'] = {int(k):v for k,v in settings['turret_objectives'].items()}

            if lumaview.scope.has_turret():
                turret_objectives = list(settings["turret_objectives"].values())
                assigned = [obj for obj in turret_objectives if obj is not None]
                if not assigned:
                    from modules.notification_center import notifications
                    notifications.warning("Turret", "No Turret Objectives Assigned",
                        "Turret positions have no objectives assigned. "
                        "Please assign objectives in Vertical Control > Turret before running protocols.")
                elif objective_id not in assigned:
                    logger.warning(f"Startup objective {objective_id} not found in turret objectives ({turret_objectives}).")

            self.ids['objective_spinner'].text = objective_id

            vertical_control_id = ctx.motion_settings.ids['verticalcontrol_id']
            v_control_objective_spinner = vertical_control_id.ids['objective_spinner2']
            v_control_objective_spinner.text = objective_id

            objective_helper = ctx.objective_helper
            objective = objective_helper.get_objective_info(objective_id=objective_id)
            self.ids['magnification_id'].text = f"{objective['magnification']}"

            # Load previous turret position objectives
            for turret_pos, objective_id in settings["turret_objectives"].items():
                if objective_id is None:
                    button_text = f"{turret_pos}"
                else:
                    magnification = objective_helper.get_objective_info(objective_id=objective_id)['magnification']
                    button_text = f"{magnification}x"

                vertical_control_id.ids[f"turret_pos_{turret_pos}_btn"].text = button_text

            if settings['scale_bar']['enabled']:
                self.ids['enable_scale_bar_btn'].state = 'down'
            else:
                self.ids['enable_scale_bar_btn'].state = 'normal'

            # Single hardware initialization call — replaces scattered
            # scope.set_frame_size / set_binning_size / set_stage_offset /
            # set_turret_config / set_objective / set_scale_bar / set_acceleration_limit
            labware_id, labware = get_selected_labware()
            scope_config = self.scopes.get(settings.get('microscope'))
            config = ScopeInitConfig.from_settings(settings, labware, scope_config=scope_config)
            lumaview.scope.initialize(config)

            protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
            protocol_settings.ids['capture_period'].text = str(settings['protocol']['period'])
            protocol_settings.ids['capture_dur'].text = str(settings['protocol']['duration'])
            protocol_settings.ids['labware_spinner'].text = settings['protocol']['labware']
            protocol_settings.select_labware()

            zstack_settings = ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id']
            zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']
            zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
            zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])

            z_reference = common_utils.convert_zstack_reference_position_setting_to_config(text_label=settings['zstack']['position'])

            zstack_config = ZStackConfig(
                range=settings['zstack']['range'],
                step_size=settings['zstack']['step_size'],
                current_z_reference=z_reference,
                current_z_value=None
            )

            zstack_settings.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

            if "show_tooltips" in settings:
                if settings["show_tooltips"]:
                    self.ids['show_tooltips_btn'].state = 'down'
                    ctx.show_tooltips = True
                else:
                    self.ids['show_tooltips_btn'].state = 'normal'
                    ctx.show_tooltips = False


            if "protocol_led_on" in settings:
                if settings["protocol_led_on"]:
                    self.ids['protocol_led_on_btn'].state = 'down'
                else:
                    self.ids['protocol_led_on_btn'].state = 'normal'
            else:
                self.ids['protocol_led_on_btn'].state = 'normal'
                settings["protocol_led_on"] = False

            if "stimulation_enabled" in settings:
                if settings["stimulation_enabled"]:
                    self.ids['stimulation_settings_btn'].state = 'down'
                else:
                    self.ids['stimulation_settings_btn'].state = 'normal'
                    # Apply the disabled state to all layers
                    self.update_stimulation_settings()
            else:
                self.ids['stimulation_settings_btn'].state = 'normal'
                settings["stimulation_enabled"] = False

            # Protocol accordions are permanently disabled (no longer a setting)
            settings.pop("disable_protocol_accordions", None)

            for layer in common_utils.get_layers():

                layer_obj = ctx.image_settings.layer_lookup(layer=layer)

                # Set initializing flag to prevent apply_settings during load
                layer_obj._initializing = True

                if (layer in common_utils.get_fluorescence_layers()):
                    layer_obj.ids['composite_threshold_slider'].value = settings[layer]['composite_brightness_threshold']

                if 'ill' in settings[layer]:
                    layer_obj.ids['ill_slider'].value = settings[layer]['ill']

                layer_obj.ids['gain_slider'].max = max_gain

                if settings[layer]['gain'] <= max_gain:
                    layer_obj.ids['gain_slider'].value = settings[layer]['gain']
                else:
                    layer_obj.ids['gain_slider'].value = max_gain
                    settings[layer]['gain'] = max_gain

                layer_obj.ids['exp_slider'].max = max_exposure

                if settings[layer]['exp'] <= max_exposure:
                    layer_obj.ids['exp_slider'].value = settings[layer]['exp']
                else:
                    layer_obj.ids['exp_slider'].value = max_exposure
                    settings[layer]['exp'] = max_exposure

                layer_obj.ids['false_color'].active = settings[layer]['false_color']

                if 'sum' in settings[layer]:
                    layer_obj.ids['sum_slider'].value = settings[layer]['sum']
                else:
                    layer_obj.ids['sum_slider'].value = 1

                if settings[layer]['acquire'] == "image":
                    layer_obj.ids['acquire_image'].active = True
                elif settings[layer]['acquire'] == "video":
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
                    layer_obj.ids['stim_ill_slider'].value = float(stim_config.get('illumination', 100))
                    layer_obj.ids['stim_freq_text'].text = str(stim_config['frequency'])
                    layer_obj.ids['stim_freq_slider'].value = float(stim_config['frequency'])
                    layer_obj.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                    layer_obj.ids['stim_pulse_width_slider'].value = float(stim_config['pulse_width'])
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

        except Exception:
            logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')

        self.set_ui_features_for_scope()

    def update_separate_folders_per_channel(self):
        settings = _app_ctx.ctx.settings

        if self.ids['separate_folder_per_channel_id'].state == 'down':
            self._seperate_folder_per_channel = True
        else:
            self._seperate_folder_per_channel = False
        gui_logger.toggle('SEPARATE_FOLDERS', self._seperate_folder_per_channel)

        settings['separate_folder_per_channel'] = self._seperate_folder_per_channel


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

    def update_full_pixel_depth_state(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings

        if self.ids['enable_full_pixel_depth_btn'].state == 'down':
            use_full_pixel_depth = True
        else:
            use_full_pixel_depth = False
        gui_logger.toggle('FULL_PIXEL_DEPTH', use_full_pixel_depth)

        ctx.scope_display.use_full_pixel_depth = use_full_pixel_depth

        # Route through camera executor to prevent race with live view grab loop
        def _set_pixel_format():
            if use_full_pixel_depth:
                if not ctx.lumaview.scope.set_pixel_format('Mono12'):
                    formats = ctx.lumaview.scope.get_supported_pixel_formats()
                    if formats:
                        ctx.lumaview.scope.set_pixel_format(formats[0])
            else:
                if not ctx.lumaview.scope.set_pixel_format('Mono8'):
                    formats = ctx.lumaview.scope.get_supported_pixel_formats()
                    if formats:
                        ctx.lumaview.scope.set_pixel_format(formats[0])
        ctx.camera_executor.put(IOTask(action=_set_pixel_format))

        settings['use_full_pixel_depth'] = use_full_pixel_depth

    def update_false_color_16bit_state(self):
        settings = _app_ctx.ctx.settings
        enabled = self.ids['false_color_16bit_btn'].state == 'down'
        settings['false_color_16bit'] = enabled

    def select_live_image_output_format(self):
        settings = _app_ctx.ctx.settings
        settings['image_output_format']['live'] = self.ids['live_image_output_format_spinner'].text


    def select_sequenced_image_output_format(self):
        settings = _app_ctx.ctx.settings
        settings['image_output_format']['sequenced'] = self.ids['sequenced_image_output_format_spinner'].text

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

        ctx.lumaview.scope.set_scale_bar(enabled=enabled)
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
        if self.ids['enable_live_image_histogram_equalization_btn'].state == 'down':
            ctx.scope_display.use_live_image_histogram_equalization = True
            ctx.live_histo_setting = True
        else:
            ctx.scope_display.use_live_image_histogram_equalization = False
            ctx.live_histo_setting = False


    def update_show_tooltips(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        if self.ids['show_tooltips_btn'].state == 'down':
            ctx.show_tooltips = True
            settings["show_tooltips"] = True
        else:
            ctx.show_tooltips = False
            settings["show_tooltips"] = False

    def update_protocol_led_on(self):
        settings = _app_ctx.ctx.settings
        if self.ids['protocol_led_on_btn'].state == 'down':
            settings["protocol_led_on"] = True
        else:
            settings["protocol_led_on"] = False

    def update_stimulation_settings(self):
        """Toggle stimulation features globally across all channels."""
        settings = _app_ctx.ctx.settings
        stimulation_enabled = self.ids['stimulation_settings_btn'].state == 'down'
        settings["stimulation_enabled"] = stimulation_enabled

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
    def save_settings(self, file="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.save_settings()')
        ctx = _app_ctx.ctx
        settings = ctx.settings

        if isinstance(file, str) and (file[-5:].lower() != '.json'):
                file = file+'.json'

        t0 = time.monotonic()
        with ctx.settings_lock:
            settings_snapshot = copy.deepcopy(settings)
        # Resolve relative paths against source_path instead of relying on CWD
        if not os.path.isabs(file):
            file = os.path.join(ctx.source_path, file)
        with open(file, "w") as write_file:
            json.dump(settings_snapshot, write_file, indent = 4, cls=CustomJSONizer)
        dt = time.monotonic() - t0
        if dt > 0.1:
            logger.warning(f'[LVP Main  ] save_settings took {dt*1000:.0f}ms')


    def load_binning_sizes(self):
        spinner = self.ids['binning_spinner']
        # Use Lumascope API to get available binning sizes
        try:
            sizes = _app_ctx.ctx.lumaview.scope.get_available_binning_sizes()
        except Exception:
            logger.warning('[LVP Main  ] Could not read camera binning sizes, using defaults.')
            sizes = [1, 2, 4]
        spinner.values = [f'{s}x{s}' for s in sizes]


    def select_binning_size(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings

        lumaview = ctx.lumaview
        orig_binning_size = lumaview.scope.get_binning_size()
        orig_frame_size = get_current_frame_dimensions()

        new_binning_size_str = self.ids['binning_spinner'].text

        new_binning_size = binning.binning_size_str_to_int(new_binning_size_str)
        settings['binning']['size'] = new_binning_size_str
        ratio = new_binning_size / orig_binning_size
        new_frame_size = {
            'width': math.floor(orig_frame_size['width'] / ratio),
            'height': math.floor(orig_frame_size['height'] / ratio),
        }
        self.ids['frame_width_id'].text = str(new_frame_size['width'])
        self.ids['frame_height_id'].text = str(new_frame_size['height'])

        # During app init, scope.initialize() handles all hardware calls
        if ctx.initializing:
            return

        # Route through camera executor to prevent race with live view grab loop
        ctx.camera_executor.put(IOTask(
            action=lumaview.scope.set_binning_size,
            kwargs={'size': new_binning_size}
        ))
        self.frame_size()


    def load_scopes(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_scopes()')
        spinner = self.ids['scope_spinner']
        spinner.values = list(self.scopes.keys())

    def select_scope(self):
        gui_logger.select('SCOPE', self.ids['scope_spinner'].text)
        logger.info('[LVP Main  ] MicroscopeSettings.select_scope()')
        ctx = _app_ctx.ctx
        settings = ctx.settings

        spinner = self.ids['scope_spinner']
        settings['microscope'] = spinner.text

        self.set_ui_features_for_scope()
        ctx.stage.full_redraw()


    def set_ui_features_for_scope(self) -> None:
        ctx = _app_ctx.ctx
        settings = ctx.settings

        microscope_settings = ctx.motion_settings.ids['microscope_settings_id']
        scope_configs = microscope_settings.scopes
        selected_scope_config = scope_configs[settings['microscope']]

        microscope_settings.set_acceleration_control_visibility(visible=selected_scope_config['XYStage'])

        motion_settings = ctx.motion_settings
        motion_settings.set_turret_control_visibility(visible=selected_scope_config['Turret'])
        motion_settings.set_xystage_control_visibility(visible=selected_scope_config['XYStage'])
        motion_settings.set_tiling_control_visibility(visible=selected_scope_config['XYStage'])

        image_settings = ctx.image_settings
        layers_config = selected_scope_config['Layers']
        image_settings.set_df_layer_control_visibility(visible=layers_config['Darkfield'])
        image_settings.set_lumi_layer_control_visibility(visible=layers_config['Lumi'])
        image_settings.set_fluoresence_layer_controls_visibility(visible=layers_config['Fluorescence'])

        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        protocol_settings.set_labware_selection_visibility(visible=selected_scope_config['XYStage'])
        protocol_settings.set_show_protocol_step_locations_visibility(visible=selected_scope_config['XYStage'])

        ctx.motion_settings.ids['post_processing_id'].ids['stitch_controls_id'].set_button_enabled_state(state=selected_scope_config['XYStage'])

        if selected_scope_config['XYStage'] is False:
            ctx.stage.remove_parent()
            protocol_settings.select_labware(labware="Center Plate")
            ctx.motion_settings.ids['post_processing_id'].hide_stitch()

        ctx.stage.set_motion_capability(enabled=selected_scope_config['XYStage'])


    def load_objectives(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_objectives()')
        spinner = self.ids['objective_spinner']
        objective_helper = _app_ctx.ctx.objective_helper
        spinner.values = objective_helper.get_objectives_list()


    def select_objective(self):
        try:
            objective_id = self.ids['objective_spinner'].text
            gui_logger.select('OBJECTIVE', objective_id)
            logger.info('[LVP Main  ] MicroscopeSettings.select_objective()')
            ctx = _app_ctx.ctx

            lumaview = ctx.lumaview
            settings = ctx.settings
            objective_helper = ctx.objective_helper

            # If turret is present, objective must be assigned to a turret position (#606)
            if lumaview.scope.has_turret():
                turret_objectives = list(settings.get("turret_objectives", {}).values())
                assigned = [obj for obj in turret_objectives if obj is not None]
                if assigned and objective_id not in assigned:
                    from modules.notification_center import notifications
                    notifications.warning("Objective", "Objective Not in Turret",
                        f"'{objective_id}' is not assigned to any turret position. "
                        f"Assign it in Vertical Control > Turret before using.")

            objective = objective_helper.get_objective_info(objective_id=objective_id)
            settings['objective_id'] = objective_id
            microscope_settings_id = ctx.motion_settings.ids['microscope_settings_id']
            microscope_settings_id.ids['magnification_id'].text = f"{objective['magnification']}"

            # Update selected to be consistent with other selector
            vc_objective_spinner = ctx.motion_settings.ids['verticalcontrol_id'].ids['objective_spinner2']
            vc_objective_spinner.text = objective_id

            if lumaview.scope.has_turret():
                lumaview.scope.set_turret_config(turret_config=settings["turret_objectives"])

            lumaview.scope.set_objective(objective_id=objective_id)

            fov_size = common_utils.get_field_of_view(
                focal_length=objective['focal_length'],
                frame_size=settings['frame'],
                binning_size=get_binning_from_ui(),
            )
            self.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
            self.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))
        except Exception as e:
            logger.error(f'[UI] select_objective failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Error", message=str(e))

    def frame_size(self):
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        ctx = _app_ctx.ctx

        lumaview = ctx.lumaview
        settings = ctx.settings
        objective_helper = ctx.objective_helper

        if not lumaview.scope.camera_is_connected():
            return

        try:
            current_frame_size = get_current_frame_dimensions()
        except ValueError:
            current_frame_size = {
                'width': settings['frame']['width'],
                'height': settings['frame']['height'],
            }

        width = int(min(current_frame_size['width'], lumaview.scope.get_max_width()))
        height = int(min(current_frame_size['height'], lumaview.scope.get_max_height()))

        try:
            min_frame_size = lumaview.scope.camera_min_frame_size
            width = max(width, min_frame_size['width'])
            height = max(height, min_frame_size['height'])

            max_frame_size = lumaview.scope.camera_max_frame_size
            width = min(width, max_frame_size['width'])
            height = min(height, max_frame_size['height'])
        except Exception:
            logger.warning('[LVP Main  ] Could not clamp frame size to camera limits.')

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame_width_id'].text = str(width)
        self.ids['frame_height_id'].text = str(height)

        objective_id = settings['objective_id']
        objective = objective_helper.get_objective_info(objective_id=objective_id)

        fov_size = common_utils.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=settings['frame'],
            binning_size=get_binning_from_ui(),
        )
        self.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
        self.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))

        # Route through camera executor to prevent race with live view grab loop
        ctx.camera_executor.put(IOTask(
            action=lumaview.scope.set_frame_size,
            args=(width, height)
        ))

    def generate_support_report(self):
        """Show confirmation dialog, then generate a tech support report."""
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
                    Clock.schedule_once(
                        lambda dt: self._update_report_progress(pct, msg), 0)

                path = report.generate(callback=progress, include_bandwidth_test=False)
                Clock.schedule_once(lambda dt: self._report_done(path), 0)
            except Exception as e:
                logger.error(f"Support report failed: {e}", exc_info=True)
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

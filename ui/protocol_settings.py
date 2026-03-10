# Copyright Etaluma, Inc.
import copy
import json
import logging
import os
import pathlib
import threading
import time
import typing

import pandas as pd

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import BooleanProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from lumaviewpro import CompositeCapture

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from modules.sequential_io_executor import IOTask
from modules.tiling_config import TilingConfig
from modules.timedelta_formatter import strfdelta
from ui.progress_popup import show_popup

logger = logging.getLogger('LVP.ui.protocol_settings')


class ProtocolSettings(CompositeCapture):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):

        super(ProtocolSettings, self).__init__(**kwargs)
        logger.info('[LVP Main  ] ProtocolSettings.__init__()')

        # Create trigger for debounced UI updates to prevent memory leaks
        self._update_step_ui_trigger = Clock.create_trigger(self._do_update_step_ui, 0.05)

        # Thread-safe flag to prevent duplicate file completion handlers
        self._scan_files_completed_event = threading.Event()

        import lumaviewpro
        source_path = lumaviewpro.source_path

        os.chdir(source_path)
        try:
            with open('./data/labware.json', "r") as read_file:
                self.labware = json.load(read_file)
        except Exception:
            logger.exception("[LVP Main  ] Error reading labware definition file 'data/labware.json'")
            if not os.path.isdir('./data'):
                raise FileNotFoundError("Couldn't find 'data' directory.")
            else:
                raise

        self.curr_step = -1


        self.tiling_config = TilingConfig(
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )

        self.tiling_min = {
            "x": 120000,
            "y": 80000
        }
        self.tiling_max = {
            "x": 0,
            "y": 0
        }

        self.tiling_count = self.tiling_config.get_mxn_size(self.tiling_config.default_config())

        self._protocol = None

        self.exposures = 1  # 1 indexed
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
        new_step_num = str(self.curr_step+1)
        if self.ids['step_number_input'].text != new_step_num:
            self.ids['step_number_input'].text = new_step_num

        new_total = str(num_steps)
        if self.ids['step_total_input'].text != new_total:
            self.ids['step_total_input'].text = new_total

        self.generate_step_name_input()


    def _init_ui(self, dt=0):
        import lumaviewpro
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx
        source_path = lumaviewpro.source_path

        self.ids['tiling_size_spinner'].values = self.tiling_config.available_configs()
        self.ids['tiling_size_spinner'].text = self.tiling_config.default_config()

        try:
            filepath = settings['protocol']['filepath']
            protocol_success = ctx.motion_settings.ids['protocol_settings_id'].load_protocol(filepath=filepath)

            if not protocol_success:
                logger.warning('[LVP Main  ] Unable to load protocol at startup')
                # If protocol file is missing or incomplete, file name and path are cleared from memory.
                filepath=''
                settings['protocol']['filepath']=''

                protocol_config = lumaviewpro.get_sequenced_capture_config_from_ui()
                self._protocol = Protocol.create_empty(
                    config=protocol_config,
                    tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
                )

        except Exception:
            logger.exception('[LVP Main  ] Error loading protocol at startup')
            filepath=''
            settings['protocol']['filepath']=''
            protocol_config = lumaviewpro.get_sequenced_capture_config_from_ui()
            self._protocol = Protocol.create_empty(
                    config=protocol_config,
                    tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
                )


        self.select_labware()
        self.update_step_ui()


    # Update Protocol Period
    def update_period(self):
        import lumaviewpro
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] ProtocolSettings.update_period()')
        try:
            settings['protocol']['period'] = float(self.ids['capture_period'].text)
        except Exception:
            logger.exception('[LVP Main  ] Update Period is not an acceptable value')

        time_params = lumaviewpro.get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    # Update Protocol Duration
    def update_duration(self):
        import lumaviewpro
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] ProtocolSettings.update_duration()')
        try:
            settings['protocol']['duration'] = float(self.ids['capture_dur'].text)
        except Exception:
            logger.warning('[LVP Main  ] Update Duration is not an acceptable value')

        time_params = lumaviewpro.get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )


    def step_name_validation(self, text: str):
        cleaned_str = Protocol.sanitize_step_name(input=text)

        if hasattr(self, '_protocol') and (self._protocol is not None) and (self._protocol.num_steps() > 0 and self.curr_step >= 0):
            self._protocol.modify_name(
                step_idx=self.curr_step,
                step_name=cleaned_str
            )
            self.ids['step_name_input'].text = cleaned_str
        else:
            self.ids['step_name_input'].text = ""

    def update_capture_root(self, text: str):
        # Sanitize and store capture root on protocol to avoid invalid path chars
        sanitized = Protocol.sanitize_step_name(input=text)
        self.ids['capture_root'].text = sanitized
        if hasattr(self, '_protocol') and (self._protocol is not None):
            self._protocol.modify_capture_root(capture_root=sanitized)


    # Labware Selection
    def select_labware(self, labware: str = None):
        import lumaviewpro
        settings = _app_ctx.ctx.settings
        wellplate_loader = _app_ctx.ctx.wellplate_loader

        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        if labware is None:
            spinner = self.ids['labware_spinner']
            spinner.values = wellplate_loader.get_plate_list()
            settings['protocol']['labware'] = spinner.text
        else:
            center_plate_str = 'Center Plate'
            spinner = self.ids['labware_spinner']
            spinner.values = list(center_plate_str,)
            spinner.text = center_plate_str
            settings['protocol']['labware'] = labware

        labware_id, labware = lumaviewpro.get_selected_labware()

        if labware is None:
            logger.error(f"Labware could not be loaded")
            return

        lumaviewpro.lumaview.scope.set_labware(labware=labware)

        if self._protocol is not None:
            self._protocol.modify_labware(labware_id=labware_id)

        lumaviewpro.stage.full_redraw()


    def set_labware_selection_visibility(self, visible):
        labware_spinner = self.ids['labware_spinner']
        labware_spinner.visible = visible
        labware_spinner.size_hint_y = None if visible else 0
        labware_spinner.height = '30dp' if visible else 0
        labware_spinner.opacity = 1 if visible else 0
        labware_spinner.disabled = not visible

        if not visible:
            labware_spinner.text = 'Center Plate'


    def set_show_protocol_step_locations_visibility(self, visible: bool) -> None:
        if visible:
            self.ids['show_step_locations_id'].disabled = False
            self.ids['show_step_locations_id'].opacity = 1
            self.ids['show_step_locations_label_id'].opacity = 1
        else:
            self.ids['show_step_locations_id'].disabled = True
            self.ids['show_step_locations_id'].opacity = 0
            self.ids['show_step_locations_label_id'].opacity = 0


    def apply_tiling(self):
        import lumaviewpro
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] Apply tiling to protocol')

        axes_config = lumaviewpro.lumaview.scope.get_axes_config()
        _, labware = lumaviewpro.get_selected_labware()
        stage_offset = settings['stage_offset']

        tile_status = self._protocol.apply_tiling(
            tiling=self.ids['tiling_size_spinner'].text,
            frame_dimensions=lumaviewpro.get_current_frame_dimensions(),
            binning_size=lumaviewpro.get_binning_from_ui(),
            curr_step_idx=self.curr_step,
            axes_config=axes_config,
            labware=labware,
            stage_offset=stage_offset
        )

        tiles_skipped = tile_status['tiles_skipped']

        if tiles_skipped > 0:
            error_msg = f"Tiling application skipped {tiles_skipped} new tiles due to bounds outside of labware."
            from ui.notification_popup import show_notification_popup
            Clock.schedule_once(lambda dt: show_notification_popup(title="Protocol Tiling Warning", message=error_msg), 0)

        self._protocol.optimize_step_ordering()
        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step(protocol=False)


    def apply_zstacking(self):
        import lumaviewpro

        logger.info('[LVP Main  ] Apply Z-Stacking to protocol')
        zstack_params = lumaviewpro.get_zstack_params()

        if zstack_params['range'] < 0 or zstack_params['step_size'] < 0:
            error_msg = f"Z-Stacking parameters are not valid. Please ensure range and step size are positive values."
            logger.warning(error_msg)
            from ui.notification_popup import show_notification_popup
            Clock.schedule_once(lambda dt: show_notification_popup(title="Z-Stacking Warning", message=error_msg), 0)
            return
        elif zstack_params['range'] == 0 or zstack_params['step_size'] == 0:
            logger.warning(f"Z-stacking parameters are zero. No changes applied.")
            return

        self._protocol.apply_zstacking(
            zstack_params=zstack_params,
        )

        self._protocol.optimize_step_ordering()
        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step(protocol=False)


    def generate_step_name_input(self):
        num_steps = self._protocol.num_steps()
        if num_steps > 0:
            step = self.get_curr_step()
            if step['Name'] == '':
                new_text = step["Name"]
                new_hint = self.get_default_name_for_curr_step()
            elif step['Custom Step'] and step["Name"].startswith("custom"):
                # For custom added steps where the user did not change the default name (i.e. custom####)
                new_text = ""
                new_hint = self.get_default_name_for_curr_step()
            else:
                new_text = step["Name"]
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
        import lumaviewpro

        logger.info('[LVP Main  ] ProtocolSettings.new_protocol()')

        # Check if file writing is in progress
        file_io_executor = lumaviewpro.file_io_executor
        if file_io_executor.is_protocol_queue_active():
            logger.warning('[LVP Main  ] Cannot create new protocol - files still being written')
            from ui.notification_popup import show_notification_popup
            show_notification_popup(
                title="Operation Blocked",
                message="Please wait - files are still being written to disk from the previous scan."
            )
            return

        config = lumaviewpro.get_sequenced_capture_config_from_ui()
        source_path = lumaviewpro.source_path
        protocol = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )

        protocol_executor = _app_ctx.ctx.protocol_executor
        protocol_executor.put(IOTask(
            action=self.new_protocol_ex,
            args=(protocol),
            callback=self.update_step_ui,

        ))

    def new_protocol_ex(self, protocol):
        import lumaviewpro
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        if (lumaviewpro.lumaview.scope.has_turret()) and (not lumaviewpro.lumaview.scope.is_current_turret_position_objective_set()):
            error_msg = f"Cannot create new protocol. Please set objective for current turret position."
            logger.error(error_msg)

            from ui.notification_popup import show_notification_popup
            Clock.schedule_once(lambda dt: show_notification_popup(title="Protocol Creation Error", message=error_msg), 0)
            return

        if not self._validate_objectives_in_protocol(protocol_df=protocol.steps()):
            error_msg = f"Cannot create new protocol. Not all objectives are in turret config."
            logger.error(error_msg)
            Clock.schedule_once(lambda dt:
                Popup(
                    title="Protocol Creation Error",
                    content=Label(text=error_msg),
                    size_hint=(0.85,0.85),
                ), 0)

            return

        self._protocol = protocol

        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())
        def temp():
            self.ids['protocol_filename'].text = ''
            self.ids['capture_root'].text = ''

        settings['protocol']['filepath'] = ''
        Clock.schedule_once(lambda dt: temp(), 0)
        self.curr_step = 0
        self.go_to_step(protocol=False)


    def _validate_labware(self, labware: str):
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        scope_configs = ctx.motion_settings.ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]

        # If XY motion is available, any type of labware is acceptable
        if selected_scope_config['XYStage'] is True:
            return True, labware

        # If XY motion is not available, only Center Plate
        if labware == "Center Plate":
            return True, labware
        else:
            return False, "Center Plate"


    @show_popup
    def _show_popup_message(self, popup, title, message, delay_sec):
        popup.title = title
        popup.text = message
        time.sleep(delay_sec)
        self.done = True


    def _validate_objectives_in_protocol(self, protocol_df: pd.DataFrame) -> bool:
        import lumaviewpro

        # Validation for objectives with multi-objective protocol
        protocol_objective_ids = set(protocol_df['Objective'].to_list())

        # For single objective protocols, don't perform any objective validation (legacy)
        if len(protocol_objective_ids) == 1:
            return True

        # Otherwise, check all the objectives used in the protocol and confirm
        # they are all part of the current turret config
        turret_objective_ids = set(lumaviewpro.lumaview.scope.get_turret_config().values())
        return protocol_objective_ids.issubset(turret_objective_ids)

    # Load Protocol from File
    def load_protocol(self, filepath="./data/new_default_protocol.tsv"):
        import lumaviewpro
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx
        source_path = lumaviewpro.source_path

        logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

        if not pathlib.Path(filepath).exists():
            raise FileNotFoundError(f"Protocol not found at {filepath}")

        try:
            protocol = Protocol.from_file(
                file_path=filepath,
                tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            )
        except IOError:
            # Guard to prevent LVP startup notification popup
            return False

        except Exception as e:
            error_title = "Protocol Loading Error"
            error_msg = f"Cannot load protocol from file: {e}"
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title=error_title, message=error_msg)
            return False

        if protocol is False:
            error_title = "Empty Protocol Steps"
            error_msg = f"Warning: Selected protocol had no steps. Empty protocol loaded."
            protocol_config = lumaviewpro.get_sequenced_capture_config_from_ui()

            protocol = Protocol.create_empty(
                config=protocol_config,
                tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            )

        if protocol is None:
            logger.error(f"Unable to load protocol at {filepath}")
            return

        if not self._validate_objectives_in_protocol(protocol_df=protocol.steps()):
            error_msg = f"Cannot load protocol. Not all objectives are in turret config."
            logger.error(error_msg)
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Protocol Loading Error", message=error_msg)
            return False

        self._protocol = protocol

        settings['protocol']['filepath'] = filepath
        self.ids['protocol_filename'].text = os.path.basename(filepath)

        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            self.curr_step = -1
        else:
            self.curr_step = 0

        period = round(self._protocol.period().total_seconds() / 60, 2)
        duration = round(self._protocol.duration().total_seconds() / 3600, 2)
        labware = self._protocol.labware()

        scope_configs = ctx.motion_settings.ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]

        # If the scope has no XY stage, then don't allow the protocol to modify the labware
        if not selected_scope_config['XYStage']:
            labware = "Center Plate"

        self.ids['capture_period'].text = str(period)
        self.ids['capture_dur'].text = str(duration)

        settings['protocol']['period'] = period
        settings['protocol']['duration'] = duration
        settings['protocol']['labware'] = labware
        self.ids['labware_spinner'].text = settings['protocol']['labware']
        self.ids['capture_root'].text = self._protocol.capture_root()

        # Set all layers to acquire as set in loaded protocol
        for layer in common_utils.get_layers():
            settings[layer]['acquire'] = None
            if "stim_config" in settings[layer]:
                if settings[layer]['stim_config'] is not None:
                    settings[layer]['stim_config']['enabled'] = False

        lumaviewpro.reset_acquire_ui()
        lumaviewpro.reset_stim_ui()

        # Make steps available for drawing locations
        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())

        self.update_step_ui()
        # Skip go_to_step during startup - will be handled by on_start if initializing,
        # or called explicitly by user action if loading protocol later
        if not lumaviewpro._app_initializing:
            self.go_to_step(protocol=False)

        return True


    def get_default_name_for_curr_step(self):
        import lumaviewpro

        step = self.get_curr_step()

        if lumaviewpro.lumaview.scope.has_turret():
            objective_id = step['Objective']
            objective_info = lumaviewpro.objective_helper.get_objective_info(objective_id=objective_id)
            if objective_info is None:
                objective_short_name = objective_id
            else:
                objective_short_name = objective_info['short_name']
        else:
            objective_short_name = None

        if step['Well'] == "":
            custom_name_prefix = step['Name']
        else:
            custom_name_prefix = None

        return common_utils.generate_default_step_name(
            well_label=step["Well"],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            objective_short_name=objective_short_name,
            tile_label=step['Tile'],
            custom_name_prefix=custom_name_prefix,
        )


    # Save Protocol to File
    def save_protocol(self, filepath='', update_protocol_filepath: bool = True):
        import lumaviewpro
        settings = _app_ctx.ctx.settings

        logger.info('[LVP Main  ] ProtocolSettings.save_protocol()')

        time_params = lumaviewpro.get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration']
        )

        if (isinstance(filepath, str)) and len(filepath)==0:
            # If there is no current file path, "save" button will act as "save as"
            if len(settings['protocol']['filepath']) == 0:
                from ui.file_dialogs import FileSaveBTN
                FileSaveBTN_instance=FileSaveBTN()
                FileSaveBTN_instance.choose('saveas_protocol')
                return
            filepath = settings['protocol']['filepath']
        else:

            if (isinstance(filepath, str)) and (filepath[-4:].lower() != '.tsv'):
                filepath = filepath+'.tsv'

            if update_protocol_filepath:
                settings['protocol']['filepath'] = filepath

        if (isinstance(filepath, str)) and (filepath[-4:].lower() != '.tsv'):
            filepath = filepath+'.tsv'

        result = self._protocol.to_file(
            file_path=filepath
        )

        if result: # Had an error saving
            from ui.notification_popup import show_notification_popup
            show_notification_popup(title="Protocol Saving Error", message=result)

        self.ids['protocol_filename'].text = os.path.basename(filepath)


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
    def handle_step_ui_input_change(
        self
    ):
        obj = self.ids['step_number_input']
        try:
            val = int(obj.text)
        except Exception:
            num_steps = self._protocol.num_steps()
            if num_steps < 1:
                val = 0
            else:
                val = 1

            obj.text = f"{val}"
            return

        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            val = 0
            obj.text = f"{val}"
        elif val < 1:
            val = 1
            obj.text = f"{val}"
        elif val > num_steps:
            val = num_steps
            obj.text = f"{val}"

        self.curr_step = val-1
        self.go_to_step(protocol=False)


    def go_to_step(
        self,
        protocol=True
    ):
        import lumaviewpro
        lumaviewpro.go_to_step(
            protocol=self._protocol,
            step_idx=self.curr_step,
            ignore_auto_gain=False,
            include_move=True,
            called_from_protocol=protocol
        )

    # Goto to Previous Step
    def prev_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            self.curr_step = -1
            self.update_step_ui()
            return

        self.curr_step = max(self.curr_step-1, 0)
        self.update_step_ui()
        self.go_to_step(protocol=False)

    # Go to Next Step
    def next_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            return

        self.curr_step = min(self.curr_step+1, num_steps-1)
        self.update_step_ui()
        self.go_to_step(protocol=False)


    # Delete Current Step of Protocol
    def delete_step(self):
        import lumaviewpro

        logger.info('[LVP Main  ] ProtocolSettings.delete_step()')

        if self._protocol.num_steps() <= 0:
            return

        self._protocol.delete_step(
            step_idx=self.curr_step
        )

        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())

        if self._protocol.num_steps() <= 0:
            self.curr_step = -1
        else:
            self.curr_step = max(self.curr_step-1, 0)

        self.update_step_ui()
        self.go_to_step(protocol=False)


    def modify_step(self):
        import lumaviewpro

        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if self._protocol.num_steps() < 1:
            return

        io_executor = _app_ctx.ctx.io_executor
        io_executor.put(IOTask(
            action=self.modify_step_ex,
            callback=self.update_step_ui
        ))


    def modify_step_ex(self):
        import lumaviewpro
        from ui.notification_popup import show_notification_popup

        active_layer, active_layer_config = lumaviewpro.get_active_layer_config()
        stim_was_active = False

        if 'stim_config' in active_layer_config:
            if active_layer_config['stim_config'] is not None:
                if active_layer_config['stim_config']['enabled']:
                    # We want to keep the same acquire channel when we are only modifying the stim config.
                    true_step_layer = self._protocol.step(idx=self.curr_step)['Color']
                    active_layer = true_step_layer
                    active_layer_config = lumaviewpro.get_layer_configs()[active_layer]
                    stim_was_active = True

        plate_position = lumaviewpro.get_current_plate_position()
        objective_id, _ = lumaviewpro.get_current_objective_info()

        #logger.error(f"CURRENT Z POSITION IN UM {plate_position['z']}")

        if (lumaviewpro.lumaview.scope.has_turret()) and (not lumaviewpro.lumaview.scope.is_current_turret_position_objective_set()):
            error_msg = f"Cannot modify protocol step. Please set objective for current turret position."
            logger.error(error_msg)
            show_notification_popup(title="Protocol Step Modification Error", message=error_msg)
            return

        step_name = self.ids['step_name_input'].text

        # If the stim layer was active and the original acquire channel remains enabled,
        # preserve the existing step name to avoid unintended renaming.
        if stim_was_active:
            original_step = self._protocol.step(idx=self.curr_step)
            original_layer = original_step['Color']
            layer_configs_all = lumaviewpro.get_layer_configs()
            if original_layer in layer_configs_all and (layer_configs_all[original_layer]['acquire'] is not None):
                step_name = original_step['Name']

        self._protocol.modify_step(
            step_idx=self.curr_step,
            step_name=step_name,
            layer=active_layer,
            layer_config=active_layer_config,
            stim_configs=lumaviewpro.get_stim_configs(),
            plate_position=plate_position,
            objective_id=objective_id,
        )

        # Validate the modified step and warn the user if there are errors
        errors = self._protocol.validate_steps()
        if errors:
            msg = '\n'.join(errors)
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Protocol Validation Warning",
                message=f"Step modified with validation issues:\n\n{msg}"
            ), 0)

        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())


    # add_step
    def insert_step(self, after_current_step: bool = True):
        import lumaviewpro

        logger.info('[LVP Main  ] ProtocolSettings.insert_step()')
        io_executor = _app_ctx.ctx.io_executor
        io_executor.put(IOTask(
            action=self.insert_step_ex,
            args=(after_current_step),
            callback=self.update_step_ui
        ))


    def insert_step_ex(self, after_current_step: bool = True):
        import lumaviewpro
        from ui.notification_popup import show_notification_popup

        plate_position = lumaviewpro.get_current_plate_position()
        objective_id, _ = lumaviewpro.get_current_objective_info()

        if (lumaviewpro.lumaview.scope.has_turret()) and (not lumaviewpro.lumaview.scope.is_current_turret_position_objective_set()):
            error_msg = f"Cannot add step to protocol. Please set objective for current turret position."
            logger.error(error_msg)
            Clock.schedule_once(lambda dt: show_notification_popup(title="Protocol Add Step Error", message=error_msg), 0)
            return

        if after_current_step:
            after_step = self.curr_step
            before_step = None
        else:
            after_step = None
            before_step = self.curr_step

        layer_configs = lumaviewpro.get_layer_configs()

        # Early return if no channels have acquire enabled (#548)
        if not any(lc['acquire'] is not None for lc in layer_configs.values()):
            return

        for layer, layer_config in layer_configs.items():
            if layer_config['acquire'] is None:
                continue

            _ = self._protocol.insert_step(
                step_name=None,
                layer=layer,
                layer_config=layer_config,
                stim_configs=lumaviewpro.get_stim_configs(),
                plate_position=plate_position,
                objective_id=objective_id,
                before_step=before_step,
                after_step=after_step,
                include_objective_in_step_name=lumaviewpro.lumaview.scope.has_turret(),
            )

            if after_current_step or (self.curr_step < 0):
                self.curr_step += 1

        # Validate after inserting and warn the user if there are errors
        errors = self._protocol.validate_steps()
        if errors:
            msg = '\n'.join(errors)
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Protocol Validation Warning",
                message=f"Step added with validation issues:\n\n{msg}"
            ), 0)

        lumaviewpro.stage.set_protocol_steps(df=self._protocol.steps())
        self.go_to_step(protocol=False)


    def update_acquire_zstack(self):
        pass


    def update_show_step_locations(self):
        import lumaviewpro
        lumaviewpro.stage.show_protocol_steps(enable=self.ids['show_step_locations_id'].active)


    def update_tiling_selection(self):
        pass


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
        import lumaviewpro
        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.clear()

        self.ids['run_autofocus_btn'].state = 'normal'
        self.ids['run_autofocus_btn'].text = 'Autofocus All Steps'
        self.ids['run_autofocus_btn'].disabled = False
        lumaviewpro.stage.set_motion_capability(True)


    def _reset_run_scan_button(self, **kwargs):
        import lumaviewpro
        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.clear()
        self.ids['run_scan_btn'].state = 'normal'
        self.ids['run_scan_btn'].text = 'Run One Scan'
        self.ids['run_scan_btn'].disabled = False
        lumaviewpro.stage.set_motion_capability(True)


    def _reset_run_protocol_button(self, **kwargs):
        import lumaviewpro
        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.clear()
        self.ids['run_protocol_btn'].state = 'normal'
        self.ids['run_protocol_btn'].text = 'Run Full Protocol'
        self.ids['run_protocol_btn'].disabled = False
        self.ids['run_protocol_btn'].background_down = 'atlas://data/images/defaulttheme/button_pressed'
        lumaviewpro.stage.set_motion_capability(True)


    def _is_protocol_valid(self) -> bool:
        if self._protocol.num_steps() == 0:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            return False

        return True


    def _autofocus_run_complete_callback(self, **kwargs):
        import lumaviewpro

        # Don't reset immediately - keep running until files complete

        # Reset completion event for this run (thread-safe)
        self._scan_files_completed_event.clear()

        # Copy the Z-heights from the autofocus scan into the protocol first
        focused_protocol = kwargs['protocol']
        self._protocol.steps()['Z'] = focused_protocol.steps()['Z']

        file_io_executor = lumaviewpro.file_io_executor

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._file_write_status_event = Clock.schedule_interval(
                self._update_autofocus_write_status,
                0.5  # Update every 500ms
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
            version = lumaviewpro.version
            Window.set_title(f"Lumaview Pro {version}   |   Writing protocol scan files to disk...")
        else:
            # No files pending - proceed with normal reset
            lumaviewpro.live_histo_reverse()
            lumaviewpro.reset_acquire_ui()
            self._reset_run_autofocus_scan_button()


    def _update_autofocus_write_status(self, dt):
        """Update UI to show file writing progress for autofocus."""
        import lumaviewpro
        file_io_executor = lumaviewpro.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_autofocus_btn'].text = f'Writing Files... ({queue_size})'
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._autofocus_files_complete()


    def _autofocus_files_complete(self, **kwargs):
        """Called when ALL files are written to disk for autofocus run."""
        import lumaviewpro

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
        lumaviewpro.live_histo_reverse()
        lumaviewpro.reset_acquire_ui()
        Clock.schedule_once(lambda dt: lumaviewpro.reset_title(), 0)


    def debug_func(self):
        import lumaviewpro
        logger.error(f"DEBUG VAL: {lumaviewpro.lumaview.scope.get_led_status()}")

    def run_autofocus_scan_from_ui(self):
        import lumaviewpro
        from ui.notification_popup import show_notification_popup

        logger.info('[LVP Main  ] ProtocolSettings.run_autofocus_scan_from_ui()')
        trigger_source = 'autofocus_scan'
        run_not_started_func = self._reset_run_autofocus_scan_button

        sequenced_capture_executor = lumaviewpro.sequenced_capture_executor
        file_io_executor = lumaviewpro.file_io_executor
        ctx = _app_ctx.ctx

        run_trigger_source = sequenced_capture_executor.run_trigger_source()

        lumaviewpro.live_histo_off()
        lumaviewpro.stage.set_motion_capability(False)

        # Only block if starting NEW autofocus scan (button is 'down'), not if aborting (button is 'normal')
        if self.ids['run_autofocus_btn'].state == 'down' and file_io_executor.is_protocol_queue_active():
            run_not_started_func()
            lumaviewpro.live_histo_reverse()
            logger.warning(f"Cannot start autofocus scan - files still being written to disk")
            show_notification_popup(
                title="Operation Blocked",
                message="Please wait - files are still being written to disk."
            )
            return

        if self.ids['run_autofocus_btn'].state == 'normal' or (sequenced_capture_executor.run_in_progress() and run_trigger_source == trigger_source):
            self._cleanup_at_end_of_protocol(autofocus_scan=True)
            return

        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            lumaviewpro.live_histo_reverse()
            logger.warning(f"Cannot start autofocus scan. Run already in progress from {run_trigger_source}")
            return

        if not self._is_protocol_valid():
            run_not_started_func()
            lumaviewpro.live_histo_reverse()
            return



        self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

        settings = _app_ctx.ctx.settings

        callbacks = {
            'move_position': lumaviewpro._handle_ui_update_for_axis,
            'update_scope_display': ctx.scope_display.update_scopedisplay,
            # Pause live UI during recording-heavy runs for throughput
            'pause_live_ui': lambda: (
                Clock.unschedule(ctx.scope_display.update_scopedisplay),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1)
            ),
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'scan_iterate_post': run_not_started_func,
            'update_step_number': lumaviewpro._update_step_number_callback,
            'go_to_step': lumaviewpro.go_to_step,
            'run_complete': self._autofocus_run_complete_callback,
            'files_complete': self._autofocus_files_complete,
            'leds_off': lumaviewpro._handle_ui_for_leds_off,
            'led_state': lumaviewpro._handle_ui_for_led,
            'reset_autofocus_btns': lumaviewpro.update_autofocus_selection_after_protocol,
            'set_recording_title': lumaviewpro.set_recording_title,
            'set_writing_title': lumaviewpro.set_writing_title,
            'reset_title': lumaviewpro.reset_title,
        }

        autogain_settings = lumaviewpro.get_auto_gain_settings()

        sequence = copy.deepcopy(self._protocol)
        sequence.modify_autofocus_all_steps(enabled=True)

        sequenced_capture_executor.run(
            protocol=sequence,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='af_scan',
            parent_dir=None,
            image_capture_config=lumaviewpro.get_image_capture_config_from_ui(),
            enable_image_saving=False,
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            update_z_pos_from_autofocus=True,
            leds_state_at_end="off",
            video_as_frames=settings['video_as_frames']
        )


    def _scan_run_complete(self, **kwargs):
        import lumaviewpro

        # Don't reset protocol_running_global yet - keep it True until files complete

        # Reset completion event for this scan (thread-safe)
        self._scan_files_completed_event.clear()

        protocol_running_global = _app_ctx.ctx.protocol_running
        file_io_executor = lumaviewpro.file_io_executor
        version = lumaviewpro.version

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._file_write_status_event = Clock.schedule_interval(
                self._update_file_write_status,
                0.5  # Update every 500ms
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
            Window.set_title(f"Lumaview Pro {version}   |   Writing protocol scan files to disk...")
        else:
            # No files pending - proceed with normal reset
            protocol_running_global.clear()
            self._reset_run_scan_button()
            lumaviewpro.create_hyperstacks_if_needed()
            lumaviewpro.live_histo_reverse()
            lumaviewpro.reset_acquire_ui()
            self.reset_autofocus_ui()
            lumaviewpro.stage.set_motion_capability(True)


    def _update_file_write_status(self, dt):
        """Update UI to show file writing progress."""
        import lumaviewpro
        file_io_executor = lumaviewpro.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_scan_btn'].text = f'Writing Files... ({queue_size})'
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._scan_files_complete()


    def _scan_files_complete(self, **kwargs):
        """Called when ALL files are written to disk (deferred callback)."""
        import lumaviewpro

        # Guard against multiple calls using thread-safe event
        if self._scan_files_completed_event.is_set():
            return
        self._scan_files_completed_event.set()

        # Cancel status update if still scheduled
        if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
            Clock.unschedule(self._file_write_status_event)
            self._file_write_status_event = None

        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.clear()

        # Now actually reset the button
        self._reset_run_scan_button()

        # Re-enable other buttons that were disabled during file writing
        self.ids['run_protocol_btn'].disabled = False
        self.ids['run_autofocus_btn'].disabled = False

        # Complete remaining cleanup
        lumaviewpro.create_hyperstacks_if_needed()
        lumaviewpro.live_histo_reverse()
        lumaviewpro.reset_acquire_ui()
        self.reset_autofocus_ui()
        lumaviewpro.stage.set_motion_capability(True)
        lumaviewpro.reset_title()


    def run_scan_from_ui(self):
        import lumaviewpro
        from ui.notification_popup import show_notification_popup

        logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui()')
        trigger_source = 'scan'
        run_complete_func = self._scan_run_complete
        run_not_started_func = self._reset_run_scan_button

        sequenced_capture_executor = lumaviewpro.sequenced_capture_executor
        file_io_executor = lumaviewpro.file_io_executor
        ctx = _app_ctx.ctx

        # Only block if starting NEW scan (button is 'down'), not if aborting (button is 'normal')
        if self.ids['run_scan_btn'].state == 'down' and file_io_executor.is_protocol_queue_active():
            run_not_started_func()
            logger.warning(f"Cannot start scan - files still being written to disk")
            show_notification_popup(
                title="Operation Blocked",
                message="Please wait - files are still being written to disk from the previous scan."
            )
            return

        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.set()

        # Disable ability for user to move stage manually
        lumaviewpro.stage.set_motion_capability(False)

        # State of button immediately changed upon press, so we are checking if the button was previously not pressed, and if autofocus is happening
        if self.ids['run_scan_btn'].state == 'down' and sequenced_capture_executor._autofocus_executor.in_progress():
            run_not_started_func()
            logger.warning(f"Cannot start scan. Autofocus still in progress.")
            return

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if (sequenced_capture_executor.run_in_progress() and (run_trigger_source != trigger_source)):
            run_not_started_func()
            logger.warning(f"Cannot start scan. Run already in progress from {run_trigger_source}")
            return

        if not self._is_protocol_valid():
            run_not_started_func()
            return

        if self.ids['run_scan_btn'].state == 'normal':
            logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui() - User ending scan early')
            self._cleanup_at_end_of_protocol(autofocus_scan=False)
            return

        self.ids['run_scan_btn'].text = 'Abort One Scan'
        self.ids['run_scan_btn'].background_down = './data/icons/abort_protocol_background.png'

        callbacks = {
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'scan_iterate_post': run_not_started_func,
            'run_complete': run_complete_func,
            'files_complete': self._scan_files_complete,
            'leds_off': lumaviewpro._handle_ui_for_leds_off,
            'led_state': lumaviewpro._handle_ui_for_led,
            'pause_live_ui': lambda: (
                Clock.unschedule(ctx.scope_display.update_scopedisplay),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1)
            ),
        }

        self.run_sequenced_capture(
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            run_trigger_source=trigger_source,
            max_scans=1,
            callbacks=callbacks,
        )


    def _protocol_run_complete(self, **kwargs):
        import lumaviewpro

        # Don't reset protocol_running_global yet - keep it True until files complete

        # Reset completion event for this run (thread-safe)
        self._scan_files_completed_event.clear()

        protocol_running_global = _app_ctx.ctx.protocol_running
        file_io_executor = lumaviewpro.file_io_executor
        version = lumaviewpro.version

        # Check if files are still being written
        if file_io_executor.is_protocol_queue_active():
            # Schedule periodic update to show remaining file count
            self._file_write_status_event = Clock.schedule_interval(
                self._update_protocol_write_status,
                0.5  # Update every 500ms
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
            Window.set_title(f"Lumaview Pro {version}   |   Writing protocol scan files to disk...")
        else:
            # No files pending - proceed with normal reset
            protocol_running_global.clear()
            self._reset_run_protocol_button()
            lumaviewpro.live_histo_reverse()
            lumaviewpro.create_hyperstacks_if_needed()
            lumaviewpro.reset_acquire_ui()
            self.reset_autofocus_ui()
            lumaviewpro.stage.set_motion_capability(True)


    def _update_protocol_write_status(self, dt):
        """Update UI to show file writing progress for protocol."""
        import lumaviewpro
        file_io_executor = lumaviewpro.file_io_executor

        if file_io_executor.is_protocol_queue_active():
            queue_size = file_io_executor.protocol_queue_size()
            self.ids['run_protocol_btn'].text = f'Writing Files... ({queue_size})'
        else:
            # Queue is empty - cancel this scheduled update and trigger completion
            if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
                Clock.unschedule(self._file_write_status_event)
                self._file_write_status_event = None
                # Trigger completion directly since queue is done
                self._protocol_files_complete()


    def _protocol_files_complete(self, **kwargs):
        """Called when ALL files are written to disk for protocol run."""
        import lumaviewpro

        # Guard against multiple calls using thread-safe event
        if self._scan_files_completed_event.is_set():
            return
        self._scan_files_completed_event.set()

        # Cancel status update if still scheduled
        if hasattr(self, '_file_write_status_event') and self._file_write_status_event:
            Clock.unschedule(self._file_write_status_event)
            self._file_write_status_event = None

        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.clear()

        # Reset the protocol button
        self._reset_run_protocol_button()

        # Re-enable other buttons
        self.ids['run_scan_btn'].disabled = False
        self.ids['run_autofocus_btn'].disabled = False

        # Complete remaining cleanup
        lumaviewpro.live_histo_reverse()
        lumaviewpro.create_hyperstacks_if_needed()
        lumaviewpro.reset_acquire_ui()
        self.reset_autofocus_ui()
        lumaviewpro.stage.set_motion_capability(True)
        lumaviewpro.reset_title()


    def run_protocol_from_ui(self):
        import lumaviewpro
        from ui.notification_popup import show_notification_popup

        logger.info('[LVP Main  ] ProtocolSettings.run_protocol_from_ui()')
        trigger_source = 'protocol'
        run_complete_func = self._protocol_run_complete
        run_not_started_func = self._reset_run_protocol_button

        sequenced_capture_executor = lumaviewpro.sequenced_capture_executor
        file_io_executor = lumaviewpro.file_io_executor
        ctx = _app_ctx.ctx

        protocol_running_global = _app_ctx.ctx.protocol_running
        protocol_running_global.set()

        lumaviewpro.stage.set_motion_capability(False)

        # Only block if starting NEW protocol run (button is 'down'), not if aborting (button is 'normal')
        if self.ids['run_protocol_btn'].state == 'down' and file_io_executor.is_protocol_queue_active():
            run_not_started_func()
            logger.warning(f"Cannot start protocol run - files still being written to disk")
            show_notification_popup(
                title="Operation Blocked",
                message="Please wait - files are still being written to disk."
            )
            return

        run_trigger_source = sequenced_capture_executor.run_trigger_source()

        # State of button immediately changed upon press, so we are checking if the button was previously not pressed, and if autofocus is happening
        if self.ids['run_protocol_btn'].state == 'down' and sequenced_capture_executor._autofocus_executor.in_progress():

            run_not_started_func()
            logger.warning(f"Cannot start protocol run. Autofocus still in progress.")
            return

        if (sequenced_capture_executor.run_in_progress() and (run_trigger_source != trigger_source)):
            run_not_started_func()
            logger.warning(f"Cannot start protocol run. Run already in progress from {run_trigger_source}")
            return

        if not self._is_protocol_valid():
            run_not_started_func()
            return

        if self.ids['run_protocol_btn'].state == 'normal':
            self._cleanup_at_end_of_protocol(autofocus_scan=False)
            return

        # Note: This will be quickly overwritten by the remaining number of scans
        self.ids['run_protocol_btn'].text = 'Running Protocol'

        settings = _app_ctx.ctx.settings

        callbacks = {
            'protocol_iterate_pre': self._update_protocol_run_button_status,
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'run_complete': run_complete_func,
            'files_complete': self._protocol_files_complete,
            'leds_off': lumaviewpro._handle_ui_for_leds_off,
            'led_state': lumaviewpro._handle_ui_for_led,
            'pause_live_ui': lambda: (
                Clock.unschedule(ctx.scope_display.update_scopedisplay),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
            ),
            'resume_live_ui': lambda: (
                ctx.scope_display.start(),
                Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui),
                Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1)
            ),
        }

        time_params = lumaviewpro.get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

        self.run_sequenced_capture(
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            run_trigger_source=trigger_source,
            max_scans=None,
            callbacks=callbacks,
        )

    def reset_autofocus_ui(self, **kwargs):
        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            layer_obj.ids["autofocus"].state = "down" if settings[layer]["autofocus"] else "normal"

    def _update_protocol_run_button_status(
        self,
        **kwargs,
    ):
        protocol_running_global = _app_ctx.ctx.protocol_running

        if not protocol_running_global.is_set():
            return

        remaining_scans = kwargs['remaining_scans']
        scan_interval = kwargs['interval']
        remaining_duration = remaining_scans * scan_interval
        remaining_duration_str = strfdelta(
            tdelta=remaining_duration,
            fmt='{H}h {M}m',
            inputtype='timedelta',
        )
        scan_word = "scan" if remaining_scans == 1 else "scans"

        self.ids['run_protocol_btn'].text = f"{remaining_scans} {scan_word} ({remaining_duration_str}) remaining.\nPress to ABORT"
        self.ids['run_protocol_btn'].background_down = './data/icons/abort_protocol_background.png'


    def _run_scan_pre_callback(self):
        import lumaviewpro
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


    def run_sequenced_capture(
        self,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        callbacks: dict[str, typing.Callable],
        disable_saving_artifacts: bool = False,
        return_to_position: dict | None = None,
    ):
        import lumaviewpro

        lumaviewpro.live_histo_off()

        logger.info('[LVP Main  ] ProtocolSettings.run_sequenced_capture()')

        settings = _app_ctx.ctx.settings
        ctx = _app_ctx.ctx
        sequenced_capture_executor = lumaviewpro.sequenced_capture_executor

        callbacks.update(
            {
                'move_position': lumaviewpro._handle_ui_update_for_axis,
                'leds_off': lumaviewpro._handle_ui_for_leds_off,
                'led_state': lumaviewpro._handle_ui_for_led,
                'update_step_number': lumaviewpro._update_step_number_callback,
                'go_to_step': lumaviewpro.go_to_step,
                'update_scope_display': ctx.scope_display.update_scopedisplay,
                'reset_autofocus_btns': lumaviewpro.update_autofocus_selection_after_protocol,
                'set_recording_title': lumaviewpro.set_recording_title,
                'set_writing_title': lumaviewpro.set_writing_title,
                'reset_title': lumaviewpro.reset_title,
                'restore_autofocus_state': lambda layer, value: settings[layer].__setitem__('autofocus', value),
            }
        )

        parent_dir = pathlib.Path(settings['live_folder']).resolve() / "ProtocolData"

        sequence_name = self.ids['protocol_filename'].text

        image_capture_config = lumaviewpro.get_image_capture_config_from_ui()
        autogain_settings = lumaviewpro.get_auto_gain_settings()

        # Snapshot autofocus states from settings on the UI thread before passing to protocol thread
        initial_autofocus_states = {layer: settings[layer]['autofocus'] for layer in common_utils.get_layers()}

        sequenced_capture_executor.run(
            protocol=self._protocol,
            run_mode=run_mode,
            run_trigger_source=run_trigger_source,
            max_scans=max_scans,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=lumaviewpro.is_image_saving_enabled(),
            separate_folder_per_channel=ctx.motion_settings.ids['microscope_settings_id']._seperate_folder_per_channel,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            disable_saving_artifacts=disable_saving_artifacts,
            return_to_position=return_to_position,
            leds_state_at_end="off",
            video_as_frames=settings['video_as_frames'],
            initial_autofocus_states=initial_autofocus_states,
        )

        lumaviewpro.set_last_save_folder(dir=sequenced_capture_executor.run_dir())

        if run_mode == SequencedCaptureRunMode.FULL_PROTOCOL:
            self._update_protocol_run_button_status(
                remaining_scans=sequenced_capture_executor.remaining_scans(),
                interval=sequenced_capture_executor.protocol_interval(),
            )


    def _cleanup_at_end_of_protocol(self, autofocus_scan: bool):
        import lumaviewpro

        sequenced_capture_executor = lumaviewpro.sequenced_capture_executor
        sequenced_capture_executor.reset()
        sequenced_capture_executor._autofocus_executor.reset()
        lumaviewpro.live_histo_reverse()
        self._reset_run_protocol_button()
        self._reset_run_scan_button()
        self._reset_run_autofocus_scan_button()
        self.reset_autofocus_ui()
        self._autofocus_complete_callback()
        lumaviewpro.stage.set_motion_capability(True)


        if not autofocus_scan:
            try:
                lumaviewpro.create_hyperstacks_if_needed()
            except Exception as e:
                logger.error(f"Error occurred while creating hyperstacks: {e}", exc_info=True)

    def cancel_all_protocols(self):
        logger.info('[LVP Main  ] ProtocolSettings.cancel_all_protocols()')
        self._cleanup_at_end_of_protocol(autofocus_scan=False)

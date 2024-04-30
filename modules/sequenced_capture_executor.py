
import datetime
import pathlib
import os
import time
import typing

from kivy.clock import Clock

from lumascope_api import Lumascope
import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
import modules.labware_loader as labware_loader
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from lvp_logger import logger


class SequencedCaptureExecutor:

    LOGGER_NAME = "SeqCapExec"

    def __init__(
        self,
        scope: Lumascope,
        stage_offset: dict,
    ):
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._wellplate_loader = labware_loader.WellPlateLoader()
        self._stage_offset = stage_offset

        self._scope = scope
        self._run_trigger_source = None
        self._reset_vars()


    def _reset_vars(
        self
    ):
        self._run_dir = None
        self._run_trigger_source = None
        self._run_in_progress = False
        self._curr_step = 0
        self._n_scans = 0
        self._scan_count = 0
        self._scan_in_progress = False
        self._autofocus_count = 0
        self._autogain_countdown = 0


    @staticmethod
    def _calculate_num_scans(
        protocol: Protocol,
        run_mode: SequencedCaptureRunMode,
        max_scans: int | None,
    ) -> int:
        if run_mode in (
            SequencedCaptureRunMode.FULL_PROTOCOL,
        ):
            n_scans = int(protocol.duration()/protocol.period())

            if max_scans is not None:
                n_scans = min(n_scans, max_scans)
        else:
            n_scans = max_scans

        return n_scans
    

    def num_scans(self) -> int:
        return self._n_scans
    

    def scan_count(self) -> int:
        return self._scan_count
    

    def remaining_scans(self) -> int:
        return self._n_scans - self._scan_count
    

    def _init_for_new_scan(
        self,
        max_scans: int
    ) -> bool:
        self._reset_vars()
        self._n_scans = self._calculate_num_scans(
            protocol=self._protocol,
            run_mode=self._run_mode,
            max_scans=max_scans,
        )
        self._start_t = datetime.datetime.now()

        try:
            self._parent_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            err_str = f"Unable to save data to {str(self._parent_dir)}. Please select an accessible capture location."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
        
        result = self._create_run_dir()
        if not result['status']:
            return result
        
        try:
            self._initialize_run_dir()
        except:
            err_str = f"Unable to initialize sequenced run directory."
            return {
                'status': False,
                'data': None,
                'error': err_str
            }
        
        return {
            'status': True,
            'data': None,
            'error': None
        }
    

    def run_dir(self):
        return self._run_dir
        

    def _create_run_dir(self):
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._parent_dir / time_string

        try:
            self._run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            err_str = f"Unable to save data to {str(self._run_dir)}, already exists."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
        except FileNotFoundError:
            err_str = f"Unable to save data to {str(self._run_dir)}. Please select an accessible capture location."
            return {
                'status': False,
                'data': None,
                'error': err_str,
            }
        
        return {
            'status': True,
            'data': None,
            'error': None,
        }
    

    def _initialize_run_dir(self):
        if self._sequence_name in (None, ""):
            self._sequence_name = 'unsaved_protocol'
            
        filename = self._sequence_name
        if not filename.endswith(".tsv"):
            filename += ".tsv"

        protocol_file_loc = self._run_dir / filename
        self._protocol.to_file(
            file_path=protocol_file_loc
        )

        protocol_record_file_loc = self._run_dir / ProtocolExecutionRecord.DEFAULT_FILENAME
        self._protocol_execution_record = ProtocolExecutionRecord(outfile=protocol_record_file_loc)

        return True


    def reset(self):
        if not self._run_in_progress:
            return
        
        self._cleanup()
    

    def run(
        self,
        protocol: Protocol,
        run_trigger_source: str,
        run_mode: SequencedCaptureRunMode,
        sequence_name: str,
        parent_dir: pathlib.Path,
        image_capture_config: dict,
        enable_image_saving: bool = True,
        separate_folder_per_channel: bool = False,
        autogain_target_brightness: float = 0.3,
        autogain_max_duration: datetime.timedelta = 1.0,
        callbacks: dict[str, typing.Callable] | None = None,
        max_scans: int | None = None,
        return_to_position: dict | None = None,
    ):
        if self._run_in_progress:
            logger.error(f"[{self.LOGGER_NAME} ] Cannot start new run, run already in progress")
            return
        
        self._protocol = protocol
        self._run_mode = run_mode
        self._sequence_name = sequence_name
        self._parent_dir = parent_dir
        self._image_capture_config = image_capture_config
        self._enable_image_saving = enable_image_saving
        self._separate_folder_per_channel = separate_folder_per_channel
        self._autogain_target_brightness = autogain_target_brightness
        self._autogain_max_duration = autogain_max_duration
        self._callbacks = callbacks
        self._return_to_position = return_to_position

        self._cancel_all_scheduled_events()
        result = self._init_for_new_scan(max_scans=max_scans)
        if not result['status']:
            self._run_in_progress = False
            return 
        
        self._run_trigger_source = run_trigger_source
        self._run_in_progress = True
        self._scope.camera.update_auto_gain_target_brightness(self._autogain_target_brightness)

        self._run_scan()
        Clock.schedule_interval(self._protocol_iterate, 1)

    
    def _protocol_iterate(self, dt):
        if self._scan_in_progress:
            return
        
        remaining_scans=self.remaining_scans()
        if remaining_scans == 0:
            self._cleanup()
            return 

        if 'protocol_iterate_pre' in self._callbacks:
            self._callbacks['protocol_iterate_pre'](remaining_scans=remaining_scans)

        current_time = datetime.datetime.now()
        elapsed_time = current_time - self._start_t

        # If the next period hasn't been reached, then return
        if elapsed_time <= self._protocol.period():
            return

        # reset the start time and update number of scans remaining
        self._start_t = current_time
                
        self._run_scan()


    def _run_scan(
        self
    ):
        if self._scan_in_progress == True:
            self._scan_count += 1

        self._scan_in_progress = True 
        self._curr_step = 0

        # reset the is_complete flag on autofocus
        if 'run_scan_pre' in self._callbacks:
            self._callbacks['run_scan_pre']()

        self._go_to_step(step_idx=self._curr_step)

        Clock.schedule_interval(self._scan_iterate, 0.1)
    

    def _scan_iterate(self, dt):       
        if self._scope.is_focusing:
            return

        # Check if at desired position
        x_status = self._scope.get_target_status('X')
        y_status = self._scope.get_target_status('Y')
        z_status = self._scope.get_target_status('Z')

        # Check if target location has not been reached yet
        if (not x_status) or (not y_status) or (not z_status) or self._scope.get_overshoot():
            return
        
        step = self._protocol.step(idx=self._curr_step)
        
        # Set camera settings
        self._scope.set_auto_gain(
            state=step['Auto_Gain'],
            target_brightness=self._autogain_target_brightness,
        )
        self._scope.led_on(
            channel=self._scope.color2ch(step['Color']),
            mA=step['Illumination'],
        )

        if not step['Auto_Gain']:
            self._scope.set_gain(step['Gain'])
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            self._scope.set_exposure_time(step['Exposure'])

        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            self._auto_gain_countdown -= 0.1
        
        # If the autofocus is selected, is not currently running and has not completed, begin autofocus
        autofocus_is_complete = not self._scope.is_focusing
        if step['Auto_Focus'] and not autofocus_is_complete:

            if 'autofocus_in_progress' in self._callbacks:
                self._callbacks['autofocus_in_progress']()

            self._scope.autofocus()
            
            return
        
        # Check if autogain has time-finished after auto-focus so that they can run in parallel
        if step['Auto_Gain'] and self._auto_gain_countdown > 0:
            return
        
        # Reset the autogain countdown
        self._auto_gain_countdown = self._autogain_max_duration.total_seconds
        
        # reset the is_complete flag on autofocus
        if 'autofocus_complete' in self._callbacks:
            self._callbacks['autofocus_complete']()
        # lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

        if self._separate_folder_per_channel:
            save_folder = self._run_dir / step["Color"]
            save_folder.mkdir(parents=True, exist_ok=True)
        else:
            save_folder = self._run_dir

        if step["Auto_Focus"]:
            self._autofocus_count += 1

        image_filepath = self._capture(
            save_folder=save_folder,
            step=step,
            scan_count=self._scan_count,
        )

        if self._enable_image_saving == True:
            if image_filepath is None:
                image_filepath_name = "unsaved"

            elif self._separate_folder_per_channel:
                image_filepath_name = pathlib.Path(step["Color"]) / image_filepath.name

            else:
                image_filepath_name = image_filepath.name
        else:
            image_filepath_name = "unsaved"

        self._protocol_execution_record.add_step(
            image_file_name=image_filepath_name,
            step_name=step['Name'],
            step_index=self._curr_step,
            scan_count=self._scan_count,
            timestamp=datetime.datetime.now()
        )

        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            self._scope.set_auto_gain(state=False)

        num_steps = self._protocol.num_steps()
        if self._curr_step < num_steps-1:

            # increment to the next step. Don't let it exceed the number of steps in the protocol
            self._curr_step = min(self._curr_step+1, num_steps-1)

            if 'update_step_number' in self._callbacks:
                self._callbacks['update_step_number'](self._curr_step+1)
            self._go_to_step(step_idx=self._curr_step)
            return


        # At the end of a scan, if we've performed more than 100 AFs, cycle the Z-axis to re-distribute grease
        if self._autofocus_count >= 100:
            self._perform_grease_redistribution()
            self._autofocus_count = 0

        self._scan_count += 1
        
        if 'scan_iterate_post' in self._callbacks is not None:
            self._callbacks['scan_iterate_post']()

        Clock.unschedule(self._scan_iterate)
        self._scan_in_progress = False


    def run_in_progress(self) -> bool:
        return self._run_in_progress
    

    def run_trigger_source(self) -> str:
        return self._run_trigger_source
    
    
    def _default_move(
        self,
        px: float | None = None,
        py: float | None = None,
        z: float | None = None,
    ):
        labware = self._wellplate_loader.get_plate(plate_key=self._protocol.labware())

        if (px is not None) and (py is not None):
            sx, sy = self._coordinate_transformer.plate_to_stage(
                labware=labware,
                stage_offset=self._stage_offset,
                px=px,
                py=py
            )

            self._scope.move_absolute_position('X', sx)
            self._scope.move_absolute_position('Y', sy)
            self._callbacks['move_position']('X')
            self._callbacks['move_position']('Y')

        if z is not None:
            self._scope.move_absolute_position('Z', z)
            self._callbacks['move_position']('Z')


    def _go_to_step(
        self,
        step_idx: int,
    ):
        if 'go_to_step' in self._callbacks:
            self._callbacks['go_to_step'](
                protocol=self._protocol,
                step_idx=step_idx,
                include_move=True,
                ignore_auto_gain=True
            )

        else:
            step = self._protocol.step(idx=step_idx)
            self._default_move(
                px=step['X'],
                py=step['Y'],
                z=step['Z'],
            )


    def _perform_grease_redistribution(self):
        axis='Z'
        z_orig = self._scope.get_current_position(axis=axis)

        self._scope.move_absolute_position(
            axis=axis,
            pos=0,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            self._callbacks['move_position'](axis)

        self._scope.move_absolute_position(
            axis=axis,
            pos=z_orig,
            wait_until_complete=True,
            overshoot_enabled=True
        )

        if 'move_position' in self._callbacks:
            self._callbacks['move_position'](axis)


    def _cancel_all_scheduled_events(self):
        Clock.unschedule(self._protocol_iterate)
        Clock.unschedule(self._scan_iterate)


    def _cleanup(self):
        if not self._run_in_progress:
            return
        
        self._cancel_all_scheduled_events()
        self._scope.leds_off()

        if 'leds_off' in self._callbacks:
            self._callbacks['leds_off']()

        self._protocol_execution_record.complete()

        if self._return_to_position is not None:
            self._default_move(
                px=self._return_to_position['x'],
                py=self._return_to_position['y'],
                z=self._return_to_position['z'],
            )

        self._run_in_progress = False

        if 'run_complete' in self._callbacks:
            self._callbacks['run_complete']()


    def _capture(
        self,
        save_folder,
        step,
        scan_count = None,
    ):
        if not step['Auto_Gain']:
            self._scope.set_gain(step['Gain'])
            self._scope.set_exposure_time(step['Exposure'])
    
        name = common_utils.generate_default_step_name(
            well_label=step['Well'],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            scan_count=scan_count,
            custom_name_prefix=step['Name'],
            tile_label=step['Tile']
        )

        # Illuminate
        if self._scope.led:
            channel = self._scope.color2ch(step['Color'])
            self._scope.led_on(channel, step['Illumination'])
            logger.info(f"[{self.LOGGER_NAME} ] scope.led_on({channel}, {step['Illumination']})")
        else:
            logger.warning('LED controller not available.')

        # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        # Grab image and save
        time.sleep(2*step['Exposure']/1000+0.2)

        if 'update_scope_display' in self._callbacks:
            self._callbacks['update_scope_display']()
        
        use_color = step['Color'] if step['False_Color'] else 'BF'

        if self._enable_image_saving == True:
            use_full_pixel_depth = self._image_capture_config['use_full_pixel_depth']

            image_filepath = self._scope.save_live_image(
                save_folder=save_folder,
                file_root=None,
                append=name,
                color=use_color,
                tail_id_mode=None,
                force_to_8bit=not use_full_pixel_depth,
                output_format=self._image_capture_config['output_format']
            )
        else:
            image_filepath = None

        if 'leds_off' in self._callbacks:
            self._callbacks['leds_off']()

        return image_filepath

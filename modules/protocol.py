# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import json
import datetime
import io
import numpy as np
import pandas as pd
import os
import pathlib
import re
import copy

from lvp_logger import logger
from modules.exceptions import ProtocolError

import modules.color_channels as color_channels
import modules.common_utils as common_utils
import modules.labware_loader as labware_loader
from modules.tiling_config import TilingConfig
from modules.objectives_loader import ObjectiveLoader
from modules.zstack_config import ZStackConfig
from modules.coord_transformations import CoordinateTransformer

coordinate_transformer = CoordinateTransformer()

class ProtocolFormatError(Exception):
    pass



class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"
    COLUMNS = {
        1: ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Channel', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective'],
        2: ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID'],
        3: ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config'],
        4: ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config'],
        5: ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config', 'Stim_Config'],
    }
    CURRENT_VERSION = 5
    CURRENT_COLUMNS = COLUMNS[CURRENT_VERSION]
    STEP_NAME_PATTERN = re.compile(r"^(?P<well_label>[A-Z][0-9]+)(_(?P<color>(Blue|Green|Red|BF|DF|PC|Lumi)))(_T(?P<tile_label>[A-Z][0-9]+))?(_Z(?P<z_slice>[0-9]+))?(_([0-9]*))?(.tif[f])?$")
    
    def __init__(
        self,
        tiling_configs_file_loc: pathlib.Path,
        config: dict = None
    ):

        self._objective_loader = ObjectiveLoader()

        self._tiling_config = TilingConfig(
            tiling_configs_file_loc=tiling_configs_file_loc
        )

        if config is None:
            self._config = {}
        else:
            self._config = config

        
    @staticmethod
    def _build_z_height_map(values) -> dict:
            z_height_map = {}

            z_heights = sorted(set(values.astype('float').tolist()))
            if len(z_heights) >= 2:
                z_height_map = {z_height: idx for idx, z_height in enumerate(z_heights)}

            return z_height_map


    def period(self) -> datetime.timedelta:
        return self._config['period']
    

    def duration(self) -> datetime.timedelta:
        return self._config['duration']
    

    def labware(self) -> str:
        return self._config['labware_id']
    

    @staticmethod
    def sanitize_step_name(input: str) -> str:
        return re.sub(r'[^a-zA-Z0-9-_]', '', input)
    

    def capture_root(self) -> str:
        return self._config.get('capture_root', '')


    def to_file(self, file_path: pathlib.Path):   
        
        if self.period() is None:
            period_minutes = -1
        else:
            period_minutes = round(self.period().total_seconds() / 60.0, 2)

        if self.duration() is None:
            duration_hours = -1
        else:
            duration_hours = round(self.duration().total_seconds() / 3600.0, 6)

        try:
            with open(file_path, 'w') as fp:
                csvwriter = csv.writer(fp, delimiter='\t', lineterminator='\n') # access the file using the CSV library

                csvwriter.writerow(['LumaViewPro Protocol'])
                csvwriter.writerow(['Version', self._config['version']])
                csvwriter.writerow(['Period', period_minutes])
                csvwriter.writerow(['Duration', duration_hours])
                csvwriter.writerow(['Labware', self._config['labware_id']])
                csvwriter.writerow(['Capture Root', self.capture_root() if self.capture_root() is not None or self.capture_root() != '' else ''])
                
                fp.write('\nSteps\n')

                protocol_table_str = self.steps().to_csv(
                    sep='\t',
                    lineterminator='\n',
                    index=False
                )
                fp.write(protocol_table_str)
        except Exception as e:
            logger.error(f"[Protocol] Error saving protocol to file {file_path}. File may be open in another window: {e}")
            return f"Error saving protocol to file {file_path}.\n File may be open in another window.\n"
        
        return None



    def optimize_step_ordering(self):
        steps = self.steps()

        if len(steps) == 0:
            return

        # First group by X/Y location
        grouped_df = steps.groupby(by=['X','Y'], sort=False)
        
        # For each X/Y location, sort by Z-height
        grouped_list = []
        for _, group_df in grouped_df:
            group_df = group_df.sort_values(by=['Objective', 'Z'], ascending=[True,True])
            grouped_list.append(group_df)

        # Re-combine into single dataframe
        df = pd.concat(grouped_list, ignore_index=True).reset_index(drop=True)
        self._config['steps'] = df
    

    def _create_empty_steps_df() -> pd.DataFrame:
        dtypes = np.dtype(
            [
                ("Name", str),
                ("X", float),
                ("Y", float),
                ("Z", float),
                ("Auto_Focus", bool),
                ("Color", str),
                ("False_Color", bool),
                ("Illumination", float),
                ("Gain", float),
                ("Auto_Gain", bool),
                ("Exposure", float),
                ("Sum", int),
                ("Objective", str),
                ("Well", str),
                ("Tile", str),
                ("Z-Slice", int),
                ("Custom Step", bool),
                ("Tile Group ID", int),
                ("Z-Stack Group ID", int),
                ("Acquire", str),
                ("Video Config", object),
                ("Stim_Config", object),
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df
    

    # Valid values for field validation
    VALID_COLORS = {c.name for c in color_channels.ColorChannel}
    VALID_ACQUIRE_MODES = {'image', 'video'}

    def validate_steps(self, objectives_file: str = None) -> list:
        """Validate all step fields and return a list of error strings.

        Returns an empty list if all steps are valid.
        """
        errors = []
        steps = self.steps()
        if steps is None or len(steps) == 0:
            return errors

        # Load valid objectives
        valid_objectives = set()
        try:
            obj_loader = ObjectiveLoader()
            valid_objectives = set(obj_loader.get_objectives_list())
        except Exception:
            pass  # skip objective validation if loader fails

        for idx, step in steps.iterrows():
            label = f"Step {idx + 1} ({step.get('Name', '?')})"

            # Color
            color = step.get('Color', '')
            if color not in self.VALID_COLORS:
                errors.append(
                    f"{label}: Color '{color}' is not valid. "
                    f"Must be one of: {', '.join(sorted(self.VALID_COLORS))}")

            # Objective
            obj = step.get('Objective', '')
            if valid_objectives and obj not in valid_objectives:
                errors.append(
                    f"{label}: Objective '{obj}' not found in objectives.json")

            # Exposure
            try:
                exposure = float(step.get('Exposure', 0))
                if exposure <= 0:
                    errors.append(f"{label}: Exposure must be > 0, got {exposure}")
            except (ValueError, TypeError):
                errors.append(f"{label}: Exposure is not a valid number")

            # Illumination
            try:
                illum = float(step.get('Illumination', 0))
                if illum < 0 or illum > 1000:
                    errors.append(
                        f"{label}: Illumination must be 0–1000 mA, got {illum}")
            except (ValueError, TypeError):
                errors.append(f"{label}: Illumination is not a valid number")

            # Gain
            try:
                gain = float(step.get('Gain', 0))
                if gain < 0:
                    errors.append(f"{label}: Gain must be >= 0, got {gain}")
            except (ValueError, TypeError):
                errors.append(f"{label}: Gain is not a valid number")

            # Sum
            try:
                sum_count = int(step.get('Sum', 1))
                if sum_count < 1:
                    errors.append(f"{label}: Sum must be >= 1, got {sum_count}")
            except (ValueError, TypeError):
                errors.append(f"{label}: Sum is not a valid integer")

            # Acquire mode
            acquire = step.get('Acquire', 'image')
            if acquire not in self.VALID_ACQUIRE_MODES:
                errors.append(
                    f"{label}: Acquire must be 'image' or 'video', got '{acquire}'")

            # Video Config (only validate if acquire == 'video')
            if acquire == 'video':
                vc = step.get('Video Config', {})
                if isinstance(vc, dict):
                    fps = vc.get('fps', 0)
                    duration = vc.get('duration', 0)
                    if not isinstance(fps, (int, float)) or fps <= 0:
                        errors.append(f"{label}: Video Config fps must be > 0")
                    if not isinstance(duration, (int, float)) or duration <= 0:
                        errors.append(f"{label}: Video Config duration must be > 0")
                else:
                    errors.append(f"{label}: Video Config is not a valid dict")

            # Name length
            name = step.get('Name', '')
            if len(str(name)) > 200:
                errors.append(
                    f"{label}: Name exceeds 200 characters ({len(str(name))})")

        return errors

    def validate_for_run(self, axis_limits: dict = None) -> list:
        """Validate protocol is safe to execute on hardware.

        Checks positions are within axis travel limits. Call this before
        starting a protocol run. validate_steps() checks field format;
        this method checks runtime safety.

        Args:
            axis_limits: dict mapping axis name to {'min': float, 'max': float}
                in µm. Example: {'X': {'min': 0, 'max': 120000}, ...}

        Returns:
            List of error strings. Empty list if all checks pass.
        """
        errors = []
        steps = self.steps()
        if steps is None or len(steps) == 0:
            return errors

        # Validate step field values first
        errors.extend(self.validate_steps())

        if axis_limits:
            for idx, step in steps.iterrows():
                label = f"Step {idx + 1} ({step.get('Name', '?')})"

                for axis in ('X', 'Y', 'Z'):
                    if axis not in axis_limits:
                        continue
                    try:
                        pos = float(step.get(axis, 0))
                    except (ValueError, TypeError):
                        errors.append(f"{label}: {axis} position is not a valid number")
                        continue
                    limits = axis_limits[axis]
                    if pos < limits['min'] or pos > limits['max']:
                        errors.append(
                            f"{label}: {axis} position {pos} µm is outside travel limits "
                            f"({limits['min']}–{limits['max']} µm)")

        # Validate labware exists
        labware_key = self.labware()
        if labware_key:
            try:
                from modules import labware_loader
                loader = labware_loader.WellPlateLoader()
                plate_list = loader.get_plate_list()
                if labware_key not in plate_list:
                    errors.append(
                        f"Labware '{labware_key}' not found. "
                        f"Available: {', '.join(plate_list)}")
            except Exception:
                pass  # skip labware validation if loader fails

        return errors

    def num_steps(self) -> int:
        return len(self._config['steps'])
    

    def steps(self) -> pd.DataFrame:
        return self._config['steps']
    

    def modify_autofocus(self, step_idx: int, enabled: bool):
        self._config['steps'].at[step_idx, "Auto_Focus"] = enabled

    
    def modify_autofocus_all_steps(self, enabled: bool):
        for idx, _ in self._config['steps'].iterrows():
            self.modify_autofocus(step_idx=idx, enabled=enabled)
    

    def delete_step(self, step_idx: int):
        num_steps = self.num_steps()

        if num_steps < 1:
            return
        
        if step_idx >= self.num_steps():
            raise ProtocolError(f"Cannot delete step idx {step_idx}. Protocol only has {self.num_steps()}.")
        
        self._config['steps'].drop(index=step_idx, axis=0, inplace=True)
        self._config['steps'].reset_index(drop=True, inplace=True)
    

    def modify_labware(
        self,
        labware_id: str,
    ):
        self._config['labware_id'] = labware_id


    def modify_time_params(
        self,
        period: datetime.timedelta,
        duration: datetime.timedelta,
    ):
        self._config['period'] = period
        self._config['duration'] = duration


    def modify_step_z_height(
        self,
        step_idx: int,
        z: float
    ):
        self._config['steps'].at[step_idx, "Z"] = z

    def modify_capture_root(
        self,
        capture_root: str
    ):
        self._config['capture_root'] = capture_root

    def modify_name(
            self,
            step_idx: int,
            step_name: str,
    ):
        if step_idx < 0:
                raise ProtocolError(f"Step idx must be > 0")
            
        if step_idx >= self.num_steps():
            raise ProtocolError(f"Cannot modify step idx {step_idx}. Protocol only has {self.num_steps()}.")

        Protocol.sanitize_step_name(step_name)
        self._config['steps'].at[step_idx, "Name"] = step_name
        
        
    def modify_step(
        self,
        step_idx: int,
        step_name: str,
        layer: str,
        layer_config: dict,
        plate_position: dict,
        objective_id: str,
        stim_configs: dict,
    ):
        def _validate_inputs():
            if step_idx < 0:
                raise ProtocolError(f"Step idx must be > 0")
            
            if step_idx >= self.num_steps():
                raise ProtocolError(f"Cannot modify step idx {step_idx}. Protocol only has {self.num_steps()}.")
            
        _validate_inputs()

        self._config['steps'].at[step_idx, "Name"] = step_name
        self._config['steps'].at[step_idx, "X"] = plate_position['x']
        self._config['steps'].at[step_idx, "Y"] = plate_position['y']
        self._config['steps'].at[step_idx, "Z"] = plate_position['z']

        self._config['steps'].at[step_idx, "Auto_Focus"] = layer_config['autofocus']
        self._config['steps'].at[step_idx, "Color"] = layer
        self._config['steps'].at[step_idx, "False_Color"] = layer_config['false_color']
        self._config['steps'].at[step_idx, "Illumination"] = layer_config['illumination']
        self._config['steps'].at[step_idx, "Gain"] = layer_config['gain']
        self._config['steps'].at[step_idx, "Auto_Gain"] = layer_config['auto_gain']
        self._config['steps'].at[step_idx, "Exposure"] = layer_config['exposure']
        self._config['steps'].at[step_idx, "Sum"] = int(layer_config['sum'])
        self._config['steps'].at[step_idx, "Objective"] = objective_id
        self._config['steps'].at[step_idx, "Acquire"] = layer_config['acquire']
        self._config['steps'].at[step_idx, "Video Config"] = copy.deepcopy(layer_config['video_config'])
        self._config['steps'].at[step_idx, "Stim_Config"] = copy.deepcopy(stim_configs)

    def insert_step(
        self,
        step_name: str | None,
        layer: str,
        layer_config: dict,
        plate_position: dict,
        objective_id: str,
        stim_configs: dict,
        before_step: int | None = 0,
        after_step: int | None = None,
        include_objective_in_step_name: bool = False,
    ) -> str:
        
        def _validate_inputs():
            if (before_step is None) and (after_step is None):
                raise ProtocolError(f"Must specify after_step or before_step")
            
            if (before_step is not None) and (after_step is not None):
                raise ProtocolError(f"Must specify only after_step or before_step, not both")
            
            if (before_step is not None) and (before_step < 0):
                raise ProtocolError(f"before_step cannot be < 0")
            
            if (after_step is not None) and (after_step > self.num_steps()):
                raise ProtocolError(f"after_step cannot be > num_steps")
            
        _validate_inputs()
        
        if include_objective_in_step_name:
            objective_short_name = self._objective_loader.get_objective_info(objective_id=objective_id)['short_name']
        else:
            objective_short_name = None

        if step_name is None:
            step_name = common_utils.generate_default_step_name(
                well_label="",
                custom_name_prefix=f"custom{self._config['custom_step_count']}",
                color=layer,
                objective_short_name=objective_short_name

            )
            CUSTOM_INDEX_WIDTH = 4
            step_name = f"custom{self._config['custom_step_count']:0{CUSTOM_INDEX_WIDTH}d}"
            self._config['custom_step_count'] += 1

        well = ""
        tile = "" # Manually inserted step is not a tile
        zslice = -1
        custom_step = True
        tile_group_id = -1
        zstack_group_id = -1

        step_dict = self._create_step_dict(
            name=step_name,
            x=plate_position['x'],
            y=plate_position['y'],
            z=plate_position['z'],
            af=layer_config['autofocus'],
            color=layer,
            fc=layer_config['false_color'],
            ill=layer_config['illumination'],
            gain=layer_config['gain'],
            auto_gain=layer_config['auto_gain'],
            exp=layer_config['exposure'],
            sum=layer_config['sum'],
            objective=objective_id,
            well=well,
            tile=tile,
            zslice=zslice,
            custom_step=custom_step,
            tile_group_id=tile_group_id,
            zstack_group_id=zstack_group_id,
            acquire=layer_config['acquire'],
            video_config=layer_config['video_config'],
            stim_config=copy.deepcopy(stim_configs),
        )

        if before_step is not None:
            pos_index = before_step-0.5
        else:
            pos_index = after_step+0.5

        line = pd.DataFrame(data=step_dict, index=[pos_index])
        line = line.astype({'Video Config': 'object'})
        line.at[pos_index, 'Video Config'] = step_dict['Video Config']
        
        line = line.astype({'Stim_Config': 'object'})
        line.at[pos_index, 'Stim_Config'] = step_dict['Stim_Config']

        self._config['steps'] = pd.concat([self._config['steps'], line], ignore_index=False, axis=0)
        self._config['steps'] = self._config['steps'].sort_index().reset_index(drop=True)

        return step_name
    

    def step(
        self,
        idx: int
    ):
        def _validate():
            if idx < 0:
                raise ProtocolError(f"Step index cannot be < 0")
            
            if idx >= self.num_steps():
                raise ProtocolError(f"Step idx {idx} does not exist. Protocol only has {self.num_steps()}.")

        _validate(

        )            
        return self._config['steps'].iloc[idx]
    

    def apply_tiling(
        self,
        tiling: str,
        frame_dimensions: dict,
        binning_size: int,
        curr_step_idx: int,
        axes_config: dict,
        labware,
        stage_offset,
    ) -> dict:
        "Returns status dict"
        "If"

        status = {
            "tiles_skipped": 0,
                  }

        if tiling == '1x1':
            return status

        try:
            orig_steps_df = self.steps()

            curr_step = orig_steps_df.iloc[curr_step_idx]

            # Add objective focal length to steps dataframe
            objectives = self._objective_loader.get_objectives_dataframe()['focal_length']

            orig_steps_df["focal_length"] = orig_steps_df["Objective"].map(objectives)

        except Exception as e:
            logger.error(f"Error adding objective focal length to steps dataframe: {e}")
            return status


        try:
            x_limits = axes_config['X']['limits']
            y_limits = axes_config['Y']['limits']

        except Exception as e:
            logger.error(f"Error getting axes limits from axes_config: {e}")
            return status

        existing_max_tile_group_id = orig_steps_df['Tile Group ID'].max()
        tile_group_id = existing_max_tile_group_id + 1

        new_steps = []

        
        for idx, row in orig_steps_df.iterrows():
            try:
                tiles = self._tiling_config.get_tile_centers(
                    config_label=tiling,
                    focal_length=row['focal_length'],
                    frame_size=frame_dimensions,
                    fill_factor=TilingConfig.DEFAULT_FILL_FACTORS['position'],
                    binning_size=binning_size,
                )
            except Exception as e:
                logger.error(f"Error getting tile centers for step {idx}: {e}")
                tiles = {}

            orig_step_df = orig_steps_df.iloc[idx]
            orig_step_dict = orig_step_df.to_dict()

            if len(tiles) == 1: # No tiles generated
                new_steps.append(orig_step_dict)
                continue

            # If already a tile, copy it over to the new protocol
            if orig_step_df['Tile'] not in (None, ""):
                new_steps.append(orig_step_dict)
                continue
            
            x = orig_step_df["X"]
            y = orig_step_df["Y"]

            for tile_label, tile_position in tiles.items():   
                
                x_tile = round(x + tile_position["x"]/1000, common_utils.max_decimal_precision('x')) # in 'plate' coordinates
                y_tile = round(y + tile_position["y"]/1000, common_utils.max_decimal_precision('y')) # in 'plate' coordinates 

                sx, sy = coordinate_transformer.plate_to_stage(
                    labware=labware,
                    stage_offset=stage_offset,
                    px=x_tile,
                    py=y_tile
                )

                # Check if tile is within stage limits
                if (sx > x_limits['max']) or (sx < x_limits['min']) or (sy > y_limits['max']) or (sy < y_limits['min']):
                    logger.info(f"[Protocol] Skipping tile {tile_label} for step {idx} - out of stage limits")
                    status['tiles_skipped'] += 1
                    continue
                
                name = common_utils.generate_default_step_name(
                        well_label=orig_step_df['Well'],
                        color=orig_step_df['Color'],
                        z_height_idx=orig_step_df['Z-Slice'],
                        tile_label=tile_label,
                        objective_short_name=None,  # Can add this if needed
                        custom_name_prefix=None if not orig_step_df['Custom Step'] else orig_step_df['Name'],
                    )
                    
                # if not orig_step_df['Custom Step']:
                #     name = common_utils.generate_default_step_name(
                #         well_label=orig_step_df['Well'],
                #         color=orig_step_df['Color'],
                #         tile_label=tile_label,
                #         objective_short_name=None,
                #     )
                # else:
                #     name = orig_step_df['Name']
                
                new_step_dict = self._create_step_dict(
                    name=name,
                    x=x_tile,
                    y=y_tile,
                    z=orig_step_df['Z'],
                    af=orig_step_df['Auto_Focus'],
                    color=orig_step_df['Color'],
                    fc=orig_step_df['False_Color'],
                    ill=orig_step_df['Illumination'],
                    gain=orig_step_df['Gain'],
                    auto_gain=orig_step_df['Auto_Gain'],
                    exp=orig_step_df['Exposure'],
                    sum=orig_step_df['Sum'],
                    objective=orig_step_df['Objective'],
                    well=orig_step_df['Well'],
                    tile=tile_label,
                    zslice=orig_step_df['Z-Slice'],
                    custom_step=orig_step_df['Custom Step'],
                    tile_group_id=tile_group_id,
                    zstack_group_id=orig_step_df['Z-Stack Group ID'],
                    acquire=orig_step_df['Acquire'],
                    video_config=orig_step_df['Video Config'],
                    stim_config=orig_step_df['Stim_Config'],
                )

                new_steps.append(new_step_dict)
            
            tile_group_id += 1

        self._config['steps'] = pd.DataFrame.from_dict(new_steps)

        return status
 

    def apply_zstacking(
        self,
        zstack_params: dict,
    ):
        
        if zstack_params['step_size'] <= 0 or zstack_params['range'] <= 0:
            return
        
        steps = self.steps()
        existing_max_zstack_group_id = steps['Z-Stack Group ID'].max()

        zstack_group_id = existing_max_zstack_group_id + 1

        num_steps = self.num_steps()
        new_steps = list()

        for row_idx in range(num_steps):
            orig_step_df = self.step(idx=row_idx)
            orig_step_dict = orig_step_df.to_dict()

            # If already part of a Z-Stack, copy it over to the new protocol
            if orig_step_df['Z-Slice'] not in (None, "", -1):
                new_steps.append(orig_step_dict)
                continue

            zstack_config = ZStackConfig(
                range=zstack_params['range'],
                step_size=zstack_params['step_size'],
                current_z_reference=zstack_params['z_reference'],
                current_z_value=orig_step_df["Z"],
            )

            if zstack_config.number_of_steps() == 0:
                continue

            zstack_positions = zstack_config.step_positions()
            
            # Create a z-stack  
            for zstack_slice, zstack_position in zstack_positions.items():

                name = common_utils.generate_default_step_name(
                            well_label=orig_step_df['Well'],
                            color=orig_step_df['Color'],
                            z_height_idx=zstack_slice,
                            tile_label=orig_step_df['Tile'],
                            objective_short_name=None,  # Can add this if needed
                            custom_name_prefix=None if not orig_step_df['Custom Step'] else orig_step_df['Name'],
                        )
                
                new_step_dict = self._create_step_dict(
                    name=name,
                    x=orig_step_df["X"],
                    y=orig_step_df["Y"],
                    z=zstack_position,
                    af=orig_step_df['Auto_Focus'],
                    color=orig_step_df['Color'],
                    fc=orig_step_df['False_Color'],
                    ill=orig_step_df['Illumination'],
                    gain=orig_step_df['Gain'],
                    auto_gain=orig_step_df['Auto_Gain'],
                    exp=orig_step_df['Exposure'],
                    sum=orig_step_df['Sum'],
                    objective=orig_step_df['Objective'],
                    well=orig_step_df['Well'],
                    tile=orig_step_df['Tile'],
                    zslice=zstack_slice,
                    custom_step=orig_step_df['Custom Step'],
                    tile_group_id=orig_step_df['Tile Group ID'],
                    zstack_group_id=zstack_group_id,
                    acquire=orig_step_df['Acquire'],
                    video_config=orig_step_df['Video Config'],
                    stim_config=orig_step_df['Stim_Config'],
                )

                new_steps.append(new_step_dict)
            
            zstack_group_id += 1

        self._config['steps'] = pd.DataFrame.from_dict(new_steps)

    
    @classmethod
    def from_config(
        cls,
        input_config: dict,
        tiling_configs_file_loc : pathlib.Path
    ):
        
        tiling_config = TilingConfig(
            tiling_configs_file_loc=tiling_configs_file_loc
        )

        if 'positions' in input_config:
            positions = input_config['positions']
        else:
            positions = None

        labware_id = input_config['labware_id']
        objective_id = input_config['objective_id']
        zstack_params = input_config['zstack_params']
        use_zstacking = input_config['use_zstacking']
        tiling = input_config['tiling']
        layer_configs = input_config['layer_configs']
        period = input_config['period']
        duration = input_config['duration']
        frame_dimensions = input_config['frame_dimensions']
        binning_size = input_config['binning_size']
        stim_config = input_config['stim_config']

        objective_loader = ObjectiveLoader()
        objective = objective_loader.get_objective_info(objective_id=objective_id)

        tiles = tiling_config.get_tile_centers(
            config_label=tiling,
            focal_length=objective['focal_length'],
            frame_size=frame_dimensions,
            fill_factor=TilingConfig.DEFAULT_FILL_FACTORS['position'],
            binning_size=binning_size,
        )

        config = {
            'version': cls.CURRENT_VERSION,
            'period': period,
            'duration': duration,
            'positions': positions,
            'labware_id': labware_id,
            'custom_step_count': 0,
            'capture_root': '',
        }
        
        if positions is not None:
            actual_positions = positions
            position_source = 'from_manual'
        else:
            wellplate_loader = labware_loader.WellPlateLoader()
            labware_obj = wellplate_loader.get_plate(plate_key=labware_id)
            labware_obj.set_positions()
            well_positions = labware_obj.get_positions_with_labels()

            tmp = []
            for well_position in well_positions:
                well_x, well_y, well_label = well_position
                tmp.append(
                    {
                        'x': well_x,
                        'y': well_y,
                        'z': None,
                        'name': well_label,
                    }
                )

            actual_positions = tmp
            position_source = 'from_labware'            

        tile_group_id = 0
        zstack_group_id = 0
        # custom_step_count = 0
        steps = []

         # Iterate through all the positions in the scan
        for pos in actual_positions:
            for tile_label, tile_position in tiles.items():

                if not use_zstacking:
                    zstack_position_offsets = {None: None}
                else:
                    zstack_config = ZStackConfig(
                        range=zstack_params['range'],
                        step_size=zstack_params['step_size'],
                        current_z_reference=zstack_params['z_reference'],
                        current_z_value=0,
                    )
                    zstack_position_offsets = zstack_config.step_positions()

                for zstack_slice, zstack_position_offset in zstack_position_offsets.items():
                    for layer_name, layer_config in layer_configs.items():
                        if layer_config['acquire'] not in ['image', 'video']:
                            continue
                        
                        x = round(pos['x'] + tile_position["x"]/1000, common_utils.max_decimal_precision('x')) # in 'plate' coordinates
                        y = round(pos['y'] + tile_position["y"]/1000, common_utils.max_decimal_precision('y')) # in 'plate' coordinates

                        if pos['z'] is None:
                            z = layer_config['focus']
                        else:
                            z = pos['z']

                        if zstack_slice is not None:
                            z += zstack_position_offset

                        z = round(z, common_utils.max_decimal_precision('z'))

                        autofocus = layer_config['autofocus']
                        false_color = layer_config['false_color']
                        illumination = round(layer_config['illumination'], common_utils.max_decimal_precision('illumination'))
                        sum = int(layer_config['sum'])
                        gain = round(layer_config['gain'], common_utils.max_decimal_precision('gain'))
                        auto_gain = common_utils.to_bool(layer_config['auto_gain'])
                        exposure = round(layer_config['exposure'], common_utils.max_decimal_precision('exposure'))
                        video_config = layer_config['video_config']


                        well_label = pos['name']
                        if position_source == 'from_labware':
                            custom_step = False
                        else:
                            custom_step = True

                        if zstack_slice in ("", None):
                            zstack_slice_label = -1
                        else:
                            zstack_slice_label = zstack_slice

                        if tile_label == "":
                            tile_group_id_label = -1
                        else:
                            tile_group_id_label = tile_group_id

                        if zstack_slice is None:
                            zstack_group_id_label = -1
                        else:
                            zstack_group_id_label = zstack_group_id
                        
                        step_name = common_utils.generate_default_step_name(
                            well_label=well_label,
                            color=layer_name,
                            z_height_idx=zstack_slice_label,
                            tile_label=tile_label,
                            objective_short_name=None,  # Can add this if needed
                            custom_name_prefix=None if not custom_step else well_label,
                        )

                        step_dict = cls._create_step_dict(
                            name=step_name,
                            x=x,
                            y=y,
                            z=z,
                            af=autofocus,
                            color=layer_name,
                            fc=false_color,
                            ill=illumination,
                            gain=gain,
                            auto_gain=auto_gain,
                            exp=exposure,
                            sum=sum,
                            objective=objective_id,
                            well=well_label,
                            tile=tile_label,
                            zslice=zstack_slice_label,
                            custom_step=custom_step,
                            tile_group_id=tile_group_id_label,
                            zstack_group_id=zstack_group_id_label,
                            acquire=layer_config['acquire'],
                            video_config=video_config,
                            stim_config=stim_config,
                        )
                        steps.append(step_dict)
                
                if zstack_slice is not None:
                    zstack_group_id += 1

            if tile_label != "":
                tile_group_id += 1

        steps_df = cls._create_empty_steps_df()
        steps_df = cls._add_steps_to_steps_df(
            steps_df=steps_df,
            new_steps = steps
        )

        config['steps'] = steps_df

        protocol = cls(
            tiling_configs_file_loc=tiling_configs_file_loc,
            config=config
        )

        # Validate step fields — reject protocol if any errors found
        validation_errors = protocol.validate_steps()
        if validation_errors:
            for err in validation_errors:
                logger.error(f"Protocol validation: {err}")
            raise ProtocolFormatError(
                f"Protocol has {len(validation_errors)} validation error(s): {validation_errors[0]}"
            )

        return protocol


    @classmethod
    def create_empty(
        cls,
        config: dict,
        tiling_configs_file_loc : pathlib.Path,
    ):
        
        tc = TilingConfig(
            tiling_configs_file_loc=tiling_configs_file_loc
        )
        
        labware_id = config['labware_id']
        objective_id = config['objective_id']
        zstack_params = {'range': 0, 'step_size': 0}
        use_zstacking = False
        tiling = tc.no_tiling_label()
        layer_configs = {}
        period = config['period']
        duration = config['duration']
        frame_dimensions = config['frame_dimensions']
        stim_config = {}

        input_config = {
            'labware_id': labware_id,
            'objective_id': objective_id,
            'zstack_params': zstack_params,
            'use_zstacking': use_zstacking,
            'tiling': tiling,
            'layer_configs': layer_configs,
            'period': period,
            'duration': duration,
            'frame_dimensions': frame_dimensions,
            'binning_size': config['binning_size'],
            'stim_config': stim_config,
        }

        return cls.from_config(
            input_config=input_config,
            tiling_configs_file_loc=tiling_configs_file_loc
        )


    @staticmethod
    def _create_step_dict(
        name,
        x,
        y,
        z,
        af,
        color,
        fc,
        ill,
        gain,
        auto_gain,
        exp,
        sum,
        objective,
        well,
        tile,
        zslice,
        custom_step,
        tile_group_id,
        zstack_group_id,
        acquire,
        video_config,
        stim_config
    ):
        return {
            "Name": name,
            "X": x,
            "Y": y,
            "Z": z,
            "Auto_Focus": af,
            "Color": color,
            "False_Color": fc,
            "Illumination": ill,
            "Gain": gain,
            "Auto_Gain": auto_gain,
            "Exposure": exp,
            "Sum": sum,
            "Objective": objective,
            "Well": well,
            "Tile": tile,
            "Z-Slice": zslice,
            "Custom Step": custom_step,
            "Tile Group ID": tile_group_id,
            "Z-Stack Group ID": zstack_group_id,
            "Acquire": acquire,
            "Video Config": copy.deepcopy(video_config),
            "Stim_Config": copy.deepcopy(stim_config),
        }
        

    """
    stim_config = {
        "Red": {
            "enabled": True,
            "illumination": 100,
            "frequency": 1,
            "pulse_width": 10,
            "pulse_count": 1,
        },
        "Green": {
            "enabled": True,
            "illumination": 100,
            "frequency": 1,
            "pulse_width": 10,
            "pulse_count": 1,
        },
        "Blue": {
            "enabled": True,
            "illumination": 100,
            "frequency": 1,
            "pulse_width": 10,
            "pulse_count": 1,
        }
    }

    """


    @staticmethod
    def _add_steps_to_steps_df(
        steps_df: pd.DataFrame,
        new_steps: list[dict]
    ):
        new_steps_df = pd.DataFrame(new_steps)
        steps_df = pd.concat([steps_df, new_steps_df], ignore_index=True).reset_index(drop=True)
        return steps_df


    @classmethod
    def from_file(
        cls,
        file_path: pathlib.Path,
        tiling_configs_file_loc : pathlib.Path | None
    ):
        """
        Returns Protocol object loaded from file on success
        Raises ProtocolFormatError on format issues
        """
        
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
        MAX_STEP_COUNT = 10_000

        config = {}

        # Check file size before reading
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"Protocol file exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB "
                f"({file_size:,} bytes). File may be corrupt."
            )

        # Filter out blank lines
        file_content = None
        fp = None

        try:
            with open(file_path, 'r') as fp_orig:
                file_data_lines = [line for line in fp_orig.readlines() if line.strip()]
                file_content = ''.join(file_data_lines)
                fp = io.StringIO(file_content)
        except Exception as e:
            logger.error(f"Error reading protocol file {file_path}: {e}")
            raise IOError(f"Error reading protocol file {file_path}") from e

        csvreader = csv.reader(fp, delimiter='\t')

        try:
            verify = next(csvreader)
        except StopIteration:
            logger.error(f"Protocol file {file_path} is empty or invalid.")
            raise ProtocolFormatError(f"Protocol file is empty or invalid.")
        
        if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
            raise ProtocolFormatError(f"Not a valid LumaViewPro Protocol")
        
        try:
            version_row = next(csvreader)
        except StopIteration:
            logger.error(f"Protocol file {file_path} is missing version information.")
            raise ProtocolFormatError(f"Protocol file is missing version information.")
        
        if version_row[0] != "Version":
            logger.error(f"Unable to load {file_path} which is missing 'Version' row.")
            raise ProtocolFormatError(f"Protocol format is missing 'Version' row.")

        try:
            config['version'] = int(version_row[1])
        except ValueError as ve:
            logger.error(f"Invalid 'Version' value in protocol file {file_path}. 'Version' must be an integer: {ve}")
            raise ProtocolFormatError(f"Invalid 'Version' value in protocol file {file_path}") from ve

        allowed = False
        if config['version'] == cls.CURRENT_VERSION:
            allowed = True

        elif (config['version'] in (2, 3, 4)) and (cls.CURRENT_VERSION == 5):
            allowed = True
                
        if not allowed:
            logger.error(f"Unable to load {file_path} which contains protocol version {config['version']}.\nPlease create a new protocol using this version of LumaViewPro.")
            raise ProtocolFormatError(f"Protocol version {config['version']} is not supported.")


        # Read Period
        try:
            period_row = next(csvreader)
            if period_row[0] != "Period":
                logger.error(f"Missing 'Period' row in protocol file {file_path}")
                raise ProtocolFormatError(f"Missing 'Period' row in protocol file")
            
            minutes = float(period_row[1])
            if minutes <= 0:
                logger.error(f"Invalid 'Period' value in protocol file {file_path}: must be > 0")
                raise ProtocolFormatError(f"Invalid 'Period' value in protocol file: must be > 0")
            
            config['period'] = datetime.timedelta(minutes=minutes)

        except StopIteration:
            logger.error(f"Missing 'Period' row in protocol file {file_path}")
            raise ProtocolFormatError(f"Missing 'Period' row in protocol file")
        
        except ValueError as ve:
            logger.error(f"Invalid 'Period' value in protocol file {file_path} 'Period' must be numeric: {ve}")
            raise ProtocolFormatError(f"Invalid 'Period' value in protocol file") from ve
        
        except ProtocolFormatError as pfe:
            raise pfe


        # Read Duration
        try:
            duration = next(csvreader)
            if duration[0] != "Duration":
                logger.error(f"Missing 'Duration' row in protocol file {file_path}")
                raise ProtocolFormatError(f"Missing 'Duration' row in protocol file")
            
            hours = float(duration[1])
            if hours <= 0:
                logger.error(f"Invalid 'Duration' value in protocol file {file_path}: must be > 0")
                raise ProtocolFormatError(f"Invalid 'Duration' value in protocol file: must be > 0")
            
            config['duration'] = datetime.timedelta(hours=hours)

        except StopIteration:
            logger.error(f"Missing 'Duration' row in protocol file {file_path}")
            raise ProtocolFormatError(f"Missing 'Duration' row in protocol file")
        
        except ValueError as ve:
            logger.error(f"Invalid 'Duration' value in protocol file {file_path}. 'Duration' must be numeric: {ve}")
            raise ProtocolFormatError(f"Invalid 'Duration' value in protocol file") from ve
        
        except ProtocolFormatError as pfe:
            raise pfe
        

        # Read Labware
        try:
            labware = next(csvreader)
            if labware[0] != "Labware":
                logger.error(f"Invalid 'Labware' row in protocol file {file_path}")
                raise ProtocolFormatError(f"Invalid 'Labware' row in protocol file")
            
            config['labware_id'] = labware[1]

        except StopIteration:
            logger.error(f"Missing 'Labware' row in protocol file {file_path}")
            raise ProtocolFormatError(f"Missing 'Labware' row in protocol file")
        
        except ProtocolFormatError as pfe:
            raise pfe

        # Optional fields not present in older/newer files
        # Ensure defaults to avoid KeyErrors
        try:
            next_row = next(csvreader)
            if next_row[0] == "Capture Root":
                config['capture_root'] = next_row[1]
            else:
                config['capture_root'] = ''
        except StopIteration:
            logger.error(f"Protocol file {file_path} is incomplete.")
            raise ProtocolFormatError(f"Protocol file is incomplete.")


        # Search for "Steps" to indicate start of steps
        if next_row[0] != "Steps": # Check to ensure compatibility with no capture_root field
            try:
                while True:
                    tmp = next(csvreader)
                    if len(tmp) == 0:
                        continue

                    if tmp[0] == "Steps":
                        break
            except StopIteration:
                logger.error(f"Missing 'Steps' section in protocol file {file_path}")
                raise ProtocolFormatError(f"Missing 'Steps' section in protocol file")
        
        table_lines = []
        for line in fp:
            table_lines.append(line)
        
        table_str = ''.join(table_lines)
        protocol_df = pd.read_csv(io.StringIO(table_str), sep='\t', lineterminator='\n').fillna('')
        protocol_df['Name'] = protocol_df['Name'].astype(str)

        if len(protocol_df) == 0:
            # Will create an empty protocol
            return False

        # Added in v3
        DEFAULT_VIDEO_CONFIG = {
            'fps': 5,
            'duration': 5
        }

        # Added in v4
        DEFAULT_SUM_CONFIG = 1

        # Added in v5
        DEFAULT_STIM_CONFIG = {
            "Red": {
                "enabled": False,
                "illumination": 100,
                "frequency": 1,
                "pulse_width": 10,
                "pulse_count": 100,
            },
            "Green": {
                "enabled": False,
                "illumination": 100,
                "frequency": 1,
                "pulse_width": 10,
                "pulse_count": 100,
            },
            "Blue": {
                "enabled": False,
                "illumination": 100,
                "frequency": 1,
                "pulse_width": 10,
                "pulse_count": 100,
            }
        }

        if (config['version'] < cls.CURRENT_VERSION):
            logger.info(f"Converting loaded protocol from {config['version']} to {cls.CURRENT_VERSION}")

        if (config['version'] == 2) and (cls.CURRENT_VERSION >= 4):
            protocol_df['Acquire'] = "image"
            # Ensure each row gets its own Video Config dict
            protocol_df['Video Config'] = copy.deepcopy(DEFAULT_VIDEO_CONFIG)
        else:

            # Convert Video Config strings per step to dictionary
            try:
                protocol_df['Video Config'] = protocol_df.apply(lambda x: json.loads(x['Video Config']), axis=1)
            except Exception as ex:
                logger.error(f"Unable to parse video config, using default instead: {ex}")
                protocol_df['Video Config'] = copy.deepcopy(DEFAULT_VIDEO_CONFIG)


        if (config['version'] in (2,3,)) and (cls.CURRENT_VERSION >= 4):
            protocol_df['Sum'] = DEFAULT_SUM_CONFIG

        if (config['version'] in (2,3,4)) and (cls.CURRENT_VERSION >= 5):
            # Ensure each row gets its own Stim Config dict
            protocol_df['Stim_Config'] = copy.deepcopy(DEFAULT_STIM_CONFIG)
        else:
            # Convert Stim Config strings per step to dictionary
            try:
                protocol_df['Stim_Config'] = protocol_df.apply(lambda x: json.loads(x['Stim_Config']), axis=1)
            except Exception as ex:
                logger.error(f"Unable to parse stim config, using default instead: {ex}")
                protocol_df['Stim_Config'] = copy.deepcopy(DEFAULT_STIM_CONFIG)

        if config['version'] in (2, 3, 4, 5):
            protocol_df['Step Index'] = protocol_df.index

            # Extract tiling config from step names
            tc = TilingConfig(
                tiling_configs_file_loc=tiling_configs_file_loc
            )
            
            if len(protocol_df) > 0:
                
                config['tiling'] = tc.determine_tiling_label_from_names(
                    names=protocol_df['Name'].to_list()
                )
            else:
                config['tiling'] = tc.no_tiling_label()

        config['steps'] = protocol_df
        config['custom_step_count'] = 0

        if len(protocol_df) > MAX_STEP_COUNT:
            raise ValueError(
                f"Protocol contains {len(protocol_df):,} steps, exceeding the maximum of "
                f"{MAX_STEP_COUNT:,}. File may be corrupt."
            )

        return cls(
            tiling_configs_file_loc=tiling_configs_file_loc,
            config=config
        )
    

    def mark_zstack_starts_and_ends(self) -> None:
        df = self.steps().copy()
        df['Z-Stack Group Index'] = df.groupby(by=['Z-Stack Group ID']).cumcount()
        df['First Z'] = df['Z-Stack Group Index'].apply(lambda x: True if x==0 else False)
        df['Last Z'] = df.groupby(by=['Z-Stack Group ID'])['Z-Stack Group Index'].transform('max') == df['Z-Stack Group Index']
        df = df.drop(columns=['Z-Stack Group Index'])
        self._config['steps'] = df


    def remove_zstack_starts_and_ends(self) -> None:
        df = self.steps()
        df = df.drop(columns=['First Z','Last Z'])
        self._config['steps'] = df


    def has_zstacks(self) -> bool:
        max_group_id = self.steps()['Z-Stack Group ID'].max()
        if max_group_id > -1:
            return True
        else:
            return False
    
        

    @classmethod
    def extract_data_from_step_name(cls, s):
        result = cls.STEP_NAME_PATTERN.match(string=s['name'])
        if result is None:
            return s
        
        details = result.groupdict()

        if 'well_label' in details:
            s['well'] = details['well_label']
        
        if ('z_slice' in details) and (details['z_slice'] is not None):
            s['z_slice'] = details['z_slice']
        else:
            s['z_slice'] = None

        if 'tile_label' in details:
            s['tile'] = details['tile_label']

        return s


if __name__ == "__main__":
    protocol = Protocol.from_file(file_path=pathlib.Path("modules/protocol_test6.tsv"))


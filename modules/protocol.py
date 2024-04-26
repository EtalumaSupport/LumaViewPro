

import csv
import datetime
import io
import numpy as np
import pandas as pd
import os
import pathlib
import re

from lvp_logger import logger

import modules.color_channels as color_channels
import modules.common_utils as common_utils
import modules.labware_loader as labware_loader
from modules.tiling_config import TilingConfig
from modules.objectives_loader import ObjectiveLoader


class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"
    COLUMNS_V1 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Channel', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective']
    COLUMNS_V2 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID']
    # COLUMNS_V2 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Step Index', 'Color', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID']
    CURRENT_VERSION = 2
    CURRENT_COLUMNS = COLUMNS_V2
    STEP_NAME_PATTERN = re.compile(r"^(?P<well_label>[A-Z][0-9]+)(_(?P<color>(Blue|Green|Red|BF|EP|PC)))(_T(?P<tile_label>[A-Z][0-9]+))?(_Z(?P<z_slice>[0-9]+))?(_([0-9]*))?(.tif[f])?$")
    
    def __init__(self, config=None):

        self._objective_loader = ObjectiveLoader()

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
    
    # @staticmethod
    # def _get_column_index_map(column_names: list) -> dict[str, int]:
    #     return {column_name: idx for idx, column_name in enumerate(column_names)}
    

    # Not used yet
    def to_file(self, file_path: pathlib.Path):
        raise NotImplementedError(f"Not implemented")
    
        period_minutes = self._config['period'].total_seconds() / 60.0
        duration_hours = self._config['duration'].total_seconds() / 3600.0

        steps = self._config['steps'][['name','x','y','z','auto_focus','false_color','illumination','gain','auto_gain','exposure','well','tile','z_slice','step_index','color']]
        with open(file_path, 'w') as fp:
            csvwriter = csv.writer(fp, delimiter='\t', lineterminator='\n')
            csvwriter.writerow(['LumaViewPro Protocol'])
            csvwriter.writerow(['Version', self.CURRENT_VERSION])
            csvwriter.writerow(['Period', period_minutes])
            csvwriter.writerow(['Duration', duration_hours])
            csvwriter.writerow(['Labware', self._config['labware']])
            csvwriter.writerow(['Tiling', self._config['tiling']])
            csvwriter.writerow(self.CURRENT_COLUMNS)

            for _, row in steps.iterrows():
                csvwriter.writerow(row)


    @staticmethod
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
                ("Objective", str),
                ("Well", str),
                ("Tile", str),
                ("Z-Slice", int),
                ("Custom Step", bool),
                ("Tile Group ID", int),
                ("Z-Stack Group ID", int)
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df
    

    def num_steps(self) -> int:
        return len(self.config['steps'])
    

    def steps(self) -> pd.DataFrame:
        return self.config['steps']
    

    def delete_step(self, step_idx: int):
        num_steps = self.num_steps()

        if num_steps < 1:
            return
        
        if step_idx >= self.num_steps():
            raise Exception(f"Cannot delete step idx {step_idx}. Protocol only has {self.num_steps()}.")
        
        self.config['steps'].drop(index=step_idx, axis=0, inplace=True)
        self.config['steps'].reset_index(drop=True, inplace=True)
    

    def modify_step_z_height(
        self,
        step_idx: int,
        z: float
    ):
        self.config['steps'].at[step_idx, "Z"] = z
        
        
    def modify_step(
        self,
        step_idx: int,
        step_name: str,
        layer: str,
        layer_config: dict,
        plate_position: dict,
        objective_id: str,
    ):
        def _validate_inputs():
            if step_idx < 0:
                raise Exception(f"Step idx must be > 0")
            
            if step_idx >= self.num_steps():
                raise Exception(f"Cannot modify step idx {step_idx}. Protocol only has {self.num_steps()}.")
            
        _validate_inputs()

        self.config['steps'].at[step_idx, "Name"] = step_name
        self.config['steps'].at[step_idx, "X"] = plate_position['x']
        self.config['steps'].at[step_idx, "Y"] = plate_position['y']
        self.config['steps'].at[step_idx, "Z"] = plate_position['z']

        self.config['steps'].at[step_idx, "Auto_Focus"] = layer_config['autofocus']
        self.config['steps'].at[step_idx, "Color"] = layer
        self.config['steps'].at[step_idx, "False_Color"] = layer_config['false_color']
        self.config['steps'].at[step_idx, "Illumination"] = layer_config['illumination']
        self.config['steps'].at[step_idx, "Gain"] = layer_config['gain']
        self.config['steps'].at[step_idx, "Auto_Gain"] = layer_config['auto_gain']
        self.config['steps'].at[step_idx, "Exposure"] = layer_config['exposure']
        self.config['steps'].at[step_idx, "Objective"] = objective_id


    def insert_step(
        self,
        step_name: str | None,
        layer: str,
        layer_config: dict,
        plate_position: dict,
        objective_id: str,
        before_step: int | None = 0,
        after_step: int | None = None,
    ) -> str:
        
        def _validate_inputs():
            if (before_step is None) and (after_step is None):
                raise Exception(f"Must specify after_step or before_step")
            
            if (before_step is not None) and (after_step is not None):
                raise Exception(f"Must specify only after_step or before_step, not both")
            
            if before_step < 0:
                raise Exception(f"before_step cannot be <0")
            
        _validate_inputs()
        
        if step_name is None:
            step_name = f"custom{self.config['custom_step_count']}"
            self.config['custom_step_count'] += 1

        well = ""
        tile = "" # Manually inserted step is not a tile
        zslice = -1
        custom_step = True
        tile_group_id = -1
        zstack_group_id = -1

        step_dict = self.create_step_dict(
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
            objective=objective_id,
            well=well,
            tile=tile,
            zslice=zslice,
            custom_step=custom_step,
            tile_group_id=tile_group_id,
            zstack_group_id=zstack_group_id
        )

        if before_step is not None:
            pos_index = before_step-0.5
        else:
            pos_index = after_step+0.5

        line = pd.DataFrame(data=step_dict, index=[pos_index])
        self.config['steps'] = pd.concat([self.config['steps'], line], ignore_index=False, axis=0)
        self.config['steps'] = self.config['steps'].sort_index().reset_index(drop=True)

        return step_name
    

    def step(
        self,
        idx: int
    ):
        def _validate():
            if idx < 0:
                raise Exception(f"Step ndex cannot be < 0")
            
            if idx >= self.num_steps():
                raise Exception(f"Step idx {idx} does not exist. Protocol only has {self.num_steps()}.")

        _validate(

        )            
        return self.config['steps'].iloc[idx]


    @classmethod
    def from_config(
        cls,
        input_config: dict,
        tiling_configs_file_loc : pathlib.Path
    ):
        
        tiling_config = TilingConfig(
            tiling_configs_file_loc=tiling_configs_file_loc
        )

        labware = input_config['labware']
        objective_id = input_config['objective']
        zstack_positions = input_config['zstack_positions']
        tiling = input_config['tiling']
        layer_configs = input_config['layer_configs']
        period = input_config['period']
        duration = input_config['duration']
        frame = input_config['frame']

        objective_loader = ObjectiveLoader()
        objective = objective_loader.get_objective_info(objective_id=objective_id)

        tiles = tiling_config.get_tile_centers(
            config_label=tiling,
            focal_length=objective['focal_length'],
            frame_size=frame,
            fill_factor=TilingConfig.DEFAULT_FILL_FACTORS['position']
        )

        config = {
            'version': 2,
            'period': period,
            'duration': duration,
            'labware': labware,
            'tiling': tiling,
            'custom_step_count': 0,
        }
        
        wellplate_loader = labware_loader.WellPlateLoader()
        labware_obj = wellplate_loader.get_plate(plate_key=labware)
        labware_obj.set_positions()

        tile_group_id = 0
        zstack_group_id = 0
        # custom_step_count = 0
        steps = []

         # Iterate through all the positions in the scan
        for pos in labware_obj.pos_list:
            for tile_label, tile_position in tiles.items():
                for zstack_slice, zstack_position in zstack_positions.items():
                    for layer_name, layer_config in layer_configs.items():
                        if layer_config['acquire'] == False:
                            continue
                        
                        x = round(pos[0] + tile_position["x"]/1000, common_utils.max_decimal_precision('x')) # in 'plate' coordinates
                        y = round(pos[1] + tile_position["y"]/1000, common_utils.max_decimal_precision('y')) # in 'plate' coordinates

                        if zstack_slice is None:
                            z = layer_config['focus']
                        else:
                            z = zstack_position

                        z = round(z, common_utils.max_decimal_precision('z'))

                        autofocus = layer_config['autofocus']
                        false_color = layer_config['false_color']
                        illumination = round(layer_config['illumination'], common_utils.max_decimal_precision('illumination'))
                        gain = round(layer_config['gain'], common_utils.max_decimal_precision('gain'))
                        auto_gain = common_utils.to_bool(layer_config['auto_gain'])
                        exposure = round(layer_config['exposure'], common_utils.max_decimal_precision('exposure'))
                        custom_step = False
                        well_label = labware.get_well_label(x=pos[0], y=pos[1])

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
                        
                        step_dict = cls._create_step_dict(
                            name="",
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
                            objective=objective_id,
                            well=well_label,
                            tile=tile_label,
                            zslice=zstack_slice_label,
                            custom_step=custom_step,
                            tile_group_id=tile_group_id_label,
                            zstack_group_id=zstack_group_id_label
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

        return Protocol(
            config=config
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
        objective,
        well,
        tile,
        zslice,
        custom_step,
        tile_group_id,
        zstack_group_id
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
            "Objective": objective,
            "Well": well,
            "Tile": tile,
            "Z-Slice": zslice,
            "Custom Step": custom_step,
            "Tile Group ID": tile_group_id,
            "Z-Stack Group ID": zstack_group_id
        }


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
        
        config = {}

        # Filter out blank lines
        file_content = None
        fp = None
        with open(file_path, 'r') as fp_orig:
            file_data_lines = [line for line in fp_orig.readlines() if line.strip()]
            file_content = ''.join(file_data_lines)
            fp = io.StringIO(file_content)

        csvreader = csv.reader(fp, delimiter='\t')
        verify = next(csvreader)
        if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
            raise Exception(f"Not a valid LumaViewPro Protocol")
        
        version_row = next(csvreader)
        if version_row[0] != "Version":
            logger.error(f"Unable to load {file_path} which contains an older protocol format that is no longer supported.\nPlease create a new protocol using this version of LumaViewPro.")
            return

        config['version'] = int(version_row[1])
        if config['version'] != 2:
            logger.error(f"Unable to load {file_path} which contains a protocol version that is not supported.\nPlease create a new protocol using this version of LumaViewPro.")
            return

        period_row = next(csvreader)
        config['period'] = datetime.timedelta(minutes=float(period_row[1]))
        
        duration = next(csvreader)
        config['duration'] = datetime.timedelta(hours=float(duration[1]))

        labware = next(csvreader)
        config['labware'] = labware[1]

        # Search for "Steps" to indicate start of steps
        while True:
            tmp = next(csvreader)
            if len(tmp) == 0:
                continue

            if tmp[0] == "Steps":
                break
        
        table_lines = []
        for line in fp:
            table_lines.append(line)
        
        table_str = ''.join(table_lines)
        protocol_df = pd.read_csv(io.StringIO(table_str), sep='\t', lineterminator='\n').fillna('')

        if config['version'] == 1:
            protocol_df['Step Index'] = protocol_df.index

            if not use_version_1a:
                protocol_df['color'] = protocol_df.apply(lambda s: color_channels.ColorChannel(s['channel']).name, axis=1)
                protocol_df = protocol_df.drop(columns=['channel'])

            # Extract tiling config from step names
            tc = tiling_config.TilingConfig(
                tiling_configs_file_loc=tiling_configs_file_loc
            )
            config['tiling'] = tc.determine_tiling_label_from_names(
                names=protocol_df['Name'].to_list()
            )

            if not use_version_1a:
                protocol_df = protocol_df.apply(cls.extract_data_from_step_name, axis=1)
                protocol_df['Custom Step'] = False

        # Index and build a map of Z-heights. Indicies will be used in step/file naming
        # Only build the height map if we have at least 2 heights in the protocol.
        # Otherwise, we don't want "_Z<slice>" added to the name
        # config['z_height_map'] = cls._build_z_height_map(values=steps_df['z'])
        
        elif config['version'] == 2:
            protocol_df['Step Index'] = protocol_df.index

            # Extract tiling config from step names
            tc = tiling_config.TilingConfig(
                tiling_configs_file_loc=tiling_configs_file_loc
            )
            config['tiling'] = tc.determine_tiling_label_from_names(
                names=protocol_df['Name'].to_list()
            )

        config['steps'] = protocol_df
        config['custom_step_count'] = 0 # TODO determine custom step count

        return Protocol(
            config=config
        )
    

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
    
    
    def get_tile_groups(self):
        steps = self._config['steps']
        tile_groups = steps.groupby(
            by=[
                'Tile Group ID',
                'Well',
                'Color',
                'Z-Slice',
                'Objective',
                # 'Z-Stack Group ID',
                'Custom Step'
            ],
            dropna=False
        )

        tile_dict = {}
        for idx, (_, group_info) in enumerate(tile_groups):
            tile_dict[idx] = group_info[['Name','X','Y','Color','Well','Tile','Step Index','Z-Slice','Objective','Tile Group ID','Z-Stack Group ID','Custom Step']]
        
        return tile_dict
    

if __name__ == "__main__":
    protocol = Protocol.from_file(file_path=pathlib.Path("modules/protocol_test6.tsv"))
    tile_groups = protocol.get_tile_groups()



import csv
import datetime
import numpy as np
import pandas as pd
import os
import pathlib
import re

import modules.color_channels as color_channels
import modules.common_utils as common_utils
import modules.tiling_config as tiling_config


class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"
    COLUMNS_V1 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Channel', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective']
    COLUMNS_V2 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Step Index', 'Color']
    CURRENT_VERSION = 2
    CURRENT_COLUMNS = COLUMNS_V2
    STEP_NAME_PATTERN = re.compile(r"^(?P<well_label>[A-Z][0-9]+)(_(?P<color>(Blue|Green|Red|BF|EP|PC)))(_T(?P<tile_label>[A-Z][0-9]+))?(_Z(?P<z_slice>[0-9]+))?(_([0-9]*))?(.tif[f])?$")
    
    def __init__(self, config=None):

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
    
    @staticmethod
    def _get_column_index_map(column_names: list) -> dict[str, int]:
        return {column_name: idx for idx, column_name in enumerate(column_names)}
    

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



    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        
        config = {}
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t')
            verify = next(csvreader)
            if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
                raise Exception(f"Not a valid LumaViewPro Protocol")
            
            version_row = next(csvreader)
            if version_row[0] == "Version":
                config['version'] = int(version_row[1])
                period_row = next(csvreader)
            else:
                config['version'] = 1 # Version 1 did not include versioning
                period_row = version_row

            config['period'] = datetime.timedelta(minutes=float(period_row[1]))
            
            duration = next(csvreader)
            config['duration'] = datetime.timedelta(hours=float(duration[1]))

            labware = next(csvreader)
            config['labware'] = labware[1]

            if config['version'] >= 2:
                tiling_config_row = next(csvreader)
                config['tiling'] = tiling_config_row[1]

            # orig_labware = labware
            # labware_valid, labware = self._validate_labware(labware=orig_labware)
            # if not labware_valid:
                # raise Exception(f"Invalid labware: {labware}")
                # logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

            columns = next(csvreader)
            column_map = cls._get_column_index_map(column_names=columns)

            steps = []
            for row in csvreader:
                step = {
                    'name': row[column_map['Name']],
                    'x': common_utils.to_float(val=row[column_map['X']]),
                    'y': common_utils.to_float(val=row[column_map['Y']]),
                    'z': common_utils.to_float(val=row[column_map['Z']]),
                    'auto_focus': common_utils.to_bool(val=row[column_map['Auto_Focus']]),
                    # 'channel': int(float(row[column_map['Channel']])),
                    'false_color': common_utils.to_bool(val=row[column_map['False_Color']]),
                    'illumination': common_utils.to_float(val=row[column_map['Illumination']]),
                    'gain': common_utils.to_float(val=row[column_map['Gain']]),
                    'auto_gain': common_utils.to_bool(val=row[column_map['Auto_Gain']]),
                    'exposure': common_utils.to_float(val=row[column_map['Exposure']]),
                    'objective': row[column_map['Objective']]
                }

                if config['version'] == 1:
                    step['channel'] = common_utils.to_int(val=row[column_map['Channel']])

                if config['version'] >= 2:
                    step['well'] = row[column_map['Well']]
                    step['tile'] = row[column_map['Tile']]
                    step['z_slice'] = row[column_map['Z-Slice']]
                    step['step_index'] = row[column_map['Step Index']]
                    step['color'] = row[column_map['Color']]

                steps.append(step) 

        steps_df = pd.DataFrame(steps)

        if config['version'] == 1:
            steps_df['step_index'] = steps_df.index
            steps_df['color'] = steps_df.apply(lambda s: color_channels.ColorChannel(s['channel']).name, axis=1)
            steps_df = steps_df.drop(columns=['channel'])

            # Extract tiling config from step names 
            tc = tiling_config.TilingConfig()
            config['tiling'] = tc.determine_tiling_label_from_names(
                names=steps_df['name'].to_list()
            )

            steps_df = steps_df.apply(cls.extract_data_from_step_name, axis=1)

        # Index and build a map of Z-heights. Indicies will be used in step/file naming
        # Only build the height map if we have at least 2 heights in the protocol.
        # Otherwise, we don't want "_Z<slice>" added to the name
        # config['z_height_map'] = cls._build_z_height_map(values=steps_df['z'])
        config['steps'] = steps_df

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
                'well',
                'color',
                'z_slice',
                'objective'
            ]
        )

        tile_dict = {}
        for idx, (_, group_info) in enumerate(tile_groups):
            tile_dict[idx] = group_info[['name','x','y','color','well','tile','step_index','z_slice','objective']]
        
        return tile_dict
    

if __name__ == "__main__":
    protocol = Protocol.from_file(file_path=pathlib.Path("modules/protocol_test6.tsv"))
    tile_groups = protocol.get_tile_groups()

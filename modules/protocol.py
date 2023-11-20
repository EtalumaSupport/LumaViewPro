

import csv
import datetime
import numpy as np
import pandas as pd
import os
import pathlib
import re

import modules.color_channels as color_channels
import modules.tiling_config as tiling_config
import modules.protocol_step as protocol_step


class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"
    STEP_NAME_PATTERN = re.compile(r"^(?P<well_label>[A-Z][0-9]+)_(?P<color>(Blue|Green|Red|BF|EP|PC))_(Z(?P<z_slice>[0-9]+)_)?([0-9]*_)?T(?P<tile_label>[A-Z][0-9]+)(.tif[f])?$")
    
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
            

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        
        config = {}
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t')
            verify = next(csvreader)
            if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
                raise Exception(f"Not a valid LumaViewPro Protocol")
            
            period = next(csvreader)        
            config['period'] = datetime.timedelta(minutes=float(period[1]))
            
            duration = next(csvreader)
            config['duration'] = datetime.timedelta(hours=float(duration[1]))

            labware = next(csvreader)
            config['labware'] = labware[1]

            # orig_labware = labware
            # labware_valid, labware = self._validate_labware(labware=orig_labware)
            # if not labware_valid:
                # raise Exception(f"Invalid labware: {labware}")
                # logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

            columns = next(csvreader)
            column_map = cls._get_column_index_map(column_names=columns)

            steps = []
            for row in csvreader:
                steps.append(
                    {
                        'name': row[column_map['Name']],
                        'x':float(row[column_map['X']]),
                        'y': float(row[column_map['Y']]),
                        'z': float(row[column_map['Z']]),
                        'auto_focus': bool(float(row[column_map['Auto_Focus']])),
                        'channel': int(float(row[column_map['Channel']])),
                        'false_color': bool(float(row[column_map['False_Color']])),
                        'illumination': float(row[column_map['Illumination']]),
                        'gain': float(row[column_map['Gain']]),
                        'auto_gain': bool(float(row[column_map['Auto_Gain']])),
                        'exposure': float(row[column_map['Exposure']])
                    }
                )

        steps_df = pd.DataFrame(steps)

        steps_df['color'] = steps_df.apply(lambda s: color_channels.ColorChannel(s['channel']).name, axis=1)

        # Index and build a map of Z-heights. Indicies will be used in step/file naming
        # Only build the height map if we have at least 2 heights in the protocol.
        # Otherwise, we don't want "_Z<slice>" added to the name
        config['z_height_map'] = cls._build_z_height_map(values=steps_df['z'])

        # Extract tiling config from step names 
        tc = tiling_config.TilingConfig()
        config['tiling_config_label'] = tc.determine_tiling_label_from_names(
             names=steps_df['name'].to_list()
        )

        steps_df = steps_df.apply(cls.extract_data_from_step_name, axis=1)
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
            s['z_slice'] = -1

        if 'tile_label' in details:
            s['tile'] = details['tile_label']

        return s
    
    
    def get_tile_groups(self):
        steps = self._config['steps']
        tile_groups = steps.groupby(
            by=[
                'well',
                'color',
                'z'
            ]
        )

        tile_dict = {}
        for idx, (group_name, group_info) in enumerate(tile_groups):
            tile_dict[idx] = group_info[['name','x','y','color','well','tile']]
        
        return tile_dict
    

    
        


if __name__ == "__main__":
    protocol = Protocol.from_file(file_path=pathlib.Path("modules/protocol_test6.tsv"))
    tile_groups = protocol.get_tile_groups()
    print("Done")

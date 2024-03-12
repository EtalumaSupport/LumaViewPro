

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
import modules.tiling_config as tiling_config


class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"
    COLUMNS_V1 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Channel', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective']
    COLUMNS_V2 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID']
    # COLUMNS_V2 = ['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure', 'Objective', 'Well', 'Tile', 'Z-Slice', 'Step Index', 'Color', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID']
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



    @classmethod
    def from_file(cls, file_path: pathlib.Path, tiling_configs_file_loc : pathlib.Path | None):
        
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

        # if config['version'] >= 2:
        #     tiling_config_row = next(csvreader)
        #     config['tiling'] = tiling_config_row[1]

        # empty = next(csvreader)

        # Search for "Steps" to indicate start of steps
        while True:
            tmp = next(csvreader)
            if len(tmp) == 0:
                continue

            if tmp[0] == "Steps":
                break
        

        # orig_labware = labware
        # labware_valid, labware = self._validate_labware(labware=orig_labware)
        # if not labware_valid:
            # raise Exception(f"Invalid labware: {labware}")
            # logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

        table_lines = []
        for line in fp:
            table_lines.append(line)
        
        table_str = ''.join(table_lines)
        protocol_df = pd.read_csv(io.StringIO(table_str), sep='\t', lineterminator='\n').fillna('')

        #     columns = next(csvreader)
        #     column_map = cls._get_column_index_map(column_names=columns)

            # Check if version is 1A (Extra columns added)
            # use_version_1a = True
            # if config['version'] == 1:

            #     for column in ('Color', 'Well', 'Tile', 'Z-Slice', 'Custom Step', 'Tile Group ID', 'Z-Stack Group ID'):
            #         if column not in protocol_df.columns:
            #             use_version_1a = False

        #     steps = []
        #     for row in csvreader:
        #         step = {
        #             'name': row[column_map['Name']],
        #             'x': round(common_utils.to_float(val=row[column_map['X']]), common_utils.max_decimal_precision('x')),
        #             'y': round(common_utils.to_float(val=row[column_map['Y']]), common_utils.max_decimal_precision('y')),
        #             'z': round(common_utils.to_float(val=row[column_map['Z']]), common_utils.max_decimal_precision('z')),
        #             'auto_focus': common_utils.to_bool(val=row[column_map['Auto_Focus']]),
        #             # 'channel': int(float(row[column_map['Channel']])),
        #             'false_color': common_utils.to_bool(val=row[column_map['False_Color']]),
        #             'illumination': round(common_utils.to_float(val=row[column_map['Illumination']]), common_utils.max_decimal_precision('illumination')),
        #             'gain': round(common_utils.to_float(val=row[column_map['Gain']]), common_utils.max_decimal_precision('gain')),
        #             'auto_gain': common_utils.to_bool(val=row[column_map['Auto_Gain']]),
        #             'exposure': round(common_utils.to_float(val=row[column_map['Exposure']]), common_utils.max_decimal_precision('exposure')),
        #             'objective': row[column_map['Objective']]
        #         }

        #         if config['version'] == 1:
        #             if not use_version_1a:
        #                 step['channel'] = common_utils.to_int(val=row[column_map['Channel']])

        #             if use_version_1a:
        #                 step['color'] = row[column_map['Color']]
        #                 step['well'] = row[column_map['Well']]
        #                 step['tile'] = row[column_map['Tile']]
        #                 step['z_slice'] = row[column_map['Z-Slice']]
        #                 step['custom_step'] = row[column_map['Custom Step']]
        #                 step['tile_group_id'] = row[column_map['Tile Group ID']]
        #                 step['zstack_group_id'] = row[column_map['Z-Stack Group ID']]

        #         if config['version'] >= 2:
        #             step['well'] = row[column_map['Well']]
        #             step['tile'] = row[column_map['Tile']]
        #             step['z_slice'] = row[column_map['Z-Slice']]
        #             step['step_index'] = row[column_map['Step Index']]
        #             step['color'] = row[column_map['Color']]
        #             step['custom_step'] = row[column_map['Custom Step']]
        #             step['tile_group_id'] = row[column_map['Tile Group ID']]
        #             step['zstack_group_id'] = row[column_map['Z-Stack Group ID']]

        #         steps.append(step) 

        # steps_df = pd.DataFrame(steps)

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
                'Z-Stack Group ID',
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

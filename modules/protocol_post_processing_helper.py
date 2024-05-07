
import os
import pathlib

import pandas as pd

import modules.artifact_locations as artifact_locations
import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


class ProtocolPostProcessingHelper:

    def __init__(self):
        self._name = self.__class__.__name__


    @staticmethod
    def _get_image_filenames_from_folder(
        path: pathlib.Path, 
        exclude_subpaths: list = [],
        include_subpaths: list = [],
    ) -> dict[str, list[pathlib.Path]]:
        
        raw_image_names = []
        post_image_names = []
        raw_image_dirs = [
            '.',
            *common_utils.get_layers()
        ]

        if (len(include_subpaths) != 0) and (len(exclude_subpaths) != 0):
            raise Exception(f"Specify only include_subpaths OR exclude_subpaths. Not both.")
        
        tiff_images = path.rglob('*.tif[f]')
        ome_tiff_images = path.rglob('*.ome.tif[f]')
        images = []
        images.extend(tiff_images)
        images.extend(ome_tiff_images)
        for image in images:
            image_name = pathlib.Path(os.path.relpath(image, path))
            parent_dir = str(image_name.parent)

            if len(exclude_subpaths) > 0 and (parent_dir in exclude_subpaths):
                continue

            elif len(include_subpaths) > 0 and (parent_dir not in include_subpaths):
                continue
            
            if parent_dir not in raw_image_dirs:
                post_image_names.append(image_name)
            else:
                raw_image_names.append(image_name)

        return {
            'raw': raw_image_names,
            'post': post_image_names,
        }


    @staticmethod
    def generate_output_dir_name(record: pd.Series) -> pathlib.Path:
        # Filter to only the true values
        record = record[record==True]

        # Get the post-processing function names, in alphabetical order
        used_functions = sorted(record.keys().to_list())

        return pathlib.Path('-'.join(used_functions))



    def _find_protocol_tsvs(self, path: pathlib.Path) -> dict[str, pathlib.Path] | None:

        # If provided a file, change to the parent folder
        try:
            if not path.is_dir():
                path = path.parent
        except:
            return None

        loc_data = {}

        # Search for the protocol execution record TSV in the current directory and the parent directory
        protocol_execution_record_filename = ProtocolExecutionRecord.DEFAULT_FILENAME
        protocol_execution_record_file_loc = path / protocol_execution_record_filename
        if protocol_execution_record_file_loc.is_file():
            protocol_root_dir = path
        else:
            try:
                protocol_execution_record_file_loc = path.parent / protocol_execution_record_filename
                if protocol_execution_record_file_loc.is_file():
                    protocol_root_dir = path.parent
                else:
                    return None
            except:
                return None
            
        loc_data['protocol_root_dir'] = protocol_root_dir
        loc_data['protocol_execution_record'] = protocol_execution_record_file_loc
        protocol_execution_record = ProtocolExecutionRecord.from_file(file_path=protocol_execution_record_file_loc)
            
        # Search for the post-processing record TSV
        post_record_filename = ProtocolPostRecord.DEFAULT_FILENAME
        post_record_file_loc = protocol_root_dir / post_record_filename
        if post_record_file_loc.is_file():
            loc_data['protocol_post_record'] = post_record_file_loc
        else:
            loc_data['protocol_post_record'] = None

        # Search for the protocol TSV
        protocol_file_relative_loc = protocol_execution_record.protocol_file_loc()
        protocol_file_loc = protocol_root_dir / protocol_file_relative_loc
        if not protocol_file_loc.is_file():
            return None
        
        loc_data['protocol'] = protocol_file_loc
       
        return loc_data
 

    def _get_raw_images_df(
        self,
        image_names: list,
        protocol: Protocol,
        protocol_execution_record: ProtocolExecutionRecord,
    ) -> pd.DataFrame | None:
                
        image_data = []
        for image_name in image_names:
            file_data = protocol_execution_record.get_data_from_filename(file_path=image_name)
            if file_data is None:
                logger.warning(f"No info found in protocol execution record for {image_name}")
                continue

            step_idx = file_data['Step Index']
            step = protocol.step(idx=step_idx)

            image_data.append(
                {
                    'Filepath': image_name,
                    'Name': step['Name'],
                    'Scan Count': file_data['Scan Count'],
                    'Step Index': step_idx,
                    'X': step['X'],
                    'Y': step['Y'],
                    'Z': step['Z'],
                    'Z-Slice': step['Z-Slice'],
                    'Well': step['Well'],
                    'Color': step['Color'],
                    'Objective': step['Objective'],
                    'Tile': step['Tile'],
                    'Tile Group ID': step['Tile Group ID'],
                    'Z-Stack Group ID': step['Z-Stack Group ID'],
                    'Custom Step': step['Custom Step'],
                    'Timestamp': file_data['Timestamp'],
                }
            )

        df = pd.DataFrame(image_data)
        df = df.fillna('')

        return df
    

    def _get_post_images_df(
        self,
        image_names: list[pathlib.Path],
        protocol_post_record: ProtocolPostRecord,
    ) -> pd.DataFrame | None:
        
        df = protocol_post_record.records()
        if len(df) == 0:
            return df

        # Filter out any images that are missing from the filesystem
        # This is not strictly needed since the following filter using 'image_names'
        # will also inherently remove non-existent files
        df = df[df['File Exists'] == True]

        # Filter out any images that are not path of the selected images provided
        df = df[df['Filepath'].isin(image_names)]

        return df


    @staticmethod
    def _add_zproject_group_index(df: pd.DataFrame) -> pd.DataFrame:
        df['Z-Project Group Index'] = df.groupby(
            by=[
                'Scan Count',
                'Well',
                'Color',
                'Objective',
                'X',
                'Y',
                'Tile',
                'Custom Step'
            ],
            dropna=False
        ).ngroup()
        return df


    def load_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
    ) -> dict:
        selected_path = pathlib.Path(path)
        logger.info(f'{self._name}: Loading folder {selected_path}')

        protocol_tsvs = self._find_protocol_tsvs(path=selected_path)

        if protocol_tsvs is None:
            logger.error(f"{self._name}: Protocol and/or protocol record not found in folder")
            return {
                'status': False,
                'message': 'Protocol and/or Protocol Record not found in folder'
            }
        
        root_path = protocol_tsvs['protocol_root_dir']
                
        # Special handling for logging this since it may be None or a pathlib file
        protocol_post_record_str = "None" if protocol_tsvs['protocol_post_record'] is None else protocol_tsvs['protocol_post_record'].name

        logger.info(f"""{self._name}: Found ->
            Selected dir:                      {selected_path}
            Protocol root dir:                 {root_path}
            Protocol:                          {protocol_tsvs['protocol'].name}
            Protocol execution record:         {protocol_tsvs['protocol_execution_record'].name}
            Protocol post-processing metadata: {protocol_post_record_str}
        """)

        protocol = Protocol.from_file(
            file_path=protocol_tsvs['protocol'],
            tiling_configs_file_loc=tiling_configs_file_loc
        )

        if protocol is None:
            logger.error(f"{self._name}: Protocol not loaded")
            return {
                'status': False,
                'message': 'Protocol not loaded'
            }
        
        protocol_execution_record = ProtocolExecutionRecord.from_file(
            file_path=protocol_tsvs['protocol_execution_record'],
        )

        if protocol_execution_record is None:
            logger.error(f"{self._name}: Protocol Execution Record not loaded")
            return {
                'status': False,
                'message': 'Protocol Execution Record not loaded'
            }
        
        
        protocol_post_record = None
        if protocol_tsvs['protocol_post_record'] is not None:
            try:
                protocol_post_record = ProtocolPostRecord.from_file(
                    file_path=protocol_tsvs['protocol_post_record'],
                )
                logger.info(f"Loaded existing protocol post record {protocol_tsvs['protocol_post_record']}")
            except:
                logger.error(f"Unable to load the protocol post record file {protocol_tsvs['protocol_post_record']}. Creating new record.")

        if protocol_post_record is None:
            protocol_post_record = ProtocolPostRecord(
                file_loc=root_path / ProtocolPostRecord.DEFAULT_FILENAME
            )

        
        if selected_path == root_path:
            include_subpaths=[]
        else:
            include_subpaths = [
                selected_path.name
            ]

        image_names = self._get_image_filenames_from_folder(
            path=root_path,
            exclude_subpaths=[],
            include_subpaths=include_subpaths
        )

        raw_images_df = self._get_raw_images_df(
            image_names=image_names['raw'],
            protocol=protocol,
            protocol_execution_record=protocol_execution_record,
        )
        raw_images_df['Raw'] = True

        post_processing_names = PostFunction.list_values()
        raw_images_df[post_processing_names] = False

        post_images_df = self._get_post_images_df(
            image_names=image_names['post'],
            protocol_post_record=protocol_post_record,
        )
        post_images_df['Raw'] = False

        df_list = [raw_images_df, post_images_df]
        images_df = pd.concat([df for df in df_list if not df.empty])

        return {
            'status': True,
            'root_path': root_path,
            'selected_path': selected_path,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'protocol_post_record': protocol_post_record,
            'images_df': images_df,
        }

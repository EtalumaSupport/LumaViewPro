
import os
import pathlib

import pandas as pd

import modules.artifact_locations as artifact_locations
import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
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
    ) -> list:
        
        if (len(include_subpaths) != 0) and (len(exclude_subpaths) != 0):
            raise Exception(f"Specify only include_subpaths OR exclude_subpaths. Not both.")
        
        tiff_images = path.rglob('*.tif[f]')
        ome_tiff_images = path.rglob('*.ome.tif[f]')
        images = []
        images.extend(tiff_images)
        images.extend(ome_tiff_images)
        image_names = []
        for image in images:
            image_name = os.path.relpath(image, path)
            parent_dir = str(pathlib.Path(image_name).parent)

            if len(exclude_subpaths) > 0 and (parent_dir in exclude_subpaths):
                continue

            elif len(include_subpaths) > 0 and (parent_dir not in include_subpaths):
                continue
            
            image_names.append(image_name)

        return image_names


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


    @staticmethod
    def _is_stitched_image(image_filename: str) -> bool:
        if 'stitched' in image_filename:

            # For now, dont allow images that are both stitched and composite
            if 'Composite' in image_filename:
                return False
            
            return True
        
        return False
    

    @staticmethod
    def _is_composite_image(image_filename: str) -> bool:
        if 'Composite' in image_filename:

            # For now, dont allow images that are both stitched and composite
            if 'stitched' in image_filename:
                return False
            
            return True
        
        return False
    

    def _get_post_generated_images(
        self,
        parent_path: pathlib.Path,
        artifact_subfolder: str,
        metadata_filename: str
    ) -> pd.DataFrame | None:
        images_path = parent_path / artifact_subfolder
        image_metadata_path = images_path / metadata_filename

        load_images = True
        if not images_path.exists():
            logger.info(f'{self._name}: No folder found at {images_path}')
            load_images = False
         
        if load_images and not image_metadata_path.exists():
            logger.error(f'{self._name}: No metadata found at {image_metadata_path}')
            load_images = False

        if not load_images:
            return None

        parse_dates = ['Timestamp']
        df = pd.read_csv(
            filepath_or_buffer=image_metadata_path,
            sep='\t',
            lineterminator='\n',
            parse_dates=parse_dates
        )

        # Add subfolder to filename path
        df['Filename'] = df.apply(lambda row: str(pathlib.Path(artifact_subfolder, row['Filename'])), axis=1)

        df = df.fillna('')
        return df
    

    def _get_image_tile_groups(
        self,
        image_names: list,
        protocol_tile_groups,
        protocol_execution_record,
    ) -> pd.DataFrame | None:
                
        image_tile_groups = []
        for image_name in image_names:
            file_data = protocol_execution_record.get_data_from_filename(filename=image_name)
            if file_data is None:
                logger.warning(f"No info found in protocol execution record for {image_name}")
                continue

            for protocol_group_index, protocol_group_data in protocol_tile_groups.items():
                match = protocol_group_data[protocol_group_data['Step Index'] == file_data['Step Index']]
                if len(match) == 0:
                    continue

                if len(match) > 1:
                    raise Exception(f"Expected 1 match, but found multiple")
                
                first_row = match.iloc[0]
                image_tile_groups.append(
                    {
                        'Filename': image_name,
                        'Name': first_row['Name'],
                        'Protocol Group Index': protocol_group_index,
                        'Scan Count': file_data['Scan Count'],
                        'Step Index': first_row['Step Index'],
                        'X': first_row['X'],
                        'Y': first_row['Y'],
                        'Z-Slice': first_row['Z-Slice'],
                        'Well': first_row['Well'],
                        'Color': first_row['Color'],
                        'Objective': first_row['Objective'],
                        'Tile': first_row['Tile'],
                        'Tile Group ID': first_row['Tile Group ID'],
                        'Z-Stack Group ID': first_row['Z-Stack Group ID'],
                        'Custom Step': first_row['Custom Step'],
                        'Timestamp': file_data['Timestamp'],
                        'Stitched': False,
                        'Composite': False
                    }
                )

        df = pd.DataFrame(image_tile_groups)

        df = self._add_stitch_group_index(df=df)
        df = self._add_composite_group_index(df=df)
        df = self._add_video_group_index(df=df)
        df = self._add_zproject_group_index(df=df)
        df = df.fillna('')

        return df
    



    @staticmethod
    def _add_stitch_group_index(df: pd.DataFrame) -> pd.DataFrame:
        df['Stitch Group Index'] = df.groupby(
            by=[
                'Protocol Group Index',
                'Scan Count',
                'Z-Slice',
                'Well',
                'Color',
                'Objective',
                'Tile Group ID',
                'Custom Step'
            ],
            dropna=False
        ).ngroup()
        return df
    

    @staticmethod
    def _add_composite_group_index(df: pd.DataFrame) -> pd.DataFrame:
        df['Composite Group Index'] = df.groupby(
            by=[
                'Scan Count',
                'Z-Slice',
                'Well',
                'Objective',
                'X',
                'Y',
                'Tile',
                'Custom Step'
            ],
            dropna=False
        ).ngroup()
        return df
    

    @staticmethod
    def _add_video_group_index(df: pd.DataFrame) -> pd.DataFrame:
        df['Video Group Index'] = df.groupby(
            by=[
                'Protocol Group Index',
                'Z-Slice',
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
        include_stitched_images: bool = False,
        include_composite_images: bool = False,
        include_composite_and_stitched_images: bool = False,
    ) -> dict:
        selected_path = pathlib.Path(path)
        logger.info(f'{self._name}: Loading folder {selected_path}')

        protocol_tsvs = self._find_protocol_tsvs(path=selected_path)

        root_path = protocol_tsvs['protocol_root_dir']

        if protocol_tsvs is None:
            logger.error(f"{self._name}: Protocol and/or protocol record not found in folder")
            return {
                'status': False,
                'message': 'Protocol and/or Protocol Record not found in folder'
            }
                
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
                outfile=root_path / ProtocolPostRecord.DEFAULT_FILENAME
            )

        protocol_tile_groups = protocol.get_tile_groups()
        # exclude_paths = [
        #     artifact_locations.composite_output_dir(),
        #     artifact_locations.stitcher_output_dir(),
        #     artifact_locations.composite_and_stitched_output_dir()
        # ]

        
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

        image_tile_groups_df = self._get_image_tile_groups(
            image_names=image_names,
            protocol_tile_groups=protocol_tile_groups,
            protocol_execution_record=protocol_execution_record,
        )

        return {
            'status': True,
            'root_path': root_path,
            'selected_path': selected_path,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'protocol_post_record': protocol_post_record,
            'image_names': image_names,
            'protocol_tile_groups': protocol_tile_groups,
            'image_tile_groups':image_tile_groups_df,
            # 'stitched_images': stitched_images_df,
            # 'composite_images': composite_images_df,
            # 'composite_and_stitched_images': composite_and_stitched_images_df,
        }

                

        if include_stitched_images:
            stitched_images_df = self._get_post_generated_images(
                parent_path=path,
                artifact_subfolder=artifact_locations.stitcher_output_dir(),
                metadata_filename=artifact_locations.stitcher_output_metadata_filename()
            )

            if stitched_images_df is not None:
                # Create empty 'Tile label' for already stitched images
                stitched_images_df['Tile'] = ""

                stitched_images_df = self._add_stitch_group_index(df=stitched_images_df)
                stitched_images_df = self._add_composite_group_index(df=stitched_images_df)
                stitched_images_df = self._add_video_group_index(df=stitched_images_df)
                stitched_images_df = self._add_zproject_group_index(df=stitched_images_df)
                stitched_images_df['Stitched'] = True
        else:
            stitched_images_df = None


        if include_composite_images:
            composite_images_df = self._get_post_generated_images(
                parent_path=path,
                artifact_subfolder=artifact_locations.composite_output_dir(),
                metadata_filename=artifact_locations.composite_output_metadata_filename()
            )

            if composite_images_df is not None:
                composite_images_df = self._add_stitch_group_index(df=composite_images_df)
                composite_images_df = self._add_composite_group_index(df=composite_images_df)
                composite_images_df = self._add_video_group_index(df=composite_images_df)
                composite_images_df = self._add_zproject_group_index(df=composite_images_df)
                composite_images_df['Composite'] = True
        else:
            composite_images_df = None

        if include_composite_and_stitched_images:
            # Create empty 'Tile label' for already stitched images
            if stitched_images_df is not None:
                stitched_images_df['Tile'] = ""

            composite_and_stitched_images_df = self._get_post_generated_images(
                parent_path=path,
                artifact_subfolder=artifact_locations.composite_and_stitched_output_dir(),
                metadata_filename=artifact_locations.composite_and_stitched_output_metadata_filename()
            )

            if composite_and_stitched_images_df is not None:
                composite_and_stitched_images_df = self._add_stitch_group_index(df=composite_and_stitched_images_df)
                composite_and_stitched_images_df = self._add_composite_group_index(df=composite_and_stitched_images_df)
                composite_and_stitched_images_df = self._add_video_group_index(df=composite_and_stitched_images_df)
                composite_and_stitched_images_df = self._add_zproject_group_index(df=composite_and_stitched_images_df)
                composite_and_stitched_images_df['Composite'] = True
                composite_and_stitched_images_df['Stitched'] = True
        else:
            composite_and_stitched_images_df = None


        protocol_tile_groups = protocol.get_tile_groups()
        exclude_paths = [
            artifact_locations.composite_output_dir(),
            artifact_locations.stitcher_output_dir(),
            artifact_locations.composite_and_stitched_output_dir()
        ]

        image_names = self._get_image_filenames_from_folder(
            path=path,
            exclude_paths=exclude_paths
        )

        image_tile_groups_df = self._get_image_tile_groups(
            image_names=image_names,
            protocol_tile_groups=protocol_tile_groups,
            protocol_execution_record=protocol_execution_record,
        )

        return {
            'status': True,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'image_names': image_names,
            'protocol_tile_groups': protocol_tile_groups,
            'image_tile_groups':image_tile_groups_df,
            'stitched_images': stitched_images_df,
            'composite_images': composite_images_df,
            'composite_and_stitched_images': composite_and_stitched_images_df,
        }

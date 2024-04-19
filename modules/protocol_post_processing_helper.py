
import os
import pathlib

import pandas as pd

import modules.artifact_locations as artifact_locations
import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

from lvp_logger import logger


class ProtocolPostProcessingHelper:

    def __init__(self):
        self._name = self.__class__.__name__


    @staticmethod
    def _get_image_filenames_from_folder(path: pathlib.Path, exclude_paths: list = []) -> list:
        tiff_images = path.rglob('*.tif[f]')
        ome_tiff_images = path.rglob('*.ome.tif[f]')
        images = []
        images.extend(tiff_images)
        images.extend(ome_tiff_images)
        image_names = []
        for image in images:
            image_name = os.path.relpath(image, path)
            parent_dir = str(pathlib.Path(image_name).parent)

            if parent_dir in exclude_paths:
                continue
            
            image_names.append(image_name)

        return image_names


    def _find_protocol_tsvs(self, path: pathlib.Path) -> dict[str, pathlib.Path] | None:
        tsv_files = list(path.glob('*.tsv'))
        if len(tsv_files) !=  2:
            return None
        
        tsv_file_names = [tsv_file.name for tsv_file in tsv_files]
        
        # Confirm one of the two files matches the protocol execution record filename
        if ProtocolExecutionRecord.DEFAULT_FILENAME not in tsv_file_names:
            return None
        
        # Find the other filename as the protocol file
        for tsv_file_name in tsv_file_names:
            if tsv_file_name != ProtocolExecutionRecord.DEFAULT_FILENAME:
                protocol_file = tsv_file_name
                break

        return {
            'protocol_execution_record': path / ProtocolExecutionRecord.DEFAULT_FILENAME,
            'protocol': path / protocol_file
        }
        
    
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


    def load_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        include_stitched_images: bool = False,
        include_composite_images: bool = False,
        include_composite_and_stitched_images: bool = False,
    ) -> dict:
        logger.info(f'{self._name}: Loading folder {path}')
        path = pathlib.Path(path)

        protocol_tsvs = self._find_protocol_tsvs(path=path)
        if protocol_tsvs is None:
            logger.error(f"{self._name}: Protocol and/or protocol record not found in folder")
            return {
                'status': False,
                'message': 'Protocol and/or Protocol Record not found in folder'
            }
        
        logger.info(f"{self._name}: Found -> protocol {protocol_tsvs['protocol']}, protocol execution record {protocol_tsvs['protocol_execution_record']}")
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
        
        protocol_execution_record = ProtocolExecutionRecord.from_file(file_path=protocol_tsvs['protocol_execution_record'])
        if protocol_execution_record is None:
            logger.error(f"{self._name}: Protocol Execution Record not loaded")
            return {
                'status': False,
                'message': 'Protocol Execution Record not loaded'
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

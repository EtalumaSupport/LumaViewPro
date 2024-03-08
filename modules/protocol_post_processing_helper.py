
import os
import pathlib

import pandas as pd

from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord

import modules.common_utils as common_utils
from lvp_logger import logger


class ProtocolPostProcessingHelper:

    def __init__(self):
        self._name = self.__class__.__name__


    @staticmethod
    def _get_image_filenames_from_folder(path: pathlib.Path) -> list:
        tiff_images = path.rglob('*.tif[f]')
        ome_tiff_images = path.rglob('*.ome.tif[f]')
        images = []
        images.extend(tiff_images)
        images.extend(ome_tiff_images)
        image_names = []
        for image in images:
            image_names.append(os.path.relpath(image, path))

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
    

    def load_folder(
        self,
        path: str | pathlib.Path,
        include_stitched_images: bool = False,
        include_composite_images: bool = False
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
        protocol = Protocol.from_file(file_path=protocol_tsvs['protocol'])
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

        image_names = self._get_image_filenames_from_folder(path=path)

        protocol_tile_groups = protocol.get_tile_groups()
        image_tile_groups = []

        stitched_images = []
        composite_image_names = []
        composite_images = []

        logger.info(f"{self._name}: Matching images to stitching groups")
        for image_name in image_names:
            file_data = protocol_execution_record.get_data_from_filename(filename=image_name)
            if file_data is None:
                if (include_stitched_images == True) and (self._is_stitched_image(image_filename=image_name) == True):
                    well = common_utils.get_well_label_from_name(name=image_name)
                    color = common_utils.get_layer_from_name(name=image_name)

                    # Assumes stitched image name is of format <well>_<layer>_<z_slice>_<scan_count>_stitched.tiff
                    # Not a valid method for non-stitched images
                    scan_count = int(image_name.split('_')[-2])

                    # Extract Z-slice if applicable
                    tmp = image_name.split('_')[-3]
                    if tmp.startswith('Z'):
                        z_slice = int(tmp[1:])
                    else:
                        z_slice = None

                    stitched_images.append({
                        'Filename': image_name,
                        'Well': well,
                        'Color': color,
                        'Scan Count': scan_count,
                        'Z-Slice': z_slice
                    })

                elif (include_composite_images == True) and (self._is_composite_image(image_filename=image_name) == True):
                    composite_image_names.append(image_name)

                continue

            scan_count = file_data['Scan Count']

            for protocol_group_index, protocol_group_data in protocol_tile_groups.items():
                match = protocol_group_data[protocol_group_data['Step Index'] == file_data['Step Index']]
                if len(match) == 0:
                    continue

                if len(match) > 1:
                    raise Exception(f"Expected 1 match, but found multiple")
                
                image_tile_groups.append(
                    {
                        'Filename': image_name,
                        'Name': match['Name'].values[0],
                        'Protocol Group Index': protocol_group_index,
                        'Scan Count': scan_count,
                        'Step Index': match['Step Index'].values[0],
                        'X': match['X'].values[0],
                        'Y': match['Y'].values[0],
                        'Z-Slice': match['Z-Slice'].values[0],
                        'Well': match['Well'].values[0],
                        'Color': match['Color'].values[0],
                        'Objective': match['Objective'].values[0],
                        'Tile Group ID': match['Tile Group ID'].values[0],
                        'Z-Stack Group ID': match['Z-Stack Group ID'].values[0],
                        'Custom Step': match['Custom Step'].values[0]
                    }
                )

                break

        # Process composite images
        if len(composite_image_names) > 0:
            for name in composite_image_names:
                well = common_utils.get_well_label_from_name(name=name)

                # Assumes composite image name is of format <well>_<layer>_<z_slice>_<scan_count>.tiff
                # Not a valid method for non-composite images
                scan_count = int(name.split('_')[-1].split('.')[0])

                # Extract Z-slice if applicable
                tmp = name.split('_')[-2]
                if tmp.startswith('Z'):
                    z_slice = int(tmp[1:])
                else:
                    z_slice = None

                

                composite_images.append({
                    'filename': name,
                    'well': well,
                    'scan_count': scan_count,
                    'z_slice': z_slice
                })

                
        
        df = pd.DataFrame(image_tile_groups)

        if len(stitched_images) > 0:
            stitched_images_df = pd.DataFrame(stitched_images)
        else:
            stitched_images_df = None

        if len(composite_images) > 0:
            composite_images_df = pd.DataFrame(composite_images)
        else:
            composite_images_df = None


        return {
            'status': True,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'image_names': image_names,
            'protocol_tile_groups': protocol_tile_groups,
            'image_tile_groups': df,
            'stitched_images': stitched_images_df,
            'composite_images': composite_images_df
        }

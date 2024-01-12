
import pathlib

import pandas as pd

from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from lvp_logger import logger


class ProtocolPostProcessingHelper:

    def __init__(self):
        self._name = self.__class__.__name__


    @staticmethod
    def _get_image_filenames_from_folder(path: pathlib.Path) -> list:
        images = path.glob('*.tif[f]')
        image_names = []
        for image in images:
            image_names.append(image.name)

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
        

    def load_folder(self, path: str | pathlib.Path) -> dict:
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
        protocol_execution_record = ProtocolExecutionRecord.from_file(file_path=protocol_tsvs['protocol_execution_record'])

        image_names = self._get_image_filenames_from_folder(path=path)

        protocol_tile_groups = protocol.get_tile_groups()
        image_tile_groups = []

        logger.info(f"{self._name}: Matching images to stitching groups")
        for image_name in image_names:
            file_data = protocol_execution_record.get_data_from_filename(filename=image_name)
            if file_data is None:
                continue

            scan_count = file_data['scan_count']

            for protocol_group_index, protocol_group_data in protocol_tile_groups.items():
                match = protocol_group_data[protocol_group_data['step_index'] == file_data['step_index']]
                if len(match) == 0:
                    continue

                if len(match) > 1:
                    raise Exception(f"Expected 1 match, but found multiple")
                
                image_tile_groups.append(
                    {
                        'filename': image_name,
                        'protocol_group_index': protocol_group_index,
                        'scan_count': scan_count,
                        'step_index': match['step_index'].values[0],
                        'x': match['x'].values[0],
                        'y': match['y'].values[0],
                        'z_slice': match['z_slice'].values[0],
                        'well': match['well'].values[0],
                        'color': match['color'].values[0],
                        'objective': match['objective'].values[0]
                    }
                )

                break
        
        df = pd.DataFrame(image_tile_groups)

        return {
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'image_names': image_names,
            'protocol_tile_groups': protocol_tile_groups,
            'image_tile_groups': df
        }


import os
import pathlib


from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord


class Stitcher:

    def __init__(self):
        pass


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
        

    def load_folder(self, path: str | pathlib.Path):
        path = pathlib.Path(path)

        protocol_tsvs = self._find_protocol_tsvs(path=path)
        if protocol_tsvs is None:
            return False
        
        protocol = Protocol.from_file(file_path=protocol_tsvs['protocol'])
        protocol_execution_record = ProtocolExecutionRecord.from_file(file_path=protocol_tsvs['protocol_execution_record'])

        image_names = self._get_image_filenames_from_folder(path=path)
        protocol_tile_groups = protocol.get_tile_groups()
        image_tile_groups = {}

        # Build a dictionary matching up the protocol tile groups with the image filenames
        # Data structure will be tiered dictionary: group_index -> scan_count -> image filename
        for image_name in image_names:
            file_data = protocol_execution_record.get_data_from_filename(filename=image_name)
            if file_data is None:
                continue

            scan_count = file_data['scan_count']

            for group_index, group_data in protocol_tile_groups.items():
                match = group_data[group_data['step_index'] == file_data['step_index']]
                if len(match) == 0:
                    continue

                if len(match) > 1:
                    raise Exception(f"Expected 1 match, but found multiple")

                if group_index not in image_tile_groups:
                    image_tile_groups[group_index] = {}

                if scan_count not in image_tile_groups[group_index]:
                    image_tile_groups[group_index][scan_count] = {}

                image_tile_groups[group_index][scan_count][image_name] = {
                    'step_index': match['step_index'].values[0]
                }

                break

        
        print('hi')


        

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
    

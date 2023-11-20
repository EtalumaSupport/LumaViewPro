
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
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
        image_tile_groups = []

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
                        'well': match['well'].values[0],
                        'color': match['color'].values[0]
                    }
                )

                break

        df = pd.DataFrame(image_tile_groups)
        df['stitch_group_index'] = df.groupby(by=['protocol_group_index','scan_count']).ngroup()
        
        for _, stitch_group in df.groupby(by=['stitch_group_index']):
            stitched_image = self.position_stitcher(
                path=path,
                df=stitch_group[['filename', 'x', 'y']],
                pos2pix=2630
            )

            stitched_filename = self._generate_stitched_filename(df=stitch_group)

            cv2.imwrite(
                filename=str(path / stitched_filename),
                img=stitched_image
            )

    
    @staticmethod
    def _generate_stitched_filename(df: pd.DataFrame) -> str:
        row0 = df.iloc[0]
        name = common_utils.generate_default_step_name(
            well_label=row0['well'],
            color=row0['color'],
            # z_height_idx=,
            scan_count=row0['scan_count']
        )

        return f"{name}_stitched.tiff"


    @staticmethod
    def position_stitcher(path: pathlib.Path, df: pd.DataFrame, pos2pix):
        
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['filename']
            images[row['filename']] = cv2.imread(str(image_filepath))

        df = df.copy()

        df['x_pos_range'] = df['x'] - df['x'].min()
        df['y_pos_range'] = df['y'] - df['y'].min()
        df['x_pix_range'] = df['x_pos_range']*pos2pix
        df['y_pix_range'] = df['y_pos_range']*pos2pix
        df = df.sort_values(['x','y'], ascending=False)
        df[['x','y']] = df[['x','y']].apply(np.floor).astype(int)

        source_image_sample = df['filename'].values[0]
        stitched_im_x = images[source_image_sample].shape[1] + df['x'].max()
        stitched_im_y = images[source_image_sample].shape[0] + df['y'].max()

        reverse_y = True
        if reverse_y:
            df['y'] = stitched_im_y - df['y']

        stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype='uint8')
        for _, row in df.iterrows():
            filename = row['filename']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            if reverse_y:
                stitched_img[row['y']-im_y:row['y'], row['x']:row['x']+im_x,:] = image
            else:
                stitched_img[row['y']:row['y']+im_y, row['x']:row['x']+im_x,:] = image

        return stitched_img
            

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
    print('hi')
    

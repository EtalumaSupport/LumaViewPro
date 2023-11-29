
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
import modules.tiling_config as tiling_config
from lvp_logger import logger


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
        

    def load_folder(self, path: str | pathlib.Path) -> dict:
        logger.info(f'Stitcher: Loading folder {path}')
        path = pathlib.Path(path)

        protocol_tsvs = self._find_protocol_tsvs(path=path)
        if protocol_tsvs is None:
            logger.error(f"Stitcher: Protocol and/or protocol record not found in folder")
            return {
                'status': False,
                'message': 'Protocol and/or Protocol Record not found in folder'
            }
        
        logger.info(f"Stitcher: Found -> protocol {protocol_tsvs['protocol']}, protocol execution record {protocol_tsvs['protocol_execution_record']}")
        protocol = Protocol.from_file(file_path=protocol_tsvs['protocol'])
        protocol_execution_record = ProtocolExecutionRecord.from_file(file_path=protocol_tsvs['protocol_execution_record'])

        image_names = self._get_image_filenames_from_folder(path=path)
        protocol_tile_groups = protocol.get_tile_groups()
        image_tile_groups = []

        logger.info(f"Stitcher: Matching images to stitching groups")
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
        df['stitch_group_index'] = df.groupby(by=['protocol_group_index','scan_count']).ngroup()
        
        logger.info(f"Stitcher: Generating stitched images")
        for _, stitch_group in df.groupby(by=['stitch_group_index']):
            # pos2pix = self._calc_pos2pix_from_objective(objective=stitch_group['objective'].values[0])

            stitched_image = self.simple_position_stitcher(
                path=path,
                df=stitch_group[['filename', 'x', 'y', 'z_slice']]
            )

            # stitched_image = self.position_stitcher(
            #     path=path,
            #     df=stitch_group[['filename', 'x', 'y']],
            #     pos2pix=int(pos2pix * tiling_config.TilingConfig.DEFAULT_FILL_FACTORS['position'])
            # )
            
            stitched_filename = self._generate_stitched_filename(df=stitch_group)
            logger.debug(f"Stitcher: - {stitched_filename}")

            cv2.imwrite(
                filename=str(path / stitched_filename),
                img=stitched_image
            )
        
        logger.info(f"Stitcher: Complete")
        return {
            'status': True,
            'message': 'Success'
        }


    @staticmethod
    def _calc_pos2pix_from_objective(objective: str) -> int:
        # TODO
        if objective == '4x':
            return 625 #550
        else:
            raise NotImplementedError(f"Image stitching for objectives other than 4x is not implemented")
    

    @staticmethod
    def _generate_stitched_filename(df: pd.DataFrame) -> str:
        row0 = df.iloc[0]
        name = common_utils.generate_default_step_name(
            well_label=row0['well'],
            color=row0['color'],
            z_height_idx=row0['z_slice'],
            scan_count=row0['scan_count']
        )

        return f"{name}_stitched.tiff"


    @staticmethod
    def simple_position_stitcher(path: pathlib.Path, df: pd.DataFrame):
        '''
        Performs a simple concatenation of images, given a set of X/Y positions the images were captured from.
        Assumes no overlap between images.
        '''
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['filename']
            images[row['filename']] = cv2.imread(str(image_filepath))

        df = df.copy()

        num_x_tiles = df['x'].nunique()
        num_y_tiles = df['y'].nunique()

        source_image_sample = df['filename'].values[0]
        source_image_w = images[source_image_sample].shape[1]
        source_image_h = images[source_image_sample].shape[0]
        
        df = df.sort_values(['x','y'], ascending=False)
        df['x_index'] = df.groupby(by=['x']).ngroup()
        df['y_index'] = df.groupby(by=['y']).ngroup()
        df['x_pix_range'] = df['x_index'] * source_image_w
        df['y_pix_range'] = df['y_index'] * source_image_h
            
        stitched_im_x = source_image_w * num_x_tiles
        stitched_im_y = source_image_h * num_y_tiles

        reverse_x = True
        reverse_y = False
        if reverse_x:
            df['x_pix_range'] = stitched_im_x - df['x_pix_range']

        if reverse_y:
            df['y_pix_range'] = stitched_im_y - df['y_pix_range']

        stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype='uint8')
        for _, row in df.iterrows():
            filename = row['filename']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                else:
                    stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image

            else:

                if reverse_x:
                    stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                else:
                    stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image


        return stitched_img


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
        df = df.sort_values(['x_pix_range','y_pix_range'], ascending=False)
        df[['x_pix_range','y_pix_range']] = df[['x_pix_range','y_pix_range']].apply(np.floor).astype(int)

        source_image_sample = df['filename'].values[0]
        stitched_im_x = images[source_image_sample].shape[1] + df['x_pix_range'].max()
        stitched_im_y = images[source_image_sample].shape[0] + df['y_pix_range'].max()

        reverse_x = True
        reverse_y = False
        if reverse_x:
            df['x_pix_range'] = stitched_im_x - df['x_pix_range']

        if reverse_y:
            df['y_pix_range'] = stitched_im_y - df['y_pix_range']

        stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype='uint8')
        for _, row in df.iterrows():
            filename = row['filename']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                else:
                    stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image

            else:

                if reverse_x:
                    stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                else:
                    stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image


        return stitched_img
            

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
    print('hi')
    

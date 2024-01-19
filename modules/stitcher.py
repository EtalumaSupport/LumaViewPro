
import itertools
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

from lvp_logger import logger


class Stitcher:

    def __init__(self):
        self._name = self.__class__.__name__
        self._protocol_post_processing_helper = ProtocolPostProcessingHelper()
        

    def load_folder(self, path: str | pathlib.Path) -> dict:
        results = self._protocol_post_processing_helper.load_folder(
            path=path,
            include_stitched_images=False,
            include_composite_images=False
        )
       
        df = results['image_tile_groups']
        df['stitch_group_index'] = df.groupby(by=['protocol_group_index','scan_count']).ngroup()

        # composite_images_df = results['composite_images']
        # if composite_images_df is not None:
        #     composite_images_df['stitch_group_index'] = composite_images_df.groupby(by=['scan_count', 'z_slice', 'well']).ngroup()
        #     loop_list = itertools.chain(
        #         df.groupby(by=['stitch_group_index']),
        #         composite_images_df.groupby(by=['stitch_group_index'])
        #     )
        # else:
        #     loop_list = df.groupby(by=['stitch_group_index'])
        
        logger.info(f"{self._name}: Generating stitched images")
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
            logger.debug(f"{self._name}: - {stitched_filename}")

            cv2.imwrite(
                filename=str(path / stitched_filename),
                img=stitched_image
            )
        
        logger.info(f"{self._name}: Complete")
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

        outfile = f"{name}_stitched.tiff"

        # Handle case of individual folders per channel
        split_name = os.path.split(row0['filename'])
        if len(split_name) == 2:
            outfile = os.path.join(split_name[0], outfile)
        
        return outfile


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

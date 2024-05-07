
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import image_utils

from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord


class Stitcher(ProtocolPostProcessingExecutor):

    def __init__(self):
        super().__init__(
            post_function=PostFunction.STITCHED
        )
        self._name = self.__class__.__name__
        

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Scan Count',
                'Z-Slice',
                'Well',
                'Color',
                'Objective',
                'Tile Group ID',
                'Custom Step',
                'Raw',
                *PostFunction.list_values()
            ],
            dropna=False
        )
    

    @staticmethod
    def _generate_filename(df: pd.DataFrame) -> str:
        row0 = df.iloc[0]

        name = common_utils.generate_default_step_name(
            custom_name_prefix=row0['Name'],
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count'],
            tile_label=None,
            stitched=True,
        )
        
        outfile = f"{name}.tiff"
        return outfile
    

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already stitched outputs
        df = df[df[self._post_function.value] == False]

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return Stitcher._simple_position_stitcher(
            path=path,
            df=df[['Filepath', 'X', 'Y']]
        )


    @staticmethod
    def _add_record(
        protocol_post_record: ProtocolPostRecord,
        alg_metadata: dict,
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        row0: pd.Series,
        **kwargs: dict,
    ):
        protocol_post_record.add_record(
            root_path=root_path,
            file_path=file_path,
            timestamp=row0['Timestamp'],
            name=row0['Name'],
            scan_count=row0['Scan Count'],
            x=alg_metadata['center']['x'],
            y=alg_metadata['center']['y'],
            z=row0['Z'],
            z_slice=row0['Z-Slice'],
            well=row0['Well'],
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile="",
            custom_step=row0['Custom Step'],
            **kwargs,
        )


    @staticmethod
    def _simple_position_stitcher(path: pathlib.Path, df: pd.DataFrame):
        '''
        Performs a simple concatenation of images, given a set of X/Y positions the images were captured from.
        Assumes no overlap between images.
        '''
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        df = df.copy()

        num_x_tiles = df['X'].nunique()
        num_y_tiles = df['Y'].nunique()

        # Used to find the center of the image in X/Y coordinates
        x_center = df['X'].unique().mean()
        y_center = df['Y'].unique().mean()
        center = {
            'x': round(x_center, common_utils.max_decimal_precision(parameter='x')),
            'y': round(y_center, common_utils.max_decimal_precision(parameter='y')),
        }

        source_image_sample_row = df.iloc[0]
        source_image_sample_filename = source_image_sample_row['Filepath']
        source_image_sample = images[source_image_sample_filename]
        source_image_h = source_image_sample.shape[0]
        source_image_w = source_image_sample.shape[1]
        
        df = df.sort_values(['X','Y'], ascending=False)
        df['x_index'] = df.groupby(by=['X']).ngroup()
        df['y_index'] = df.groupby(by=['Y']).ngroup()
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

        
        is_color_image = image_utils.is_color_image(image=source_image_sample)
        if is_color_image:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=source_image_sample.dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=source_image_sample.dtype)

        for _, row in df.iterrows():
            filename = row['Filepath']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x] = image
            else:

                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image

        return {
            'status': True,
            'error': None,
            'image': stitched_img,
            'metadata': {
                'center': center,
            }
        }


    @staticmethod
    def position_stitcher(path: pathlib.Path, df: pd.DataFrame, pos2pix):
        
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        df = df.copy()

        df['x_pos_range'] = df['X'] - df['X'].min()
        df['y_pos_range'] = df['Y'] - df['Y'].min()
        df['x_pix_range'] = df['x_pos_range']*pos2pix
        df['y_pix_range'] = df['y_pos_range']*pos2pix
        df = df.sort_values(['x_pix_range','y_pix_range'], ascending=False)
        df[['x_pix_range','y_pix_range']] = df[['x_pix_range','y_pix_range']].apply(np.floor).astype(int)

        source_image_sample_filename = df['Filepath'].values[0]
        source_image_sample = images[source_image_sample_filename]
        stitched_im_x = source_image_sample.shape[1] + df['x_pix_range'].max()
        stitched_im_y = source_image_sample.shape[0] + df['y_pix_range'].max()

        reverse_x = True
        reverse_y = False
        if reverse_x:
            df['x_pix_range'] = stitched_im_x - df['x_pix_range']

        if reverse_y:
            df['y_pix_range'] = stitched_im_y - df['y_pix_range']

        
        is_color_image = image_utils.is_color_image(image=source_image_sample)

        if is_color_image:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=source_image_sample.dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=source_image_sample.dtype)

        for _, row in df.iterrows():
            filename = row['Filepath']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x] = image
            else:

                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image


        return stitched_img
            

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))

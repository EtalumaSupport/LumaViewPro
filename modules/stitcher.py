# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.stitch_algorithms import stitch_registered_tiles

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_record import ProtocolPostRecord


class Stitcher(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            post_function=PostFunction.STITCHED,
            *args,
            **kwargs,
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
                *PostFunction.list_values(),
            ],
            dropna=False,
        )

    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        row0 = df.iloc[0]
        objective_short_name = self._get_objective_short_name_if_has_turret(
            objective_id=row0['Objective']
        )

        # Use custom root + step name if available
        custom_root = row0.get('Custom Root', '') if 'Custom Root' in row0 else ''
        prefix = f'{custom_root}_{row0["Name"]}' if custom_root not in (None, '') else row0['Name']
        name = common_utils.generate_default_step_name(
            custom_name_prefix=prefix,
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count'],
            objective_short_name=objective_short_name,
            tile_label=None,
            stitched=True,
        )

        outfile = f'{name}.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already stitched outputs
        df = df[df[self._post_function.value] == False]

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]

        return df

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        position_result = self._position_stitcher(
            path=path,
            df=df[['Filepath', 'X', 'Y', 'Objective', 'Color']],
            output_file_loc=kwargs.get('output_file_loc'),
        )
        if position_result['status']:
            return position_result

        logger_msg = position_result['error']
        import logging
        logging.getLogger('LVP.ui.protocol_settings').warning(
            f"[Stitch] Position-aware stitch failed ({logger_msg}); "
            "falling back to simple grid stitch"
        )
        return Stitcher._simple_position_stitcher(
            path=path,
            df=df[['Filepath', 'X', 'Y']],
            output_file_loc=kwargs.get('output_file_loc'),
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
            tile='',
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    @staticmethod
    def _simple_position_stitcher(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path | None = None,
    ):
        """
        Performs a simple concatenation of images, given a set of X/Y positions the images were captured from.
        Assumes no overlap between images.

        When output_file_loc is provided, writes the stitched output via
        tifffile and returns image=None per the protocol_post_processor
        subclass-write bypass contract (matches composite_generation +
        zprojector). When None (test / legacy callers), returns the
        stitched array for the caller to save.
        """
        # Load source images via tifffile (RGB-native; returns mono 2D
        # for single-channel TIFFs). Matches the canonical read path
        # used by composite_generation + zprojector, replacing the
        # legacy cv2.imread BGR-native call.
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = tf.imread(str(image_filepath))

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

        df = df.sort_values(['X', 'Y'], ascending=False)
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
            stitched_img = np.zeros(
                (stitched_im_y, stitched_im_x, 3), dtype=source_image_sample.dtype
            )
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
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x] = image
            else:
                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val : y_val + im_y, x_val : x_val + im_x, :] = image
                    else:
                        stitched_img[y_val : y_val + im_y, x_val : x_val + im_x] = image

        # Self-write when output_file_loc is provided (canonical path
        # under protocol_post_processor). Matches composite_generation +
        # zprojector. tifffile auto-detects photometric: 2D mono ->
        # minisblack, 3D shape[-1]=3 -> rgb. Signal subclass-wrote via
        # image=None so the base class skips its own write branch.
        if output_file_loc is not None:
            # Widen mono fluorescence to RGB before save so the stitched
            # output matches the per-tile capture's false-color shape.
            # Without this, the bare tifffile write below produces
            # grayscale for Blue/Green/Red/Lumi stitches.
            output_image = image_utils.maybe_apply_false_color(
                data=stitched_img,
                color=df['Color'].iloc[0],
            )
            output_file_loc_abs = path / output_file_loc
            output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
            tf.imwrite(
                str(output_file_loc_abs),
                output_image,
                compression='lzw',
            )
            return_image = None
        else:
            return_image = stitched_img

        return {
            'status': True,
            'error': None,
            'image': return_image,
            'metadata': {
                'center': center,
            },
        }


    def _position_stitcher(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path | None = None,
    ):
        """Place tiles using recorded stage positions.

        Unlike _simple_position_stitcher, this preserves the pixel overlap
        implied by the stage coordinates instead of treating every adjacent
        tile as edge-to-edge. Overlapping pixels are averaged.
        """
        required_cols = {'Filepath', 'X', 'Y', 'Objective'}
        if not required_cols.issubset(df.columns):
            missing = sorted(required_cols.difference(df.columns))
            return {
                'status': False,
                'error': f"missing required columns: {missing}",
            }

        df = df.copy()
        df['X'] = df['X'].astype(float)
        df['Y'] = df['Y'].astype(float)

        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image = tf.imread(str(image_filepath))
            if image is None:
                return {
                    'status': False,
                    'error': f'unable to read image: {image_filepath}',
                }
            images[row['Filepath']] = image

        sample_row = df.iloc[0]
        sample = images[sample_row['Filepath']]
        image_h = sample.shape[0]
        image_w = sample.shape[1]

        try:
            objective = self._objectives_helper.get_objective_info(
                objective_id=sample_row['Objective']
            )
            fov = common_utils.get_field_of_view(
                focal_length=objective['focal_length'],
                frame_size={'width': image_w, 'height': image_h},
                binning_size=1,
            )
        except Exception as e:
            return {
                'status': False,
                'error': f"unable to determine objective field of view: {e}",
            }

        um_per_pixel_x = fov['width'] / image_w
        um_per_pixel_y = fov['height'] / image_h
        if um_per_pixel_x <= 0 or um_per_pixel_y <= 0:
            return {
                'status': False,
                'error': 'invalid field-of-view scale',
            }

        x_max = df['X'].max()
        y_min = df['Y'].min()
        df['x_pix'] = ((x_max - df['X']) * 1000 / um_per_pixel_x).round().astype(int)
        df['y_pix'] = ((df['Y'] - y_min) * 1000 / um_per_pixel_y).round().astype(int)

        stitched_w = int(df['x_pix'].max() + image_w)
        stitched_h = int(df['y_pix'].max() + image_h)
        if stitched_w <= 0 or stitched_h <= 0:
            return {
                'status': False,
                'error': 'invalid stitched image dimensions',
            }

        tiles = []
        for _, row in df.iterrows():
            tiles.append(
                {
                    'tile': images[row['Filepath']],
                    'x_px': int(row['x_pix']),
                    'y_px': int(row['y_pix']),
                }
            )

        center = {
            'x': round(df['X'].unique().mean(), common_utils.max_decimal_precision(parameter='x')),
            'y': round(df['Y'].unique().mean(), common_utils.max_decimal_precision(parameter='y')),
        }

        stitched_img, registered_tiles = stitch_registered_tiles(tiles)

        if output_file_loc is not None:
            color = df['Color'].iloc[0] if 'Color' in df.columns else ''
            output_image = image_utils.maybe_apply_false_color(
                data=stitched_img,
                color=color,
            )
            output_file_loc_abs = path / output_file_loc
            output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
            tf.imwrite(
                str(output_file_loc_abs),
                output_image,
                compression='lzw',
            )
            return_image = None
        else:
            return_image = stitched_img

        return {
            'status': True,
            'error': None,
            'image': return_image,
            'metadata': {
                'center': center,
                'registered_tiles': registered_tiles,
            },
        }


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import modules.image_utils as image_utils

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

        # Prepend the protocol's capture_root (passed in via kwargs by
        # ProtocolPostProcessor.load_folder) so the stitched output
        # carries the same filename root as the per-image saves.
        capture_root = kwargs.get('capture_root', '')
        prefix = f'{capture_root}_{row0["Name"]}' if capture_root else row0['Name']
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
        return Stitcher._simple_position_stitcher(
            path=path,
            df=df[['Filepath', 'X', 'Y', 'Color']],
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
        # zprojector. Routes through write_tiff so the output carries
        # the layer's PALETTE colormap (Windows Preview / FIJI render
        # the layer color) plus the source acquisition context
        # (objective, exposure, gain, pixel size, plate, instrument)
        # forwarded from the first tile. Signal subclass-wrote via
        # image=None so the base class skips its own write branch.
        if output_file_loc is not None:
            output_file_loc_abs = path / output_file_loc
            output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
            first_tile_path = path / source_image_sample_filename
            metadata = image_utils.build_postproc_output_metadata(
                input_path=first_tile_path,
                channel=source_image_sample_row['Color'],
                plate_pos_mm_override=center,
            )
            image_utils.write_tiff(
                data=stitched_img,
                file_loc=output_file_loc_abs,
                metadata=metadata,
                ome=False,
                color=source_image_sample_row['Color'],
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


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

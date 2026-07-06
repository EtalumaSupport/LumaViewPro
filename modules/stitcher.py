# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.stitch_algorithms import stitch_registered_tiles

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord


class Stitcher(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.STITCHED,
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

        # A stitch spans every tile of a (well, channel), so the per-tile token
        # is omitted (tile=None) -- by construction, never by stripping it back
        # out of a name. The channel comes from the authoritative Color column:
        # a single-channel stitch keeps its channel, and a composite-stitch
        # carries 'Composite' automatically (its Color is 'Composite'), so no
        # leaked channel token needs removing.
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                tile=None,
                scan_count=row0['Scan Count'],
                objective=objective_short_name,
                post=('stitched',),
            )
        )

        outfile = f'{self._prepend_capture_root(name, kwargs)}.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already stitched outputs
        df = df[df[self._post_function.value] == False]  # noqa: E712 -- pandas mask

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]  # noqa: E712 -- pandas mask

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]  # noqa: E712 -- pandas mask

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
            return PostProcResult.from_group_result(position_result)

        logger_msg = position_result['error']
        import logging

        logging.getLogger('LVP.ui.protocol_settings').warning(
            f'[Stitch] Position-aware stitch failed ({logger_msg}); '
            'falling back to simple grid stitch'
        )
        return PostProcResult.from_group_result(
            Stitcher._simple_position_stitcher(
                path=path,
                df=df[['Filepath', 'X', 'Y', 'Color']],
                output_file_loc=kwargs.get('output_file_loc'),
            )
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
            label=row0['Label'],
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
        # Tiles are read on demand inside the placement loop (one tile resident
        # at a time) rather than pre-loaded into a dict: the simple path places
        # each tile independently with no overlap, so peak memory is one tile +
        # the canvas instead of every tile + the canvas. Reads go through
        # tifffile (RGB-native; mono 2D for single-channel TIFFs), the canonical
        # path shared with composite_generation + zprojector.

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
        # Only the tile geometry (size + dtype + color-ness) is needed here to
        # size the canvas; the pixels and depth of every tile -- including this
        # one -- are read in the placement loop below. Read the header alone so
        # this first tile is not decoded once here and again in the loop.
        source_image_shape, source_image_dtype = image_utils.read_image_geometry(
            path / source_image_sample_filename
        )
        source_image_h = source_image_shape[0]
        source_image_w = source_image_shape[1]

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

        is_color = image_utils.is_color_shape(source_image_shape)
        if is_color:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=source_image_dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=source_image_dtype)

        input_depths = []
        for _, row in df.iterrows():
            filename = row['Filepath']
            image, significant_bits = image_utils.load_pixels(
                path / filename, collapse_legacy_false_color=False
            )
            input_depths.append(significant_bits)
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if is_color:
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val - im_x : x_val] = image
                else:
                    if is_color:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x, :] = image
                    else:
                        stitched_img[y_val - im_y : y_val, x_val : x_val + im_x] = image
            else:
                if reverse_x:
                    if is_color:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val, :] = image
                    else:
                        stitched_img[y_val : y_val + im_y, x_val - im_x : x_val] = image
                else:
                    if is_color:
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
                significant_bits=image_utils.resolve_output_depth(input_depths),
                plate_pos_mm_override=center,
            )
            image_utils.write_tiff(
                data=stitched_img,
                file_loc=output_file_loc_abs,
                metadata=metadata,
                ome=False,
                color=source_image_sample_row['Color'],
                significant_bits=metadata['significant_bits'],
                save_encoding=image_utils.resolve_output_save_encoding(stitched_img),
            )
            return_image = None
        else:
            return_image = stitched_img

        return {
            'status': True,
            'error': None,
            'image': return_image,
            'significant_bits': image_utils.resolve_output_depth(input_depths),
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
                'error': f'missing required columns: {missing}',
            }

        df = df.copy()
        df['X'] = df['X'].astype(float)
        df['Y'] = df['Y'].astype(float)

        images = {}
        input_depths = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image, significant_bits = image_utils.load_pixels(
                image_filepath, collapse_legacy_false_color=False
            )
            if image is None:
                return {
                    'status': False,
                    'error': f'unable to read image: {image_filepath}',
                }
            images[row['Filepath']] = image
            input_depths.append(significant_bits)

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
                'error': f'unable to determine objective field of view: {e}',
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
            # Route through write_tiff (matching _simple_position_stitcher,
            # zprojector, and composite_generation) so the stitched output
            # carries the layer's PALETTE colormap -- Windows Preview / FIJI
            # render the false color for 8-bit fluorescence -- plus the source
            # acquisition context (objective, exposure, gain, pixel size,
            # plate) forwarded from the first tile. A bare tf.imwrite drops
            # both, leaving a flat grayscale, metadata-less file.
            color = df['Color'].iloc[0] if 'Color' in df.columns else ''
            output_file_loc_abs = path / output_file_loc
            output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
            first_tile_path = path / sample_row['Filepath']
            metadata = image_utils.build_postproc_output_metadata(
                input_path=first_tile_path,
                channel=color,
                significant_bits=image_utils.resolve_output_depth(input_depths),
                plate_pos_mm_override=center,
            )
            image_utils.write_tiff(
                data=stitched_img,
                file_loc=output_file_loc_abs,
                metadata=metadata,
                ome=False,
                color=color,
                significant_bits=metadata['significant_bits'],
                save_encoding=image_utils.resolve_output_save_encoding(stitched_img),
            )
            return_image = None
        else:
            return_image = stitched_img

        return {
            'status': True,
            'error': None,
            'image': return_image,
            'significant_bits': image_utils.resolve_output_depth(input_depths),
            'metadata': {
                'center': center,
                'registered_tiles': registered_tiles,
            },
        }


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

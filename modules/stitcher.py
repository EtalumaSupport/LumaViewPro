# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from modules.stitch_algorithms import stitch_registered_tiles
from modules.stitching_core import (
    channel_aware_stitcher,
    simple_position_stitcher,
)

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord


class Stitcher(ProtocolPostProcessor):
    QUALITY_MODE = 'quality'
    FAST_PREVIEW_MODE = 'fast_preview'
    _FAST_PREVIEW_SUFFIX = 'FastPreview'

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

        prefix = self._prepend_capture_root(name, kwargs)
        if kwargs.get('stitching_mode') == self.FAST_PREVIEW_MODE:
            prefix = f'{prefix}_{self._FAST_PREVIEW_SUFFIX}'
        outfile = f'{prefix}.tiff'
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
        pixel_size_um = None
        try:
            objective_info = self._objectives_helper.get_objective_info(
                objective_id=df.iloc[0]['Objective']
            )
            pixel_size_um = common_utils.get_pixel_size(
                focal_length=objective_info['focal_length'],
                binning_size=1,
            )
        except Exception:
            pixel_size_um = None

        stitch_columns = [
            col
            for col in [
                'Filepath',
                'X',
                'Y',
                'Objective',
                'Color',
                'Well',
                'Tile Group ID',
            ]
            if col in df.columns
        ]

        return PostProcResult.from_group_result(
            channel_aware_stitcher(
                path=path,
                df=df[stitch_columns],
                pixel_size_um=pixel_size_um,
                output_file_loc=kwargs.get('output_file_loc'),
                stitching_mode=kwargs.get('stitching_mode', self.QUALITY_MODE),
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
        return simple_position_stitcher(
            path=path,
            df=df,
            output_file_loc=output_file_loc,
        )

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

        tiles = [
            {
                'tile': images[row['Filepath']],
                'x_px': int(row['x_pix']),
                'y_px': int(row['y_pix']),
            }
            for _, row in df.iterrows()
        ]

        center = {
            'x': round(df['X'].unique().mean(), common_utils.max_decimal_precision(parameter='x')),
            'y': round(df['Y'].unique().mean(), common_utils.max_decimal_precision(parameter='y')),
        }

        stitched_img, registered_tiles = stitch_registered_tiles(
            tiles, output_shape=(stitched_h, stitched_w)
        )

        output_depth = image_utils.resolve_output_depth(input_depths)
        if output_file_loc is not None:
            color = df['Color'].iloc[0] if 'Color' in df.columns else ''
            output_file_loc_abs = path / output_file_loc
            output_file_loc_abs.parent.mkdir(parents=True, exist_ok=True)
            first_tile_path = path / sample_row['Filepath']
            metadata = image_utils.build_postproc_output_metadata(
                input_path=first_tile_path,
                channel=color,
                significant_bits=output_depth,
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
            'significant_bits': output_depth,
            'metadata': {
                'center': center,
                'registered_tiles': registered_tiles,
            },
        }


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

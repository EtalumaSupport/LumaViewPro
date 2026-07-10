# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
from modules.stitching_core import (
    channel_aware_stitcher,
    overlap_stitcher,
    simple_position_stitcher,
)

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
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

        # A stitch spans every tile of a (well, channel), so the per-tile
        # token baked into the step name no longer identifies the output --
        # always drop it. A composite-stitch additionally spans all channels;
        # its stored Color is 'Composite', so the leaked channel token cannot
        # be matched by Color -- drop whichever channel token is present. A
        # single-channel stitch keeps its channel (a BF stitch is still BF).
        # Any custom name text is otherwise preserved.
        base_name = common_utils.strip_tile_token(row0['Name'], row0.get('Tile', ''))
        if row0.get('Color', '') == 'Composite':
            base_name = common_utils.strip_any_channel_token(base_name)

        # Prepend the protocol's capture_root (passed in via kwargs by
        # ProtocolPostProcessor.load_folder) so the stitched output
        # carries the same filename root as the per-image saves.
        capture_root = kwargs.get('capture_root', '')
        prefix = f'{capture_root}_{base_name}' if capture_root else base_name
        if kwargs.get('stitching_mode') == self.FAST_PREVIEW_MODE:
            prefix = f'{prefix}_{self._FAST_PREVIEW_SUFFIX}'
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

        return channel_aware_stitcher(
            path=path,
            df=df[stitch_columns],
            pixel_size_um=pixel_size_um,
            output_file_loc=kwargs.get('output_file_loc'),
            stitching_mode=kwargs.get('stitching_mode', self.QUALITY_MODE),
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
        required_cols = {'Filepath', 'X', 'Y', 'Objective'}
        if not required_cols.issubset(df.columns):
            missing = sorted(required_cols.difference(df.columns))
            return {
                'status': False,
                'error': f'missing required columns: {missing}',
            }
        try:
            objective = self._objectives_helper.get_objective_info(
                objective_id=df.iloc[0]['Objective']
            )
            pixel_size_um = common_utils.get_pixel_size(
                focal_length=objective['focal_length'],
                binning_size=1,
            )
        except Exception as e:
            return {
                'status': False,
                'error': f'unable to determine objective pixel size: {e}',
            }

        return overlap_stitcher(
            path=path,
            df=df,
            pixel_size_um=pixel_size_um,
            output_file_loc=output_file_loc,
        )


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
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
        post = ('stitched',)
        if kwargs.get('stitching_mode') == self.FAST_PREVIEW_MODE:
            post = ('stitched', self._FAST_PREVIEW_SUFFIX)
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                tile=None,
                scan_count=row0['Scan Count'],
                objective=objective_short_name,
                post=post,
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
        # Source pixel_size_um from the tile's own PhysicalSizeX, written at
        # capture with the actual binning already applied. Re-deriving it from
        # the objective focal length hardcoded binning_size=1, so a binned
        # (2x/4x) capture got half/quarter the true scale -- doubling the tile
        # pixel spacing and pulling the registered montage apart.
        first_tile_meta = image_utils.read_postproc_input_metadata(path / df.iloc[0]['Filepath'])
        pixel_size_um = first_tile_meta['pixel_size_um'] if first_tile_meta else None

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


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

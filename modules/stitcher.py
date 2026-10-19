# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
import modules.post_derivation as post_derivation
from modules.stitching_core import (
    channel_aware_stitcher,
    simple_position_stitcher,
)

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord
from modules.layer_record import ChannelColors

logger = logging.getLogger('LVP.modules.stitcher')


class Stitcher(ProtocolPostProcessor):
    QUALITY_MODE = 'quality'
    FAST_PREVIEW_MODE = 'fast_preview'

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.STITCHED,
            **kwargs,
        )
        self._name = self.__class__.__name__

    def _degraded_summary(self, count: int) -> str:
        # Stitching is the one post-processor with a named registration fallback
        # chain, so its degraded outputs are the montages that dropped to
        # unregistered edge-to-edge placement.
        return (
            f'{count} {self._post_function.value.lower()} artifact(s) used geometry-only '
            'fallback stitching. Review the mosaic geometry; the log names each reason.'
        )

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
                'Derivation',
            ],
            dropna=False,
        )

    def _derivation_step(self, group: pd.DataFrame, **kwargs) -> post_derivation.Step:
        return post_derivation.Step(
            PostFunction.STITCHED,
            (('mode', kwargs.get('stitching_mode', self.QUALITY_MODE)),),
        )

    def _generate_filename(self, df: pd.DataFrame, post: tuple[str, ...], **kwargs) -> str:
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
                post=post,
            )
        )

        prefix = self._prepend_capture_root(name, kwargs)
        outfile = f'{prefix}.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # A composite is a derived display/analysis artifact, not a raw channel
        # tile. Stitch raw channels first; a composite can only be generated
        # after those stitched channel geometries agree.
        df = df[df[PostFunction.COMPOSITE.value] == False]  # noqa: E712 -- pandas mask

        # Skip already stitched outputs
        df = df[df[self._post_function.value] == False]  # noqa: E712 -- pandas mask

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]  # noqa: E712 -- pandas mask

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]  # noqa: E712 -- pandas mask

        # A recording's frames are a time series, not tiles; stitching
        # them would emit a mislabeled artifact.
        df = self._without_video_frames(df)

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
        if pixel_size_um is None or pixel_size_um <= 0:
            # No PhysicalSizeX on the tiles (older / third-party / pre-metadata
            # captures) and binning is not recoverable to re-derive the scale, so
            # the stitchers cannot register by stage position and fall back to
            # unregistered edge-to-edge placement. Name the cause instead of
            # emitting a silently-misregistered montage under a generic reason.
            logger.warning(
                '[Stitch] No pixel size in tile metadata (PhysicalSizeX missing) for well=%s; '
                'montage will be unregistered edge-to-edge. Re-capture with current LVP or '
                'supply tiles carrying pixel-size metadata.',
                df.iloc[0].get('Well', '<unknown>'),
            )

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

        result = channel_aware_stitcher(
            path=path,
            df=df[stitch_columns],
            pixel_size_um=pixel_size_um,
            output_file_loc=kwargs.get('output_file_loc'),
            stitching_mode=kwargs.get('stitching_mode', self.QUALITY_MODE),
            channel_colors=kwargs['channel_colors'],
        )
        result.setdefault('metadata', {})['channel_group_id'] = self._stitched_channel_group(df)
        return PostProcResult.from_group_result(result)

    @staticmethod
    def _stitched_channel_group(df: pd.DataFrame) -> int:
        """The channel group a stitch of ``df``'s tiles belongs to: theirs when all share one, else -1.

        A stitch joins one color across positions, so its tiles may come
        from several Adds; the stitched image belongs to an Add only when
        every tile does, and a composite of stitched images then groups by
        it. -1 is no single Add.
        """
        group_ids = df['Channel Group ID'].unique()
        return int(group_ids[0]) if len(group_ids) == 1 else -1

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
            channel_group_id=alg_metadata['channel_group_id'],
            **kwargs,
        )

    @staticmethod
    def _simple_position_stitcher(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path | None = None,
        *,
        channel_colors: ChannelColors,
    ):
        return simple_position_stitcher(
            path=path,
            df=df,
            output_file_loc=output_file_loc,
            channel_colors=channel_colors,
        )


if __name__ == '__main__':
    stitcher = Stitcher(has_turret=False)
    stitcher.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

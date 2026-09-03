# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib

import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import modules.derived_output_encoding as derived_output_encoding
import modules.image_utils as image_utils

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord

import modules.zprojection as zprojection

from lvp_logger import logger


class ZProjector(ProtocolPostProcessor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            post_function=PostFunction.ZPROJECT,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Scan Count',
                'Well',
                'Color',
                'Objective',
                'X',
                'Y',
                'Tile',
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

        # A z-projection collapses every z-slice into one image, so the z token
        # is omitted (z_index=None) -- a single slice index would mislabel the
        # projection. channel and tile are kept (per-channel, per-tile output).
        # Post-output suffixes chain: a z-projection of an already-stitched
        # output carries both ('stitched', 'zproj_<method>').
        post = ('stitched',) if row0['Stitched'] else ()
        post = (*post, f'zproj_{kwargs["method"].lower()}')
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                z_index=None,
                scan_count=row0['Scan Count'],
                objective=objective_short_name,
                post=post,
            )
        )

        outfile = f'{self._prepend_capture_root(name, kwargs)}.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
        df = df[df[self._post_function.value] == False]  # noqa: E712 -- pandas mask

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]  # noqa: E712 -- pandas mask

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]  # noqa: E712 -- pandas mask

        # A recording's frames are a time series, not Z-slices; projecting
        # them would emit a mislabeled artifact.
        df = self._without_video_frames(df)

        return df

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return PostProcResult.from_group_result(
            self._zproject(
                path=path,
                df=df[['Filepath', 'Color']],
                method=kwargs['method'],
                output_file_loc=kwargs['output_file_loc'],
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
            x=row0['X'],
            y=row0['Y'],
            z='',
            z_slice='',
            well=row0['Well'],
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    @staticmethod
    def methods() -> list[str]:
        return zprojection.ZProjectMethod.list()

    def _zproject_for_multi_channel(
        self, images_data: list[np.ndarray], method: str
    ) -> np.ndarray | None:
        sample_image = images_data[0]
        used_color_planes = image_utils.get_used_color_planes(image=sample_image)
        out_image = np.zeros_like(sample_image, dtype=sample_image.dtype)

        for used_color_plane in used_color_planes:
            images_for_color_plane = []

            for image_data in images_data:
                images_for_color_plane.append(image_data[:, :, used_color_plane])

            project_result = zprojection.zproject(images_data=images_for_color_plane, method=method)

            if project_result is None:
                error = f'Failed to create Z-Projection for color plane {used_color_plane}'
                logger.error(error)
                return {
                    'status': False,
                    'error': error,
                }

            out_image[:, :, used_color_plane] = project_result

        return {
            'status': True,
            'error': None,
            'image': out_image,
            'metadata': {},
        }

    def _zproject_for_single_channel(
        self, images_data: list[np.ndarray], method: str
    ) -> np.ndarray | None:
        project_result = zprojection.zproject(images_data=images_data, method=method)

        if project_result is None:
            error = 'Failed to create Z-Projection'
            logger.error(error)
            return {
                'status': False,
                'error': error,
            }

        return {
            'status': True,
            'error': None,
            'image': project_result,
            'metadata': {},
        }

    def _zproject(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        method: str,
        output_file_loc: pathlib.Path,
    ):
        method = zprojection.ZProjectMethod[method]

        first_slice_row = df.iloc[0]
        first_slice_path = path / first_slice_row['Filepath']

        orig_images = []
        input_depths = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image, significant_bits = image_utils.load_pixels(
                image_filepath, collapse_legacy_false_color=False
            )
            orig_images.append(image)
            input_depths.append(significant_bits)
        output_depth = image_utils.resolve_output_depth(input_depths)

        try:
            # If working with color images, split the list of color images
            # into separate lists for each color plane
            if image_utils.is_color_image(image=orig_images[0]):
                result = self._zproject_for_multi_channel(
                    images_data=orig_images,
                    method=method,
                )
            else:  # Grayscale images
                result = self._zproject_for_single_channel(images_data=orig_images, method=method)
        finally:
            # Release source images immediately -- can be GBs for large stacks
            del orig_images

        if not result['status']:
            return result

        # Route through write_tiff so the projected output carries the
        # layer's PALETTE colormap plus the source acquisition context
        # forwarded from the first slice. Z position is inherited from
        # the first slice as a representative value -- the projection
        # collapses Z, so any single value is approximate.
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        metadata = image_utils.build_postproc_output_metadata(
            input_path=first_slice_path,
            channel=first_slice_row['Color'],
            significant_bits=output_depth,
        )
        image_utils.write_tiff(
            data=result['image'],
            file_loc=output_file_loc_abs,
            metadata=metadata,
            ome=False,
            color=first_slice_row['Color'],
            significant_bits=metadata['significant_bits'],
            save_encoding=derived_output_encoding.resolve_output_save_encoding(result['image']),
        )

        del result['image']
        result['significant_bits'] = output_depth

        return result

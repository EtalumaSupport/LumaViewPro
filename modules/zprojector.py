# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib

import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import modules.image_mode as image_mode
import modules.image_utils as image_utils
import modules.layer_record as layer_record
import modules.post_derivation as post_derivation

from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord

import modules.zprojection as zprojection
from modules.zproject_method import ZProjectMethod

from lvp_logger import logger


class ZProjector(ProtocolPostProcessor):
    _PLACE = 'slice'
    _ONE_PER_ADD = ' Steps Added together make one projection.'

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
                'Channel Group ID',
                'Raw',
                'Derivation',
            ],
            dropna=False,
        )

    def _derivation_step(
        self, group: pd.DataFrame, output_format: image_mode.OutputFormat, **kwargs
    ) -> post_derivation.Step:
        return post_derivation.Step(
            PostFunction.ZPROJECT,
            (
                ('method', kwargs['method'].lower()),
                *post_derivation.format_settings(output_format),
            ),
        )

    def _places(self, group: pd.DataFrame) -> pd.Series:
        # By the slice's Z, not its index: an inserted step's index is -1.
        return group['Z']

    def _generate_filename(
        self,
        df: pd.DataFrame,
        post: tuple[str, ...],
        output_format: image_mode.OutputFormat,
        **kwargs,
    ) -> str:
        row0 = df.iloc[0]
        objective_short_name = self._get_objective_short_name_if_has_turret(
            objective_id=row0['Objective']
        )

        # A z-projection collapses every z-slice into one image, so the z token
        # is omitted (z_index=None) -- a single slice index would mislabel the
        # projection. channel and tile are kept (per-channel, per-tile output).
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                z_index=None,
                scan_count=row0['Scan Count'],
                objective=objective_short_name,
                post=post,
            )
        )

        prefix = self._prepend_capture_root(name, kwargs)
        return f'{prefix}{image_utils.derived_file_suffix(output_format)}'

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
                channel_colors=kwargs['channel_colors'],
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
            channel_group_id=row0['Channel Group ID'],
            **kwargs,
        )

    def _zproject_for_multi_channel(
        self, images_data: list[np.ndarray], method: ZProjectMethod
    ) -> np.ndarray | None:
        sample_image = images_data[0]
        used_color_planes = image_utils.get_used_color_planes(image=sample_image)
        # Allocated at the projection's dtype, not the input's: a Sum of uint8
        # planes is uint16 and would wrap in a uint8 canvas.
        out_image = np.zeros(
            sample_image.shape,
            dtype=zprojection.projected_dtype(method, sample_image.dtype),
        )

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
        self, images_data: list[np.ndarray], method: ZProjectMethod
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
        method: ZProjectMethod,
        output_file_loc: pathlib.Path,
        channel_colors: layer_record.ChannelColors,
    ):
        method = ZProjectMethod(method)

        first_slice_row = df.iloc[0]

        orig_images = []
        input_depths = []
        input_scales = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image, significant_bits, white_bits = image_utils.load_pixels(
                image_filepath, collapse_legacy_false_color=False
            )
            orig_images.append(image)
            input_depths.append(significant_bits)
            input_scales.append((significant_bits, white_bits))
        input_depth = image_utils.resolve_output_depth(input_depths)
        # Every method is shown against its slices' white: a Sum is a sum, shown
        # against one slice's white as a captured sum is against one frame's,
        # and every other method keeps its values within its slices' range.
        output_white = image_utils.resolve_output_white(input_scales)
        # A Sum reaches past its inputs' range and is tagged with the bits it
        # can reach, by the rule a summed capture is tagged with; every other
        # method keeps its values within the inputs' range.
        if method == ZProjectMethod.Sum:
            output_depth = image_utils.summed_significant_bits(len(input_depths), input_depth)
        else:
            output_depth = input_depth

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
        # layer's PALETTE colormap plus the acquisition context its slices
        # agree on. The projection collapses Z, so slices at different Z
        # state none.
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        metadata = image_utils.build_postproc_output_metadata(
            input_records=[image_utils.read_file_record(path / name) for name in df['Filepath']],
            channel=first_slice_row['Color'],
            significant_bits=output_depth,
            # A slice's frame count says how many exposures each of its pixels
            # integrates; a Sum adds the slices, so beside the per-frame exposure
            # it would state a fraction of the output's integration. Every other
            # method keeps each pixel one slice's value or a statistic of them.
            adds_frames=method == ZProjectMethod.Sum,
        )
        image_utils.write_tiff(
            data=result['image'],
            file_loc=output_file_loc_abs,
            metadata=metadata,
            # The format its name states, the one its slices share.
            ome=image_utils.is_ome_tiff(output_file_loc),
            color=layer_record.output_color(first_slice_row['Color'], channel_colors),
            significant_bits=metadata['significant_bits'],
            save_encoding=image_mode.encoding_for_array(result['image']),
            white_bits=output_white,
        )

        del result['image']
        result['significant_bits'] = output_depth

        return result

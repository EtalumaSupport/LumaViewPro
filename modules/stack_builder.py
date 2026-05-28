# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import pathlib

import numpy as np
import pandas as pd
import psutil
import tifffile as tf

import modules.image_utils as image_utils

import logging

logger = logging.getLogger('lvp_logger')


def _check_hyperstack_memory(num_t, num_z, num_c, h, w, dtype):
    """Raise MemoryError if hyperstack would exceed 80% of available RAM."""
    bytes_per_element = np.dtype(dtype).itemsize
    required_bytes = num_t * num_z * num_c * h * w * bytes_per_element
    available_bytes = psutil.virtual_memory().available
    if required_bytes > available_bytes * 0.8:
        raise MemoryError(
            f'Hyperstack requires {required_bytes / 1e9:.1f} GB but only '
            f'{available_bytes / 1e9:.1f} GB available. Reduce Z-slices, '
            f'timepoints, or channels.'
        )


import modules.common_utils as common_utils
from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_record import ProtocolPostRecord


class StackBuilder(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            post_function=PostFunction.HYPERSTACK,
            *args,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Well',
                'Objective',
                'X',
                'Y',
                'Tile',
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
        # ProtocolPostProcessor.load_folder) so the stack output carries
        # the same filename root as the per-image saves.
        capture_root = kwargs.get('capture_root', '')
        prefix = f'{capture_root}_{row0["Name"]}' if capture_root else row0['Name']

        name = common_utils.generate_default_step_name(
            custom_name_prefix=prefix,
            well_label=row0['Well'],
            color=None,
            z_height_idx=None,
            scan_count=None,
            tile_label=None,
            objective_short_name=objective_short_name,
            hyperstack=True,
        )

        outfile = f'{name}.ome.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Only process raw
        df = df[df['Raw'] == True]

        return df

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return StackBuilder._create_stack(
            path=path,
            df=df,
            output_file_loc=kwargs['output_file_loc'],
            focal_length=kwargs['focal_length'],
            binning_size=kwargs['binning_size'],
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
            scan_count=-1,
            x=row0['X'],
            y=row0['Y'],
            z=-1,
            z_slice=-1,
            well=row0['Well'],
            color='Stack',
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    @staticmethod
    def _generate_image_metadata(
        df: pd.DataFrame,
        path: pathlib.Path,
        output_file_loc: pathlib.Path,
        plane_metadata: dict,
        binning_size: int,
        focal_length: float,
    ):
        channel_names = df['Color'].unique().tolist()
        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        sample_image = tf.imread(sample_image_file_loc)

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=focal_length,
                binning_size=binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )

        metadata = image_utils.build_hyperstack_output_metadata(
            reference_input_path=sample_image_file_loc,
            channel_names=channel_names,
            plane_positions={
                'PositionX': plane_metadata['PositionX'],
                'PositionY': plane_metadata['PositionY'],
                'PositionZ': plane_metadata['PositionZ'],
            },
            significant_bits=sample_image.itemsize * 8,
            pixel_size_um=pixel_size_um,
        )

        options = dict(
            photometric='minisblack',
            tile=(128, 128),
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2,
        )

        resolution = (1e4 / pixel_size_um, 1e4 / pixel_size_um)

        return {
            'metadata': metadata,
            'options': options,
            'resolution': resolution,
        }

    @staticmethod
    def _create_stack(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path,
        focal_length: float,
        binning_size: int,
        sort_order: list[str] | None = None,
    ):
        if sort_order is None:
            sort_order = ['Scan Count', 'Z-Slice', 'Color Index']

        num_t = df['Scan Count'].nunique()
        num_z = df['Z-Slice'].nunique()
        num_c = df['Color'].nunique()

        _, color_idx_map = np.unique(df['Color'], return_inverse=True)
        df['Color Index'] = color_idx_map

        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        sample_image = tf.imread(sample_image_file_loc)
        sample_image_shape = sample_image.shape
        h, w = sample_image_shape[0], sample_image_shape[1]

        _check_hyperstack_memory(num_t, num_z, num_c, h, w, sample_image.dtype)
        stacked_image = np.zeros(
            shape=(num_t, num_z, num_c, h, w),  # Hyperstack order TZCYX
            dtype=sample_image.dtype,
        )

        df = df.sort_values(by=sort_order, ascending=True)

        plane_metadata = {
            'PositionX': [],
            'PositionY': [],
            'PositionZ': [],
        }

        for _, row in df.iterrows():
            t = row['Scan Count']
            z = row['Z-Slice']
            c = row['Color Index']
            image = tf.imread(path / row['Filepath'])

            if image_utils.is_color_image(image):
                image = image_utils.rgb_image_to_gray(image=image)

            stacked_image[t, z, c, :, :] = image
            plane_metadata['PositionX'].append(row['X'])
            plane_metadata['PositionY'].append(row['Y'])
            plane_metadata['PositionZ'].append(row['Z'])

        num_planes = len(plane_metadata['PositionX'])
        plane_metadata['PositionXUnit'] = num_planes * ['mm']
        plane_metadata['PositionYUnit'] = num_planes * ['mm']
        plane_metadata['PositionZUnit'] = num_planes * ['um']

        ome_info = StackBuilder._generate_image_metadata(
            df=df,
            path=path,
            output_file_loc=output_file_loc,
            plane_metadata=plane_metadata,
            focal_length=focal_length,
            binning_size=binning_size,
        )

        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        # Route through write_tiff's hyperstack override hook so the
        # canonical LVP write path owns the file-creation side of the
        # save pipeline. metadata / ome / color are unused on this path
        # (the hyperstack_* kwargs carry the OME-XML + write options);
        # the placeholder kwargs satisfy the existing signature.
        image_utils.write_tiff(
            data=stacked_image,
            file_loc=output_file_loc_abs,
            metadata={},
            ome=True,
            color='',
            hyperstack_metadata=ome_info['metadata'],
            hyperstack_options=ome_info['options'],
            hyperstack_resolution=ome_info['resolution'],
        )

        return {'status': True, 'error': None, 'metadata': {}}

    @staticmethod
    def create_single_recording_stack(
        df: pd.DataFrame,
        path: pathlib.Path,
        output_file_loc: pathlib.Path,
        focal_length: float,
        binning_size: int,
    ):
        # Manual-recording entry point: sorts by Scan Count alone (Z and
        # Color axes collapse to single values for single recordings)
        # and accepts an absolute output_file_loc that the caller has
        # already resolved against the save folder. Delegates to
        # _create_stack for the canonical write path; output_file_loc
        # is normalized to relative-to-path so _create_stack's internal
        # `path / output_file_loc` join reconstructs the original
        # absolute target.
        try:
            rel_loc = output_file_loc.relative_to(path)
        except ValueError:
            rel_loc = pathlib.Path(output_file_loc.name)
        return StackBuilder._create_stack(
            path=path,
            df=df,
            output_file_loc=rel_loc,
            focal_length=focal_length,
            binning_size=binning_size,
            sort_order=['Scan Count'],
        )


if __name__ == '__main__':
    stack_builder = StackBuilder(has_turret=False)
    tiling_configs_file_loc = pathlib.Path(os.getenv('SOURCE_ROOT')) / 'data' / 'tiling.json'
    stack_builder.load_folder(
        path=os.getenv('SAMPLE_IMAGE_FOLDER'),
        tiling_configs_file_loc=tiling_configs_file_loc,
        focal_length=45.0,
        binning_size=1,
    )

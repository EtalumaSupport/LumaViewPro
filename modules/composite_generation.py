# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules.composite_builder import build_composite
import modules.image_utils as image_utils
from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_record import ProtocolPostRecord
from modules.settings_init import settings
from lvp_logger import logger


class CompositeGeneration(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            post_function=PostFunction.COMPOSITE,
            *args,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Scan Count',
                'Well',
                'Objective',
                'X',
                'Y',
                'Z-Slice',
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

        # Prepend custom root + step name if available
        custom_root = row0.get('Custom Root', '') if 'Custom Root' in row0 else ''
        if custom_root not in (None, ''):
            prefix = f'{custom_root}_{row0["Name"]}'
        else:
            prefix = row0['Name']
        name = common_utils.generate_default_step_name(
            custom_name_prefix=prefix,
            well_label=row0['Well'],
            color='Composite',
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count'],
            objective_short_name=objective_short_name,
            tile_label=row0['Tile'],
            stitched=row0['Stitched'],
        )

        outfile = f'{name}.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
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
        # output_file_loc is set by the base class _process_group_callback
        # before dispatch (protocol_post_processor.py:169). Forward it so the
        # composite save happens inside _create_composite_image via tifffile
        # RGB-native, mirroring the manual composite path's tifffile-based
        # save. Returning {'image': None, ...} from _create_composite_image
        # tells the base class to skip its cv2.imwrite (which would swap
        # channels because cv2.imwrite is BGR-oriented).
        output_file_loc_rel = kwargs.get('output_file_loc')
        return CompositeGeneration._create_composite_image(
            path=path,
            df=df[['Filepath', 'Color']],
            output_file_loc=path / output_file_loc_rel if output_file_loc_rel else None,
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
            x=row0['X'],
            y=row0['Y'],
            z=row0['Z'],
            z_slice=row0['Z-Slice'],
            well=row0['Well'],
            color=alg_metadata['color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    @staticmethod
    def _create_composite_image(
        path: pathlib.Path, df: pd.DataFrame, output_file_loc: pathlib.Path = None
    ):

        BF_present = False
        BF_channel = ''

        allowed_BF_layers = common_utils.get_transmitted_layers()
        allowed_layers = [
            *common_utils.get_fluorescence_layers(),
            *common_utils.get_luminescence_layers(),
        ]
        img = None

        for layer in allowed_BF_layers:
            if (df['Color'] == layer).any():
                BF_present = True
                BF_channel = layer
                allowed_layers.append(layer)
                break

        df = df[df['Color'].isin(allowed_layers)]

        # Load source images via tifffile (RGB-native, returns mono 2D for
        # single-channel TIFFs; no BGR upconvert -> no downstream cvtColor
        # ceremony to undo it). Mirrors the manual composite path's
        # in-memory mono inputs so build_composite consumes the same
        # shape from both orchestrators.
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = tf.imread(str(image_filepath))

        error = None
        status = True

        try:
            transmitted_image = None
            channel_images = {}
            brightness_thresholds = {}
            img_dtype = None

            if BF_present:
                logger.info('CompositeGeneration] Generating transmitted channel composite')
                BF_row = df[df['Color'] == BF_channel]
                BF_image_filename = BF_row['Filepath'].iloc[0]
                BF_image = images[BF_image_filename]
                img_dtype = BF_image.dtype
                # 3-channel TIFFs come from the false-color save path
                # (per-channel TIFFs widened to RGB with one plane
                # populated). The RGB2GRAY luminance collapse used
                # before attenuated single-plane data by 41-89% of its
                # original intensity, producing a "mostly green"
                # composite for Blue and Red layers. rgb_image_to_gray
                # detects the single-populated-plane case and uses
                # max-axis extraction instead, preserving the full
                # value. Mono 2D inputs (the common path) pass through.
                transmitted_image = image_utils.rgb_image_to_gray(BF_image)
            else:
                logger.info('CompositeGeneration] Generating fluorescent channel composite')

            for _, row in df.iterrows():
                layer = row['Color']

                # Skip transmitted layer (already captured above)
                if layer == BF_channel:
                    continue

                # Skip non-fluorescence layers
                if layer not in common_utils.get_image_layers():
                    continue

                f_image = images[row['Filepath']]
                if img_dtype is None:
                    img_dtype = f_image.dtype

                # Same collapse as the transmitted path: detect
                # single-populated-plane (the false-color save shape)
                # and preserve via max-axis; pass through mono 2D
                # inputs unchanged. RGB2GRAY luminance was destroying
                # 41-89% of single-plane signal pre-fix.
                img_gray = image_utils.rgb_image_to_gray(f_image)

                channel_images[layer] = img_gray

                # Compute brightness threshold
                if BF_present:
                    if img_dtype == np.uint8:
                        max_value = 255
                    else:
                        max_value = 4095
                    ctx = _app_ctx.ctx
                    if ctx is not None:
                        with ctx.settings_lock:
                            threshold = settings[layer]['composite_brightness_threshold']
                    else:
                        threshold = settings[layer]['composite_brightness_threshold']
                    brightness_thresholds[layer] = threshold / 100 * max_value

            if not channel_images and transmitted_image is None:
                status = False
                error = 'Composite Generation Error: no channel images available for this group'
            else:
                dtype = img_dtype or np.uint8
                max_value = 255 if dtype == np.uint8 else 4095

                # build_composite returns RGB. Both the manual composite
                # path (ui/composite_capture.py) and this protocol
                # post-processing path now save RGB-native via tifffile;
                # no cvtColor RGB->BGR detour (which the old cv2.imwrite
                # path required and which produced the R/B channel swap
                # at the heart of #672). Save here so the base-class
                # cv2.imwrite branch (protocol_post_processor.py:190) is
                # bypassed for the composite case -- returning
                # 'image': None below tells the base class the subclass
                # has already written.
                img = build_composite(
                    channel_images=channel_images,
                    transmitted_image=transmitted_image,
                    brightness_thresholds=brightness_thresholds,
                    dtype=dtype,
                    max_value=max_value,
                )
                if output_file_loc is not None:
                    output_file_loc.parent.mkdir(parents=True, exist_ok=True)
                    tf.imwrite(
                        str(output_file_loc),
                        img,
                        photometric='rgb',
                        compression='lzw',
                    )

        except Exception as e:
            logger.error(f'CompositeGeneration] Error generating composite: {e}')
            error = f'Error generating composite: {e}'
            status = False

        # When output_file_loc is provided: this subclass wrote the
        # composite TIFF itself; signal to the base-class
        # _process_group_callback that no further write is needed by
        # returning 'image': None. When no output_file_loc (legacy /
        # test callers): return the RGB array so the caller can save.
        if output_file_loc is not None:
            return_image = None
        else:
            return_image = img if status else None
            if status and return_image is None:
                status = False
                error = 'Composite Generation Error: No final image'

        return {
            'status': status,
            'error': error,
            'image': return_image,
            'metadata': {
                'color': 'Composite',
            },
        }


if __name__ == '__main__':
    composite_gen = CompositeGeneration(has_turret=False)
    composite_gen.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules.composite_builder import build_composite
import modules.image_utils as image_utils
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord
from modules.settings_init import settings
from lvp_logger import logger


class CompositeGeneration(ProtocolPostProcessingExecutor):

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
                *PostFunction.list_values()
            ],
            dropna=False
        )
    

    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        row0 = df.iloc[0]

        objective_short_name = self._get_objective_short_name_if_has_turret(objective_id=row0['Objective'])

        # Prepend custom root + step name if available
        custom_root = row0.get('Custom Root', '') if 'Custom Root' in row0 else ''
        if custom_root not in (None, ''):
            prefix = f"{custom_root}_{row0['Name']}"
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

        outfile = f"{name}.tiff"
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
        return CompositeGeneration._create_composite_image(
            path=path,
            df=df[['Filepath','Color']]
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
    def _create_composite_image(path: pathlib.Path, df: pd.DataFrame):

        BF_present = False
        BF_channel = ""

        allowed_BF_layers = common_utils.get_transmitted_layers()
        allowed_layers = [*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers()]
        img = None

        for layer in allowed_BF_layers:
            if (df['Color'] == layer).any():
                BF_present = True
                BF_channel = layer
                allowed_layers.append(layer)
                break

        df = df[df['Color'].isin(allowed_layers)]

        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        error = None
        status = True

        try:
            transmitted_image = None
            channel_images = {}
            brightness_thresholds = {}
            img_dtype = None

            if BF_present:
                logger.info("CompositeGeneration] Generating transmitted channel composite")
                BF_row = df[df['Color'] == BF_channel]
                BF_image_filename = BF_row['Filepath'].iloc[0]
                BF_image = images[BF_image_filename]
                img_dtype = BF_image.dtype
                # cv2.imread returns BGR — convert grayscale transmitted to plain 2D array
                if image_utils.is_color_image(BF_image):
                    transmitted_image = cv2.cvtColor(BF_image, cv2.COLOR_BGR2GRAY)
                else:
                    transmitted_image = BF_image
            else:
                logger.info("CompositeGeneration] Generating fluorescent channel composite")

            for _, row in df.iterrows():
                layer = row['Color']

                # Skip transmitted layer (already captured above)
                if layer == BF_channel:
                    continue

                # Skip non-fluorescence layers
                if layer not in ('Red', 'Green', 'Blue', 'Lumi'):
                    continue

                f_image = images[row['Filepath']]
                if img_dtype is None:
                    img_dtype = f_image.dtype

                # Convert color images to grayscale
                if image_utils.is_color_image(f_image):
                    img_gray = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = np.array(f_image)

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
                            threshold = settings[layer]["composite_brightness_threshold"]
                    else:
                        threshold = settings[layer]["composite_brightness_threshold"]
                    brightness_thresholds[layer] = threshold / 100 * max_value

            if not channel_images and transmitted_image is None:
                status = False
                error = "Composite Generation Error: No images found"
            else:
                dtype = img_dtype or np.uint8
                max_value = 255 if dtype == np.uint8 else 4095

                # build_composite returns RGB — convert to BGR for cv2.imwrite output
                img_rgb = build_composite(
                    channel_images=channel_images,
                    transmitted_image=transmitted_image,
                    brightness_thresholds=brightness_thresholds,
                    dtype=dtype,
                    max_value=max_value,
                )
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"CompositeGeneration] Error generating composite: {e}")
            error = f"Error generating composite: {e}"
            status = False

        if img is None and status:
            status = False
            error = "Composite Generation Error: No final image"

        return {
            'status': status,
            'error': error,
            'image': img,
            'metadata': {
                'color': 'Composite',
            }
        }
       

if __name__ == "__main__":
    composite_gen = CompositeGeneration(has_turret=False)
    composite_gen.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))

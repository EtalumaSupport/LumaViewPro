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


def _strip_channel_token(name: str, channel: str) -> str:
    """Remove a single channel token from a per-channel step name.

    A composite merges every channel for one (well, position) into a single
    image, so the per-channel step name (e.g. 'A1_Green') must not tag the
    composite output with one arbitrary channel. Removes the first
    '_<channel>' (or a leading '<channel>_') token; returns the name
    unchanged when channel is empty or not present.
    """
    if not channel:
        return name
    token = str(channel)
    if f'_{token}' in name:
        return name.replace(f'_{token}', '', 1)
    if name.startswith(f'{token}_'):
        return name.replace(f'{token}_', '', 1)
    return name


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

        # The step name carries the channel (e.g. 'A1_Green'); a composite
        # spans all channels, so drop that token before it becomes the prefix.
        base_name = _strip_channel_token(row0['Name'], row0.get('Color', ''))

        # Prepend the protocol's capture_root (passed in via kwargs by
        # ProtocolPostProcessor.load_folder) so post-processed outputs
        # carry the same filename root as the per-image saves.
        capture_root = kwargs.get('capture_root', '')
        if capture_root:
            prefix = f'{capture_root}_{base_name}'
        else:
            prefix = base_name
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
        # output_file_loc is set by the base class _process_group_callback
        # before dispatch (protocol_post_processor.py:169). Forward it so the
        # composite save happens inside _create_composite_image via tifffile
        # RGB-native, mirroring the manual composite path's tifffile-based
        # save. Returning {'image': None, ...} from _create_composite_image
        # tells the base class to skip its cv2.imwrite (which would swap
        # channels because cv2.imwrite is BGR-oriented).
        output_file_loc_rel = kwargs.get('output_file_loc')
        output_format = kwargs.get('output_format', 'TIFF')
        return CompositeGeneration._create_composite_image(
            path=path,
            df=df[['Filepath', 'Color']],
            output_file_loc=path / output_file_loc_rel if output_file_loc_rel else None,
            output_format=output_format,
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
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path = None,
        output_format: str = 'TIFF',
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
                    reference_input_path = path / df.iloc[0]['Filepath']
                    metadata = image_utils.build_composite_output_metadata(
                        reference_input_path=reference_input_path,
                    )
                    # Honor the run's output format. The composite is a
                    # single 2D RGB image, so only plain OME-TIFF applies --
                    # 'OME-TIFF Hyperstack' has no per-frame meaning here and
                    # falls through to a plain TIFF, matching the per-frame
                    # hyperstack downgrade. Normalize case so the run-config
                    # token ('OME-TIFF') and the path-API token ('ome-tiff')
                    # both map to ome=True.
                    ome = output_format.strip().upper() == 'OME-TIFF'
                    image_utils.write_tiff(
                        data=img,
                        file_loc=output_file_loc,
                        metadata=metadata,
                        ome=ome,
                        color='Composite',
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


    _SUPPORTED_FORMATS = ('tiff', 'ome-tiff')

    def generate_composite_from_paths(
        self,
        output_path: pathlib.Path,
        *,
        red_path: pathlib.Path | None = None,
        green_path: pathlib.Path | None = None,
        blue_path: pathlib.Path | None = None,
        transmitted_path: pathlib.Path | None = None,
        transmitted_layer: str | None = None,
        format: str = 'tiff',
    ) -> dict:
        """Build a composite from per-channel mono TIFFs at the given paths.

        Public path-based entry point that wraps the protocol-post-processor
        DataFrame pipeline (load_folder -> _group_algorithm ->
        _create_composite_image). Synthesizes the minimal DataFrame the
        underlying composite builder needs from the per-channel path args.

        Args:
            output_path: Destination file.
            red_path, green_path, blue_path: Mono TIFF inputs per fluorescence
                channel. None = channel absent from composite.
            transmitted_path: Optional brightfield / phase-contrast / darkfield
                input; folded into composite via the build_composite
                transmitted path.
            transmitted_layer: Layer name for transmitted_path (e.g. 'BF',
                'PC', 'DF'). Required when transmitted_path is set;
                rejected otherwise.
            format: Output format. Supported: 'tiff' (plain RGB) and
                'ome-tiff' (OME-TIFF with axes='YXS'). Reserved for future:
                'png', 'jpg'. ValueError for any other value.

        Returns:
            {'status': bool, 'error': str | None, 'image': np.ndarray | None}.

        Raises:
            ValueError: If format is unsupported, if transmitted_path and
                transmitted_layer are not both-set or both-None, or if no
                channel paths are provided.
        """
        output_path = pathlib.Path(output_path)

        if format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"generate_composite_from_paths: unsupported format '{format}'. "
                f'Supported: {self._SUPPORTED_FORMATS}.'
            )

        if (transmitted_path is None) != (transmitted_layer is None):
            raise ValueError(
                'generate_composite_from_paths: transmitted_path and '
                'transmitted_layer must be provided together.'
            )

        rows = []
        if red_path is not None:
            rows.append({'Color': 'Red', 'Filepath': pathlib.Path(red_path)})
        if green_path is not None:
            rows.append({'Color': 'Green', 'Filepath': pathlib.Path(green_path)})
        if blue_path is not None:
            rows.append({'Color': 'Blue', 'Filepath': pathlib.Path(blue_path)})
        if transmitted_path is not None:
            rows.append(
                {'Color': transmitted_layer, 'Filepath': pathlib.Path(transmitted_path)}
            )

        if not rows:
            raise ValueError(
                'generate_composite_from_paths: at least one channel path required.'
            )

        df = pd.DataFrame(rows)

        # Run the canonical composite builder with output_file_loc=None to
        # get the RGB array back; the wrapper handles the write with
        # format-aware kwargs. Absolute paths in df['Filepath'] survive
        # the `path / row['Filepath']` join inside _create_composite_image
        # (pathlib discards the left operand when the right is absolute).
        result = CompositeGeneration._create_composite_image(
            path=pathlib.Path('.'),
            df=df,
            output_file_loc=None,
        )

        if not result['status'] or result.get('image') is None:
            return {
                'status': False,
                'error': result.get('error') or 'Composite Generation Error: no image produced',
                'image': None,
            }

        img = result['image']
        output_path.parent.mkdir(parents=True, exist_ok=True)

        reference_input_path = (
            red_path or green_path or blue_path or transmitted_path
        )
        metadata = image_utils.build_composite_output_metadata(
            reference_input_path=pathlib.Path(reference_input_path),
        )
        image_utils.write_tiff(
            data=img,
            file_loc=output_path,
            metadata=metadata,
            ome=(format == 'ome-tiff'),
            color='Composite',
        )

        return {
            'status': True,
            'error': None,
            'image': img,
        }


if __name__ == '__main__':
    composite_gen = CompositeGeneration(has_turret=False)
    composite_gen.load_folder(pathlib.Path(os.getenv('SAMPLE_IMAGE_FOLDER')))

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib

import pandas as pd

import modules.common_utils as common_utils
from modules.composite_builder import build_composite
import modules.image_mode as image_mode
import modules.image_utils as image_utils
from modules.common_utils import PostFunction
from modules.exceptions import ConfigError
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord
from lvp_logger import logger


class CompositeGeneration(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.COMPOSITE,
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

        # A composite spans every channel, so it is named 'Composite' in place
        # of the per-channel token. The identity is built from the authoritative
        # columns (channel forced to 'Composite'), never re-parsed from the
        # prior name string, so a stale channel token cannot leak in.
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                channel='Composite',
                scan_count=row0['Scan Count'],
                objective=objective_short_name,
                post=('stitched',) if row0['Stitched'] else (),
            )
        )

        # The extension follows the run's chosen container, matching what the
        # capture path writes: an OME composite named plain .tiff reads as a
        # different format than it is, and the readers that branch on the
        # double extension would take the wrong path.
        if kwargs.get('output_format') == image_mode.OUTPUT_FORMAT_OME_TIFF:
            extension = '.ome.tiff'
        else:
            extension = '.tiff'
        outfile = f'{self._prepend_capture_root(name, kwargs)}{extension}'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
        df = df[df[self._post_function.value] == False]  # noqa: E712 -- pandas mask

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]  # noqa: E712 -- pandas mask

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]  # noqa: E712 -- pandas mask

        # A recording's frames are a time series, not per-channel stills;
        # compositing them would emit a mislabeled artifact.
        df = self._without_video_frames(df)

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
        if 'brightness_thresholds_percent' not in kwargs:
            raise ConfigError(
                'composite generation requires brightness_thresholds_percent; '
                'the caller snapshots it from settings, because this runs on a '
                'worker thread that must not read live configuration'
            )
        return PostProcResult.from_group_result(
            CompositeGeneration._create_composite_image(
                path=path,
                df=df[['Filepath', 'Color']],
                brightness_thresholds_percent=kwargs['brightness_thresholds_percent'],
                output_file_loc=path / output_file_loc_rel if output_file_loc_rel else None,
                output_format=output_format,
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
        brightness_thresholds_percent: dict,
        output_file_loc: pathlib.Path | None = None,
        output_format: str = 'TIFF',
    ):
        """Merge one group's per-channel frames into a single composite.

        brightness_thresholds_percent maps each fluorescence layer to the
        percentage below which its pixels are not composited onto a
        transmitted base. It is a required argument rather than a settings
        read: this runs unattended on a worker thread with no GUI, and the
        module-level settings global it used to read is published only by
        the app bootstrap, so every headless caller found it None and
        crashed at the threshold lookup. Passing the values in makes the
        headless and GUI paths identical by construction.
        """

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
        input_depths = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image, significant_bits = image_utils.load_pixels(
                image_filepath, collapse_legacy_false_color=False
            )
            images[row['Filepath']] = image
            input_depths.append(significant_bits)
        # Empty after layer filtering is a handled no-op below (status=False,
        # no write), so only resolve a depth when inputs were actually loaded.
        output_depth = image_utils.resolve_output_depth(input_depths) if input_depths else None
        # The composite's own depth, set from the built array (always 8-bit).
        # Distinct from output_depth, which is the INPUT depth driving downconvert.
        composite_significant_bits = None

        # Validated before the try, not inside it: a missing threshold is the
        # caller omitting an argument, and the try below turns everything it
        # catches into a per-group error dict -- which would report a coding
        # mistake as "every image group failed". Only checked when a
        # transmitted base is present, since that is the only case that
        # blends and therefore the only case that consults a threshold.
        if BF_present:
            missing = sorted(
                layer
                for layer in df['Color']
                if layer != BF_channel
                and layer in common_utils.get_image_layers()
                and layer not in brightness_thresholds_percent
            )
            if missing:
                raise ConfigError(
                    f'composite generation needs a brightness threshold for '
                    f'{", ".join(missing)}; the caller supplies one per '
                    f'fluorescence layer so the merge cannot silently pick its own'
                )

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

                # Compute brightness threshold on the composite's 8-bit output
                # scale (build_composite downconverts the channels to 8-bit).
                if BF_present:
                    brightness_thresholds[layer] = brightness_thresholds_percent[layer] / 100 * 255

            if not channel_images and transmitted_image is None:
                status = False
                error = 'Composite Generation Error: no channel images available for this group'
            else:
                # build_composite returns 8-bit RGB (a composite is a viewing
                # product; the native depth belongs to the raw single-channel
                # captures, not the merged image). Both the manual composite
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
                    significant_bits=output_depth,
                    transmitted_image=transmitted_image,
                    brightness_thresholds=brightness_thresholds,
                )
                composite_significant_bits = img.dtype.itemsize * 8
                if output_file_loc is not None:
                    output_file_loc.parent.mkdir(parents=True, exist_ok=True)
                    reference_input_path = path / df.iloc[0]['Filepath']
                    metadata = image_utils.build_composite_output_metadata(
                        reference_input_path=reference_input_path,
                        significant_bits=composite_significant_bits,
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
                        significant_bits=metadata['significant_bits'],
                        # A merged composite is a viewing product, always
                        # 8-bit RGB, so its encoding follows from that and
                        # not from the live image mode: the merge runs
                        # inside the engine on every run kind, headless
                        # included, where there is no live mode to read.
                        save_encoding=image_mode.save_encoding_for_derived_output(
                            img, image_mode.IMAGE_MODE_8BIT
                        ),
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
            'significant_bits': composite_significant_bits,
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
        brightness_thresholds_percent: dict | None = None,
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
            rows.append({'Color': transmitted_layer, 'Filepath': pathlib.Path(transmitted_path)})

        if not rows:
            raise ValueError('generate_composite_from_paths: at least one channel path required.')

        df = pd.DataFrame(rows)

        # Run the canonical composite builder with output_file_loc=None to
        # get the RGB array back; the wrapper handles the write with
        # format-aware kwargs. Absolute paths in df['Filepath'] survive
        # the `path / row['Filepath']` join inside _create_composite_image
        # (pathlib discards the left operand when the right is absolute).
        # Thresholds are only consulted when a transmitted base is present;
        # an empty mapping with one supplied raises rather than defaulting,
        # so a caller that means to blend cannot get an unstated threshold.
        result = CompositeGeneration._create_composite_image(
            path=pathlib.Path('.'),
            df=df,
            brightness_thresholds_percent=brightness_thresholds_percent or {},
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

        reference_input_path = red_path or green_path or blue_path or transmitted_path
        # Depth travels back from _create_composite_image, which read the
        # inputs via load_pixels -- no second open of the source files.
        metadata = image_utils.build_composite_output_metadata(
            reference_input_path=pathlib.Path(reference_input_path),
            significant_bits=result['significant_bits'],
        )
        image_utils.write_tiff(
            data=img,
            file_loc=output_path,
            metadata=metadata,
            ome=(format == 'ome-tiff'),
            color='Composite',
            significant_bits=metadata['significant_bits'],
            # 8-bit RGB by ruling, as above.
            save_encoding=image_mode.save_encoding_for_derived_output(
                img, image_mode.IMAGE_MODE_8BIT
            ),
        )

        return {
            'status': True,
            'error': None,
            'image': img,
        }

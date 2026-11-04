# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib

import pandas as pd

import modules.common_utils as common_utils
from modules.finite_number import is_finite_number
from modules.composite_builder import (
    CompositeChannel,
    TransmittedBase,
    brightness_cutoff_from_percent,
    build_composite,
)
import modules.image_mode as image_mode
import modules.image_utils as image_utils
import modules.layer_record as layer_record
import modules.post_derivation as post_derivation
from modules.layer_record import GREY
from modules.common_utils import PostFunction
from modules.exceptions import ConfigurationDefectError
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord
from lvp_logger import logger


def _is_valid_threshold_percent(value: object) -> bool:
    """Whether a stored brightness threshold is usable as a percentage.

    bool is rejected deliberately: it is an int subclass, so True would pass a
    plain range test and quietly mean 1 percent. NaN is rejected because every
    comparison against it is False -- a NaN cutoff admits no pixel and drops
    the layer, which is the silent outcome this check exists to prevent.
    """
    return is_finite_number(value) and 0 <= value <= 100


class CompositeGeneration(ProtocolPostProcessor):
    _PLACE = 'plane'
    _ONE_PER_ADD = ' Steps Added together make one composite.'

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
                'Channel Group ID',
                'Raw',
                'Derivation',
            ],
            dropna=False,
        )

    def _derivation_step(
        self, group: pd.DataFrame, output_format: image_mode.OutputFormat, **kwargs
    ) -> post_derivation.Step:
        # What decides a composite's pixels: its container, and each colour
        # channel's colour and, over a transmitted base, its blend threshold.
        # The container is stated first and for every format, as it always
        # was: the step's code names the file, so a composite already built
        # keeps its name.
        transmitted = common_utils.get_transmitted_layers()
        blended = group['Color'].isin(transmitted).any()
        thresholds = kwargs['brightness_thresholds_percent']
        settings = [('format', output_format)]
        for layer in sorted(set(group['Color']) - set(transmitted)):
            # Named as the run stored it; an old name's colour and threshold
            # are its stock key's.
            key = layer_record.stored_key(layer)
            settings.append(
                (f'{layer}.color', post_derivation.color_text(kwargs['channel_colors'][key]))
            )
            if blended:
                # A missing threshold is refused by the build itself, before
                # anything is written; this only names the group's output.
                settings.append((f'{layer}.threshold', str(thresholds.get(key))))
        return post_derivation.Step(PostFunction.COMPOSITE, tuple(settings))

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
                post=post,
            )
        )

        # The extension follows the container its channels share, matching
        # what the capture path writes: an OME composite named plain .tiff
        # reads as a different format than it is, and the readers that
        # branch on the double extension would take the wrong path.
        prefix = self._prepend_capture_root(name, kwargs)
        return f'{prefix}{image_utils.derived_file_suffix(output_format)}'

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
        if 'brightness_thresholds_percent' not in kwargs:
            raise ConfigurationDefectError(
                'composite_thresholds_missing',
                'composite generation requires brightness_thresholds_percent; '
                'the caller snapshots it from settings, because this runs on a '
                'worker thread that must not read live configuration',
            )
        return PostProcResult.from_group_result(
            CompositeGeneration._create_composite_image(
                path=path,
                df=df[['Filepath', 'Color']],
                brightness_thresholds_percent=kwargs['brightness_thresholds_percent'],
                output_file_loc=path / output_file_loc_rel if output_file_loc_rel else None,
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
            z=row0['Z'],
            z_slice=row0['Z-Slice'],
            well=row0['Well'],
            color=alg_metadata['color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            channel_group_id=row0['Channel Group ID'],
            **kwargs,
        )

    def _places(self, group: pd.DataFrame) -> pd.Series:
        # A composite takes one transmitted image and one image per other
        # channel, so every transmitted layer fills the one transmitted
        # plane: two of them, or two of one channel, have no one answer, and
        # keeping one would discard an acquired image behind a composite
        # that looks complete.
        transmitted = group['Color'].isin(common_utils.get_transmitted_layers())
        return group['Color'].where(~transmitted, 'transmitted')

    @staticmethod
    def _create_composite_image(
        path: pathlib.Path,
        df: pd.DataFrame,
        brightness_thresholds_percent: dict,
        output_file_loc: pathlib.Path | None = None,
        *,
        channel_colors: layer_record.ChannelColors,
    ):
        """Merge one group's per-channel frames into a single composite.

        Written to *output_file_loc*, when one is given, in the format its
        name states (``image_utils.is_ome_tiff``).

        brightness_thresholds_percent maps each fluorescence layer to the
        percentage below which its pixels are not composited onto a
        transmitted base. It is a required argument rather than a settings
        read: this runs unattended on a worker thread with no GUI, and the
        module-level settings global it used to read is published only by
        the app bootstrap, so every headless caller found it None and
        crashed at the threshold lookup. Passing the values in makes the
        headless and GUI paths identical by construction. channel_colors is
        the colour each layer is added in, handed in for the same reason.
        """

        BF_present = False
        BF_channel = ''

        # Every layer is read by the key it reads as -- its kind, colour and
        # threshold -- so an old run's colour names build as their stock keys.
        # Nothing below writes a layer's name.
        df = df.assign(Color=df['Color'].map(layer_record.stored_key))

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
        # Each file keeps its own depth and white: the sum count is set per
        # layer, so the layers of one composite can each have their own.
        images = {}
        scales = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            image, significant_bits, white_bits = image_utils.load_pixels(
                image_filepath, collapse_legacy_false_color=False
            )
            images[row['Filepath']] = image
            scales[row['Filepath']] = (significant_bits, white_bits)
        # The composite's own depth, set from the built array (always 8-bit).
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
                raise ConfigurationDefectError(
                    'composite_thresholds_missing',
                    f'composite generation needs a brightness threshold for '
                    f'{", ".join(missing)}; the caller supplies one per '
                    f'fluorescence layer so the merge cannot silently pick its own',
                )

            # A percentage outside 0-100 is a corrupt value, not a preference to
            # be honoured approximately: clamping it would silently produce an
            # image nobody asked for, and the raw captures survive on disk, so
            # refusing costs a composite that can be regenerated. Refused here
            # rather than at the conversion below for the reason the missing
            # check gives -- inside the try it would read as a failed image
            # group instead of a bad argument. The type check is part of the
            # same question: a string reaches the arithmetic otherwise.
            out_of_range = sorted(
                (layer, brightness_thresholds_percent[layer])
                for layer in df['Color']
                if layer != BF_channel
                and layer in common_utils.get_image_layers()
                and not _is_valid_threshold_percent(brightness_thresholds_percent[layer])
            )
            if out_of_range:
                named = '; '.join(f'{layer} = {value!r}' for layer, value in out_of_range)
                raise ConfigurationDefectError(
                    'composite_threshold_out_of_range',
                    f'composite generation needs a brightness threshold between '
                    f'0 and 100 percent, and got {named}',
                )

        error = None
        status = True

        try:
            transmitted = None
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
                transmitted = TransmittedBase(
                    image_utils.rgb_image_to_gray(BF_image, baked=True),
                    *scales[BF_image_filename],
                )
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
                img_gray = image_utils.rgb_image_to_gray(f_image, baked=True)

                channel_images[layer] = CompositeChannel(
                    img_gray, channel_colors[layer], *scales[row['Filepath']]
                )

                # The mapping onto the composite's 8-bit output scale belongs to
                # the builder, which owns that scale and the comparison the
                # cutoff feeds; computing it here would be a second copy free
                # to drift from either.
                if BF_present:
                    brightness_thresholds[layer] = brightness_cutoff_from_percent(
                        brightness_thresholds_percent[layer]
                    )

            if not channel_images and transmitted is None:
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
                logger.info(
                    'CompositeGeneration] Channel colours: '
                    + ', '.join(
                        f'{layer} {tuple(round(w, 3) for w in channel.color)}'
                        for layer, channel in channel_images.items()
                    )
                )
                img = build_composite(
                    channels=channel_images,
                    transmitted=transmitted,
                    brightness_thresholds=brightness_thresholds,
                )
                composite_significant_bits = img.dtype.itemsize * 8
                if output_file_loc is not None:
                    output_file_loc.parent.mkdir(parents=True, exist_ok=True)
                    metadata = image_utils.build_postproc_output_metadata(
                        input_records=[
                            image_utils.read_file_record(path / name) for name in df['Filepath']
                        ],
                        channel='Composite',
                        significant_bits=composite_significant_bits,
                        adds_frames=False,
                    )
                    image_utils.write_tiff(
                        data=img,
                        file_loc=output_file_loc,
                        metadata=metadata,
                        ome=image_utils.is_ome_tiff(output_file_loc),
                        # An RGB composite: a colour image's colour is its own.
                        color=GREY,
                        significant_bits=metadata['significant_bits'],
                        # A merged composite is a viewing product, always
                        # 8-bit RGB; like every derived file it is encoded
                        # from its own pixels, never from the live image mode.
                        save_encoding=image_mode.encoding_for_array(img),
                        # Already rendered for display, shown as it is.
                        white_bits=None,
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

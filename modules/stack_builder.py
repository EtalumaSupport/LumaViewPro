# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import concurrent.futures
import os
from collections.abc import Callable
import pathlib

import numpy as np
import pandas as pd

import modules.image_utils as image_utils
import modules.layer_record as layer_record
import modules.post_derivation as post_derivation
import modules.image_mode as image_mode
import modules.common_utils as common_utils
import modules.recording_frames as recording_frames
from modules.common_utils import PostFunction
from modules.exceptions import (
    InputFramesUnusableError,
    HyperstacksSavedNotice,
    HyperstacksSavingNotice,
)
from modules.notification_center import notifications
from modules.protocol_post_processor import ProtocolPostProcessor, operation_key
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord

import logging

logger = logging.getLogger('lvp_logger')


def build_hyperstacks_for_run(
    submit: Callable[[], concurrent.futures.Future], on_end: Callable[[], None]
) -> concurrent.futures.Future:
    """Put a finished run's per-well hyperstack build on its way by *submit*, and tell the person.

    *submit* is the run's ``PostProcessingAPI.run_hyperstack``, which puts
    the build on the post-processing lane and returns the future it
    settles; the build waits there for the run's images to land, and its
    raise when they will not -- the bound expired, or some were abandoned
    -- is answered here like any other, after the announcement, so a build
    waiting on a slow disk is not a silent hang. So is the lane's refusal
    to take it, at shutdown.

    Nobody waits on this build, so this is where its outcome is reported:
    announced as it is put on the lane, so a multi-minute build is not a
    silent hang, then answered as it settles -- the stacks saved, or what
    went wrong in its own words -- under the same operation key, so the
    answer replaces the announcement rather than opening beside it.
    *on_end* is called last, however the build ended.

    Returns:
        The future the build settles.
    """
    key = operation_key(PostFunction.HYPERSTACK)
    notifications.report_outcome(
        HyperstacksSavingNotice(), solicited=False, category='Post-processing', operation_key=key
    )

    def answer(settled: concurrent.futures.Future) -> None:
        try:
            failure = settled.exception()
            if failure is not None:
                notifications.report_outcome(
                    failure,
                    solicited=False,
                    category='Post-processing',
                    untyped_title='Hyperstacks Not Saved',
                    operation_key=key,
                )
                return
            notifications.report_outcome(
                HyperstacksSavedNotice(settled.result()),
                solicited=False,
                category='Post-processing',
                operation_key=key,
            )
        finally:
            on_end()

    try:
        build = submit()
    except Exception as refused:
        build = concurrent.futures.Future()
        build.set_exception(refused)
    build.add_done_callback(answer)
    return build


class StackBuilder(ProtocolPostProcessor):
    _PLACE = 'plane'
    _ONE_PER_ADD = ' Steps Added together make one hyperstack.'

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.HYPERSTACK,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        # A video recording's frames form their own stack per (well, scan)
        # -- one OME-TIFF per recording -- while stills keep grouping
        # across scans (their T axis IS the scan axis). The shared derived
        # key separates the two, so a well holding both stills and video
        # steps yields both artifact families.
        df = StackBuilder._with_recording_scan(df)
        return df.groupby(
            by=[
                'Well',
                'Objective',
                'X',
                'Y',
                'Tile',
                'Custom Step',
                'Channel Group ID',
                'Raw',
                'Recording Scan',
                'Derivation',
            ],
            dropna=False,
        )

    def _derivation_step(self, group: pd.DataFrame, **kwargs) -> post_derivation.Step:
        # A hyperstack holds its raw planes as captured; nothing it is given
        # changes their pixels.
        return post_derivation.Step(PostFunction.HYPERSTACK)

    def _places(self, group: pd.DataFrame) -> pd.Series:
        # One image per scan, slice and channel; a recording's frames are its
        # scans, numbered as the frames are.
        scan = group['Scan Count']
        if recording_frames.is_video_frame(group['Filepath'].iloc[0]):
            scan = group['Filepath'].map(recording_frames.frame_number)
        return scan.astype(str) + ',' + group['Z-Slice'].astype(str) + ',' + group['Color']

    def _generate_filename(self, df: pd.DataFrame, post: tuple[str, ...], **kwargs) -> str:
        row0 = df.iloc[0]

        objective_short_name = self._get_objective_short_name_if_has_turret(
            objective_id=row0['Objective']
        )

        # A hyperstack spans every channel AND every z-slice, so both the
        # channel and z tokens are omitted (channel=None, z_index=None) -- a
        # single slice index would mislabel the whole stack. The per-tile token
        # is kept: a stack is still one tile. Collapsed dimensions are dropped
        # by construction, never by stripping tokens back out of a name.
        # One stack per recording means one stack PER SCAN for recorded video
        # frames; the scan token keeps those names distinct (and mirrors the
        # on-disk recording folder's name). A stills stack spans scans, so its
        # name carries no scan token.
        scan_count = None
        if recording_frames.is_video_frame(row0['Filepath']):
            scan_count = int(row0['Scan Count'])

        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                channel=None,
                z_index=None,
                objective=objective_short_name,
                scan_count=scan_count,
                post=post,
            )
        )

        outfile = f'{self._prepend_capture_root(name, kwargs)}.ome.tiff'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Only process raw
        df = df[df['Raw'] == True]  # noqa: E712 -- pandas mask

        return df

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        # A video group's T axis is the frame ordinal within its recording
        # ('Scan Count' carries the temporal ordinal per the execution-record
        # contract; within one recording that is the frame number). The
        # loader keys every frame row to the recording's execution-record
        # row, so all rows arrive sharing the recording's scan -- remap
        # before the grid build. The Recording Scan group key puts stills
        # (sentinel) and video rows in different groups, so row 0 decides
        # for the whole group.
        if len(df) and recording_frames.is_video_frame(df['Filepath'].iloc[0]):
            df = df.assign(**{'Scan Count': df['Filepath'].map(recording_frames.frame_number)})
        return PostProcResult.from_group_result(
            StackBuilder._create_stack(
                path=path,
                df=df,
                output_file_loc=kwargs['output_file_loc'],
                save_encoding=kwargs['save_encoding'],
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
            scan_count=-1,
            x=row0['X'],
            y=row0['Y'],
            z=-1,
            z_slice=-1,
            well=row0['Well'],
            color='Stack',
            objective=row0['Objective'],
            # A stack is grouped by its Add, not its tile group: Apply Tiling
            # gives every channel of a position its own tile group.
            tile_group_id=-1,
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            channel_group_id=row0['Channel Group ID'],
            **kwargs,
        )

    @staticmethod
    def _generate_image_metadata(
        df: pd.DataFrame,
        path: pathlib.Path,
        output_file_loc: pathlib.Path,
        plane_metadata: dict,
        significant_bits: int,
        channel_colors: layer_record.ChannelColors,
    ):
        channel_names = df['Color'].unique().tolist()
        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        # The hyperstack inherits the depth the caller carried from the
        # load_pixels read of the input frames -- no second open to re-derive
        # what those pixels already came tagged with.
        sample_significant_bits = significant_bits

        # The scale travels with the input frame as PhysicalSizeX, written at
        # capture with the real binning already applied. Re-deriving it from an
        # objective focal length recomputed a value the pixels already carry,
        # and did so from whatever focal length the caller happened to hold --
        # which on a scope with no objective selector is a default, not a
        # measurement. Nothing here rebins; planes are stacked as read, so the
        # input's own scale is the output's scale.
        pixel_size_um = image_utils.read_pixel_size_um(sample_image_file_loc)

        if pixel_size_um is None:
            # Older / third-party / pre-metadata captures carry no scale, and
            # binning is not recoverable to re-derive one. Write the hyperstack
            # with no scale claim rather than an invented one: a wrong
            # PhysicalSizeX is measured off the file forever and is
            # indistinguishable from a real value.
            logger.warning(
                '[StackBuilder] No pixel size in input metadata (PhysicalSizeX missing) '
                'for %s; hyperstack will carry no scale. Re-capture with current LVP or '
                'supply frames carrying pixel-size metadata.',
                sample_image_file_loc.name,
            )
        else:
            pixel_size_um = round(
                pixel_size_um,
                common_utils.max_decimal_precision('pixel_size'),
            )

        metadata = image_utils.build_hyperstack_output_metadata(
            reference_input_path=sample_image_file_loc,
            channel_names=channel_names,
            channel_colors=[
                layer_record.output_color(name, channel_colors) for name in channel_names
            ],
            plane_positions={
                'PositionX': plane_metadata['PositionX'],
                'PositionY': plane_metadata['PositionY'],
                'PositionZ': plane_metadata['PositionZ'],
                'DeltaT': plane_metadata['DeltaT'],
            },
            significant_bits=sample_significant_bits,
            pixel_size_um=pixel_size_um,
        )

        options = {
            'photometric': 'minisblack',
            'compression': 'lzw',
            # tifffile always emits an XResolution tag, defaulting to 1/1. Under
            # CENTIMETER that reads as one pixel per centimetre -- a concrete and
            # wildly wrong scale claim. NONE is the TIFF convention for "ratio
            # only, no absolute unit", which is what an unknown scale means.
            'resolutionunit': 'CENTIMETER' if pixel_size_um is not None else 'NONE',
            # 0 for the same Windows kernel-handle-leak reason as the still save
            # paths -- tifffile's per-write ThreadPoolExecutor holds an Event
            # handle that outlives cleanup. That leak is the governing reason
            # and holds regardless of throughput. A hyperstack is ONE write()
            # call streaming every plane, so the executor's per-page overhead is
            # paid per plane; on a fast many-core dev machine dropping it
            # measured several times faster, but that margin has not been
            # measured on slower field hardware, where compression is a larger
            # share of each page and could outweigh the overhead.
            'maxworkers': 0,
        }

        # The TIFF resolution tag is a scale claim too, and it is built by
        # dividing by the pixel size. The key is always present so callers can
        # index it; None travels through to the writer, which omits the tag
        # rather than emitting a placeholder that reads as a measurement.
        return {
            'metadata': metadata,
            'options': options,
            'resolution': (
                image_utils.resolution_for_pixel_size(pixel_size_um)
                if pixel_size_um is not None
                else None
            ),
        }

    @staticmethod
    def _create_stack(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path,
        *,
        save_encoding: str,
        channel_colors: layer_record.ChannelColors,
        sort_order: list[str] | None = None,
    ):
        if sort_order is None:
            sort_order = ['Scan Count', 'Z-Slice', 'Color Index']

        # An empty frame set has no columns to read when it arrives as an
        # empty row list, and satisfies the rectangularity test below by
        # comparing zero to zero when it arrives typed -- so it reaches the
        # sample-row read and dies there. Refuse it here, where len() is
        # the only thing that must be safe.
        if len(df) == 0:
            return {
                'status': False,
                'error': 'Cannot build a hyperstack: no images were captured.',
                'metadata': {},
            }

        # 'Scan Count' is the T axis per the execution-record contract:
        # scan ordinal for protocol wells, frame ordinal for a
        # per-recording (manual frames) build.
        num_t = df['Scan Count'].nunique()
        num_z = df['Z-Slice'].nunique()
        num_c = df['Color'].nunique()

        # A hyperstack is a rectangular T x Z x C cube: every channel must be
        # captured at the same z-slices and scan counts, exactly once each. A
        # protocol that z-stacks one channel but single-shots another leaves
        # holes the dense array could only pad with black planes -- fake data
        # in a scientific image. Refuse the whole set through the post-
        # processor's status=False failure path, naming what was refused so
        # the user can align the protocol or build each channel separately.
        # A protocol's frame set names its well; a manual recording's has no
        # well, and its one way to be non-rectangular is a channel change
        # while it recorded, which each frame's row records truthfully.
        expected_planes = num_t * num_z * num_c
        captured_cells = df.groupby(['Scan Count', 'Z-Slice', 'Color']).ngroups
        if len(df) != expected_planes or captured_cells != expected_planes:
            if 'Well' in df.columns:
                # A custom step has no well, and is named by nothing here.
                well = df['Well'].iloc[0]
                where = f' for well {well}' if well else ''
                error = (
                    f'Cannot build a hyperstack{where}: its '
                    f'channels were not all captured at the same z-slices and scan '
                    f'counts ({len(df)} images for a {num_t} x {num_z} x {num_c} '
                    f'grid). Use the same z-stack settings on every channel, '
                    f'or build each channel separately.'
                )
            else:
                error = (
                    f'Cannot build a hyperstack for this recording: its frames were '
                    f'not all lit by one channel ({len(df)} frames across {num_c} '
                    f'channels). The frames are saved as recorded; keep one channel '
                    f'lit for the whole recording to get a hyperstack.'
                )
            return {'status': False, 'error': error, 'metadata': {}}

        _, color_idx_map = np.unique(df['Color'], return_inverse=True)
        df['Color Index'] = color_idx_map

        df = df.sort_values(by=sort_order, ascending=True)

        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        sample_image, _, _ = image_utils.load_pixels(
            sample_image_file_loc, collapse_legacy_false_color=False
        )
        h, w = sample_image.shape[0], sample_image.shape[1]
        # A stack whose planes mix uint8 and uint16 (an unsummed channel
        # beside a summed one on an 8-bit camera) is built at uint16, and the
        # writer needs the dtype before the first plane: the OME header is
        # written first, so every plane's dtype is read from its header here.
        # That pair is the one mixture a stack promotes; any other is refused
        # below, plane by plane.
        plane_dtypes = {image_utils.read_image_geometry(path / fp)[1] for fp in df['Filepath']}
        promotes_uint8 = plane_dtypes == {np.dtype(np.uint8), np.dtype(np.uint16)}
        stack_dtype = np.dtype(np.uint16) if promotes_uint8 else sample_image.dtype
        # The run's encoding decides how an 8-bit plane enters that stack. A
        # container-filling encoding (scaled, RGB) wrote each 16-bit plane
        # filling its container, so the 8-bit plane fills its container too,
        # as each plane's own file shows it; right-aligned keeps it at its
        # counts, beside the others' counts.
        fill_container = image_mode.encoding_fills_container(save_encoding)

        # The output's depth claim and per-plane timing resolve from every
        # input's header before the write starts -- neither can be collected
        # during the plane stream because the OME metadata is written first.
        headers = [image_utils.read_tiff_depth_and_timestamp(path / fp) for fp in df['Filepath']]
        input_depths = [depth for depth, _ in headers]
        output_depth = image_utils.resolve_output_depth(input_depths)
        timestamps = [ts for _, ts in headers]

        # Positions only: build_hyperstack_output_metadata derives every
        # unit list itself, so unit entries built here would be dead.
        plane_metadata = {
            'PositionX': df['X'].tolist(),
            'PositionY': df['Y'].tolist(),
            'PositionZ': df['Z'].tolist(),
        }

        # Timing is all-or-nothing: DeltaT is the measured list only when
        # EVERY plane carries a readable capture time, else None (no timing
        # claim). A partial list would misalign planes against times, and
        # None is the honest state for inputs that predate per-frame
        # timestamps. The key is always present so consumers index it.
        if all(ts is not None for ts in timestamps):
            t0 = min(timestamps)
            plane_metadata['DeltaT'] = [round((ts - t0).total_seconds(), 6) for ts in timestamps]
        else:
            plane_metadata['DeltaT'] = None

        ome_info = StackBuilder._generate_image_metadata(
            df=df,
            path=path,
            output_file_loc=output_file_loc,
            plane_metadata=plane_metadata,
            significant_bits=output_depth,
            channel_colors=channel_colors,
        )

        def _planes():
            # Planes stream in sorted (T, Z, C) order -- exactly the C-order
            # page sequence of the TZCYX shape declared to the writer. Ordering
            # by sorted RANK rather than indexing a cube with raw Scan Count
            # values also builds rectangular wells whose scan counts are not a
            # dense 0..N-1 range.
            for _, row in df.iterrows():
                t, z, c = row['Scan Count'], row['Z-Slice'], row['Color Index']
                image, _, _ = image_utils.load_pixels(
                    path / row['Filepath'], collapse_legacy_false_color=False
                )

                if image_utils.is_color_image(image):
                    # A layer's plane was baked in its colour; a composite's
                    # is a colour image, read by luminance.
                    image = image_utils.rgb_image_to_gray(
                        image=image, baked=row['Color'] in common_utils.get_layers()
                    )

                # Each plane must share the hyperstack's canvas; a per-plane
                # stitch divergence otherwise surfaces as a cryptic tifffile
                # error mid-write.
                image_utils.require_uniform_geometry(
                    [('first plane', sample_image), (f'plane t{t} z{z} c{c}', image)],
                    operation='assemble this hyperstack',
                )

                if promotes_uint8 and image.dtype == np.uint8:
                    # Widening, so nothing is lost: the plane enters the
                    # uint16 stack at its counts, or filling its container.
                    image = image.astype(np.uint16)
                    if fill_container:
                        image <<= 8
                elif image.dtype != stack_dtype:
                    # Any other mixture cannot share one stack: a cast into
                    # the stack dtype would truncate (uint16 into uint8) or
                    # reinterpret (float into an integer) -- wrong data under
                    # a success status.
                    raise InputFramesUnusableError(
                        f'hyperstack input frame {row["Filepath"]} is {image.dtype} but '
                        f'this stack is {stack_dtype}: mixed pixel types cannot share one '
                        f'hyperstack. Re-capture the well at one bit depth or build each '
                        f'capture group separately.',
                        'mixed_pixel_types',
                    )

                yield image

        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        # Route through the canonical hyperstack write path so LVP owns
        # the file-creation side of the save pipeline. The caller-built
        # OME dict carries the per-plane depth, so this path needs no
        # scalar significant_bits.
        image_utils.write_hyperstack_tiff(
            planes=_planes(),
            file_loc=output_file_loc_abs,
            shape=(num_t, num_z, num_c, h, w),
            dtype=stack_dtype,
            hyperstack_metadata=ome_info['metadata'],
            hyperstack_options=ome_info['options'],
            hyperstack_resolution=ome_info['resolution'],
        )

        return {
            'status': True,
            'error': None,
            'significant_bits': output_depth,
            'metadata': {},
        }

    @staticmethod
    def create_single_recording_stack(
        df: pd.DataFrame,
        path: pathlib.Path,
        output_file_loc: pathlib.Path,
        *,
        save_encoding: str,
        channel_colors: layer_record.ChannelColors,
    ) -> dict:
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
            save_encoding=save_encoding,
            channel_colors=channel_colors,
            sort_order=['Scan Count'],
        )


if __name__ == '__main__':
    stack_builder = StackBuilder(has_turret=False)
    tiling_configs_file_loc = pathlib.Path(os.getenv('SOURCE_ROOT')) / 'data' / 'tiling.json'
    stack_builder.load_folder(
        path=os.getenv('SAMPLE_IMAGE_FOLDER'),
        tiling_configs_file_loc=tiling_configs_file_loc,
        # The sample folder is a scientific (right-aligned) capture.
        save_encoding=image_mode.SAVE_ENCODING_RIGHT_ALIGNED,
    )

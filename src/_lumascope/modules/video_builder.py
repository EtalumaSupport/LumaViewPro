# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import os
import pathlib
import uuid

import pandas as pd

import _lumascope.modules.image_utils as image_utils
import _lumascope.modules.channel_record as channel_record
import _lumascope.modules.post_derivation as post_derivation
from _lumascope.modules.channel_record import GREY, ChannelColors, Color
import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.recording_frames as recording_frames
from _lumascope.modules.common_utils import PostFunction
from _lumascope.modules.api_surface import ProgressCallback
from _lumascope.modules.protocol_post_processor import ProtocolPostProcessor
from _lumascope.modules.protocol_post_processing_result import PostProcResult
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from _lumascope.modules.tiling_config import TilingGrids
from _lumascope.modules.video_writer import VideoWriter

from _lumascope.lvp_logger import logger


# Build rate for folders with no measured capture rate ('auto' on a
# protocol scan, or a recording that predates rate manifests): the
# long-standing Create Video default, kept so those builds keep working
# instead of refusing.
DEFAULT_BUILD_FPS = 5


class VideoBuilder(ProtocolPostProcessor):
    _PLACE = 'scan'
    _ONE_PER_ADD = ' Steps Added together make one video.'

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.VIDEO,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        # One group = one output video; per-recording identity comes from
        # the shared derived key (see _with_recording_scan).
        df = VideoBuilder._with_recording_scan(df)
        return df.groupby(
            by=[
                'Well',
                'channel',
                'Objective',
                'X',
                'Y',
                'Z-Slice',
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
        # A video's frames are drawn at its rate, with or without the time
        # stamped on them, in its channel's colour.
        color = channel_record.output_color(group.iloc[0]['channel'], kwargs['channel_colors'])
        return post_derivation.Step(
            PostFunction.VIDEO,
            (
                ('fps', f'{float(kwargs["frame_rate_fps"]):g}'),
                ('timestamp', 'on' if kwargs['enable_timestamp_overlay'] else 'off'),
                ('color', post_derivation.color_text(color)),
            ),
        )

    def _places(self, group: pd.DataFrame) -> pd.Series | None:
        # A timelapse takes one still per scan; interleaving two would make a
        # video of neither. A recording's frames are each their own.
        if recording_frames.is_video_frame(group['Filepath'].iloc[0]):
            return None
        return group['Scan Count']

    def _generate_filename(self, df: pd.DataFrame, post: tuple[str, ...], **kwargs) -> str:
        row0 = df.iloc[0]
        objective_short_name = self._get_objective_short_name_if_has_turret(
            objective_id=row0['Objective']
        )

        # channel, tile and z come from the authoritative columns (a video
        # keeps the source slice's z token, matching the per-image save).

        # One output per recording means one output PER SCAN for recorded
        # video frames; the scan token keeps those names distinct (and
        # mirrors the on-disk recording folder's own name). A stills
        # timelapse spans scans, so its name carries no scan token.
        scan_count = None
        if recording_frames.is_video_frame(row0['Filepath']):
            scan_count = int(row0['Scan Count'])

        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                objective=objective_short_name,
                scan_count=scan_count,
                post=post,
            )
        )

        outfile = f'{self._prepend_capture_root(name, kwargs)}.mp4'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip self outputs
        df = df[df[self._post_function.value] == False]  # noqa: E712 -- pandas mask

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]  # noqa: E712 -- pandas mask

        return df

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        # A group's video is named from what it was made from, as every
        # build's output is, so it is written whole onto its planned name:
        # the writer's picker would move a rebuild over an unrecorded file at
        # that name to a numbered copy the record never names, and a crash
        # mid-encode would leave a truncated MP4 under the real name. The
        # temporary ends in the planned suffix because the container is
        # chosen by it (so image_utils._written_whole's '.tmp' cannot serve),
        # and is unique, so the picker never moves it.
        planned = pathlib.Path(kwargs['output_file_loc'])
        temporary = planned.with_name(f'{planned.stem}.{uuid.uuid4().hex}.tmp{planned.suffix}')
        try:
            # One group is one channel (_get_groups), so its first row's
            # channel colours the whole video.
            result = self._create_video(
                path=path,
                df=df[['Filepath', 'Scan Count', 'Timestamp']],
                frame_rate_fps=kwargs['frame_rate_fps'],
                enable_timestamp_overlay=kwargs['enable_timestamp_overlay'],
                output_file_loc=temporary,
                on_progress=kwargs['on_progress'],
                total_groups=kwargs['total_groups'],
                current_group=kwargs['current_group'],
                color=channel_record.output_color(df.iloc[0]['channel'], kwargs['channel_colors']),
            )
            if result['status']:
                try:
                    os.replace(path / temporary, path / planned)
                except OSError as e:
                    return PostProcResult.failed(f'The video could not take its name: {e}')
            return PostProcResult.from_group_result(result)
        finally:
            (path / temporary).unlink(missing_ok=True)

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
            color=row0['channel'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            channel_group_id=row0['Channel Group ID'],
            **kwargs,
        )

    def _add_source_frame(
        self,
        writer: VideoWriter,
        image_path: pathlib.Path,
        *,
        enable_timestamp_overlay: bool,
        recorded_time: object = None,
    ) -> bool:
        """Read one saved frame and hand it to the writer with its payload depth.

        Both encode paths (the protocol-post-processor pipeline and the public
        path-based build) funnel through here so a uint16 frame's significant-bit
        depth is always read alongside its pixels and passed to add_frame. A frame
        stored right-aligned (12-bit data, max 4095) renders near-black if scaled
        as a full 16-bit value, so the depth is not optional context -- dropping
        it on any one path silently darkens every frame that path encodes.
        The array and its depth come from a single call so the two cannot drift
        apart, and keeping the read-and-add sequence in one place means a future
        depth change cannot quietly skip one of the two paths.

        When the timestamp overlay is on, the per-frame time comes from each
        frame's own metadata so a video built from recorded frames shows the real
        capture time per frame; when the frame carries none it is the run
        record's time for the frame's row (`image_utils.frame_time`), the one
        answer to a frame's time. The pixels, depth, and timestamp are read
        from a single file open, so an overlay-enabled build opens each frame
        once rather than twice. When the overlay is off, the timestamp is neither
        read nor stamped, so the cheaper depth-only read is used.

        Returns True when the frame was added, False when the source could not be
        read (the caller counts it as a skipped / dropped frame).
        """
        try:
            if enable_timestamp_overlay:
                image, significant_bits, white_bits, stated_ts = (
                    image_utils.load_pixels_with_timestamp(image_path)
                )
                frame_ts = image_utils.frame_time(stated_ts, recorded_time)
            else:
                image, significant_bits, white_bits = image_utils.load_pixels(image_path)
                frame_ts = None
        except Exception as e:
            logger.error(f'[{self._name}] Failed to read image: {image_path}: {e}')
            return False

        # image is mono 2D (a legacy 3-channel replica collapses to mono inside
        # load_pixels). VideoWriter applies the channel false-color, the 8-bit
        # downconvert scaled by significant_bits against white_bits (one frame's
        # white, for a sum), and any cv2 BGR-swap internally.
        writer.add_frame(
            image=image,
            timestamp=frame_ts,
            significant_bits=significant_bits,
            white_bits=white_bits,
        )
        return True

    def _create_video(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        frame_rate_fps: int,
        enable_timestamp_overlay: bool,
        output_file_loc: pathlib.Path,
        on_progress: ProgressCallback | None = None,
        total_groups=1,
        current_group=1,
        *,
        color: Color,
    ) -> dict:
        """Encode ``df``'s frames into one video, mono frames shown in ``color``.

        ``color`` is the caller's: a protocol group's channel colour, or a
        manual recording's (grey when it was recorded with false colour off).
        """

        if recording_frames.is_video_frame(df['Filepath'].values[0]):
            df['Frame Num'] = df['Filepath'].apply(recording_frames.frame_number)
            df = df.sort_values(by=['Frame Num'], ascending=True)

        else:
            df = df.sort_values(by=['Scan Count'], ascending=True)

        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        video = VideoWriter(
            output_path=output_file_loc_abs,
            fps=frame_rate_fps,
            color=color,
            include_timestamp_overlay=enable_timestamp_overlay,
        )

        # The writer's name, not the request: a taken name has moved it.
        logger.info(f'[{self._name}] Writing video to {video.output_path}')

        # Progress bar percentage calculation
        end_percentage = (current_group / total_groups) * 100
        start_percentage = ((current_group - 1) / total_groups) * 100

        percent_diff = end_percentage - start_percentage

        total_frames = len(df)
        i = 0
        skipped = 0
        for _, row in df.iterrows():
            image_path = path / row['Filepath']
            if not self._add_source_frame(
                video,
                image_path,
                enable_timestamp_overlay=enable_timestamp_overlay,
                recorded_time=row['Timestamp'],
            ):
                skipped += 1
                continue

            if on_progress is not None:
                on_progress(start_percentage + (i / total_frames) * percent_diff, None)

            i += 1

        video.close()

        if i - video.dropped_frames <= 0:
            # Not one frame reached the video: there is no video to record,
            # and a success here put a file that does not exist in the record,
            # which every later build then skipped as already made. The
            # writer's name was free when it was made, so whatever is there
            # is this build's.
            pathlib.Path(video.output_path).unlink(missing_ok=True)
            return {
                'status': False,
                'error': (
                    f'Cannot build a video: none of its {total_frames} frames could be '
                    f'read or encoded ({df.iloc[0]["Filepath"]} first).'
                ),
                'metadata': {},
            }

        total_dropped = skipped + video.dropped_frames
        if total_dropped > 0:
            # The count rides the group's metadata to the build, which says
            # the video is short to whoever asked for it.
            logger.warning(
                f'[{self._name}] {total_dropped} of {total_frames} frames missing '
                'from output (unreadable source or encode failure)'
            )

        logger.debug(f'[{self._name}] - Complete')

        return {
            'status': True,
            'error': None,
            # The writer is the authority on where the file landed (a
            # collision suffix may have moved it from the requested path);
            # the build's answer names the real file, not the request.
            'actual_output_file_loc': video.output_path,
            # Video encodes an 8-bit stream (every frame is downconverted to
            # 8 bits inside the writer), so the output artifact's depth is 8
            # regardless of the source frames' depth.
            'significant_bits': 8,
            'metadata': {'dropped_frames': total_dropped},
        }

    def build_from_folder(
        self,
        path: str | pathlib.Path,
        tiling_grids: TilingGrids,
        on_progress: ProgressCallback | None = None,
        **kwargs: dict,
    ) -> dict:
        """Create video(s) from a captured folder, dispatching by recording type.

        Manual "Frames" recordings carry no protocol_record.tsv and so are
        rejected by the protocol post-processing pipeline; route them to a
        record-less single-video build instead. Protocol scans keep the
        standard load_folder path. Mirrors load_folder's signature so the
        caller can swap one entry point for the other without rewiring.
        """
        path = pathlib.Path(path)
        is_manual = self._is_manual_recording_folder(path)
        if kwargs.get('frame_rate_fps') is None:
            # The UI's 'auto' rate, resolved HERE where the folder type
            # is known: a manual recording plays at its own measured
            # capture rate (from its manifest); folders without a
            # measured rate (protocol scans until their cutover writes
            # manifests, pre-manifest manual recordings) build at the
            # standard default.
            measured = self._read_recording_manifest(path)['measured_fps'] if is_manual else None
            if measured is not None and measured > 0:
                kwargs['frame_rate_fps'] = measured
                logger.info(
                    f'[{self._name}] auto rate: building at the recorded '
                    f'{measured:.3g} fps from the manifest'
                )
            else:
                kwargs['frame_rate_fps'] = DEFAULT_BUILD_FPS
        if is_manual:
            return self._build_manual_recording_video(path, on_progress=on_progress, **kwargs)
        return self.load_folder(
            path=path,
            tiling_grids=tiling_grids,
            on_progress=on_progress,
            **kwargs,
        )

    def _is_manual_recording_folder(self, path: pathlib.Path) -> bool:
        # Positive detection: only a folder that actually holds manually
        # recorded frames takes the record-less path. A protocol folder with a
        # missing or broken record still falls through to load_folder so the
        # user gets the informative protocol error instead of a silent raw build.
        try:
            return path.is_dir() and any(
                recording_frames.is_manual_video_frame(p.name)
                for p in image_utils.find_tiff_files(path)
            )
        except OSError:
            return False

    def _read_recording_manifest(self, path: pathlib.Path) -> dict:
        """Read a manual recording's manifest facts for the video build.

        Manual frames are saved mono at a measured cadence, so both the
        false-color channel and the true capture rate live in the
        manifest, not in the frames. Reads the engine's
        recording_manifest.json first, then the legacy
        session_manifest.json written by pre-engine releases (their
        schemas nest differently). Fields are None when the folder
        predates both manifests, the file is unreadable, or the field is
        absent -- channel None encodes grayscale; rate None means no
        measured rate is known. ``channel_color`` is the channel to colour
        the video by, None when it was recorded with false color off: a
        manifest records the channel and a ``false_color`` flag, and one
        written before the flag records the channel only when it was on.

        Returns:
            ``{'channel_color': str | None, 'measured_fps': float | None}``
        """
        manifest_path = path / 'recording_manifest.json'
        try:
            with open(manifest_path, encoding='utf-8') as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            manifest = None
        if manifest is not None:
            fps = manifest.get('measured_fps')
            if 'false_color' in manifest:
                channel_color = manifest['channel'] if manifest['false_color'] else None
            else:
                channel_color = manifest.get('channel_color')
            return {
                'channel_color': channel_color,
                'measured_fps': float(fps) if fps else None,
            }

        legacy_path = path / 'session_manifest.json'
        try:
            with open(legacy_path, encoding='utf-8') as fh:
                legacy = json.load(fh)
        except (OSError, ValueError):
            # Neither manifest generation is readable. Say so: a LOST
            # manifest is otherwise indistinguishable from deliberate
            # grayscale, and the build quietly plays colorless at the
            # default rate. Log-only -- a pre-manifest legacy folder
            # takes this path legitimately, so a popup would misfire.
            logger.warning(
                f'[{self._name}] No readable recording manifest in {path.name}; '
                f'the video will build grayscale at the default rate. The '
                f'manifest carries the channel color and measured rate.'
            )
            return {'channel_color': None, 'measured_fps': None}
        recording = legacy.get('recording', {})
        fps = recording.get('actual_fps', {}).get('mean')
        return {
            'channel_color': recording.get('channel_color'),
            'measured_fps': float(fps) if fps else None,
        }

    def _build_manual_recording_video(
        self,
        path: pathlib.Path,
        on_progress: ProgressCallback | None = None,
        *,
        frame_rate_fps: float,
        enable_timestamp_overlay: bool = False,
        channel_colors: ChannelColors,
        **_ignored: dict,
    ) -> dict:
        frame_paths = [
            p
            for p in image_utils.find_tiff_files(path)
            if recording_frames.is_manual_video_frame(p.name)
        ]
        # The TIFF finder returns lexical order, which wraps at frame
        # 10,000 (five digits sort beside four); order numerically.
        frame_paths.sort(key=lambda p: recording_frames.frame_number(p.name))
        if not frame_paths:
            raise self._refuse(
                'no_data',
                'No recorded video frames were found in the selected folder. '
                'Check that the folder contains a manual "Frames" recording.',
            )

        # Manual frames are saved as mono with no protocol record. Build the
        # minimal dataframe _create_video needs and drive the one canonical
        # encode path. 'Scan Count' carries the temporal ordinal per the
        # execution-record contract -- within one recording that is the
        # frame ordinal. The channel color comes from the recording manifest
        # (the frames themselves are mono, so the color isn't recoverable
        # from them); without it a false-colored recording would encode
        # grayscale. None (no manifest / old recording / brightfield)
        # encodes grayscale. The per-frame timestamp is read from each
        # frame's own metadata inside _create_video, so the overlay toggle
        # authoritatively controls whether the video shows a timestamp.
        channel_color = self._read_recording_manifest(path)['channel_color']
        df = pd.DataFrame(
            {
                'Filepath': [p.name for p in frame_paths],
                'Scan Count': range(len(frame_paths)),
                'Timestamp': '',
            }
        )
        output_file_loc = pathlib.Path(f'{path.name}.mp4')

        # The name holds no settings, so a rebuild keeps the earlier video:
        # the writer moves to the next free numbered name.
        try:
            result = self._create_video(
                path=path,
                df=df,
                frame_rate_fps=frame_rate_fps,
                enable_timestamp_overlay=enable_timestamp_overlay,
                output_file_loc=output_file_loc,
                on_progress=on_progress,
                total_groups=1,
                current_group=1,
                color=(
                    GREY
                    if channel_color is None
                    else channel_record.output_color(channel_color, channel_colors)
                ),
            )
        except FileExistsError as e:
            raise self._refuse(
                'names_exhausted', f'{e}. Move or remove earlier videos, then build again.'
            ) from e

        if on_progress is not None:
            on_progress(100, None)

        # The same contract as a protocol folder's build: only a complete
        # video returns; a failed or short one raises with what was written.
        if not result['status']:
            raise self._incomplete(
                group_errors=[f'{output_file_loc}: {result.get("error") or "the encode failed"}'],
                attempted=1,
                refused_count=0,
                colliding_names=set(),
                dropped_frames=0,
                artifact_paths=[],
                output_root=str(path),
                accounting_note='',
                excluded_inputs=(),
            )
        dropped = result.get('metadata', {}).get('dropped_frames', 0)
        if dropped:
            raise self._incomplete(
                group_errors=[],
                attempted=1,
                refused_count=0,
                colliding_names=set(),
                dropped_frames=dropped,
                artifact_paths=[str(result['actual_output_file_loc'])],
                output_root=str(path),
                accounting_note='',
                excluded_inputs=(),
            )
        # What a protocol folder's build answers: the file it made, where it
        # landed, so a caller is told the video and not only that it worked.
        # It reads the recording's frames alone, so nothing is left out.
        return {
            'status': True,
            'message': 'Success.',
            'new_count': 1,
            'output_root': str(path),
            'artifact_paths': [str(result['actual_output_file_loc'])],
            'excluded_inputs': (),
        }

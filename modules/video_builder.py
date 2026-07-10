# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import pathlib

import pandas as pd

import modules.image_utils as image_utils
import modules.common_utils as common_utils
from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord
from modules.video_writer import VideoWriter

from lvp_logger import logger


# Manual "Frames" recordings name each frame ManualVideo_Frame_<NNNN>_<ts>.tiff.
# The [0-9] after the prefix keeps the optional ManualVideo_Frame_HyperStack
# container out of the frame sequence; .tif* covers .tiff and .tif.
_MANUAL_FRAME_GLOB = 'ManualVideo_Frame_[0-9]*.tif*'


class VideoBuilder(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            post_function=PostFunction.VIDEO,
            **kwargs,
        )
        self._name = self.__class__.__name__

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Well',
                'Color',
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

        # Post-output suffixes chain: a video of an already-stitched output
        # carries both ('stitched', 'video'). channel, tile and z come from the
        # authoritative columns (a video keeps the source slice's z token,
        # matching the per-image save).
        post = ('stitched',) if row0['Stitched'] else ()
        post = (*post, 'video')
        name = common_utils.build_step_name(
            common_utils.step_components(
                row0,
                objective=objective_short_name,
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
        # 'Color' included so _create_video can drive VideoWriter's in-writer
        # false-color from the layer name (one group is one color per
        # _get_groups).
        return PostProcResult.from_group_result(
            self._create_video(
                path=path,
                df=df[['Filepath', 'Scan Count', 'Timestamp', 'Color']],
                frames_per_sec=kwargs['frames_per_sec'],
                enable_timestamp_overlay=kwargs['enable_timestamp_overlay'],
                output_file_loc=kwargs['output_file_loc'],
                popup=kwargs['popup'],
                total_groups=kwargs['total_groups'],
                current_group=kwargs['current_group'],
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
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    def _add_source_frame(
        self,
        writer: VideoWriter,
        image_path: pathlib.Path,
        *,
        enable_timestamp_overlay: bool,
        fallback_timestamp=None,
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
        capture time per frame; it falls back to the caller-supplied step time
        when the frame carries none. The pixels, depth, and timestamp are read
        from a single file open, so an overlay-enabled build opens each frame
        once rather than twice. When the overlay is off, the timestamp is neither
        read nor stamped, so the cheaper depth-only read is used.

        Returns True when the frame was added, False when the source could not be
        read (the caller counts it as a skipped / dropped frame).
        """
        try:
            if enable_timestamp_overlay:
                image, significant_bits, frame_ts = image_utils.load_pixels_with_timestamp(
                    image_path
                )
                if frame_ts is None:
                    frame_ts = fallback_timestamp
            else:
                image, significant_bits = image_utils.load_pixels(image_path)
                frame_ts = None
        except Exception as e:
            logger.error(f'[{self._name}] Failed to read image: {image_path}: {e}')
            return False

        # image is mono 2D (a legacy 3-channel replica collapses to mono inside
        # load_pixels). VideoWriter applies the layer false-color, the 8-bit
        # downconvert scaled by significant_bits, and any cv2 BGR-swap internally.
        writer.add_frame(image=image, timestamp=frame_ts, significant_bits=significant_bits)
        return True

    def _create_video(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        frames_per_sec: int,
        enable_timestamp_overlay: bool,
        output_file_loc: pathlib.Path,
        popup=None,
        total_groups=1,
        current_group=1,
    ) -> dict:

        def strip_filetype(filename: str):
            filename_flipped = filename[::-1]
            if '.' in filename_flipped:
                while filename_flipped[0] != '.':
                    filename_flipped = filename_flipped[1:]
                filename_flipped = filename_flipped[1:]
                return filename_flipped[::-1]
            else:
                logger.error(f'Invalid filename {filename}')
                return

        def get_frame_num(filename):
            filename = str(filename)
            stripped_filename = strip_filetype(filename)
            return stripped_filename[-4:]

        if 'video_Frame' in str(df['Filepath'].values[0]):
            df['Frame Num'] = df['Filepath'].apply(get_frame_num)
            df = df.sort_values(by=['Frame Num'], ascending=True)

        else:
            df = df.sort_values(by=['Scan Count'], ascending=True)

        # Layer color drives VideoWriter's internal false-color application.
        # One protocol-output group is one color per _get_groups, so the
        # first row carries the right value for the whole video. The df is
        # always projected with 'Color' and grouped by it, so the else-None
        # fallback is effectively unreachable. Considered fail-fast on a
        # missing/NaN color; rejected because BF/PC/DF carry a transmitted-
        # light label that VideoWriter renders as grayscale -- a hard error
        # would break legitimate brightfield-only videos. None falls through
        # to grayscale, the safe default.
        layer_color = df.iloc[0]['Color'] if 'Color' in df.columns else None
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        video = VideoWriter(
            output_path=output_file_loc_abs,
            fps=frames_per_sec,
            color=layer_color,
            include_timestamp_overlay=enable_timestamp_overlay,
        )

        logger.info(f'[{self._name}] Writing video to {output_file_loc}')

        # Progress bar percentage calculation
        end_percentage = (current_group / total_groups) * 100
        start_percentage = ((current_group - 1) / total_groups) * 100

        percent_diff = end_percentage - start_percentage

        total_frames = len(df)
        i = 0
        skipped = 0
        for _, row in df.iterrows():
            image_path = path / row['Filepath']
            # The step time is only the fallback the overlay uses when a frame
            # carries no timestamp of its own, so with the overlay off it is never
            # read -- skip the per-frame conversion entirely. Guard the dataframe
            # Timestamp: the loader fills missing values with the empty string (no
            # to_pydatetime) rather than a pandas Timestamp.
            fallback_ts = None
            if enable_timestamp_overlay:
                fallback_ts = (
                    row['Timestamp'].to_pydatetime()
                    if hasattr(row['Timestamp'], 'to_pydatetime')
                    else None
                )
            if not self._add_source_frame(
                video,
                image_path,
                enable_timestamp_overlay=enable_timestamp_overlay,
                fallback_timestamp=fallback_ts,
            ):
                skipped += 1
                continue

            if popup is not None:
                popup.progress = start_percentage + (i / total_frames) * percent_diff

            i += 1

        video.close()

        total_dropped = skipped + video.dropped_frames
        if total_dropped > 0:
            logger.warning(
                f'[{self._name}] {total_dropped} of {total_frames} frames missing '
                'from output (unreadable source or encode failure)'
            )
            from modules.notification_center import notifications

            notifications.warning(
                'Create Video',
                'Video Frames Missing',
                f'{total_dropped} of {total_frames} frames could not be added to '
                f'"{output_file_loc}". The video is shorter than the source set. '
                'Check the log for the cause.',
            )

        logger.debug(f'[{self._name}] - Complete')

        return {
            'status': True,
            'error': None,
            # The writer is the authority on where the file landed (a
            # collision suffix or the cv2 .avi fallback may have moved it
            # from the requested path); the record must point at the real
            # file, not the request.
            'actual_output_file_loc': video.output_path,
            # Video encodes an 8-bit stream (every frame is downconverted to
            # 8 bits inside the writer), so the output artifact's depth is 8
            # regardless of the source frames' depth.
            'significant_bits': 8,
            'metadata': {'dropped_frames': total_dropped},
        }

    def build_video(
        self,
        source_dir: pathlib.Path,
        output_file: pathlib.Path,
        *,
        false_color: bool = False,
        color: str | None = None,
        fps: int = 10,
        include_timestamp_overlay: bool = False,
    ) -> dict:
        """Build a single video file from mono TIFFs in source_dir.

        Public path-based entry point that wraps VideoWriter directly,
        parallel to the protocol-post-processor pipeline (load_folder ->
        _create_video). Reads source_dir/*.tiff in lexical filename order
        via the legacy-collapse helper so pre-1d 3-channel-replica files
        and post-1d mono files both produce uniform mono input.

        Args:
            source_dir: Directory of TIFF inputs, one per frame.
            output_file: Destination video file. .mp4 routes to PyAV H.264;
                cv2 fallback rewrites the suffix to .avi.
            false_color: When True, the layer false-color is applied inside
                VideoWriter. Requires ``color``; raises ValueError otherwise.
                When False, encode grayscale.
            color: Layer name ('Red', 'Green', 'Blue', 'Lumi', 'BF', ...).
                Required when ``false_color=True``; ignored otherwise.
            fps: Frames per second.
            include_timestamp_overlay: Overlay frame timestamps.

        Returns:
            {'status': bool, 'error': str | None, 'frame_count': int}.

        Raises:
            ValueError: If false_color=True and color is None, or if
                source_dir contains no .tiff / .tif files.
        """
        source_dir = pathlib.Path(source_dir)
        output_file = pathlib.Path(output_file)

        if false_color and color is None:
            raise ValueError(
                'build_video: color is required when false_color=True. '
                "Pass e.g. color='Blue' to specify which layer to false-color."
            )

        tiff_paths = sorted(source_dir.glob('*.tiff')) + sorted(source_dir.glob('*.tif'))
        if not tiff_paths:
            raise ValueError(f'build_video: no .tiff / .tif files in {source_dir}')

        writer_color = color if false_color else None
        writer = VideoWriter(
            output_path=output_file,
            fps=fps,
            color=writer_color,
            include_timestamp_overlay=include_timestamp_overlay,
        )

        frame_count = 0
        skipped = 0
        status = True
        error = None
        try:
            for tiff_path in tiff_paths:
                if not self._add_source_frame(
                    writer,
                    tiff_path,
                    enable_timestamp_overlay=include_timestamp_overlay,
                ):
                    skipped += 1
                    continue
                frame_count += 1
        except Exception as e:
            logger.exception(f'[{self._name}] build_video: encode failed')
            status = False
            error = str(e)
        finally:
            writer.close()

        # Unreadable sources + encode failures are returned so the caller can
        # tell the user the built video is short rather than assume it is whole.
        dropped = skipped + writer.dropped_frames
        if dropped:
            logger.warning(
                f'[{self._name}] build_video: {dropped} of {len(tiff_paths)} frames '
                'missing from output (unreadable source or encode failure)'
            )
        return {
            'status': status,
            'error': error,
            'frame_count': frame_count,
            'dropped_frames': dropped,
            # Where the file actually landed (collision suffix / cv2 .avi
            # fallback may have moved it from output_file).
            'output_file': writer.output_path,
        }

    def build_from_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        popup=None,
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
        if self._is_manual_recording_folder(path):
            return self._build_manual_recording_video(path, popup=popup, **kwargs)
        return self.load_folder(
            path=path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            popup=popup,
            **kwargs,
        )

    def _is_manual_recording_folder(self, path: pathlib.Path) -> bool:
        # Positive detection: only a folder that actually holds manually
        # recorded frames takes the record-less path. A protocol folder with a
        # missing or broken record still falls through to load_folder so the
        # user gets the informative protocol error instead of a silent raw build.
        try:
            return path.is_dir() and any(path.glob(_MANUAL_FRAME_GLOB))
        except OSError:
            return False

    def _read_manifest_channel_color(self, path: pathlib.Path) -> str | None:
        """Return the recording's channel color from session_manifest.json.

        Manual frames are saved mono, so the false-color channel is recorded
        in the manifest at capture time. Returns None when the manifest is
        absent (older recordings), unreadable, or carries no channel_color --
        in which case the video encodes grayscale.
        """
        manifest_path = path / 'session_manifest.json'
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            return None
        return manifest.get('recording', {}).get('channel_color')

    def _build_manual_recording_video(
        self,
        path: pathlib.Path,
        popup=None,
        *,
        frames_per_sec: int = 5,
        enable_timestamp_overlay: bool = False,
        **_ignored: dict,
    ) -> dict:
        frame_paths = sorted(path.glob(_MANUAL_FRAME_GLOB))
        if not frame_paths:
            return {
                'status': False,
                'message': (
                    'No recorded video frames were found in the selected folder. '
                    'Check that the folder contains a manual "Frames" recording.'
                ),
            }

        # Manual frames are saved as mono with no protocol record. Build the
        # minimal dataframe _create_video needs and drive the one canonical
        # encode path. The channel color comes from the session manifest (the
        # frames themselves are mono, so the color isn't recoverable from
        # them); without it a false-colored recording would encode grayscale.
        # None (no manifest / old recording / brightfield) encodes grayscale.
        # The per-frame timestamp is read from each frame's own metadata inside
        # _create_video, so the overlay toggle authoritatively controls whether
        # the video shows a timestamp.
        channel_color = self._read_manifest_channel_color(path)
        df = pd.DataFrame(
            {
                'Filepath': [p.name for p in frame_paths],
                'Scan Count': range(len(frame_paths)),
                'Timestamp': '',
                'Color': channel_color,
            }
        )
        output_file_loc = pathlib.Path(f'{path.name}.mp4')

        try:
            result = self._create_video(
                path=path,
                df=df,
                frames_per_sec=frames_per_sec,
                enable_timestamp_overlay=enable_timestamp_overlay,
                output_file_loc=output_file_loc,
                popup=popup,
                total_groups=1,
                current_group=1,
            )
        except Exception as e:
            logger.exception(f'[{self._name}] Manual-recording video build failed')
            return {'status': False, 'message': str(e)}

        if popup is not None:
            popup.progress = 100

        if not result['status']:
            return {'status': False, 'message': result.get('error') or 'Video generation failed.'}

        return {'status': True, 'message': 'Success'}

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib

import pandas as pd

import modules.image_utils as image_utils
import modules.common_utils as common_utils
from modules.common_utils import PostFunction
from modules.protocol_post_processor import ProtocolPostProcessor
from modules.protocol_post_record import ProtocolPostRecord
from modules.video_writer import VideoWriter

from lvp_logger import logger


class VideoBuilder(ProtocolPostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            post_function=PostFunction.VIDEO,
            *args,
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

        # Prepend the protocol's capture_root (passed in via kwargs by
        # ProtocolPostProcessor.load_folder) so the video output carries
        # the same filename root as the per-image saves.
        capture_root = kwargs.get('capture_root', '')
        prefix = f'{capture_root}_{row0["Name"]}' if capture_root else row0['Name']
        name = common_utils.generate_default_step_name(
            custom_name_prefix=prefix,
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=None,
            objective_short_name=objective_short_name,
            tile_label=row0['Tile'],
            stitched=row0['Stitched'],
            video=True,
        )

        outfile = f'{name}.mp4'
        return outfile

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip self outputs
        df = df[df[self._post_function.value] == False]

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]

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
        return self._create_video(
            path=path,
            df=df[['Filepath', 'Scan Count', 'Timestamp', 'Color']],
            frames_per_sec=kwargs['frames_per_sec'],
            enable_timestamp_overlay=kwargs['enable_timestamp_overlay'],
            output_file_loc=kwargs['output_file_loc'],
            popup=kwargs['popup'],
            total_groups=kwargs['total_groups'],
            current_group=kwargs['current_group'],
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
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

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
            enable_timestamp_overlay = False

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
        for _, row in df.iterrows():
            image_path = path / row['Filepath']
            try:
                image = image_utils.read_tiff_with_legacy_collapse(image_path)
            except Exception as e:
                logger.error(f'[{self._name}] Failed to read image: {image_path}: {e}')
                continue

            # Post-1d: image is mono 2D (legacy 3-channel collapses to mono
            # via read_tiff_with_legacy_collapse). VideoWriter applies the
            # layer false-color and any cv2 BGR-swap internally.

            # Timestamp overlay and 8-bit conversion handled by VideoWriter.add_frame()
            frame_ts = row['Timestamp'].to_pydatetime() if enable_timestamp_overlay else None
            video.add_frame(image=image, timestamp=frame_ts)

            if popup is not None:
                popup.progress = start_percentage + (i / total_frames) * percent_diff

            i += 1

        video.close()

        logger.debug(f'[{self._name}] - Complete')

        return {
            'status': True,
            'error': None,
            'metadata': {},
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
        status = True
        error = None
        try:
            for tiff_path in tiff_paths:
                try:
                    image = image_utils.read_tiff_with_legacy_collapse(tiff_path)
                except Exception as e:
                    logger.error(f'[{self._name}] build_video: failed to read {tiff_path}: {e}')
                    continue
                writer.add_frame(image)
                frame_count += 1
        except Exception as e:
            logger.exception(f'[{self._name}] build_video: encode failed')
            status = False
            error = str(e)
        finally:
            writer.close()

        return {
            'status': status,
            'error': error,
            'frame_count': frame_count,
        }

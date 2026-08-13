# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Frame filename contract pins: modules/recording_frames.py is the one
home that builds AND parses the recording engine's on-disk names.

The literal name shapes are pinned here on purpose: existing user
folders must keep parsing forever, so an accidental token change in the
contract module must trip these before it ships. Changing a pinned
literal is a deliberate two-generation migration, never a refactor.
"""

import pathlib

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from modules import recording_frames
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
from modules.video_builder import VideoBuilder


class TestFrameNameShapes:
    def test_protocol_template_renders_the_shipped_shape(self):
        template = recording_frames.protocol_frame_filename_template('A1_BF_0003_video')
        assert template.format(n=7) == 'A1_BF_0003_video_Frame_0007.tiff'

    def test_manual_template_renders_the_shipped_shape(self):
        template = recording_frames.manual_frame_filename_template()
        name = template.format(n=7, ts='2026-08-09_12-00-00-000')
        assert name == 'ManualVideo_Frame_0007_2026-08-09_12-00-00-000.tiff'

    def test_manual_hyperstack_filename_is_the_shipped_literal(self):
        assert (
            recording_frames.MANUAL_HYPERSTACK_FILENAME == 'ManualVideo_Frame_HyperStack.ome.tiff'
        )


class TestFrameNumber:
    def test_parses_protocol_and_manual_names(self):
        assert recording_frames.frame_number('A1_BF_0003_video_Frame_0042.tiff') == 42
        assert (
            recording_frames.frame_number('ManualVideo_Frame_10001_2026-08-09_12-00-00-000.tiff')
            == 10001
        )

    def test_name_without_a_frame_token_fails_loudly(self):
        with pytest.raises(ValueError, match='frame number'):
            recording_frames.frame_number('A1_BF_0003.tiff')


class TestFramePredicates:
    def test_vocabularies_are_disjoint_by_case(self):
        # 'ManualVideo_Frame_' carries capital-V 'Video_Frame'; the
        # protocol token is lowercase '_video_Frame_'. A consumer that
        # substring-matched one against the other silently took the
        # wrong branch -- the predicates must classify exactly.
        manual = 'ManualVideo_Frame_0001_2026-08-09_12-00-00-000.tiff'
        protocol = 'A1_BF_0003_video_Frame_0001.tiff'
        assert recording_frames.is_manual_video_frame(manual)
        assert not recording_frames.is_manual_video_frame(protocol)
        assert recording_frames.is_protocol_video_frame(protocol)
        assert not recording_frames.is_protocol_video_frame(manual)
        assert recording_frames.is_video_frame(manual)
        assert recording_frames.is_video_frame(protocol)

    def test_hyperstack_container_is_not_a_frame(self):
        assert not recording_frames.is_manual_video_frame(
            recording_frames.MANUAL_HYPERSTACK_FILENAME
        )

    def test_still_capture_names_are_not_frames(self):
        assert not recording_frames.is_video_frame('A1_BF_0003.tiff')
        assert not recording_frames.is_video_frame('B2_Green_0001_stitched.tiff')

    def test_recording_dir_name_tests_the_final_component_only(self):
        assert recording_frames.is_video_recording_dir_name('A1_BF_0003_video')
        assert not recording_frames.is_video_recording_dir_name('my_video_data')
        assert not recording_frames.is_video_recording_dir_name('A1_BF_0003')


class TestVideoDirPathClassification:
    def test_root_with_video_token_in_ancestor_is_not_stepped_up(self, tmp_path):
        # A protocol root living under a folder that carries '_video'
        # anywhere in its path must scan normally; the old
        # substring-on-the-whole-path check stepped OUT of the protocol
        # folder and computed every relative name against the wrong base.
        root = tmp_path / 'experiments_video_2026' / 'scan_run'
        (root / 'BF').mkdir(parents=True)
        (root / 'BF' / 'A1_BF_0000.tiff').touch()

        names = ProtocolPostProcessingHelper._get_image_filenames_from_folder(path=root)

        assert names['raw'] == [pathlib.Path('BF/A1_BF_0000.tiff')]


class TestManualFramesTakeTheNumericSortBranch:
    def test_shuffled_manual_df_is_consumed_in_frame_order(self, tmp_path):
        # The old branch test substring-matched lowercase 'video_Frame',
        # which cannot match 'ManualVideo_Frame' -- manual builds fell
        # through to the Scan Count sort and only produced ordered video
        # because the caller happened to fabricate Scan Count in frame
        # order. The classification must route manual frames to the
        # numeric frame-number sort regardless of caller ordering.
        frames = [3, 0, 2, 1]
        df = pd.DataFrame(
            {
                'Filepath': [
                    f'ManualVideo_Frame_{n:04}_2026-08-09_12-00-00-000.tiff' for n in frames
                ],
                'Scan Count': range(len(frames)),
            }
        )

        consumed = []

        def _record_frame(self, writer, image_path, **kwargs):
            consumed.append(recording_frames.frame_number(image_path.name))
            return True

        writer = MagicMock()
        writer.dropped_frames = 0
        with (
            patch.object(VideoBuilder, '_add_source_frame', _record_frame),
            patch('modules.video_builder.VideoWriter', return_value=writer),
        ):
            builder = VideoBuilder(has_turret=False)
            builder._create_video(
                path=tmp_path,
                df=df,
                frames_per_sec=10,
                enable_timestamp_overlay=False,
                output_file_loc=pathlib.Path('out.mp4'),
            )

        assert consumed == sorted(frames)

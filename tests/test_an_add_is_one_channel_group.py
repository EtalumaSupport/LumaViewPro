# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Steps Added together are one composite; two Adds at one position are two (#772).

An Add makes one step per channel, all at one stage position, and every
Added step was identical in every column a post-processing build groups on:
nothing recorded which Add a step came from. So every Add at one position
became one composite, which kept the first transmitted image and the last
image of each color and dropped the rest without a word -- on an LS820,
four Adds gave one composite from 4 of 11 images. Each step now carries its
Add's ``Channel Group ID``; the composite, the z-projection, the video and
the hyperstack group by it, and a group that still holds two images for one
plane (a run folder from before the column) is refused, naming them.
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

import modules.config_helpers as config_helpers
from modules import image_utils
from modules.composite_generation import CompositeGeneration
from modules.exceptions import PostProcessingFailedError, ProtocolRunRefusedError
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.stack_builder import StackBuilder
from modules.stitcher import Stitcher
from modules.video_builder import VideoBuilder
from modules.zprojector import ZProjector
from tests.settings_fixtures import complete_settings
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    session,
    turret_in_a_known_slot,
)
from tests.test_capture_collision_policy import TILING_CONFIGS
from tests.test_issue_524_extra_z_step_on_objective_change import _empty_protocol_for_add
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture
from modules.layer_record import release_color
from modules.layer_record import release_colors
from modules.coord_transformations import PlateLocation

AT = PlateLocation(x_mm=122.26, y_mm=81.48, z_um=5000.0)
OBJECTIVE = '4x Oly'
# The reporter's four Adds at one position, in order (#772's own files).
ADDS = (('BF',), ('BF', 'Green'), ('BF', 'Blue', 'Green', 'Red'), ('BF', 'Blue', 'Green', 'Red'))
THRESHOLDS = {'Blue': 25, 'Green': 25, 'Red': 25}


def _layer_configs(acquiring):
    settings = complete_settings()
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = 'image' if layer in acquiring else None
    return config_helpers.get_layer_configs(settings)


def _added(scope, adds=ADDS):
    protocol = _empty_protocol_for_add()
    for acquiring in adds:
        scope.protocols.add_step(
            protocol,
            layer_configs=_layer_configs(acquiring),
            stim_configs={},
            plate_position=AT,
            objective_id=OBJECTIVE,
            channel_order=list(acquiring),
        )
    return protocol


def _write_tiff(path, color):
    image_utils.write_tiff(
        data=np.full((8, 8), 120, dtype=np.uint8),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': color,
            'objective': '4x',
            'exposure_time_ms': 2.0,
            'gain_db': 1.0,
            'illumination_ma': 5.0,
            'z_pos_um': 5000.0,
            'plate_pos_mm': {'x': AT.x_mm, 'y': AT.y_mm},
            'datetime': '2026:10:13 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': '',
            'significant_bits': 8,
        },
        ome=False,
        color=release_color(color),
        significant_bits=8,
        save_encoding='right_aligned',
    )


def _run_folder(tmp_path, protocol, *, without_groups=False):
    """The folder one scan of ``protocol`` saves: its protocol, its record, one image per step.

    A stand-in by design: the subject is how a run folder is grouped; the
    run's own writer is pinned by the run tests.
    """
    folder = tmp_path / 'ProtocolData' / '20261013_120000'
    folder.mkdir(parents=True)
    protocol.to_file(folder / 'run.tsv')
    if without_groups:
        # A folder written before steps recorded their Add: a v8 file.
        lines = (folder / 'run.tsv').read_text().splitlines()
        header = lines.index('Steps') + 1
        drop = lines[header].split('\t').index('Channel Group ID')
        rewritten = []
        for i, line in enumerate(lines):
            if line.startswith('Version'):
                line = 'Version\t8'
            elif i >= header and line:
                cells = line.split('\t')
                del cells[drop]
                line = '\t'.join(cells)
            rewritten.append(line)
        (folder / 'run.tsv').write_text('\n'.join(rewritten) + '\n')
    record = ProtocolExecutionRecord(
        protocol_file_loc=pathlib.Path('run.tsv'),
        outfile=folder / ProtocolExecutionRecord.DEFAULT_FILENAME,
    )
    for idx in range(protocol.num_steps()):
        step = protocol.step(idx=idx)
        filename = f'{step["Name"]}_0000.tiff'
        _write_tiff(folder / filename, step['Color'])
        record.add_step(
            capture_result_file_name=pathlib.Path(filename),
            step_name=step['Name'],
            step_index=idx,
            scan_count=0,
            # A capture's clock reading carries microseconds; a whole second
            # is written without them and the record's reader refuses it (a
            # defect of its own, dispositioned apart from this test).
            timestamp=datetime.datetime(2026, 10, 13, 12, 0, idx, 1),
        )
    return folder


def _composite(folder):
    return CompositeGeneration(has_turret=False).load_folder(
        path=folder,
        tiling_configs_file_loc=TILING_CONFIGS,
        brightness_thresholds_percent=THRESHOLDS,
        channel_colors=release_colors(),
    )


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestEachAddIsItsOwnComposite:
    def test_the_reporters_four_adds_make_three_composites_from_every_image(self, scope, tmp_path):
        protocol = _added(scope)
        assert protocol.num_steps() == 11

        result = _composite(_run_folder(tmp_path, protocol))

        # The first Add is one image and composites with nothing; each of
        # the other three is its own composite.
        assert len(result['artifact_paths']) == 3, result

    def test_an_add_shares_one_group_and_the_next_add_takes_a_new_one(self, scope):
        protocol = _added(scope, adds=(('BF', 'Green'), ('BF', 'Green')))

        groups = list(protocol.steps()['Channel Group ID'])

        assert groups == [0, 0, 1, 1]

    def test_a_folder_from_before_the_column_refuses_the_merged_group_by_name(
        self, scope, tmp_path
    ):
        protocol = _added(scope)
        folder = _run_folder(tmp_path, protocol, without_groups=True)

        with pytest.raises(PostProcessingFailedError) as failed:
            _composite(folder)

        message = str(failed.value)
        assert 'more than one image for one plane' in message
        # Every doubled image is named, the earliest and the latest BF alike.
        for idx in (0, 1, 3, 7):
            assert protocol.step(idx=idx)['Name'] in message


class TestTheCompositeRefusesADoubledPlane:
    def _files(self, tmp_path, colors):
        rows = []
        for i, color in enumerate(colors):
            _write_tiff(tmp_path / f'{i}_{color}.tiff', color)
            rows.append({'Color': color, 'Filepath': f'{i}_{color}.tiff'})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize(
        'colors',
        [('BF', 'Red', 'Red'), ('BF', 'PC', 'Red'), ('BF', 'BF', 'Red')],
        ids=['two-of-one-color', 'two-transmitted-layers', 'two-of-the-transmitted-layer'],
    )
    def test_a_group_with_two_images_for_one_plane_is_not_composited(self, tmp_path, colors):
        df = self._files(tmp_path, colors)
        out = tmp_path / 'composite.tiff'

        result = CompositeGeneration._create_composite_image(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent=THRESHOLDS,
            output_file_loc=out,
            channel_colors=release_colors(),
        )

        assert result['status'] is False
        assert 'more than one image for one plane' in result['error']
        assert not out.exists()

    def test_one_image_per_plane_is_composited(self, tmp_path):
        df = self._files(tmp_path, ('BF', 'Green', 'Red'))
        out = tmp_path / 'composite.tiff'

        result = CompositeGeneration._create_composite_image(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent=THRESHOLDS,
            output_file_loc=out,
            channel_colors=release_colors(),
        )

        assert result['status'] is True, result['error']
        assert out.exists()


def _raw_frame(rows):
    """Post-processing's raw-image frame for ``rows`` of (color, group, scan, z_slice, tile, x)."""
    from modules.common_utils import PostFunction

    frame = pd.DataFrame(
        [
            {
                'Filepath': f'{i}.tiff',
                'Scan Count': scan,
                'Well': '',
                'Color': color,
                'Objective': OBJECTIVE,
                'X': x,
                'Y': AT.y_mm,
                'Z-Slice': z_slice,
                'Tile': tile,
                'Tile Group ID': -1,
                'Custom Step': True,
                'Channel Group ID': group,
            }
            for i, (color, group, scan, z_slice, tile, x) in enumerate(rows)
        ]
    )
    frame['Raw'] = True
    for column in PostFunction.list_values():
        frame[column] = False
    return frame


def _group_sizes(builder, frame):
    return sorted(len(g) for _, g in builder._get_groups(frame))


class TestTheOtherBuildsGroupByAdd:
    def test_a_z_projection_takes_one_adds_slices_only(self):
        # Two Adds of Green at one position, each z-stacked to two slices.
        frame = _raw_frame([('Green', g, 0, z, '', AT.x_mm) for g in (0, 1) for z in (0, 1)])

        assert _group_sizes(ZProjector, frame) == [2, 2]

    def test_a_timelapse_takes_one_adds_stills_only(self):
        frame = _raw_frame([('Green', g, scan, -1, '', AT.x_mm) for g in (0, 1) for scan in (0, 1)])

        assert _group_sizes(VideoBuilder, frame) == [2, 2]

    def test_a_hyperstack_takes_one_adds_channels_only(self):
        frame = _raw_frame([(c, g, 0, -1, '', AT.x_mm) for g in (0, 1) for c in ('BF', 'Green')])

        assert _group_sizes(StackBuilder, frame) == [2, 2]

    def test_a_stitch_belongs_to_an_add_only_when_every_tile_does(self):
        # Then the composite of stitched channels groups by that Add; tiles
        # from several Adds belong to none (-1).
        one_add = _raw_frame([('Green', 3, 0, -1, 't', x) for x in (10.0, 12.0)])
        several = _raw_frame([('Green', g, 0, -1, '', x) for g, x in ((0, 10.0), (1, 12.0))])

        assert Stitcher._stitched_channel_group(one_add) == 3
        assert Stitcher._stitched_channel_group(several) == -1

    def test_a_stitch_still_joins_one_color_across_adds_at_several_positions(self):
        # Each position Added on its own: one Add per tile of the mosaic.
        frame = _raw_frame(
            [('Green', g, 0, -1, '', x) for g, x in ((0, 10.0), (1, 12.0), (2, 14.0))]
        )

        assert _group_sizes(Stitcher, frame) == [3]


class TestADoubledStillIsRefusedByTheVideoAndTheStack:
    def test_a_timelapse_with_two_stills_in_one_scan_is_refused_naming_them(self, tmp_path):
        for name in ('a.tiff', 'b.tiff', 'c.tiff'):
            _write_tiff(tmp_path / name, 'Green')
        df = pd.DataFrame(
            {
                'Filepath': ['a.tiff', 'b.tiff', 'c.tiff'],
                'Scan Count': [0, 0, 1],
                'Timestamp': ['', '', ''],
                'Color': ['Green'] * 3,
            }
        )

        result = VideoBuilder(has_turret=False)._create_video(
            path=tmp_path,
            df=df,
            frames_per_sec=1,
            enable_timestamp_overlay=False,
            output_file_loc=pathlib.Path('out.mp4'),
            channel_colors=release_colors(),
        )

        assert result['status'] is False
        assert 'a.tiff' in result['error'] and 'b.tiff' in result['error']
        assert not (tmp_path / 'out.mp4').exists()


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestAnUpdateCannotDoubleAChannelInItsAdd:
    def test_changing_a_step_to_a_channel_its_add_holds_is_refused_and_nothing_changes(self, scope):
        protocol = _added(scope, adds=(('BF', 'Green'),))
        before = protocol.steps()

        with pytest.raises(ProtocolRunRefusedError) as refused:
            scope.protocols.update_step(
                protocol,
                1,
                layer='BF',
                layer_configs=_layer_configs(('BF', 'Green')),
                stim_configs={},
                plate_position=AT,
                objective_id=OBJECTIVE,
            )

        assert refused.value.reason == 'channel_already_in_add'
        pd.testing.assert_frame_equal(protocol.steps(), before)

    def test_a_step_from_before_the_column_has_no_add_to_check(self, scope):
        protocol = _added(scope, adds=(('BF', 'Green'),))
        protocol._config['steps']['Channel Group ID'] = -1

        scope.protocols.update_step(
            protocol,
            1,
            layer='BF',
            layer_configs=_layer_configs(('BF', 'Green')),
            stim_configs={},
            plate_position=AT,
            objective_id=OBJECTIVE,
        )

        assert list(protocol.steps()['Color']) == ['BF', 'BF']


class TestAGeneratedProtocolKeepsAPositionTogether:
    def test_every_channel_and_tile_of_one_well_shares_its_group_and_each_well_has_its_own(
        self, scale_capabilities
    ):
        import copy

        from modules.labware_loader import WellPlateLoader
        from modules.objectives_loader import ObjectiveLoader
        from tests.test_issue_535_z_carry_over_on_new import _build_input_config

        config = _build_input_config()
        config['tiling'] = '2x2'
        config['layer_configs']['Green'] = copy.deepcopy(config['layer_configs']['BF'])
        protocol = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=TILING_CONFIGS,
            capabilities=scale_capabilities,
            objective_helper=ObjectiveLoader(),
            wellplate_loader=WellPlateLoader(),
        )
        steps = protocol.steps()

        per_well = steps.groupby('Well')['Channel Group ID'].nunique()
        assert set(per_well) == {1}, per_well
        assert steps.groupby('Well')['Channel Group ID'].first().is_unique
        # A well's eight steps (two channels, four tiles) are one composite
        # per tile, as before: the group adds nothing the tile does not.
        assert _group_sizes(CompositeGeneration, _frame_of(steps)) == [2] * (len(per_well) * 4)


def _frame_of(steps):
    """The raw-image frame post-processing builds from these protocol steps."""
    from modules.common_utils import PostFunction

    frame = steps[
        ['Well', 'Color', 'Objective', 'X', 'Y', 'Z-Slice', 'Tile', 'Tile Group ID',
         'Custom Step', 'Channel Group ID']
    ].copy()  # fmt: skip
    frame['Filepath'] = [f'{i}.tiff' for i in range(len(frame))]
    frame['Scan Count'] = 0
    frame['Raw'] = True
    for column in PostFunction.list_values():
        frame[column] = False
    return frame


class TestAHyperstackAfterApplyTiling:
    def test_each_tile_of_an_add_is_one_stack_holding_every_channel(self, session):
        # Apply Tiling gives every channel of a position its own tile group,
        # and the stack grouped by tile group, so every stack held one image
        # and was skipped: a tiled-at-New protocol built them, this built none.
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        session.settings['BF']['acquire'] = 'image'
        session.settings['Green']['acquire'] = 'image'
        session.scope.motion.move_absolute('X', 20000)
        session.scope.motion.move_absolute('Y', 40000)
        session.scope.motion.move_absolute('Z', 5000.0)
        protocol = session.create_empty_protocol()
        session.add_step(protocol, after_step=-1)

        session.apply_tiling(protocol, '2x2')

        stacks = list(StackBuilder._get_groups(_frame_of(protocol.steps())))
        assert sorted(sorted(g['Color']) for _, g in stacks) == [['BF', 'Green']] * 4

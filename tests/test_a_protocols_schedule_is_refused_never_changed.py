# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's period and duration are its own, and a schedule it cannot run is refused.

A period is None or 0 (one scan), or at least one second; a duration is
None, 0 or more. Anything else is refused where it enters -- the protocol's
writer, its constructor, its file, the stored default's writer -- and the
protocol keeps the timing it had. Nothing raises a short period to one
second behind the caller's back: the value a caller set is the value the
run runs, or the caller is told it was not taken.

A stored default out of range is the one value replaced: by the template's,
for that key alone, and the replacement is told once, when the Session has
someone to tell.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import shutil

import pandas as pd
import pytest

import modules.config_helpers as config_helpers
from modules import settings_init
from modules.exceptions import ConfigError, Refusal
from modules.protocol import Protocol, ProtocolScheduleRefusedError, schedule_from_units
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

REPO = pathlib.Path(__file__).resolve().parent.parent
TILING = REPO / 'data' / 'tiling.json'
TEMPLATE = REPO / 'data' / 'settings.json'
PERIOD = datetime.timedelta(minutes=5)
DURATION = datetime.timedelta(hours=2)

UNRUNNABLE = [
    pytest.param(datetime.timedelta(milliseconds=500), DURATION, id='half-second-period'),
    pytest.param(datetime.timedelta(minutes=-1), DURATION, id='negative-period'),
    pytest.param(PERIOD, datetime.timedelta(hours=-1), id='negative-duration'),
    pytest.param(5, DURATION, id='period-not-a-time'),
]
RUNNABLE = [
    pytest.param(None, None, id='none'),
    pytest.param(datetime.timedelta(0), datetime.timedelta(0), id='zero'),
    pytest.param(datetime.timedelta(seconds=1), datetime.timedelta(seconds=1), id='one-second'),
]


def _protocol(period=PERIOD, duration=DURATION) -> Protocol:
    return Protocol(
        tiling_configs_file_loc=TILING,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(),
            'custom_step_count': 0,
            'period': period,
            'duration': duration,
            'capture_root': '',
            'labware_id': 'Center Plate',
        },
    )


class TestTheProtocolsWriter:
    @pytest.mark.parametrize('period, duration', UNRUNNABLE)
    def test_an_unrunnable_schedule_is_refused_and_the_old_one_kept(self, period, duration):
        protocol = _protocol()

        with pytest.raises(ProtocolScheduleRefusedError) as refused:
            protocol.modify_time_params(period=period, duration=duration)

        assert isinstance(refused.value, Refusal)
        assert (protocol.period(), protocol.duration()) == (PERIOD, DURATION)

    @pytest.mark.parametrize('period, duration', RUNNABLE)
    def test_a_runnable_schedule_is_taken_as_given(self, period, duration):
        protocol = _protocol()

        protocol.modify_time_params(period=period, duration=duration)

        assert (protocol.period(), protocol.duration()) == (period, duration)


class TestTheConstructors:
    @pytest.mark.parametrize('period, duration', UNRUNNABLE)
    def test_a_protocol_built_with_an_unrunnable_schedule_is_refused(self, period, duration):
        with pytest.raises(ProtocolScheduleRefusedError):
            _protocol(period, duration)

    def test_a_file_carrying_a_half_second_period_is_refused_naming_the_file(self, tmp_path):
        path = tmp_path / 'short.tsv'
        _protocol().to_file(path)
        text = path.read_text().replace('Period\t5.0\n', 'Period\t0.008333\n')
        assert 'Period\t0.008333\n' in text, 'the file did not carry the row this test rewrites'
        path.write_text(text)

        with pytest.raises(ProtocolScheduleRefusedError) as refused:
            Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING)

        assert str(path) in str(refused.value)

    def test_a_file_carrying_a_negative_duration_is_refused_naming_the_file(self, tmp_path):
        path = tmp_path / 'backwards.tsv'
        _protocol().to_file(path)
        text = path.read_text().replace('Duration\t2.0\n', 'Duration\t-1\n')
        assert 'Duration\t-1\n' in text, 'the file did not carry the row this test rewrites'
        path.write_text(text)

        with pytest.raises(ProtocolScheduleRefusedError) as refused:
            Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING)

        assert str(path) in str(refused.value)


class TestAReaderThatNeverRunsTheProtocol:
    """Post-processing reads a finished run's protocol and never uses its schedule.

    A protocol saved before the one-second floor existed can carry a shorter
    period; refusing that run's post-processing over a value it never reads
    would lock the person out of their own data. Loading the same file to
    run it is still refused.
    """

    def _run_folder(self, tmp_path):
        folder = tmp_path / 'run'
        folder.mkdir()
        # One BF step at A1, as a 96-well run of this release saves it, with
        # the period an earlier release could save.
        columns = [
            'Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination',
            'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice',
            'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config',
            'Stim_Config', 'Auto_Named', 'Label',
        ]  # fmt: skip
        row = [
            'A1_BF', '14.38', '11.24', '5000.0', 'False', 'BF', 'False', '5.0', '1.0', 'False',
            '2.0', '1', '20x w/collar', 'A1', '', '-1', 'False', '-1', '-1', 'image',
            '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True', '',
        ]  # fmt: skip
        (folder / 'unsaved_protocol.tsv').write_text(
            'LumaViewPro Protocol\nVersion\t8\nPeriod\t0.005\nDuration\t0.5\n'
            'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
            + '\t'.join(columns)
            + '\n'
            + '\t'.join(row)
            + '\n'
        )
        (folder / 'protocol_record.tsv').write_text(
            'LumaViewPro Protocol Execution Record\n'
            'Version\t3\n'
            'Protocol File\tunsaved_protocol.tsv\n'
            'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
            'A1_BF_0000.tiff\tA1_BF_0000\t0\t0\t2026-10-02 23:35:52.033946\t1\t0.0\n'
        )
        import cv2
        import numpy as np

        cv2.imwrite(str(folder / 'A1_BF_0000.tiff'), np.zeros((8, 8), dtype=np.uint8))
        return folder

    def test_post_processing_reads_an_old_run_with_a_sub_second_period(self, tmp_path):
        from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

        loaded = ProtocolPostProcessingHelper().load_folder(
            self._run_folder(tmp_path), tiling_configs_file_loc=TILING
        )

        assert loaded['status'], loaded.get('message')

    def test_the_same_file_is_still_refused_when_loaded_to_run(self, tmp_path):
        with pytest.raises(ProtocolScheduleRefusedError):
            Protocol.from_file(
                file_path=self._run_folder(tmp_path) / 'unsaved_protocol.tsv',
                tiling_configs_file_loc=TILING,
            )

    def test_a_reader_that_does_not_judge_keeps_the_schedule_as_the_file_has_it(self, tmp_path):
        protocol = Protocol.from_file(
            file_path=self._run_folder(tmp_path) / 'unsaved_protocol.tsv',
            tiling_configs_file_loc=TILING,
            judge_schedule=False,
        )

        assert protocol.period() == datetime.timedelta(minutes=0.005)


class TestAValueInItsUnits:
    @pytest.mark.parametrize(
        'key, value',
        [
            ('period', '0.005'),
            ('period', 'abc'),
            ('period', ''),
            ('period', 'nan'),
            ('duration', '-1'),
            ('duration', 'inf'),
        ],
    )
    def test_what_cannot_be_run_is_refused(self, key, value):
        with pytest.raises(ProtocolScheduleRefusedError):
            schedule_from_units(key, value)

    def test_a_period_is_minutes_and_a_duration_hours(self):
        assert schedule_from_units('period', '10') == datetime.timedelta(minutes=10)
        assert schedule_from_units('duration', 2) == datetime.timedelta(hours=2)
        assert schedule_from_units('period', None) is None

    def test_the_settings_reader_raises_no_period_to_a_second(self):
        with pytest.raises(ProtocolScheduleRefusedError):
            config_helpers.get_protocol_time_params_from_settings(
                {'protocol': {'period': 0.001, 'duration': 1}}
            )


def _acquire_only_bf(session):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'


class TestTheSession:
    def test_new_builds_the_schedule_it_is_given(self, session):
        _acquire_only_bf(session)

        protocol = session.new_protocol(
            period=datetime.timedelta(minutes=7), duration=datetime.timedelta(hours=3)
        )

        assert protocol.period() == datetime.timedelta(minutes=7)
        assert protocol.duration() == datetime.timedelta(hours=3)

    def test_new_with_no_schedule_takes_the_stored_default(self, session):
        _acquire_only_bf(session)
        session.settings['protocol']['period'] = 4
        session.settings['protocol']['duration'] = 6

        protocol = session.new_protocol()

        assert protocol.period() == datetime.timedelta(minutes=4)
        assert protocol.duration() == datetime.timedelta(hours=6)

    def test_new_with_an_unrunnable_schedule_is_refused(self, session):
        _acquire_only_bf(session)

        with pytest.raises(ProtocolScheduleRefusedError):
            session.new_protocol(period=datetime.timedelta(milliseconds=500), duration=DURATION)

    def test_a_stored_default_out_of_range_is_refused_and_the_store_unchanged(self, session):
        before = dict(session.settings['protocol'])

        with pytest.raises(ProtocolScheduleRefusedError):
            session.update_settings('protocol', {**before, 'period': 0.005})

        assert session.settings['protocol'] == before

    def test_a_runnable_stored_default_is_written(self, session):
        before = dict(session.settings['protocol'])

        session.update_settings('protocol', {**before, 'period': 3})

        assert session.settings['protocol']['period'] == 3

    def test_a_protocol_settings_value_that_is_not_a_mapping_is_still_a_config_error(self, session):
        with pytest.raises(ConfigError):
            session.update_settings('protocol', 5)


def _appdata_with_stored_protocol(tmp_path, **stored) -> str:
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    current['protocol'].update(stored)
    (data_dir / 'current.json').write_text(json.dumps(current))
    return str(tmp_path)


class TestAStoredDefaultOutOfRange:
    def test_that_key_alone_takes_the_templates_value(self, tmp_path):
        template = json.loads(TEMPLATE.read_text())['protocol']
        root = _appdata_with_stored_protocol(tmp_path, period=0.001, duration=5)

        settings, _rejected = settings_init.prepare_settings(
            logging.getLogger(__name__), root, fall_back_to_template=False
        )

        assert settings['protocol']['period'] == template['period']
        assert settings['protocol']['duration'] == 5
        settings_init.take_schedule_replacements()

    def test_the_replacement_is_told_once_when_the_session_has_a_listener(self, tmp_path):
        root = _appdata_with_stored_protocol(tmp_path, period=0.001)
        settings, _rejected = settings_init.prepare_settings(
            logging.getLogger(__name__), root, fall_back_to_template=False
        )
        settings['live_folder'] = str(tmp_path / 'live')
        heard = []

        built = ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)
        try:
            told = [n for n in heard if n.title == 'Saved protocol timing replaced']
            assert len(told) == 1 and told[0].shown
            assert '0.001' in told[0].message and 'period' in told[0].message
        finally:
            built.shutdown()

        again = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path / 'live2')),
            simulate=True,
            outcome_listener=heard.append,
        )
        try:
            assert len([n for n in heard if n.title == 'Saved protocol timing replaced']) == 1
        finally:
            again.shutdown()

    def test_a_stored_default_in_range_is_kept_and_nothing_is_told(self, tmp_path):
        root = _appdata_with_stored_protocol(tmp_path, period=0, duration=0)

        settings, _rejected = settings_init.prepare_settings(
            logging.getLogger(__name__), root, fall_back_to_template=False
        )

        assert (settings['protocol']['period'], settings['protocol']['duration']) == (0, 0)
        assert settings_init.take_schedule_replacements() == []


def test_the_clamp_is_gone():
    assert not hasattr(config_helpers, 'floor_protocol_time')
    assert not hasattr(config_helpers, 'protocol_time_clamped')
    assert not hasattr(config_helpers, 'MIN_PROTOCOL_TIME_SECONDS')

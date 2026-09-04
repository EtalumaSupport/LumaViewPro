# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A factory-built session is a configured session.

`ScopeSession.create_headless` (and `create`, for a scope it builds) runs
the settings-to-scope bring-up before it returns: the turret slot keys
normalized, the slot-1 objective adopted, the labware selected, the
scope initialized, the camera start gate released. A session that came
back from a factory can therefore save an image, refuses settings that
cannot configure a scope by name, and never leaves threads behind when
it refuses.
"""

import threading
import time

import numpy as np
import pytest

from modules.exceptions import ConfigError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings, complete_settings_without


def _wait_for_thread_count(target, deadline_s=2.0):
    end = time.monotonic() + deadline_s
    while threading.active_count() > target and time.monotonic() < end:
        time.sleep(0.02)
    return threading.active_count()


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create_headless(settings=complete_settings(live_folder=str(tmp_path)))
    try:
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


class TestAFactorySessionIsConfigured:
    def test_the_helpers_are_real_and_the_engine_holds_them(self, session):
        assert session.objective_helper is not None
        assert session.wellplate_loader is not None
        assert session.coordinate_transformer is not None
        assert session.sequenced_capture_runner._wellplate_loader is session.wellplate_loader

    def test_the_scope_carries_the_objective_and_labware(self, session):
        assert (
            session.scope.runtime_state.get_current_objective_id()
            == session.settings['objective_id']
        )
        assert session.scope.runtime_state.get_labware() is not None
        assert session.scope.imaging.is_streaming()

    def test_a_bare_session_captures_and_saves(self, session, tmp_path):
        from modules.image_save import save_image

        frame = session.scope.imaging.capture_and_wait()
        assert isinstance(frame, np.ndarray)
        path = save_image(
            session.scope,
            array=frame,
            save_folder=str(tmp_path),
            file_root='bringup',
            append='',
            tail_id_mode=None,
            channel='BF',
            false_color_on=False,
            save_encoding='8bit',
            significant_bits=8,
        )
        assert path is not None


class TestSettingsThatCannotConfigureAScope:
    def test_missing_frame_refuses_by_key(self):
        with pytest.raises(ConfigError, match='frame'):
            ScopeSession.create_headless(settings=complete_settings_without('frame'))

    def test_missing_objective_refuses_by_key(self):
        with pytest.raises(ConfigError, match='objective_id'):
            ScopeSession.create_headless(settings=complete_settings_without('objective_id'))

    def test_an_unshipped_objective_refuses_by_value(self):
        with pytest.raises(ConfigError, match='banana'):
            ScopeSession.create_headless(settings=complete_settings(objective_id='banana'))

    def test_string_turret_keys_are_normalized_and_slot_one_is_adopted(self, tmp_path):
        """A caller dict carries JSON string keys; the file pipeline never saw
        it. Adoption must still find slot 1 and write the objective."""
        raw = complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_id='20x Oly',
        )
        raw['turret_objectives'] = {'1': '10x Oly', '2': None, '3': None, '4': None}
        s = ScopeSession.create_headless(settings=raw)
        try:
            assert s.settings['objective_id'] == '10x Oly'
            assert s.scope.runtime_state.get_current_objective_id() == '10x Oly'
        finally:
            s.shutdown()
            s.scope.disconnect()

    def test_a_root_without_a_template_refuses_by_root(self, tmp_path):
        with pytest.raises(ConfigError, match=r'settings\.json'):
            ScopeSession.create_headless(source_path=str(tmp_path))

    def test_a_root_without_labware_refuses_by_file(self, tmp_path):
        import pathlib
        import shutil

        repo = pathlib.Path(__file__).resolve().parent.parent
        data = tmp_path / 'data'
        data.mkdir()
        shutil.copy(repo / 'data' / 'settings.json', data / 'settings.json')
        shutil.copy(repo / 'data' / 'objectives.json', data / 'objectives.json')
        with pytest.raises(ConfigError, match=r'labware\.json'):
            ScopeSession.create_headless(source_path=str(tmp_path))


class TestARefusingFactoryLeavesNothingBehind:
    def test_thread_count_returns_to_baseline(self):
        baseline = threading.active_count()
        with pytest.raises(ConfigError):
            ScopeSession.create_headless(settings=complete_settings_without('frame'))
        # Other tests' threads may finish during the wait; what matters is
        # that the refusing factory left none of its own behind.
        assert _wait_for_thread_count(baseline) <= baseline

    def test_a_callers_lanes_survive_the_refusal(self):
        from modules.sequential_io_executor import SequentialIOExecutor

        io = SequentialIOExecutor(name='IO_CALLER')
        cam = SequentialIOExecutor(name='CAMERA_CALLER')
        io.start()
        cam.start()
        try:
            with pytest.raises(ConfigError):
                ScopeSession.create(
                    settings=complete_settings_without('frame'),
                    io_executor=io,
                    camera_executor=cam,
                )
            assert io.accepts_work() and cam.accepts_work()
        finally:
            io.shutdown()
            cam.shutdown()

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression (#790): an engineering-mode standalone AF must deliver
its characterization data to disk.

The AF runner queues _save_autofocus_data via
file_io_executor.protocol_put on every exit.  protocol_put accepts
only while the executor is in protocol mode; outside it the task is
DROPPED (return None) by design.  The old standalone-AF wrapper drove
the AF thread directly with the executors live, so the save was
silently dropped: the timestamped results folder appeared on disk and
stayed empty -- "queued" is not "delivered"
(tests/test_af_char_save_on_all_exits.py pins queued-on-a-mock, which
is exactly the blind spot the drop lived in).

The standalone button now starts a one-position SINGLE_AUTOFOCUS_SCAN
run, so the file executor is in protocol mode for the AF's whole
window and the save rides the run's file queue.  This pin drives that
engine path end to end -- real simulated scope, real executors, real
AutofocusRunner on a real AutofocusThread -- and asserts the data
lands on disk.
"""

from __future__ import annotations

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it. (Harness mirrors
# tests/test_run_refusal_contract.py.)
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from tests.scope_fakes import home_sim_scope
from modules.protocol import Protocol
from modules.sequenced_capture_runner import SequencedCaptureRunner
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import SequentialIOExecutor
from tests.protocol_drives import autofocus_snapshot

COMPLETION_TIMEOUT = 60  # seconds -- a real AF sweep runs in sim time

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _make_af_step_protocol():
    import pandas as pd

    step = {
        'Name': 'AF_test',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': True,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': '10x Oly',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
    }
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': pd.DataFrame([step]),
        'period': datetime.timedelta(minutes=1.0),
        'duration': datetime.timedelta(hours=1.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)


class TestStandaloneAfDeliversCharacterizationData:
    def test_af_run_delivers_characterization_data_to_disk(self, tmp_path):
        from modules.autofocus_runner import AutofocusRunner
        from modules.autofocus_thread import AutofocusThread
        from modules.coord_transformations import CoordinateTransformer
        from modules.labware_loader import WellPlateLoader
        from modules.protocol_thread import ProtocolThread

        scope = home_sim_scope(Lumascope(simulate=True))
        scope._led_driver.set_timing_mode('fast')
        scope._motion_driver.set_timing_mode('fast')
        scope._camera_driver.set_timing_mode('fast')
        scope.imaging.start_streaming()

        io_executor = SequentialIOExecutor(name='AF790_IO')
        file_io_executor = SequentialIOExecutor(name='AF790_FILE')
        camera_executor = SequentialIOExecutor(name='AF790_CAMERA')
        for e in (io_executor, file_io_executor, camera_executor):
            e.start()
        protocol_thread = ProtocolThread()
        protocol_thread.start()

        af_runner = AutofocusRunner(
            scope=scope,
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=file_io_executor,
        )
        af_thread = AutofocusThread(afe=af_runner)
        af_thread.start()

        runner = SequencedCaptureRunner(
            scope=scope,
            stage_offset={'x': 0.0, 'y': 0.0},
            io_executor=io_executor,
            protocol_thread=protocol_thread,
            file_io_executor=file_io_executor,
            camera_executor=camera_executor,
            autofocus_thread=af_thread,
            autofocus_runner=af_runner,
        )
        runner._wellplate_loader = WellPlateLoader()
        runner._coordinate_transformer = CoordinateTransformer()

        char_dir = tmp_path / 'Autofocus Characterization'
        done = threading.Event()
        files_done = threading.Event()
        try:
            plan = runner.prepare(
                protocol=_make_af_step_protocol(),
                run_trigger_source='autofocus',
                run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
                sequence_name='autofocus',
                image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
                autogain_settings={
                    'target_brightness': 0.3,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                    'max_duration': datetime.timedelta(seconds=1),
                },
                parent_dir=char_dir,
                enable_image_saving=False,
                disable_saving_artifacts=True,
                save_autofocus_data=True,
                max_scans=1,
                callbacks={
                    'go_to_step': lambda **kw: None,
                    'move_position': lambda axis: None,
                    'run_complete': lambda **kw: done.set(),
                    'files_complete': lambda **kw: files_done.set(),
                },
                leds_state_at_end='off',
                autofocus_snapshot=autofocus_snapshot(
                    states={
                        'BF': True,
                        'PC': False,
                        'DF': False,
                        'Red': False,
                        'Green': False,
                        'Blue': False,
                        'Lumi': False,
                    },
                ),
            )
            runner.start(plan)
            assert done.wait(timeout=COMPLETION_TIMEOUT), 'AF run did not complete'
            assert files_done.wait(timeout=COMPLETION_TIMEOUT), 'AF run files_complete did not fire'

            deadline = time.monotonic() + 10.0
            files = []
            while time.monotonic() < deadline:
                files = [p for p in char_dir.rglob('*') if p.is_file()]
                if files:
                    break
                time.sleep(0.1)
            assert files, (
                'an AF run with save_autofocus_data=True must leave its '
                'characterization data on disk; an empty folder means the '
                'save was dropped instead of riding the run file queue'
            )
        finally:
            af_thread.stop()
            protocol_thread.stop(timeout=2.0)
            for e in (io_executor, file_io_executor, camera_executor):
                try:
                    e.shutdown()
                except Exception:
                    pass
            scope.imaging.stop_streaming()
            scope.disconnect()

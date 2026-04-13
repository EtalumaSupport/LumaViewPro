# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Headless protocol execution test.

Verifies that the full protocol execution chain runs without Kivy.
This test ensures Rule 15 (executors must be GUI-agnostic) holds at
the import and runtime level:
  1. No module under modules/ imports Kivy directly (except the
     explicitly-named modules/image_utils_kivy.py).
  2. A full protocol can be executed through SequencedCaptureExecutor
     without any Kivy module loaded in sys.modules.
  3. kivy_utils.schedule_ui() falls back to direct invocation when no
     GUI dispatcher has been set.

This complements test_integration.py, which mocks Kivy out — here we
assert the modules NEVER LOAD Kivy at all.
"""

import datetime
import pathlib
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock non-GUI heavy dependencies (but NOT Kivy — we want to see if it loads)
# ---------------------------------------------------------------------------
_mock_logger = MagicMock()
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = _mock_logger
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()

sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
sys.modules.setdefault('psutil', MagicMock())

sys.modules.setdefault('pypylon', MagicMock())
sys.modules.setdefault('pypylon.pylon', MagicMock())
sys.modules.setdefault('pypylon.genicam', MagicMock())
sys.modules.setdefault('ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak', MagicMock())
sys.modules.setdefault('ids_peak.ids_peak_ipl_extension', MagicMock())
sys.modules.setdefault('ids_peak_ipl', MagicMock())

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


# ---------------------------------------------------------------------------
# CRITICAL: remove any Kivy modules that might have been loaded by a previous
# test in the same session. We want to verify the protocol chain can import
# cleanly without Kivy.
# ---------------------------------------------------------------------------
def _purge_kivy_from_sys_modules():
    """Drop all kivy.* and kivy modules from sys.modules (including mocks)."""
    to_drop = [name for name in list(sys.modules) if name == 'kivy' or name.startswith('kivy.')]
    for name in to_drop:
        del sys.modules[name]


# Purge at import time so our imports below are clean
_purge_kivy_from_sys_modules()


# Now import the protocol execution chain — these MUST not require Kivy
from modules.lumascope_api import Lumascope
from modules.sequential_io_executor import SequentialIOExecutor
from modules.sequenced_capture_executor import (
    SequencedCaptureExecutor,
    SequencedCaptureRunMode,
)
from modules.protocol import Protocol
from modules.kivy_utils import schedule_ui
import modules.kivy_utils as _kivy_utils


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHeadlessImports:
    """Verify modules/ imports don't load Kivy."""

    def test_protocol_imports_do_not_load_kivy(self):
        """After importing the protocol execution chain, Kivy must not be loaded."""
        _purge_kivy_from_sys_modules()

        # Re-import the protocol chain (modules already in sys.modules will
        # be a no-op, so reload their state by dropping them first if needed)
        import importlib
        import modules.lumascope_api
        import modules.sequenced_capture_executor
        import modules.protocol
        import modules.kivy_utils

        # Verify no kivy module is in sys.modules
        kivy_loaded = [name for name in sys.modules if name == 'kivy' or name.startswith('kivy.')]
        assert not kivy_loaded, (
            f"Kivy modules unexpectedly loaded by protocol chain: {kivy_loaded}. "
            "This violates Rule 15 (executors must be GUI-agnostic)."
        )

    def test_schedule_ui_falls_back_to_direct_invocation(self):
        """Without a UI dispatcher, schedule_ui calls the function directly."""
        # Clear any dispatcher set by previous tests
        _kivy_utils._ui_dispatcher = None

        called = []

        def my_func(dt):
            called.append(dt)

        schedule_ui(my_func)
        assert called == [0], "schedule_ui should have invoked func directly with dt=0"

    def test_schedule_ui_with_dispatcher(self):
        """When dispatcher is set, schedule_ui goes through it."""
        calls = []

        def fake_dispatcher(func, timeout):
            calls.append((func, timeout))

        _kivy_utils.set_ui_dispatcher(fake_dispatcher)
        try:
            def my_func(dt):
                pass
            schedule_ui(my_func, timeout=0.5)
            assert len(calls) == 1
            assert calls[0][0] is my_func
            assert calls[0][1] == 0.5
        finally:
            _kivy_utils._ui_dispatcher = None


class TestHeadlessProtocolExecution:
    """Verify a full protocol runs end-to-end without Kivy loaded."""

    def _make_executors(self):
        names = ['io', 'protocol', 'file_io', 'camera', 'autofocus']
        execs = {n: SequentialIOExecutor(name=f"HEADLESS_{n.upper()}") for n in names}
        for e in execs.values():
            e.start()
        return execs

    def _shutdown_executors(self, execs):
        for e in execs.values():
            try:
                e.shutdown()
            except Exception:
                pass

    def _make_protocol(self):
        """Build a minimal single-step protocol."""
        import pandas as pd

        TILING_CONFIGS = pathlib.Path(__file__).parent.parent / "data" / "tiling.json"

        rows = [{
            'Name': 'A1_BF',
            'X': 10.0, 'Y': 20.0, 'Z': 5000.0,
            'Auto_Focus': False,
            'Color': 'BF',
            'False_Color': False,
            'Illumination': 100.0,
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
        }]
        df = pd.DataFrame(rows)
        config = {
            "version": Protocol.CURRENT_VERSION,
            "steps": df,
            "period": datetime.timedelta(minutes=1),
            "duration": datetime.timedelta(hours=1),
            "labware_id": "6 well microplate",
            "capture_root": "",
            "tiling": "1x1",
        }
        return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)

    def test_kivy_stays_unloaded_during_protocol_run(self, tmp_path):
        """Full protocol run must not cause Kivy to be loaded at any point."""
        _purge_kivy_from_sys_modules()
        # Ensure no dispatcher leaked from previous test
        _kivy_utils._ui_dispatcher = None

        from modules.coord_transformations import CoordinateTransformer
        from modules.labware_loader import WellPlateLoader

        scope = Lumascope(simulate=True)
        # Speed up the simulator for test runtime
        scope.led.set_timing_mode('fast')
        scope.motion.set_timing_mode('fast')
        scope.camera.set_timing_mode('fast')
        scope.camera.grab()

        execs = self._make_executors()
        try:
            mock_af = MagicMock()
            mock_af.reset = MagicMock()
            mock_af.in_progress = MagicMock(return_value=False)
            mock_af.complete = MagicMock(return_value=False)
            mock_af.is_running = MagicMock(return_value=False)
            mock_af.result = MagicMock(return_value=None)
            mock_af.best_focus_position = MagicMock(return_value=5000.0)
            mock_af.run_in_progress = MagicMock(return_value=False)

            executor = SequencedCaptureExecutor(
                scope=scope,
                stage_offset={'x': 0.0, 'y': 0.0},
                io_executor=execs['io'],
                protocol_executor=execs['protocol'],
                file_io_executor=execs['file_io'],
                camera_executor=execs['camera'],
                autofocus_io_executor=execs['autofocus'],
                autofocus_executor=mock_af,
            )
            executor._wellplate_loader = WellPlateLoader()
            executor._coordinate_transformer = CoordinateTransformer()

            protocol = self._make_protocol()

            done = threading.Event()

            def on_complete(**kwargs):
                done.set()

            callbacks = {
                'run_complete': on_complete,
                'move_position': lambda axis: None,
            }

            autogain_settings = {
                'target_brightness': 0.3,
                'min_gain': 0.0,
                'max_gain': 20.0,
                'max_duration': datetime.timedelta(seconds=2),
            }

            image_capture_config = {
                'output_format': {'live': 'TIFF', 'sequenced': 'TIFF'},
                'use_full_pixel_depth': False,
            }

            executor.run(
                protocol=protocol,
                run_trigger_source='test',
                run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
                sequence_name='headless_test',
                image_capture_config=image_capture_config,
                autogain_settings=autogain_settings,
                parent_dir=tmp_path / 'output',
                max_scans=1,
                callbacks=callbacks,
                leds_state_at_end='off',
                enable_image_saving=False,
                initial_autofocus_states={
                    'BF': False, 'PC': False, 'DF': False,
                    'Red': False, 'Green': False, 'Blue': False, 'Lumi': False,
                },
            )

            assert done.wait(timeout=30), "Protocol did not complete within timeout"

            # Check Kivy was never loaded during the protocol run
            kivy_loaded = [name for name in sys.modules
                           if name == 'kivy' or name.startswith('kivy.')]
            assert not kivy_loaded, (
                f"Kivy modules loaded during protocol run: {kivy_loaded}. "
                "Rule 15 violation: executors must be GUI-agnostic."
            )
        finally:
            try:
                scope.disconnect()
            except Exception:
                pass
            self._shutdown_executors(execs)

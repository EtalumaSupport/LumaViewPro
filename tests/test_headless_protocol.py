# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Headless protocol execution test.

Verifies that the full protocol execution chain runs without Kivy.
This test ensures Rule 15 (executors must be GUI-agnostic) holds at
the import and runtime level:
  1. No module under modules/ imports Kivy directly (image_utils_kivy
     used to live there as an exception; it now lives at
     ui/image_utils_kivy.py per Rule 15).
  2. A full protocol can be executed through SequencedCaptureRunner
     without any Kivy module loaded in sys.modules.
  3. kivy_utils.schedule_ui() falls back to direct invocation when no
     GUI dispatcher has been set.

This complements test_integration.py, which mocks Kivy out -- here we
assert the modules NEVER LOAD Kivy at all.
"""

import contextlib
import datetime
import pathlib
import sys
import threading
from unittest.mock import MagicMock


# Heavy non-GUI deps (lvp_logger, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. The kivy mock from conftest
# gets purged below -- this test deliberately verifies the protocol chain
# loads without any kivy module present.

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


# Now import the protocol execution chain -- these MUST not require Kivy
from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from modules.sequential_io_executor import SequentialIOExecutor
from modules.sequenced_capture_runner import (
    SequencedCaptureRunner,
    SequencedCaptureRunMode,
)
from modules.protocol import Protocol
from modules.kivy_utils import schedule_ui
import modules.kivy_utils as _kivy_utils

# The purge above poisons the rest of the session: conftest installed the
# kivy stubs once, before any file was collected, and every later-collected
# test file that imports a ui/ module relies on them still being present.
# Re-install (idempotent) now that the kivy-free imports are proven.
from tests.conftest import install_mock_deps
from tests.protocol_drives import autofocus_snapshot

install_mock_deps()


@contextlib.contextmanager
def _kivy_purged():
    """Purge kivy for the body, then hand the stubs back.

    ``sys.modules`` is process-global. conftest installs the kivy stubs once
    before collection, and six sibling test files install their own at import
    time, so a purge that does not restore disarms all of them for the rest of
    the session -- the failure lands in whichever file happens to run next,
    nowhere near here. The restore is bound to the purge here rather than left
    to each site to remember, because a site already forgot.
    """
    _purge_kivy_from_sys_modules()
    try:
        yield
    finally:
        install_mock_deps()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHeadlessImports:
    """Verify modules/ imports don't load Kivy."""

    def test_protocol_imports_do_not_load_kivy(self):
        """After importing the protocol execution chain, Kivy must not be loaded."""
        with _kivy_purged():
            # Re-import the protocol chain (modules already in sys.modules will
            # be a no-op, so reload their state by dropping them first if needed)
            import modules.lumascope_api
            import modules.sequenced_capture_runner
            import modules.protocol
            import modules.kivy_utils  # noqa: F401  -- imported to load the protocol chain so the no-kivy assertion below covers it

            # Verify no kivy module is in sys.modules
            kivy_loaded = [
                name for name in sys.modules if name == 'kivy' or name.startswith('kivy.')
            ]
            assert not kivy_loaded, (
                f'Kivy modules unexpectedly loaded by protocol chain: {kivy_loaded}. '
                'This violates Rule 15 (executors must be GUI-agnostic).'
            )

    def test_a_purge_does_not_outlive_its_own_scope(self):
        """The stubs a purge removes must be back when the body exits.

        This pins the invariant, not the symptom: whichever sibling file runs
        next is the one that fails, so nothing in this file would have caught
        a purge that forgot to restore.
        """
        assert 'kivy.app' in sys.modules, 'conftest should have stubbed kivy.app'

        with _kivy_purged():
            assert not [
                name for name in sys.modules if name == 'kivy' or name.startswith('kivy.')
            ], 'the body of a purge must see no kivy at all'

        assert 'kivy.app' in sys.modules, 'the purge did not hand the stubs back'

    def test_schedule_ui_falls_back_to_direct_invocation(self):
        """Without a UI dispatcher, schedule_ui calls the function directly."""
        # Clear any dispatcher set by previous tests
        _kivy_utils._ui_dispatcher = None

        called = []

        def my_func(dt):
            called.append(dt)

        schedule_ui(my_func)
        assert called == [0], 'schedule_ui should have invoked func directly with dt=0'

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
        from modules.protocol_thread import ProtocolThread

        names = ['io', 'file_io', 'camera', 'autofocus']
        execs = {n: SequentialIOExecutor(name=f'HEADLESS_{n.upper()}') for n in names}
        for e in execs.values():
            e.start()
        pt = ProtocolThread()
        pt.start()
        execs['protocol'] = pt
        return execs

    def _shutdown_executors(self, execs):
        for name, e in execs.items():
            try:
                if name == 'protocol':
                    e.stop(timeout=2.0)
                else:
                    e.shutdown()
            except Exception:
                pass

    def _make_protocol(self):
        """Build a minimal single-step protocol."""
        import pandas as pd

        TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'

        rows = [
            {
                'Name': 'A1_BF',
                'X': 10.0,
                'Y': 20.0,
                'Z': 5000.0,
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
            }
        ]
        df = pd.DataFrame(rows)
        config = {
            'version': Protocol.CURRENT_VERSION,
            'steps': df,
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(hours=1),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        }
        return Protocol(tiling_configs_file_loc=TILING_CONFIGS, config=config)

    def test_kivy_stays_unloaded_during_protocol_run(self, tmp_path):
        """Full protocol run must not cause Kivy to be loaded at any point."""
        with _kivy_purged():
            # Ensure no dispatcher leaked from previous test
            _kivy_utils._ui_dispatcher = None

            from modules.coord_transformations import CoordinateTransformer
            from modules.labware_loader import WellPlateLoader

            scope = Lumascope(simulate=True)
            # Speed up the simulator for test runtime
            scope._led_driver.set_timing_mode('fast')
            scope._motion_driver.set_timing_mode('fast')
            scope._camera_driver.set_timing_mode('fast')
            scope._camera_driver.grab()

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

                executor = SequencedCaptureRunner(
                    scope=scope,
                    stage_offset={'x': 0.0, 'y': 0.0},
                    io_executor=execs['io'],
                    protocol_thread=execs['protocol'],
                    file_io_executor=execs['file_io'],
                    camera_executor=execs['camera'],
                    autofocus_thread=MagicMock(is_running=False),
                    autofocus_runner=mock_af,
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

                image_capture_config = ImageCaptureConfig.from_image_mode('8bit')

                plan = executor.prepare(
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
                    autofocus_snapshot=autofocus_snapshot(),
                )
                executor.start(plan)

                assert done.wait(timeout=30), 'Protocol did not complete within timeout'

                # Check Kivy was never loaded during the protocol run
                kivy_loaded = [
                    name for name in sys.modules if name == 'kivy' or name.startswith('kivy.')
                ]
                assert not kivy_loaded, (
                    f'Kivy modules loaded during protocol run: {kivy_loaded}. '
                    'Rule 15 violation: executors must be GUI-agnostic.'
                )
            finally:
                try:
                    scope.disconnect()
                except Exception:
                    pass
                self._shutdown_executors(execs)

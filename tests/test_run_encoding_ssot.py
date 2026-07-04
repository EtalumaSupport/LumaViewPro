# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run-encoding single source of truth.

A run's capture depth and save encoding travel as ONE immutable
ImageCaptureConfig from prepare() to every save inside the run. Before
this contract, SequencedCaptureRunner.start() re-read save_encoding from
ctx.settings (or hardcoded 8-bit when ctx was None), so a headless or
API run configured for a 12-bit mode silently saved 8-bit / right-aligned
(dark) stills -- silent data loss. These tests pin the contract:

1. Headless (ctx is None): the run's stills honor the run config's mode.
2. One run, one encoding: stills and video legs read the same held config;
   ctx.settings has no say.
3. GUI-equivalent source: a live ctx.settings image_mode never overrides
   the run's prepared config.
4. Illegal states are unrepresentable on ImageCaptureConfig itself.
5. AST guard: the run-pipeline modules contain no settings/UI config
   re-read to regress back to.
"""

import ast
import dataclasses
import pathlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.exceptions import ConfigError
from modules.image_mode import (
    ImageCaptureConfig,
    SAVE_ENCODING_MSB_ALIGNED,
    SAVE_ENCODING_RGB,
    SAVE_ENCODING_RIGHT_ALIGNED,
)
from modules.lumascope_api import Lumascope
from modules.protocol import Protocol
from modules.protocol_callbacks import ProtocolCallbacks
from modules.sequenced_capture_runner import (
    SequencedCaptureRunMode,
    SequencedCaptureRunner,
)
from modules.sequential_io_executor import SequentialIOExecutor


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TILING_CONFIGS = REPO_ROOT / 'data' / 'tiling.json'

COMPLETION_TIMEOUT = 20  # seconds
SAVE_FLUSH_TIMEOUT = 5  # seconds to wait for the file-IO thread's save


# ---------------------------------------------------------------------------
# Full-stack harness (mirrors tests/test_protocol_roundtrip.py)
# ---------------------------------------------------------------------------


def _make_step(color='BF'):
    return {
        'Name': 'A1_BF',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': False,
        'Color': color,
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
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
        'Auto_Named': True,
        'Label': '',
    }


def _build_protocol():
    import datetime

    import pandas as pd

    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_make_step()]),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s.imaging.start_streaming()
    yield s
    s.imaging.stop_streaming()
    s.disconnect()


@pytest.fixture
def executors():
    from modules.protocol_thread import ProtocolThread

    execs = {
        'io': SequentialIOExecutor(name='SSOT_IO'),
        'file_io': SequentialIOExecutor(name='SSOT_FILE'),
        'camera': SequentialIOExecutor(name='SSOT_CAMERA'),
        'autofocus': SequentialIOExecutor(name='SSOT_AF'),
    }
    for e in execs.values():
        e.start()
    pt = ProtocolThread()
    pt.start()
    execs['protocol'] = pt
    yield execs
    for name, e in execs.items():
        try:
            if name == 'protocol':
                e.stop(timeout=2.0)
            else:
                e.shutdown()
        except Exception:
            pass


@pytest.fixture
def executor(scope, executors):
    mock_af = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.run_in_progress = MagicMock(return_value=False)

    exc = SequencedCaptureRunner(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=executors['io'],
        protocol_thread=executors['protocol'],
        file_io_executor=executors['file_io'],
        camera_executor=executors['camera'],
        autofocus_thread=MagicMock(is_running=False),
        autofocus_runner=mock_af,
    )
    mock_transformer = MagicMock()
    mock_transformer.plate_to_stage = MagicMock(return_value=(0.0, 0.0))
    exc._wellplate_loader = MagicMock()
    exc._coordinate_transformer = mock_transformer
    return exc


def _spy_save_image(monkeypatch, tmp_path):
    """Record every save_image call the protocol writer makes."""
    recorded = []

    def _record(scope, **kwargs):
        recorded.append(kwargs)
        return tmp_path / 'out.tiff'

    monkeypatch.setattr('modules.protocol_image_writer.save_image', _record)
    return recorded


def _run_one_still(executor, tmp_path, config):
    done = threading.Event()

    def on_complete(**kwargs):
        done.set()

    plan = executor.prepare(
        protocol=_build_protocol(),
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='encoding_ssot',
        image_capture_config=config,
        autogain_settings={'target_brightness': 0.3},
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={
            'run_complete': on_complete,
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        },
        leds_state_at_end='off',
        initial_autofocus_states={
            'BF': False,
            'PC': False,
            'DF': False,
            'Red': False,
            'Green': False,
            'Blue': False,
            'Lumi': False,
        },
    )
    executor.start(plan)
    assert done.wait(timeout=COMPLETION_TIMEOUT), 'run did not complete'


def _wait_for_saves(recorded):
    """The save runs on the file-IO thread; poll briefly for it to land."""
    deadline = time.monotonic() + SAVE_FLUSH_TIMEOUT
    while not recorded and time.monotonic() < deadline:
        time.sleep(0.05)
    assert recorded, 'the run must reach save_image for its one still'
    return recorded


# ---------------------------------------------------------------------------
# 1. Headless still honors the run mode (ctx is None)
# ---------------------------------------------------------------------------


class TestHeadlessStillHonorsRunMode:
    """With no app context, the encoding saved is the run config's encoding.

    The old code hardcoded 8-bit at start() when ctx was None, so a headless
    12-bit run silently saved 8-bit stills.
    """

    @pytest.mark.parametrize(
        'mode, expected_encoding',
        [
            ('12bit_false_color_rgb', SAVE_ENCODING_RGB),
            ('12bit_scaled', SAVE_ENCODING_MSB_ALIGNED),
        ],
    )
    def test_headless_still_saves_run_config_encoding(
        self, executor, tmp_path, monkeypatch, mode, expected_encoding
    ):
        monkeypatch.setattr('modules.app_context.ctx', None)
        recorded = _spy_save_image(monkeypatch, tmp_path)

        _run_one_still(executor, tmp_path, ImageCaptureConfig.from_image_mode(mode))

        saves = _wait_for_saves(recorded)
        assert saves[0]['save_encoding'] == expected_encoding, (
            f'a headless {mode} run must save {expected_encoding}, got {saves[0]["save_encoding"]}'
        )


# ---------------------------------------------------------------------------
# 2. One run, one encoding (still and video legs read the same held config)
# ---------------------------------------------------------------------------


class TestOneRunOneEncoding:
    """Both write_capture legs read the writer's held config -- never
    ctx.settings, which here disagrees on purpose."""

    @staticmethod
    def _writer(config):
        from modules.protocol_image_writer import ProtocolImageWriter

        return ProtocolImageWriter(
            scope=MagicMock(),
            callbacks=ProtocolCallbacks(),
            aborted=threading.Event(),
            file_io_executor=MagicMock(),
            abort_fn=lambda: None,
            execution_record=None,
            leds_off_fn=lambda: None,
            is_run_in_progress_fn=lambda: True,
            image_capture_config=config,
        )

    def test_still_and_video_legs_read_the_same_held_config(self, monkeypatch, tmp_path):
        config = ImageCaptureConfig.from_image_mode('12bit_scaled')
        writer = self._writer(config)

        # A live settings source that says 8-bit must have no say.
        monkeypatch.setattr(
            'modules.app_context.ctx',
            MagicMock(settings={'image_mode': '8bit'}, settings_lock=threading.Lock()),
        )

        still_saves = _spy_save_image(monkeypatch, tmp_path)
        video_writes = {}
        monkeypatch.setattr(
            'modules.protocol_image_writer.write_video',
            lambda **kwargs: video_writes.update(kwargs) or (tmp_path / 'vid'),
        )

        step = {'Name': 'A1_BF', 'Color': 'BF', 'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        writer.write_capture(
            enable_image_saving=True,
            is_video=False,
            captured_image=np.zeros((4, 4), dtype=np.uint16),
            step=step,
            name='A1_BF',
            save_folder=str(tmp_path),
            use_color='BF',
            output_format='TIFF',
        )
        writer.write_capture(
            enable_image_saving=True,
            is_video=True,
            video_as_frames=True,
            video_result=SimpleNamespace(captured_frames=1, duration_sec=1.0),
            step=step,
            name='A1_BF_vid',
            save_folder=tmp_path,
        )

        assert still_saves, 'the still leg must reach save_image'
        assert video_writes, 'the video leg must reach write_video'
        assert (
            still_saves[0]['save_encoding']
            == video_writes['save_encoding']
            == config.save_encoding
            == SAVE_ENCODING_MSB_ALIGNED
        ), 'one run has one encoding: both legs read the writer-held run config'


# ---------------------------------------------------------------------------
# 3. GUI-equivalent source: live settings must not override the run config
# ---------------------------------------------------------------------------


class TestSettingsDoNotOverrideRunConfig:
    def test_run_config_wins_over_live_settings_image_mode(self, executor, tmp_path, monkeypatch):
        """ctx present and its settings say 8bit; the run was prepared for
        12bit_scientific. The still must save right_aligned -- the old start()
        re-read settings and let them win."""
        ctx = MagicMock(
            settings={'image_mode': '8bit', 'protocol': {}},
            settings_lock=threading.Lock(),
            # Real repo root so components that resolve data files from the
            # app context (e.g. the protocol's objective loader) keep working.
            source_path=str(REPO_ROOT),
        )
        monkeypatch.setattr('modules.app_context.ctx', ctx)
        recorded = _spy_save_image(monkeypatch, tmp_path)

        _run_one_still(executor, tmp_path, ImageCaptureConfig.from_image_mode('12bit_scientific'))

        saves = _wait_for_saves(recorded)
        assert saves[0]['save_encoding'] == SAVE_ENCODING_RIGHT_ALIGNED, (
            'the prepared run config must win over live ctx.settings; '
            f'got {saves[0]["save_encoding"]}'
        )


# ---------------------------------------------------------------------------
# 4. Illegal states are unrepresentable
# ---------------------------------------------------------------------------


class TestIllegalConfigStatesUnrepresentable:
    def test_unknown_mode_is_refused(self):
        with pytest.raises(ConfigError):
            ImageCaptureConfig.from_image_mode('not_a_mode')

    def test_frozen_instance_rejects_assignment(self):
        config = ImageCaptureConfig.from_image_mode('8bit')
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.save_encoding = 'rgb'

    def test_direct_init_with_mismatched_pair_is_refused(self):
        # capture 12 / save 8bit under an 8bit mode is exactly the split
        # this type exists to make unrepresentable.
        with pytest.raises(ConfigError):
            ImageCaptureConfig(
                image_mode='8bit',
                capture_depth=12,
                save_encoding='8bit',
            )


# ---------------------------------------------------------------------------
# 5. AST guard: no settings/UI config re-read inside the run pipeline
# ---------------------------------------------------------------------------


RUN_PIPELINE_MODULES = [
    'modules/sequenced_capture_runner.py',
    'modules/protocol_image_writer.py',
    'modules/protocol_step_runner.py',
    'modules/video_capture.py',
]

FORBIDDEN_CONFIG_SOURCES = {
    'get_image_capture_config_from_settings',
    'get_image_capture_config_from_ui',
}


def _referenced_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.mark.parametrize('module_path', RUN_PIPELINE_MODULES)
def test_run_pipeline_has_no_config_source_but_the_run_config(module_path):
    """Structural lock on the fix: the run pipeline must not re-derive its
    capture config from settings or the UI mid-run. A reintroduced call is
    exactly the old silent-encoding-swap defect."""
    tree = ast.parse((REPO_ROOT / module_path).read_text())
    forbidden = _referenced_names(tree) & FORBIDDEN_CONFIG_SOURCES
    assert not forbidden, (
        f'{module_path} references {sorted(forbidden)} -- the run pipeline '
        'must read the run ImageCaptureConfig, never settings/UI'
    )


def test_sequenced_capture_runner_has_no_8bit_fallback():
    """The ctx-is-None hardcoded 8-bit fallback must stay dead."""
    tree = ast.parse((REPO_ROOT / 'modules/sequenced_capture_runner.py').read_text())
    assert 'SAVE_ENCODING_8BIT' not in _referenced_names(tree), (
        'sequenced_capture_runner must not fall back to SAVE_ENCODING_8BIT; '
        'the run config is the only encoding source'
    )


# ---------------------------------------------------------------------------
# 6. No silent headless default: a config-less run is refused loudly
# ---------------------------------------------------------------------------


class TestNoSilentHeadlessDefault:
    """A headless run with no image_capture_config raises ConfigError naming
    image_mode BEFORE any executor starts or hardware moves. The old default
    silently resolved to 8-bit, quietly downgrading scripts that captured
    full depth on earlier releases."""

    def test_configless_run_raises_before_anything_starts(self, tmp_path):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        try:
            runner = session.create_protocol_runner()
            with pytest.raises(ConfigError, match='image_mode'):
                runner.run_protocol(_build_protocol(), parent_dir=str(tmp_path))
            assert not runner.is_running(), 'a refused config-less run must not be running'
            assert not runner._owned_resources_started, (
                'the raise must precede executor startup -- nothing was committed'
            )
        finally:
            session.shutdown_executors()

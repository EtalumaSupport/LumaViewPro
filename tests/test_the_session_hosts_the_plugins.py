# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session hosts the plugins, so every host gets the same plugins and the same runs.

The plugin registry used to live on the GUI's application context. The GUI
discovered the plugins, handed each the context, and passed the session two
callbacks back into it. The one thing that ran a post-processing plugin
after a protocol was the Run button's handler, so a protocol started over
REST, from a script or by a plugin never ran one, and it ran on the GUI's
thread inside the run's own delivery. Engineering mode split the same way:
the engineering plugin turned on the context's flag, never the session's.

Now a host asks the session for plugins, the session hands each plugin
itself, and a Full Protocol from any starter hands its folder to the
opted-in processors on the post-processing lane once its images and its
hyperstack build are done.
"""

from __future__ import annotations

import datetime
import threading
import types
from unittest.mock import patch

import pandas as pd
import pytest

import modules.image_mode as image_mode
import modules.stack_builder as stack_builder
from modules.plugins import PluginSpec, ProcessorResult
from modules.protocol import Protocol
from tests.test_a_late_write_records_its_frame import REPO_ROOT, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session, single_run_dir

WAIT_S = 60.0


class _EntryPoint:
    def __init__(self, module):
        self.name = module.spec.name
        self._module = module

    def load(self):
        return self._module


def _plugin(name, *, processor=None, auto_run=False, subscribes_to=(), on_register=None):
    """A plugin module that records what the host calls it with."""
    module = types.ModuleType(f'fake_plugin_{name}')
    module.spec = PluginSpec(
        name=name,
        version='0.1.0',
        requires_lvp_version='>=4.0.0',
        description=name,
        subscribes_to=subscribes_to,
        auto_run_on_protocol_complete=auto_run,
    )
    module.calls = []

    def register(ctx):
        module.calls.append(('register', ctx))
        if processor is not None:
            ctx.plugins.post_processing.register(module.spec, processor)
        if on_register is not None:
            on_register(ctx)

    def unregister(ctx):
        module.calls.append(('unregister', ctx))

    def on_settings_changed(ctx, settings):
        module.calls.append(('settings', ctx, settings))

    module.register = register
    module.unregister = unregister
    module.on_settings_changed = on_settings_changed
    return module


def _load(session, *modules):
    with patch('importlib.metadata.entry_points', return_value=[_EntryPoint(m) for m in modules]):
        session.load_plugins()


def _recording_processor():
    """A processor that records each call and the thread it ran on."""
    calls = []
    called = threading.Event()

    def processor(input_dir, manifest, output_dir):
        calls.append((input_dir, dict(manifest), threading.current_thread().name))
        called.set()
        return ProcessorResult(success=True, message='done')

    processor.calls = calls
    processor.called = called
    return processor


def _one_scan_protocol(name='P1'):
    """A Full Protocol whose duration ends it after its first scan."""
    return Protocol(
        tiling_configs_file_loc=REPO_ROOT / 'data' / 'tiling.json',
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_step(name, 0, x=20.0, gain=1.0)]),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(seconds=1.0),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _drain_lane(session):
    """Return once everything put on the post-processing lane before now has run."""
    from modules.sequential_io_executor import IOTask

    session.post_processing.lane.call(IOTask(action=lambda: None), 'test.drain', WAIT_S)


def _run_protocol(runner, tmp_path, *, sequenced_format='TIFF'):
    runner.session.update_settings('image_output_format.sequenced', sequenced_format)
    run = runner.run_protocol(
        protocol=_one_scan_protocol(),
        parent_dir=str(tmp_path / 'runs'),
        run_trigger_source='api_protocol',
    )
    files = run.wait_for_files(timeout_s=WAIT_S)
    assert files is not None, 'the run never finished its files'
    return run, files


class TestAFullProtocolRunsTheOptedInProcessors:
    def test_a_protocol_started_through_the_session_hands_its_folder_on_once(self, tmp_path):
        opted_in = _recording_processor()
        not_opted_in = _recording_processor()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(
                session,
                _plugin('opted_in', processor=opted_in, auto_run=True),
                _plugin('not_opted_in', processor=not_opted_in),
            )
            run, files = _run_protocol(runner, tmp_path)
            assert opted_in.called.wait(WAIT_S), 'the opted-in processor never ran'
            _drain_lane(session)

        assert files.outcome == 'written'
        assert len(opted_in.calls) == 1
        input_dir, manifest, thread_name = opted_in.calls[0]
        assert input_dir == str(run.run_dir)
        assert manifest['trigger_source'] == 'api_protocol'
        assert manifest['protocol_name'] == 'protocol'
        assert thread_name.startswith('POSTPROC'), thread_name
        assert not_opted_in.calls == []

    def test_back_to_back_protocols_each_hand_on_their_own_folder(self, tmp_path):
        opted_in = _recording_processor()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(session, _plugin('opted_in', processor=opted_in, auto_run=True))
            first, _files = _run_protocol(runner, tmp_path)
            second, _files = _run_protocol(runner, tmp_path)
            _drain_lane(session)
            for _ in range(int(WAIT_S / 0.05)):
                if len(opted_in.calls) == 2:
                    break
                threading.Event().wait(0.05)

        assert sorted(call[0] for call in opted_in.calls) == sorted(
            [str(first.run_dir), str(second.run_dir)]
        )

    def test_a_scan_hands_nothing_on(self, tmp_path):
        opted_in = _recording_processor()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(session, _plugin('opted_in', processor=opted_in, auto_run=True))
            scan = runner.run_single_scan(
                protocol=_one_scan_protocol(), parent_dir=str(tmp_path / 'runs')
            )
            assert scan.wait_for_files(timeout_s=WAIT_S) is not None
            # The positive barrier: a Full Protocol after the scan is handed
            # on, so a hand-off the scan owed would have come first.
            protocol, _files = _run_protocol(runner, tmp_path)
            assert opted_in.called.wait(WAIT_S), 'the protocol after the scan was not handed on'
            _drain_lane(session)

        assert [call[0] for call in opted_in.calls] == [str(protocol.run_dir)]

    def test_a_session_without_plugins_runs_its_protocol(self, tmp_path):
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            assert session.plugins is None
            _run, files = _run_protocol(runner, tmp_path)

        assert files.outcome == 'written'

    def test_a_processor_that_raises_is_reported_and_the_next_still_runs(self, tmp_path):
        def raises(*_args):
            raise RuntimeError('the processor broke')

        after = _recording_processor()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(
                session,
                _plugin('a_raises', processor=raises, auto_run=True),
                _plugin('b_after', processor=after, auto_run=True),
            )
            with patch.object(session.plugins, 'record_runtime_error') as recorded:
                _run_protocol(runner, tmp_path)
                assert after.called.wait(WAIT_S), 'the processor after the failure never ran'
                _drain_lane(session)

        assert [c.args[:2] for c in recorded.call_args_list] == [
            ('a_raises', 'auto_run_on_protocol_complete')
        ]

    def test_the_processor_runs_after_the_hyperstack_build(self, tmp_path, monkeypatch):
        build_ended = threading.Event()
        real_build = stack_builder.build_hyperstacks_for_run

        def _build(**kwargs):
            real_build(**kwargs)
            build_ended.set()

        monkeypatch.setattr(stack_builder, 'build_hyperstacks_for_run', _build)
        seen_build_ended = []

        def processor(*_args):
            seen_build_ended.append(build_ended.is_set())
            return ProcessorResult(success=True, message='done')

        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(session, _plugin('after_build', processor=processor, auto_run=True))
            _run_protocol(runner, tmp_path, sequenced_format=image_mode.OUTPUT_FORMAT_HYPERSTACK)
            _drain_lane(session)
            deadline = threading.Event()
            for _ in range(int(WAIT_S / 0.05)):
                if seen_build_ended:
                    break
                deadline.wait(0.05)

        assert seen_build_ended == [True]

    def test_the_run_does_not_wait_for_its_processors(self, tmp_path):
        release = threading.Event()
        started = threading.Event()

        def held(*_args):
            started.set()
            release.wait(WAIT_S)
            return ProcessorResult(success=True, message='done')

        try:
            with open_composite_session(headless_settings(tmp_path)) as (session, runner):
                _load(session, _plugin('held', processor=held, auto_run=True))
                _run_protocol(runner, tmp_path)
                assert started.wait(WAIT_S), 'the processor never started'
                # The run's files are done while its processor is still held,
                # and the scope takes the next run.
                second = runner.run_single_scan(
                    protocol=_one_scan_protocol('P2'), parent_dir=str(tmp_path / 'runs')
                )
                assert second.wait_for_files(timeout_s=WAIT_S) is not None
                release.set()
                _drain_lane(session)
        finally:
            release.set()


class TestOnlyALoadedPluginIsHandedTheFolder:
    def _registry_with(self, *modules):
        from tests.plugin_test_harness import _make_ctx

        host = _make_ctx()
        with patch(
            'importlib.metadata.entry_points', return_value=[_EntryPoint(m) for m in modules]
        ):
            host.plugins.load(host, '4.0.0')
        return host

    def test_an_unloaded_plugins_processor_is_not_run(self):
        processor = _recording_processor()
        host = self._registry_with(_plugin('unloaded', processor=processor, auto_run=True))
        # The positive control: loaded, it is handed the folder.
        host.plugins.run_protocol_complete_processors('/run/1', {}, '/run/1', 'written')
        assert len(processor.calls) == 1

        host.plugins.unload(host)
        host.plugins.run_protocol_complete_processors('/run/2', {}, '/run/2', 'written')

        assert [call[0] for call in processor.calls] == ['/run/1']

    def test_a_plugin_whose_register_failed_is_not_run(self):
        processor = _recording_processor()

        def fail_after_registering(ctx):
            raise RuntimeError('register broke after the processor was in')

        host = self._registry_with(
            _plugin('half', processor=processor, auto_run=True, on_register=fail_after_registering)
        )
        assert 'half' in [p.name for p in host.plugins.not_loaded()]

        host.plugins.run_protocol_complete_processors('/run/1', {}, '/run/1', 'written')

        assert processor.calls == []

    def test_a_run_whose_images_did_not_all_save_is_handed_on_as_incomplete(
        self, tmp_path, monkeypatch
    ):
        import modules.protocol_image_writer as protocol_image_writer

        def _fail(scope, **kwargs):
            raise OSError('the save drive went away')

        monkeypatch.setattr(protocol_image_writer, 'save_image', _fail)
        processor = _recording_processor()
        handed = []
        handed_on = threading.Event()
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            _load(session, _plugin('opted_in', processor=processor, auto_run=True))
            real = session.plugins.run_protocol_complete_processors

            def _spy(input_dir, manifest, output_dir, files):
                handed.append(files)
                real(input_dir, manifest, output_dir, files)
                handed_on.set()

            monkeypatch.setattr(session.plugins, 'run_protocol_complete_processors', _spy)
            run = runner.run_protocol(
                protocol=_one_scan_protocol(), parent_dir=str(tmp_path / 'runs')
            )
            files = run.wait_for_files(timeout_s=WAIT_S)
            assert files is not None and files.outcome == 'incomplete'
            assert handed_on.wait(WAIT_S), 'the run was never handed on'

        assert handed == ['incomplete']
        assert processor.calls == []


class TestTheSessionIsThePluginsHost:
    def test_a_plugin_is_handed_the_session(self, tmp_path):
        plugin = _plugin('host')
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session, plugin)
            assert plugin.calls == [('register', session)]

    def test_a_setting_changed_before_the_first_save_is_told_at_it(self, tmp_path):
        plugin = _plugin('listener', subscribes_to=('image_output_format.sequenced',))
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session, plugin)
            session.update_settings('image_output_format.sequenced', 'OME-TIFF')
            session.save_settings(str(tmp_path / 'current.json'))

        told = [call for call in plugin.calls if call[0] == 'settings']
        assert len(told) == 1
        _kind, ctx, settings = told[0]
        assert ctx is session
        assert settings['image_output_format']['sequenced'] == 'OME-TIFF'

    def test_a_save_that_changes_nothing_subscribed_tells_nothing(self, tmp_path):
        plugin = _plugin('listener', subscribes_to=('image_output_format.sequenced',))
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session, plugin)
            session.save_settings(str(tmp_path / 'current.json'))

        assert [call for call in plugin.calls if call[0] == 'settings'] == []

    def test_health_is_the_registrys_once_plugins_load(self, tmp_path):
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            assert session.plugin_health() is None
            _load(session, _plugin('healthy'))
            loaded = [s.name for ns in session.plugin_health().namespaces for s in ns.loaded]
            assert 'stitcher' in loaded

    def test_unload_tells_each_plugin_once_and_shutdown_unloads(self, tmp_path):
        unloaded_by_host = _plugin('by_host')
        unloaded_by_shutdown = _plugin('by_shutdown')
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session, unloaded_by_host)
            session.unload_plugins()
            session.unload_plugins()
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session, unloaded_by_shutdown)

        assert [c[0] for c in unloaded_by_host.calls] == ['register', 'unregister']
        assert [c[0] for c in unloaded_by_shutdown.calls] == ['register', 'unregister']

    def test_a_built_in_that_fails_to_register_is_reported(self, tmp_path, monkeypatch):
        from modules.plugins.builtin import stitcher_plugin

        def _raises(ctx):
            raise RuntimeError('the built-in broke')

        monkeypatch.setattr(stitcher_plugin, 'register', _raises)
        with open_composite_session(headless_settings(tmp_path)) as (session, _runner):
            _load(session)
            assert 'stitcher' in [p.name for p in session.plugins.not_loaded()]


class TestOneEngineeringMode:
    def test_a_plugin_that_turns_engineering_mode_on_turns_it_on_for_the_session(self, tmp_path):
        def turn_on(session):
            session.engineering_mode = True

        settings = headless_settings(tmp_path)
        # Where a still is saved: the one key a still reads that a run does not.
        settings['separate_folder_per_channel'] = False
        with open_composite_session(settings) as (session, runner):
            assert session.engineering_mode is False
            _load(session, _plugin('engineering', on_register=turn_on))
            runner.run_composite(sequence_name='eng', parent_dir=str(tmp_path / 'runs'))
            token = f'Turret{int(session.scope.motion.get_current_position("T"))}'
            still = session.manual_capture.capture(layer='BF', false_color_on=False).result(
                timeout=WAIT_S
            )

        frames = sorted(p.name for p in single_run_dir(tmp_path / 'runs').glob('*.tiff'))
        assert frames, 'the run wrote no frames'
        for name in frames:
            assert token in name, f'{name} carries no turret position; expected {token}'
        assert token in still[0].name, f'{still[0].name} carries no turret position'

    def test_the_application_context_holds_no_engineering_flag(self):
        from modules.app_context import AppContext

        with pytest.raises(TypeError):
            AppContext(engineering_mode=True)

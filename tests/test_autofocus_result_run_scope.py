# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The autofocus result belongs to the run that produced it (#816).

``AutofocusRunner.best_focus_position()`` answers one question: what did
THIS run's autofocus find? A float when it found something, None when it
did not.

Bug history
-----------
The attribute was reset only at ``AutofocusRunner.run()`` entry, and
``AutofocusRunner.reset()`` had no production caller. So a value stayed
valid from one autofocus's start to the NEXT autofocus's start -- not for
the sequenced-capture run, the well, the layer or the session. Two
consumers read it across that gap:

* ``ui/vertical_control.py`` did not read it at all. It sampled
  ``get_current_position('Z')`` while the non-blocking pre-autofocus
  restore was still in flight and committed that in-transit coordinate to
  ``settings[layer]['focus']``, which persists to ``current.json``. The
  bench trace on #816 shows the stored value equal to the in-transit
  coordinate to the last digit.
* ``modules/protocol_step_runner.py`` gates "skip this channel's
  autofocus and use the brightfield result" on the value being non-None,
  so a PRIOR RUN's value opened the gate and was written into the step Z.

The run now clears the result before producing one, and the GUI asks the
autofocus what it found instead of asking the stage where it is.

Known open, deliberately
------------------------
Scoping the result to the RUN does not scope it to the POSITION. Within
one run with ``bf_af_for_fluorescence`` on, well A1's brightfield result
still suppresses well B1's autofocus. That is a larger change than the
one this file covers; ``test_cross_position_reuse_is_still_open``
locks it as known-open so closing it cannot pass unnoticed.
"""

from __future__ import annotations

import ast
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it.
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {'BF': {'autofocus': False}, 'Green': {'autofocus': False}}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from tests.af_drives import af_runner_and_scope, drive_af
from tests.ast_seams import parse_module
from tests.scope_fakes import spec_scope
from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

STALE_Z = 4321.0  # a prior run's result; no drive ever produces this value


def _runner_holding_a_stale_result():
    """A sequenced-capture runner whose autofocus already carries a
    result from an earlier run, as the real object would."""
    af_runner, _scope = af_runner_and_scope()
    af_runner._best_focus_position = STALE_Z
    return bare_capture_runner(autofocus_runner=af_runner), af_runner


# ---------------------------------------------------------------------------
# The run clears the result it is about to produce
# ---------------------------------------------------------------------------


class TestRunStartScopesTheResultToTheRun:
    def test_start_clears_a_prior_runs_result(self):
        """with a value left from run N, best_focus_position() is
        None at the start of run N+1."""
        runner, af_runner = _runner_holding_a_stale_result()
        assert af_runner.best_focus_position() == STALE_Z, 'precondition: the stale value is there'

        runner.start(runner.prepare(**scr_run_kwargs()))

        assert af_runner.best_focus_position() is None, (
            'start() must clear the autofocus result before the run that '
            f'will produce one; got {af_runner.best_focus_position()}'
        )

    def test_a_run_that_fails_at_start_reports_with_no_result(self, monkeypatch):
        """the clear sits BEFORE the failure window, so a run that
        fails at start dispatches its completion callbacks with the
        result already None.

        Ordering is the whole point: a clear placed after the failure
        window would leave the stale value in place on exactly the path
        where no autofocus ever runs to overwrite it.
        """
        runner, af_runner = _runner_holding_a_stale_result()

        def _boom():
            raise OSError('run dir init failed')

        monkeypatch.setattr(runner, '_setup_run_dir', _boom)

        seen = []
        runner.start(
            runner.prepare(
                **scr_run_kwargs(
                    callbacks={
                        'run_complete': lambda **kw: seen.append(
                            (kw.get('status'), af_runner.best_focus_position())
                        )
                    }
                )
            )
        )

        assert len(seen) == 1, f'run_complete must fire exactly once; got {seen}'
        status, result_at_completion = seen[0]
        assert status == 'failed_at_start', f'expected a failed-at-start run; got {status}'
        assert result_at_completion is None, (
            'a run that fails at start must reach its completion callbacks '
            f'with the autofocus result already cleared; got {result_at_completion}'
        )

    def test_a_prior_runs_result_cannot_open_the_bf_reuse_gate(self):
        """Narrowed: the brightfield-reuse gate in the step runner
        is `best_focus_position() is not None`. After a new run starts,
        a prior run's value can no longer satisfy it.

        The gate expression itself is evaluated here rather than
        restated, so a change to what the gate reads fails this test.
        """
        runner, af_runner = _runner_holding_a_stale_result()
        assert af_runner.best_focus_position() is not None, (
            'precondition: before the fix the stale value opens the gate'
        )

        runner.start(runner.prepare(**scr_run_kwargs()))

        assert af_runner.best_focus_position() is None, (
            "a fluorescence step must not adopt a prior RUN's brightfield result"
        )

    def test_cross_position_reuse_is_still_open(self):
        """The other half: scoping to the RUN is not scoping to the
        POSITION, and K16 is NOT closed.

        The gate reads the result with no check of which position
        measured it, so within one run well A1's brightfield focus still
        suppresses well B1's autofocus. Locked as known-open: when the
        result learns to carry its position, this test goes red and
        whoever closes K16 updates it deliberately.
        """
        tree = parse_module('modules/protocol_step_runner.py')
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'best_focus_position'
        ]
        assert reads, 'the brightfield-reuse gate no longer reads best_focus_position at all'
        assert all(not call.args and not call.keywords for call in reads), (
            'best_focus_position() now takes an argument -- K16 (the result '
            'carrying the position it was measured at) looks closed. Re-read '
            'the gate at protocol_step_runner.py and update this lock.'
        )


class TestClearResultIsUnconditional:
    def test_clear_result_does_not_skip_while_an_autofocus_is_in_flight(self):
        """The clear must NOT carry reset()'s in-flight guard.

        A prior autofocus can still be unwinding when the next run starts
        (protocol_cleanup proceeds after a 5 s timeout), and that is
        precisely the case where a stale value is most likely to be read.
        A guard would make the clear a no-op exactly there.

        Safe without the guard because run() only WRITES the result; it
        never reads it, which is why clearing the result is not the
        _params wipe reset() refuses to do mid-run.
        """
        af_runner, _scope = af_runner_and_scope()
        af_runner._best_focus_position = STALE_Z
        af_runner._af_in_progress.set()

        af_runner.clear_result()

        assert af_runner.best_focus_position() is None, (
            'clear_result() must be unconditional; an in-flight guard would '
            'skip it on the one path where the stale value gets read'
        )

    def test_clear_result_leaves_the_scan_params_alone(self):
        """The narrow clear is not AFE.reset(): _params stays, because
        run() reads it on the AF thread."""
        af_runner, _scope = af_runner_and_scope()
        drive_af(af_runner)
        params_before = dict(af_runner._params)
        assert params_before, 'precondition: a drive populates _params'

        af_runner.clear_result()

        assert af_runner._params == params_before, (
            'clear_result() must touch only the result; wiping _params would '
            'race AFE.run() on the AF thread'
        )


# ---------------------------------------------------------------------------
# The GUI asks the autofocus, not the stage
# ---------------------------------------------------------------------------


def _vertical_control_stub(af_runner, layer='Green', stored_focus=1234.0):
    """The collaborators _autofocus_run_complete actually touches."""
    import modules.app_context as app_context

    layer_obj = MagicMock()
    settings = {layer: {'focus': stored_focus}}
    # A specced scope, not a bare MagicMock: these tests assert the stage
    # is NOT consulted, and a double that answers any attribute at all
    # would make that assertion vacuous.
    ctx = SimpleNamespace(
        scope=spec_scope(),
        settings=settings,
        settings_lock=threading.Lock(),
        image_settings=MagicMock(),
        autofocus_runner=af_runner,
        autofocus_thread=MagicMock(),
    )
    ctx.image_settings.layer_lookup.return_value = layer_obj
    stub = SimpleNamespace(
        _unschedule_af_safety_timer=lambda: None,
        _reset_run_autofocus_button=lambda: None,
    )
    return stub, ctx, settings, layer_obj, app_context


def _call_af_run_complete(monkeypatch, stub, ctx, app_context, opened_layer=None):
    import ui.vertical_control as vc

    monkeypatch.setattr(app_context, 'ctx', ctx)
    monkeypatch.setattr(vc, 'live_histo_reverse', lambda: None)
    monkeypatch.setattr(
        vc.common_utils,
        'get_opened_layer',
        opened_layer if opened_layer is not None else (lambda image_settings: 'Green'),
    )
    vc.VerticalControl._autofocus_run_complete(stub, protocol=MagicMock(), status='completed')


class TestGuiStoresWhatTheAutofocusFound:
    def test_a_result_is_stored_exactly_as_found(self, monkeypatch):
        """Found: the stored focus is the autofocus's answer, not a
        stage sample taken while the pre-AF restore is in flight."""
        af_runner, _scope = af_runner_and_scope()
        af_runner._best_focus_position = 777.5
        stub, ctx, settings, _layer_obj, app_context = _vertical_control_stub(af_runner)
        # The stage is mid-restore and reads something else entirely --
        # the value the old code committed.
        ctx.scope.motion.get_current_position.return_value = 91.25

        _call_af_run_complete(monkeypatch, stub, ctx, app_context)

        assert settings['Green']['focus'] == 777.5, (
            'the layer focus must be the autofocus result, not the stage '
            f'position; got {settings["Green"]["focus"]}'
        )

    def test_no_result_leaves_the_stored_focus_untouched(self, monkeypatch):
        """Not found: an autofocus that found nothing writes
        nothing. The old code wrote wherever the stage happened to be."""
        af_runner, _scope = af_runner_and_scope()
        af_runner._best_focus_position = None
        stub, ctx, settings, _layer_obj, app_context = _vertical_control_stub(
            af_runner, stored_focus=1234.0
        )
        ctx.scope.motion.get_current_position.return_value = 91.25

        _call_af_run_complete(monkeypatch, stub, ctx, app_context)

        assert settings['Green']['focus'] == 1234.0, (
            'an autofocus with no result must leave the stored focus alone; '
            f'got {settings["Green"]["focus"]}'
        )

    def test_the_camera_widget_resync_runs_on_every_terminal_path(self, monkeypatch):
        """The widget re-sync is not about focus: autofocus restores the
        camera from committed settings on EVERY terminal path, so an
        uncommitted text edit must be re-pointed at the truth even when
        no focus was found."""
        af_runner, _scope = af_runner_and_scope()
        af_runner._best_focus_position = None
        stub, ctx, _settings, layer_obj, app_context = _vertical_control_stub(af_runner)
        ctx.scope.motion.get_current_position.return_value = 91.25

        _call_af_run_complete(monkeypatch, stub, ctx, app_context)

        assert layer_obj.sync_widgets_from_settings.called, (
            'the camera widget re-sync must run even with no autofocus result'
        )

    def test_the_defensive_af_thread_abort_runs_before_the_store_write(self, monkeypatch):
        """The store write no longer sits in a broad handler, so it can
        raise. The AF-thread unwind must already have happened.

        Its old handler swallowed every failure into a GUI log line, so a
        focus update that never happened looked identical to one that did.
        """
        af_runner, _scope = af_runner_and_scope()
        af_runner._best_focus_position = 777.5
        stub, ctx, _settings, _layer_obj, app_context = _vertical_control_stub(af_runner)

        def _boom(image_settings):
            raise RuntimeError('layer resolution exploded')

        with pytest.raises(RuntimeError, match='layer resolution exploded'):
            _call_af_run_complete(monkeypatch, stub, ctx, app_context, opened_layer=_boom)

        assert ctx.autofocus_thread.abort.called, (
            'the defensive AF-thread unwind must not be skippable by a '
            'failure in the focus update below it'
        )

    def test_the_completion_handler_never_samples_the_stage(self):
        """An AST lock: _autofocus_run_complete must not call
        get_current_position -- the whole defect was reading the stage
        instead of the result."""
        tree = parse_module('ui/vertical_control.py')
        handler = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_autofocus_run_complete'
            ),
            None,
        )
        assert handler is not None, 'ui/vertical_control.py: _autofocus_run_complete is gone'
        called = {
            node.func.attr
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert 'get_current_position' not in called, (
            '_autofocus_run_complete samples the stage again. The pre-AF '
            'restore is non-blocking, so the sample is an in-transit '
            'coordinate; read best_focus_position() instead (#816).'
        )


# ---------------------------------------------------------------------------
# The protocol Z copy-back is gated on the run finishing (#824)
# ---------------------------------------------------------------------------


def _protocol_settings_stub(stored_z, focused_z):
    """The collaborators _autofocus_run_complete_callback touches on the
    no-files-pending path."""
    import modules.app_context as app_context

    protocol = MagicMock()
    protocol.steps.return_value = {'Z': list(stored_z)}
    focused_protocol = MagicMock()
    focused_protocol.steps.return_value = {'Z': list(focused_z)}

    file_io_executor = MagicMock()
    file_io_executor.is_protocol_queue_active.return_value = False
    ctx = SimpleNamespace(file_io_executor=file_io_executor)

    stub = SimpleNamespace(
        _protocol=protocol,
        _scan_files_completed_event=threading.Event(),
        _reset_run_autofocus_scan_button=lambda: None,
    )
    return stub, ctx, protocol, focused_protocol, app_context


def _call_scan_complete(monkeypatch, stub, ctx, focused_protocol, app_context, status):
    import ui.protocol_settings as ps

    monkeypatch.setattr(app_context, 'ctx', ctx)
    monkeypatch.setattr(ps, 'live_histo_reverse', lambda: None)
    ps.ProtocolSettings._autofocus_run_complete_callback(
        stub, protocol=focused_protocol, status=status
    )


class TestProtocolZCopyBackIsGatedOnCompletion:
    def test_a_completed_run_copies_the_focused_z_back(self, monkeypatch):
        """Completed: the scan finished, so its Z column is the
        answer."""
        stub, ctx, protocol, focused, app_context = _protocol_settings_stub(
            stored_z=[10.0, 20.0], focused_z=[11.5, 21.5]
        )

        _call_scan_complete(monkeypatch, stub, ctx, focused, app_context, status='completed')

        assert protocol.steps()['Z'] == [11.5, 21.5], (
            f'a completed autofocus scan must copy its Z column back; got {protocol.steps()["Z"]}'
        )

    @pytest.mark.parametrize('status', ['aborted', 'failed', 'failed_at_start'])
    def test_an_unfinished_run_leaves_the_protocol_z_untouched(self, monkeypatch, status):
        """Not completed: a scan the user aborted, or one that
        failed, has a partial Z column. Copying it back overwrites the
        user's protocol with the steps that never ran (#824)."""
        stub, ctx, protocol, focused, app_context = _protocol_settings_stub(
            stored_z=[10.0, 20.0], focused_z=[11.5, 20.0]
        )

        _call_scan_complete(monkeypatch, stub, ctx, focused, app_context, status=status)

        assert protocol.steps()['Z'] == [10.0, 20.0], (
            f'a {status} scan must not overwrite the protocol Z column; got {protocol.steps()["Z"]}'
        )

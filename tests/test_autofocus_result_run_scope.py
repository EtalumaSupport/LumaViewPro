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

The run now clears the result before producing one, and reports what it
found in its outcome (``af_focus_z_um``). ``run_autofocus`` writes no focus
itself; the Autofocus button saves the outcome's Z, on ``run_ended``, as the
focus of the layer it started the run on, through
``ScopeSession.save_layer_focus``.

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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it.
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {'BF': {'autofocus': False}, 'Green': {'autofocus': False}}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.run_events import RunEvents
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
        fails at start sends its run_ended with the result already None.

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
                    events=RunEvents(
                        run_ended=lambda outcome, *_: seen.append(
                            (outcome.status, af_runner.best_focus_position())
                        )
                    )
                )
            )
        )

        assert len(seen) == 1, f'run_ended must fire exactly once; got {seen}'
        status, result_at_completion = seen[0]
        assert status == 'failed_at_start', f'expected a failed-at-start run; got {status}'
        assert result_at_completion is None, (
            'a run that fails at start must reach its run_ended '
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
# run_autofocus reports the focus; the Autofocus button saves it
# ---------------------------------------------------------------------------


def test_run_autofocus_reports_the_focus_and_writes_none(tmp_path):
    """The member a script or REST calls writes no layer focus: what it
    chose is in its outcome, for the caller to keep or not."""
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        stored = session.settings['BF']['focus']
        motion = session.scope.motion
        limits = motion.get_axis_limits('Z')
        # Mid-travel, so the sweep's window fits inside Z's travel.
        motion.move_absolute('Z', (limits['min'] + limits['max']) / 2.0)

        outcome = runner.run_autofocus(layer='BF').wait(timeout_s=60)

        assert outcome is not None and outcome.af_focus_z_um is not None, (
            f'precondition: the sweep chose a focus; got {outcome}'
        )
        assert outcome.af_focus_z_um != stored, 'precondition: the chosen Z differs from the stored'
        assert session.settings['BF']['focus'] == stored, (
            "run_autofocus wrote a layer focus; the caller keeps the outcome's Z or not"
        )


def _outcome(z):
    return SimpleNamespace(af_focus_z_um=z)


class _Session:
    def __init__(self):
        self.saved = []

    def save_layer_focus(self, layer, z_um):
        self.saved.append((layer, z_um))


@pytest.fixture
def gui(monkeypatch):
    """The Autofocus button over a recording session and a specced scope.

    The scope is specced, not a bare MagicMock: these tests assert the stage
    is NOT consulted, and a double that answers any attribute at all would
    make that assertion vacuous.
    """
    import modules.app_context as app_context
    import ui.vertical_control as vc

    layer_obj = MagicMock()
    ctx = SimpleNamespace(
        scope=spec_scope(),
        session=_Session(),
        image_settings=MagicMock(),
        engineering_mode=False,
    )
    ctx.image_settings.layer_lookup.return_value = layer_obj
    # The stage is mid-restore and reads something else entirely -- the
    # value the old code committed.
    ctx.scope.motion.get_current_position.return_value = 91.25
    monkeypatch.setattr(app_context, 'ctx', ctx)
    opened = {'layer': 'Green'}
    monkeypatch.setattr(vc.common_utils, 'get_opened_layer', lambda _settings: opened['layer'])
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    button = SimpleNamespace(
        _autofocus_run=None,
        _unschedule_af_safety_timer=lambda: None,
    )
    button._autofocus_run_ended = lambda layer, outcome: vc.VerticalControl._autofocus_run_ended(
        button, layer, outcome
    )
    return SimpleNamespace(
        vc=vc, ctx=ctx, layer_obj=layer_obj, opened=opened, button=button, saved=ctx.session.saved
    )


def _press(gui, monkeypatch):
    """Press Autofocus; return the events the button handed run_autofocus."""
    member = MagicMock()
    gui.ctx.session.create_protocol_runner = lambda: member
    gui.button._submit_autofocus_request = lambda call, stop=False: call()
    gui.vc.VerticalControl.run_autofocus_from_ui(gui.button)
    return member.run_autofocus.call_args.kwargs['events']


class TestTheButtonSavesTheFocusItsRunChose:
    def test_the_outcomes_z_is_saved_for_the_layer_the_run_started_on(self, gui, monkeypatch):
        """The drawer changed and the stage moved before the handler ran:
        the focus still goes to the layer the run focused, at the Z the
        run chose."""
        events = _press(gui, monkeypatch)
        gui.opened['layer'] = 'BF'

        events.run_ended(_outcome(777.5), None, MagicMock())

        assert gui.saved == [('Green', 777.5)]

    def test_an_outcome_with_no_focus_saves_nothing(self, gui, monkeypatch):
        """Not found: an autofocus that found nothing writes nothing. The
        old code wrote wherever the stage happened to be (#816)."""
        events = _press(gui, monkeypatch)

        events.run_ended(_outcome(None), None, MagicMock())

        assert gui.saved == []

    def test_the_camera_widget_resync_runs_on_every_terminal_path(self, gui):
        """The widget re-sync is not about focus: autofocus restores the
        camera from committed settings on EVERY terminal path, so an
        uncommitted text edit must be re-pointed at the truth even when
        no focus was found."""
        gui.vc.VerticalControl._autofocus_run_ended(gui.button, 'Green', _outcome(None))

        gui.layer_obj.sync_widgets_from_settings.assert_called_once()

    def test_a_refused_focus_is_reported_once_as_the_refusal(self, gui, monkeypatch):
        """The handler is a run_ended subscriber: a refusal of the Z is
        reported once by the run's delivery, as the refusal it is."""
        from modules import run_events
        from modules.exceptions import PositionOutOfRangeError
        from modules.notification_center import notifications

        refused = PositionOutOfRangeError('Z', 1e9, 0.0, 14000.0)

        def _refuse(layer, z_um):
            raise refused

        gui.ctx.session.save_layer_focus = _refuse
        monkeypatch.setattr(run_events, 'schedule_ui', lambda fn, *a: fn(0))
        reported = []
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda ex, **kw: reported.append((ex, kw))
        )
        events = _press(gui, monkeypatch)

        run_events.deliver(events.run_ended, 'run_ended', _outcome(1e9), None, MagicMock())

        assert reported == [(refused, {'solicited': False, 'category': 'run_ended'})]

    def test_the_completion_handler_never_samples_the_stage(self):
        """An AST lock: _autofocus_run_ended must not call
        get_current_position -- the whole defect was reading the stage
        instead of the result."""
        tree = parse_module('ui/vertical_control.py')
        handler = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_autofocus_run_ended'
            ),
            None,
        )
        assert handler is not None, 'ui/vertical_control.py: _autofocus_run_ended is gone'
        called = {
            node.func.attr
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert 'get_current_position' not in called, (
            '_autofocus_run_ended samples the stage again. Save the Z the '
            "run's outcome carries instead (#816)."
        )

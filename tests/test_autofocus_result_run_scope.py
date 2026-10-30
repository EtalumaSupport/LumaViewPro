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
* ``modules/protocol_step_runner.py`` gated "skip this channel's
  autofocus and use the brightfield result" on the value being non-None,
  so a PRIOR RUN's value opened the gate and was written into the step Z.
  That reuse has since been removed (#324).

The run now clears the result before producing one, and reports what it
found in its outcome (``af_focus_z_um``). ``start_autofocus`` writes no focus
itself; the Autofocus button saves the outcome's Z, on ``run_ended``, as the
focus of the layer it started the run on, through
``ScopeSession.save_channel_focus``.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


from modules.run_events import RunEvents
from tests.af_drives import af_runner_on, sweep
from tests.ast_seams import parse_module
from tests.protocol_drives import bare_capture_runner, scr_run_kwargs
from tests.stock_channels import LED1


@pytest.fixture
def runner_holding_a_stale_result(sim_scope):
    """A sequenced-capture runner whose autofocus carries the result of an
    earlier sweep, as the real object would after run N."""
    af_runner = af_runner_on(sim_scope)
    sweep(af_runner, sim_scope)
    return bare_capture_runner(autofocus_runner=af_runner), af_runner


# ---------------------------------------------------------------------------
# The run clears the result it is about to produce
# ---------------------------------------------------------------------------


class TestRunStartScopesTheResultToTheRun:
    def test_start_clears_a_prior_runs_result(self, runner_holding_a_stale_result):
        """with a value left from run N, best_focus_position() is
        None at the start of run N+1."""
        runner, af_runner = runner_holding_a_stale_result
        assert af_runner.best_focus_position() is not None, 'precondition: the stale value is there'

        runner.start(runner.prepare(**scr_run_kwargs()))

        assert af_runner.best_focus_position() is None, (
            'start() must clear the autofocus result before the run that '
            f'will produce one; got {af_runner.best_focus_position()}'
        )

    def test_a_run_that_fails_at_start_reports_with_no_result(
        self, runner_holding_a_stale_result, monkeypatch
    ):
        """the clear sits BEFORE the failure window, so a run that
        fails at start sends its run_ended with the result already None.

        Ordering is the whole point: a clear placed after the failure
        window would leave the stale value in place on exactly the path
        where no autofocus ever runs to overwrite it.
        """
        runner, af_runner = runner_holding_a_stale_result

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


class TestClearResultIsUnconditional:
    def test_clear_result_does_not_skip_while_an_autofocus_is_in_flight(
        self, sim_scope, monkeypatch
    ):
        """The clear must NOT carry reset()'s in-flight guard.

        A prior autofocus can still be unwinding when the next run starts
        (protocol_cleanup proceeds after a 5 s timeout), and that is
        precisely the case where a stale value is most likely to be read.
        A guard would make the clear a no-op exactly there.

        Safe without the guard because run() only WRITES the result; it
        never reads it, which is why clearing the result is not the
        _params wipe reset() refuses to do mid-run.
        """
        af_runner = af_runner_on(sim_scope)
        seen = []
        restore = sim_scope.camera.restore_camera_state

        def clear_while_unwinding(snapshot):
            # The sweep has its result and is restoring the camera: in flight.
            found = af_runner.best_focus_position()
            af_runner.clear_result()
            seen.append((af_runner.in_progress(), found, af_runner.best_focus_position()))
            return restore(snapshot)

        monkeypatch.setattr(sim_scope.camera, 'restore_camera_state', clear_while_unwinding)

        sweep(af_runner, sim_scope)

        ((in_flight, found, after_the_clear),) = seen
        assert in_flight and found is not None, 'precondition: a result, while in flight'
        assert after_the_clear is None, (
            'clear_result() must be unconditional; an in-flight guard would '
            'skip it on the one path where the stale value gets read'
        )

    def test_clear_result_leaves_the_scan_params_alone(self, sim_scope, monkeypatch):
        """The narrow clear is not AFE.reset(): the scan parameters stay,
        because run() reads them on the AF thread -- a sweep cleared
        mid-flight still finds its focus."""
        import modules.autofocus_functions as autofocus_functions

        from tests.af_drives import FOCAL_Z_UM

        af_runner = af_runner_on(sim_scope)
        score = autofocus_functions.focus_function

        def clear_mid_sweep(image):
            af_runner.clear_result()
            return score(image)

        monkeypatch.setattr('modules.autofocus_functions.focus_function', clear_mid_sweep)

        assert sweep(af_runner, sim_scope) == pytest.approx(FOCAL_Z_UM, abs=10.0), (
            'clear_result() must touch only the result; wiping the scan '
            'parameters would race AFE.run() on the AF thread'
        )


# ---------------------------------------------------------------------------
# start_autofocus reports the focus; the Autofocus button saves it
# ---------------------------------------------------------------------------


def test_start_autofocus_reports_the_focus_and_writes_none(tmp_path):
    """The member a script or REST calls writes no layer focus: what it
    chose is in its outcome, for the caller to keep or not."""
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        stored = session.settings['BF']['focus']
        motion = session.scope.motion
        limits = motion.get_axis_limits('Z')
        # Mid-travel, so the sweep's window fits inside Z's travel.
        mid = (limits.min_um + limits.max_um) / 2.0
        motion.move_z(mid)
        # A focus inside the sweep's window, off its start: a featureless
        # frame has no peak, and an autofocus over it fails.
        camera = session.scope._camera_driver
        camera.set_test_pattern(enabled=True, pattern='focus_target')
        camera.set_focal_z(mid + 30.0)

        outcome = runner.start_autofocus(channel='BF').result(timeout=60)

        assert outcome is not None and outcome.af_focus_z_um is not None, (
            f'precondition: the sweep chose a focus; got {outcome}'
        )
        assert outcome.af_focus_z_um != stored, 'precondition: the chosen Z differs from the stored'
        assert session.settings['BF']['focus'] == stored, (
            "start_autofocus wrote a layer focus; the caller keeps the outcome's Z or not"
        )


def _outcome(z):
    return SimpleNamespace(af_focus_z_um=z)


@pytest.fixture
def gui(gui_ctx, monkeypatch):
    """The Autofocus button over a simulated session, unhomed: its Z is not a
    focus any run chose, so a Z saved from the stage would show."""
    import ui.vertical_control as vc

    layer_obj = MagicMock()
    gui_ctx.image_settings = MagicMock()
    gui_ctx.image_settings.layer_lookup.return_value = layer_obj
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    button = SimpleNamespace(
        _autofocus_run=None,
    )
    button._autofocus_run_ended = lambda layer, outcome: vc.VerticalControl._autofocus_run_ended(
        button, layer, outcome
    )
    return SimpleNamespace(vc=vc, ctx=gui_ctx, layer_obj=layer_obj, button=button)


def _focus(gui, layer):
    return gui.ctx.session.settings[layer]['focus']


def _press(gui, monkeypatch):
    """Press Autofocus; return the events the button handed start_autofocus."""
    member = MagicMock()
    gui.ctx.session.create_protocol_runner = lambda: member
    gui.button._submit_autofocus_request = lambda call, stop=False: call()
    gui.vc.VerticalControl.run_autofocus_from_ui(gui.button)
    return member.start_autofocus.call_args.kwargs['events']


class TestTheButtonSavesTheFocusItsRunChose:
    def test_the_outcomes_z_is_saved_for_the_layer_the_run_started_on(self, gui, monkeypatch):
        """The shown channel changed before the handler ran: the focus still
        goes to the layer the run focused, at the Z the run chose."""
        started_on = gui.ctx.session.shown_channel.channel
        other = LED1 if started_on != LED1 else 'BF'
        kept = _focus(gui, other)
        events = _press(gui, monkeypatch)
        gui.ctx.session.show_channel(other)

        events.run_ended(_outcome(777.5), None, MagicMock())

        assert _focus(gui, started_on) == 777.5
        assert _focus(gui, other) == kept

    def test_an_outcome_with_no_focus_saves_nothing(self, gui, monkeypatch):
        """Not found: an autofocus that found nothing writes nothing. The
        old code wrote wherever the stage happened to be (#816)."""
        layer = gui.ctx.session.shown_channel.channel
        kept = _focus(gui, layer)
        events = _press(gui, monkeypatch)

        events.run_ended(_outcome(None), None, MagicMock())

        assert _focus(gui, layer) == kept

    def test_the_camera_widget_resync_runs_on_every_terminal_path(self, gui):
        """The widget re-sync is not about focus: autofocus restores the
        camera from committed settings on EVERY terminal path, so an
        uncommitted text edit must be re-pointed at the truth even when
        no focus was found."""
        gui.vc.VerticalControl._autofocus_run_ended(gui.button, LED1, _outcome(None))

        gui.layer_obj.sync_widgets_from_settings.assert_called_once()

    def test_a_refused_focus_is_reported_once_as_the_refusal(self, gui, monkeypatch):
        """The handler is a run_ended subscriber: a refusal of the Z is
        reported once by the run's delivery, as the refusal it is."""
        from modules import run_events
        from modules.exceptions import PositionOutOfRangeError
        from modules.notification_center import notifications

        # A Z past the travel: the session refuses to save it.
        monkeypatch.setattr(run_events, 'schedule_ui', lambda fn: fn(0))
        reported = []
        monkeypatch.setattr(
            notifications, 'report_outcome', lambda ex, **kw: reported.append((ex, kw))
        )
        events = _press(gui, monkeypatch)

        run_events.deliver(events, 'run_ended', _outcome(1e9), None, MagicMock())

        assert [(type(ex), kw) for ex, kw in reported] == [
            (PositionOutOfRangeError, {'solicited': False, 'category': 'run_ended'})
        ]

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

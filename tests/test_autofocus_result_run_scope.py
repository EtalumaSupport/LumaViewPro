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
* ``src/_lumascope/modules/protocol_step_runner.py`` gated "skip this channel's
  autofocus and use the brightfield result" on the value being non-None,
  so a PRIOR RUN's value opened the gate and was written into the step Z.
  That reuse has since been removed (#324).

The run now clears the result before producing one, and reports what it
found in its outcome (``af_focus_z_um``). ``start_autofocus`` saves that Z as
the focus of the channel it focused, through ``ScopeSession.save_channel_focus``,
before the run lets go of the scope; the Autofocus button only displays.
"""

from __future__ import annotations

import ast
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


from _lumascope.modules.run_events import RunEvents
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
        import _lumascope.modules.autofocus_functions as autofocus_functions

        from tests.af_drives import FOCAL_Z_UM

        af_runner = af_runner_on(sim_scope)
        score = autofocus_functions.autofocus_score

        def clear_mid_sweep(image):
            af_runner.clear_result()
            return score(image)

        monkeypatch.setattr(
            '_lumascope.modules.autofocus_functions.autofocus_score', clear_mid_sweep
        )

        assert sweep(af_runner, sim_scope) == pytest.approx(FOCAL_Z_UM, abs=10.0), (
            'clear_result() must touch only the result; wiping the scan '
            'parameters would race AFE.run() on the AF thread'
        )


# ---------------------------------------------------------------------------
# start_autofocus saves the focus it chose; the Autofocus button displays
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _session_over_a_focus_target(tmp_path, *, in_focus=True):
    """A headless session, Z mid-travel; with *in_focus*, a sample whose focus
    lies inside the sweep's window, off its start; without, a featureless
    frame, which has no peak, so the sweep chooses none."""
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        motion = session.scope.motion
        limits = motion.get_axis_limits('Z')
        # Mid-travel, so the sweep's window fits inside Z's travel.
        mid = (limits.min_um + limits.max_um) / 2.0
        motion.move_z(mid)
        if in_focus:
            camera = session.scope._camera_driver
            camera.set_test_pattern(enabled=True, pattern='focus_target')
            camera.set_focal_z(mid + 30.0)
        yield session, runner


class TestStartAutofocusSavesTheFocusItChose:
    def test_the_chosen_z_is_the_channels_focus_when_the_run_returns(self, tmp_path):
        """REST, the SDK and the button get one behaviour: the run saves the Z
        it chose as the focus of the channel it focused, before its result
        is returned, and says so."""
        with _session_over_a_focus_target(tmp_path) as (session, runner):
            stored = session.settings['BF']['focus']

            outcome = runner.start_autofocus(channel='BF').result(timeout=60)

            assert outcome.af_focus_z_um is not None, (
                f'precondition: the sweep chose a focus; got {outcome}'
            )
            assert outcome.af_focus_z_um != stored, 'precondition: the chosen Z differs'
            assert session.saved_focus_z_um('BF') == outcome.af_focus_z_um
            assert outcome.focus_written is True, outcome

    def test_a_run_that_chose_no_z_leaves_the_focus_as_it_was(self, tmp_path):
        """#816's abort path, kept: no Z chosen, no focus saved, and the
        outcome says the save it was asked for did not happen."""
        with _session_over_a_focus_target(tmp_path, in_focus=False) as (session, runner):
            stored = session.settings['BF']['focus']

            outcome = runner.start_autofocus(channel='BF').result(timeout=60)

            assert outcome.af_focus_z_um is None, f'precondition: no focus chosen; {outcome}'
            assert session.settings['BF']['focus'] == stored
            assert outcome.focus_written is False, outcome

    def test_a_stopped_run_saves_exactly_the_z_it_reports(self, tmp_path, monkeypatch):
        """Stopped mid-sweep, the run's saved focus and its reported Z are one
        read: the reported Z when it carries one, else the focus untouched."""
        import _lumascope.modules.autofocus_functions as autofocus_functions

        with _session_over_a_focus_target(tmp_path) as (session, runner):
            stored = session.settings['BF']['focus']
            score = autofocus_functions.autofocus_score
            started = []

            def stop_at_the_first_score(image):
                if started and not started[0].done():
                    started[0].stop()
                return score(image)

            monkeypatch.setattr(
                '_lumascope.modules.autofocus_functions.autofocus_score',
                stop_at_the_first_score,
            )
            started.append(runner.start_autofocus(channel='BF'))
            outcome = started[0].result(timeout=60)

            assert outcome.status == 'aborted', f'precondition: the stop ended it; {outcome}'
            expected = stored if outcome.af_focus_z_um is None else outcome.af_focus_z_um
            assert session.settings['BF']['focus'] == expected
            assert outcome.focus_written is (outcome.af_focus_z_um is not None), outcome

    def test_a_refused_save_is_reported_once_and_the_scope_is_released(self, tmp_path, monkeypatch):
        """A save the session refuses is told once, in its own words; the
        outcome says the focus was not saved, and the run still lets go of
        the scope, so the next run starts."""
        from _lumascope.modules.exceptions import PositionOutOfRangeError
        from tests.test_autofocus_all_steps_writes_the_focus import _reports_of

        with _session_over_a_focus_target(tmp_path) as (session, runner):
            stored = session.settings['BF']['focus']
            store = session._store_channel_focus
            # A stand-in by design: the simulator's sweep never chooses a Z
            # outside Z's travel, so the refusal is made by asking the real
            # store for one.
            monkeypatch.setattr(session, '_store_channel_focus', lambda ch, _z: store(ch, 1e9))

            with _reports_of(PositionOutOfRangeError) as reported:
                outcome = runner.start_autofocus(channel='BF').result(timeout=60)

            assert outcome.af_focus_z_um is not None, f'precondition: a Z was chosen; {outcome}'
            assert len(reported) == 1, reported
            assert outcome.focus_written is False, outcome
            assert session.settings['BF']['focus'] == stored
            assert runner.start_autofocus(channel='BF').result(timeout=60) is not None

    def test_a_run_under_a_diagnostics_claim_saves_nothing(self, tmp_path):
        """A run lent a diagnostic's claim acts inside that diagnostic, and
        what it keeps is the diagnostic's call: the channel's focus stays as
        it was, and the run was asked to save none."""
        with _session_over_a_focus_target(tmp_path) as (session, runner):
            stored = session.settings['BF']['focus']

            with session.diagnostic_claim() as claim:
                outcome = runner.start_autofocus(channel='BF', claim=claim).result(timeout=60)

            assert outcome.af_focus_z_um is not None, (
                f'precondition: the sweep chose a focus; got {outcome}'
            )
            assert session.settings['BF']['focus'] == stored
            assert outcome.focus_written is None, outcome


@pytest.fixture
def gui(gui_ctx, monkeypatch):
    """The Autofocus button's run-ended handler over a simulated session."""
    import ui.vertical_control as vc

    channel_control = MagicMock()
    gui_ctx.image_settings = MagicMock()
    gui_ctx.image_settings.channel_control_lookup.return_value = channel_control
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    button = SimpleNamespace(_autofocus_run=None)
    return SimpleNamespace(vc=vc, ctx=gui_ctx, channel_control=channel_control, button=button)


class TestTheButtonDisplaysTheRunsEnd:
    def test_the_camera_widget_resync_runs_on_every_terminal_path(self, gui):
        """The widget re-sync is not about focus: autofocus restores the
        camera from committed settings on EVERY terminal path, so an
        uncommitted text edit must be re-pointed at the truth even when
        no focus was found."""
        gui.vc.VerticalControl._autofocus_run_ended(gui.button, LED1)

        gui.channel_control.sync_widgets_from_settings.assert_called_once()

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
            '_autofocus_run_ended samples the stage again; the run saves the Z it chose (#816).'
        )
        assert 'save_channel_focus' not in called, (
            '_autofocus_run_ended saves a focus; the run already saved the Z it '
            'chose, for every client (start_autofocus).'
        )

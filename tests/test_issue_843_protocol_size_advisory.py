# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A protocol large enough to fill a disk should say so before it is run.

A 307,200-step protocol (384 wells x 4 channels x 200 slices) was created in
silence. Its size surfaced 2 minutes 36 seconds and two Scan presses later,
and only because it happened to exceed the disk. One that FITS is still
created with no comment at all.

The size was knowable the moment the protocol was constructed, and was
computed nowhere until a run started -- the whole-protocol estimate was an
expression inside one consumer. It is now a capability of a protocol, which
is what lets a second consumer ask without growing a twin.

Advisory only: nothing here refuses anything, and the run-start disk guard is
unchanged.

Reuses the Protocol builders from test_protocol_roundtrip.
"""

import pathlib
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
from modules.common_utils import (
    PROTOCOL_SIZE_ADVISORY_STEPS,
    estimate_step_write_mb,
    format_disk_size_mb,
)
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from ui.protocol_settings import ProtocolSettings

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _mixed_steps():
    """Image steps, two videos, and two malformed rows the estimator must survive."""
    return [
        _make_step(name='A1_BF'),
        _make_step(name='A2_BF'),
        _make_step(
            name='A3_VID',
            acquire='video',
            video_config={'duration': 1, 'fps': 30},
        ),
        _make_step(
            name='A4_VID',
            acquire='video',
            video_config={'duration': 600, 'fps': 30},
        ),
        # Malformed Video Config and a malformed frame value: the per-step
        # estimator sizes both to the single-image estimate rather than
        # raising, and the whole-protocol total inherits that contract.
        _make_step(name='A5_VID', acquire='video', video_config={'duration': 'nonsense'}),
        _make_step(name='A6_VID', acquire='video', video_config=None),
    ]


def _per_step_sum(protocol, *, video_as_frames, global_max_fps):
    return sum(
        estimate_step_write_mb(
            protocol.step(idx=i),
            video_as_frames=video_as_frames,
            global_max_fps=global_max_fps,
        )
        for i in range(protocol.num_steps())
    )


class TestTheWholeProtocolEstimateHasOneOwner:
    """T1: the run guard's total and a per-step sum must agree exactly.

    This is the real lock on moving the run loop onto the shared estimator.
    The grep it replaced could only see that a name was mentioned.
    """

    @pytest.mark.parametrize('video_as_frames', [False, True])
    @pytest.mark.parametrize('global_max_fps', [0, 10])
    def test_total_equals_the_sum_of_its_steps(self, video_as_frames, global_max_fps):
        protocol = _build_protocol(_mixed_steps())
        assert protocol.estimate_write_mb(
            video_as_frames=video_as_frames, global_max_fps=global_max_fps
        ) == _per_step_sum(protocol, video_as_frames=video_as_frames, global_max_fps=global_max_fps)


class TestTheThresholdBoundary:
    """T2: the advisory fires above the threshold, not at it."""

    @pytest.mark.parametrize(
        'n_steps,expect_advisory',
        [
            (PROTOCOL_SIZE_ADVISORY_STEPS - 1, False),
            (PROTOCOL_SIZE_ADVISORY_STEPS, False),
            (PROTOCOL_SIZE_ADVISORY_STEPS + 1, True),
        ],
    )
    def test_advisory_only_above_the_threshold(self, n_steps, expect_advisory):
        protocol = _build_protocol([_make_step(name=f'S{i}') for i in range(n_steps)])
        advisory = protocol.size_advisory(global_max_fps=0)
        assert (advisory is not None) is expect_advisory


class TestTheMessageNamesBothNumbers:
    """T3: the sentence carries the step count and the projected size."""

    def test_the_message_names_the_count_and_the_size(self):
        n_steps = PROTOCOL_SIZE_ADVISORY_STEPS + 1
        protocol = _build_protocol([_make_step(name=f'S{i}') for i in range(n_steps)])
        advisory = protocol.size_advisory(global_max_fps=0)

        assert advisory.num_steps == n_steps
        assert f'{n_steps:,}' in advisory.message
        assert format_disk_size_mb(advisory.projected_mb) in advisory.message

    def test_the_reported_disk_figure_is_binary(self):
        # The literal the issue's protocol produces. Pinned so the acceptance
        # criterion and the code cannot disagree: binary gives 2.34, decimal
        # gives 2.46, and only one of them is what the disk checks mean.
        assert format_disk_size_mb(2_457_600) == '2.3 TB'


class TestTheGuiRendersAndDecidesNothing:
    """T4 + T5: whether a protocol is large is the session's answer.

    Behavioural, not a source pin: the render method is called UNBOUND on a
    SimpleNamespace carrying only what it touches, so a mutation of any branch
    turns exactly the test for that branch red. Widgets cannot be instantiated
    under the stubbed Kivy base; the unbound call sidesteps that.
    """

    def _fake_self(self):
        return SimpleNamespace(
            ids={'protocol_size_advisory_label': SimpleNamespace(text='UNSET')},
            _protocol=object(),
            protocol_size_advisory_active=None,
        )

    def _fake_ctx(self, *, advisory, run_lockout=False, asked=None):
        def protocol_size_advisory(protocol):
            if asked is not None:
                asked.append(protocol)
            return advisory

        return SimpleNamespace(
            session=SimpleNamespace(
                run_lockout=run_lockout,
                protocol_size_advisory=protocol_size_advisory,
            )
        )

    def test_it_renders_exactly_what_the_session_returned(self, monkeypatch):
        # The sentence is the session's, verbatim -- the GUI does not compose,
        # reformat or threshold it. A deliberately odd message makes that
        # impossible to pass by accident.
        advisory = SimpleNamespace(num_steps=1, projected_mb=1.0, message='SESSION SENTENCE')
        fake = self._fake_self()
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=advisory))

        ProtocolSettings._update_protocol_size_advisory(fake)

        assert fake.ids['protocol_size_advisory_label'].text == 'SESSION SENTENCE'
        assert fake.protocol_size_advisory_active is True

    def test_no_advisory_clears_the_line_and_collapses_the_row(self, monkeypatch):
        fake = self._fake_self()
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=None))

        ProtocolSettings._update_protocol_size_advisory(fake)

        assert fake.ids['protocol_size_advisory_label'].text == ''
        assert fake.protocol_size_advisory_active is False

    def test_the_gui_holds_no_threshold_of_its_own(self, monkeypatch):
        # A protocol the GUI might think is large, with the session saying no:
        # nothing is shown. The decision is the session's, not a step count
        # compared here.
        fake = self._fake_self()
        fake._protocol = _build_protocol(
            [_make_step(name=f'S{i}') for i in range(PROTOCOL_SIZE_ADVISORY_STEPS + 1)]
        )
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=None))

        ProtocolSettings._update_protocol_size_advisory(fake)

        assert fake.ids['protocol_size_advisory_label'].text == ''
        assert fake.protocol_size_advisory_active is False

    def test_it_does_not_recompute_while_a_run_holds_the_lock(self, monkeypatch):
        # Without the early return the estimate runs at the step-navigation
        # refresh rate for the whole length of every run. The label must keep
        # its last text, which stays correct because a run cannot change the
        # protocol.
        asked = []
        fake = self._fake_self()
        fake.ids['protocol_size_advisory_label'].text = 'PREVIOUS TEXT'
        monkeypatch.setattr(
            _app_ctx, 'ctx', self._fake_ctx(advisory=None, run_lockout=True, asked=asked)
        )

        ProtocolSettings._update_protocol_size_advisory(fake)

        assert asked == [], 'the session must not be asked while a run holds the lock'
        assert fake.ids['protocol_size_advisory_label'].text == 'PREVIOUS TEXT'

    def test_a_missing_label_is_survivable(self, monkeypatch):
        # The method runs from the step-navigation refresh, which fires before
        # the kv ids are populated on some paths.
        fake = SimpleNamespace(ids={}, _protocol=object(), protocol_size_advisory_active=None)
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=None))

        ProtocolSettings._update_protocol_size_advisory(fake)

    def test_the_advisory_raises_no_popup(self, monkeypatch):
        # Eric's ruling was a non-blocking line, explicitly not a modal that
        # must be acknowledged. Any popup helper reached from this path fails.
        import ui.protocol_settings as ps_module

        def _explode(*args, **kwargs):
            raise AssertionError('the size advisory must not raise a popup')

        monkeypatch.setattr(ps_module, 'show_notification_popup', _explode, raising=False)
        advisory = SimpleNamespace(num_steps=1, projected_mb=1.0, message='x')
        fake = self._fake_self()
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=advisory))

        ProtocolSettings._update_protocol_size_advisory(fake)


class TestTheEstimateIsTotal:
    """T6: it runs inside the run loop's broad except, so it must never raise.

    A raise there does not surface -- it silently disables the free-space
    guard and the run proceeds onto a full disk.
    """

    def test_an_all_video_protocol_estimates(self):
        protocol = _build_protocol(
            [
                _make_step(name=f'V{i}', acquire='video', video_config={'duration': 5, 'fps': 30})
                for i in range(3)
            ]
        )
        assert protocol.estimate_write_mb(global_max_fps=0) > 0

    def test_an_empty_protocol_estimates_zero(self):
        protocol = _build_protocol([_make_step(name='A1_BF')])
        protocol.delete_step(0)
        assert protocol.estimate_write_mb(global_max_fps=0) == 0.0

    def test_an_execution_copy_estimates(self):
        # copy_for_execution's object is what the run loop actually holds, and
        # it is the shape a cached total broke: an __init__-only attribute is
        # absent there, raising into the swallowing except.
        protocol = _build_protocol(_mixed_steps())
        assert protocol.copy_for_execution().estimate_write_mb(
            global_max_fps=0
        ) == protocol.estimate_write_mb(global_max_fps=0)


class TestTheEstimateTracksAnInPlaceEdit:
    """T7: trivially true while the estimate is stateless -- which is the point.

    It fails the moment anyone adds a cache keyed on something an in-place
    mutator does not touch. The step count does not change here, so a cache
    keyed on num_steps would pass every other test in this file and fail this
    one. Driven by mutating the frame the way the in-place mutators do, rather
    than through modify_step, so no invalidation hook can mask a stale cache.
    """

    def test_flipping_a_step_to_video_moves_the_estimate(self):
        protocol = _build_protocol([_make_step(name='A1_BF'), _make_step(name='A2_BF')])
        before = protocol.estimate_write_mb(global_max_fps=0)

        steps = protocol.steps()
        steps.at[1, 'Acquire'] = 'video'
        steps.at[1, 'Video Config'] = {'duration': 600, 'fps': 30}

        after = protocol.estimate_write_mb(global_max_fps=0)
        assert after > before
        assert protocol.num_steps() == 2, 'the step count is unchanged; only the size moved'


class TestTheAbortMessageNamesTheRealStepCount:
    """T2b: the run-loop guard's refusal must name the count, never 0.

    The step count is bound inside the memoization branch and read outside it,
    in the abort message. Dropping that binding while moving the guard onto
    the shared estimator would make the refusal read "for 0 steps"; removing
    its initialiser instead makes it a NameError swallowed by the same broad
    except, so the abort never fires and the run proceeds onto a full disk.

    Locked as source rather than by driving the loop: the guard sits inside
    the scan loop and reaching it needs a full fake run. Named here so the
    substitution is visible -- this checks the binding survives, not that the
    rendered message is correct at runtime.
    """

    def _run_loop_source(self):
        # pin-justified: the count is interpolated into an f-string inside the
        # scan loop's body, which has no importable seam -- reaching it for
        # real needs a fully faked run.
        return (_REPO_ROOT / 'modules/protocol_run_loop.py').read_text()

    def test_the_message_interpolates_the_step_count(self):
        assert '{num_steps} steps' in self._run_loop_source()

    def test_the_step_count_is_assigned_before_the_message(self):
        source = self._run_loop_source()
        assigned = source.index('num_steps = p._protocol.num_steps()')
        message = source.index('{num_steps} steps')
        assert assigned < message

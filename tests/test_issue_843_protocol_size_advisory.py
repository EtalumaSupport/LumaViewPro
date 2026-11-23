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

Advisory only: nothing here refuses anything. A run whose whole size does not
fit its disk is refused at its start (``tests/test_write_size.py``); the
advisory gives one scan's size and a full run's.

Reuses the Protocol builders from test_protocol_roundtrip.
"""

from types import SimpleNamespace

import pytest

import _lumascope.modules.app_context as _app_ctx
from _lumascope.modules.common_utils import (
    PROTOCOL_SIZE_ADVISORY_STEPS,
    format_disk_size_mb,
)
from _lumascope.modules.image_mode import RecordingFormat
from _lumascope.modules.scope_session import ControlsLock
from _lumascope.modules.write_size import estimate_step_write_mb
from tests.protocol_drives import a_frame_sizing
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from ui.protocol_settings import ProtocolSettings


def _mixed_steps():
    """Image steps, two videos, and two malformed rows the estimator must survive."""
    return [
        _make_step(name='A1_BF'),
        _make_step(name='A2_BF'),
        _make_step(
            name='A3_VID',
            acquire='video',
            video_config={'duration': 1},
        ),
        _make_step(
            name='A4_VID',
            acquire='video',
            video_config={'duration': 600},
        ),
        # A malformed duration and no Video Config: the per-step estimator
        # sizes both at the MP4 floor, or one frame saved as frames, rather
        # than raising, and the whole-protocol total inherits that contract.
        _make_step(name='A5_VID', acquire='video', video_config={'duration': 'nonsense'}),
        _make_step(name='A6_VID', acquire='video', video_config=None),
    ]


# The rate a video step is sized at; any known rate, since the sum and the
# total are asked at the same one.
VIDEO_FPS = 40.0
# The run's frame sizing; any, since the sum and the total are asked at the
# same one.
SIZING = a_frame_sizing()


def _per_step_sum(protocol, *, recording_format, video_fps):
    return sum(
        estimate_step_write_mb(
            protocol.step_row(step_index=i),
            sizing=SIZING,
            recording_format=recording_format,
            video_fps=video_fps,
        )
        for i in range(protocol.num_steps)
    )


class TestTheWholeProtocolEstimateHasOneOwner:
    """T1: the run guard's total and a per-step sum must agree exactly.

    This is the real lock on moving the run loop onto the shared estimator.
    The grep it replaced could only see that a name was mentioned.
    """

    # A stack's second copy is the run's, not a step's, so the formats that
    # build none are the ones whose total is its steps' sum.
    @pytest.mark.parametrize('recording_format', [RecordingFormat.MP4, RecordingFormat.FRAMES])
    def test_total_equals_the_sum_of_its_steps(self, recording_format):
        protocol = _build_protocol(_mixed_steps())
        assert protocol.estimate_write_mb(
            sizing=SIZING, recording_format=recording_format, video_fps=VIDEO_FPS
        ) == _per_step_sum(protocol, recording_format=recording_format, video_fps=VIDEO_FPS)


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
        advisory = protocol.size_advisory(sizing=SIZING, video_fps=None)
        assert (advisory is not None) is expect_advisory


class TestTheMessageNamesBothNumbers:
    """T3: the sentence carries the step count, one scan's size and a full
    run's (M6: one scan was given as the protocol's)."""

    def test_the_message_names_the_count_a_scan_and_the_full_run(self):
        n_steps = PROTOCOL_SIZE_ADVISORY_STEPS + 1
        # 48 h at a 20 min period: 144 scans.
        protocol = _build_protocol(
            [_make_step(name=f'S{i}') for i in range(n_steps)], period_min=20, duration_hrs=48
        )
        advisory = protocol.size_advisory(sizing=SIZING, video_fps=None)

        assert advisory.num_steps == n_steps
        assert advisory.scan_mb == protocol.estimate_write_mb(sizing=SIZING, video_fps=None)
        assert advisory.run_mb == 144 * advisory.scan_mb
        assert f'{n_steps:,}' in advisory.message
        assert f'{format_disk_size_mb(advisory.scan_mb)} of images a scan' in advisory.message
        assert f'{format_disk_size_mb(advisory.run_mb)} for the full run of 144 scans' in (
            advisory.message
        )

    def test_a_video_step_with_no_rate_leaves_the_protocol_unsized(self):
        # No limit set and nothing delivered: the rate a video step records
        # at is not known. The advisory gives no figure rather than one that
        # leaves the video out or sizes it at a rate of 0.
        n_steps = PROTOCOL_SIZE_ADVISORY_STEPS + 2
        steps = [_make_step(name=f'S{i}') for i in range(n_steps - 2)]
        steps += [
            _make_step(name=f'V{i}', acquire='video', video_config={'duration': 600})
            for i in range(2)
        ]
        protocol = _build_protocol(steps)

        advisory = protocol.size_advisory(sizing=SIZING, video_fps=None)

        assert advisory.scan_mb is None and advisory.run_mb is None
        assert 'not sized while the camera is not delivering frames' in advisory.message

    def test_an_unknown_frame_size_leaves_the_protocol_unsized(self):
        n_steps = PROTOCOL_SIZE_ADVISORY_STEPS + 1
        protocol = _build_protocol([_make_step(name=f'S{i}') for i in range(n_steps)])

        advisory = protocol.size_advisory(sizing=None, video_fps=None)

        assert advisory.scan_mb is None and advisory.run_mb is None
        assert "not sized while the camera's frame size is not known" in advisory.message

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
                controls_lock=ControlsLock(
                    run_named='A protocol run is in progress.' if run_lockout else None,
                    control_named=None,
                    recording=False,
                ),
                protocol_size_advisory=protocol_size_advisory,
            )
        )

    def test_it_renders_exactly_what_the_session_returned(self, monkeypatch):
        # The sentence is the session's, verbatim -- the GUI does not compose,
        # reformat or threshold it. A deliberately odd message makes that
        # impossible to pass by accident.
        advisory = SimpleNamespace(num_steps=1, scan_mb=1.0, run_mb=1.0, message='SESSION SENTENCE')
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
        advisory = SimpleNamespace(num_steps=1, scan_mb=1.0, run_mb=1.0, message='x')
        fake = self._fake_self()
        monkeypatch.setattr(_app_ctx, 'ctx', self._fake_ctx(advisory=advisory))

        ProtocolSettings._update_protocol_size_advisory(fake)


class TestTheEstimateSizesEveryShape:
    """T6: every protocol shape a run can hold is sized, and a video step
    with no rate to size it at is refused rather than sized at none.

    The run's disk check sizes the protocol outside any catch, so a raise
    ends the run where it happens rather than switching the guard off.
    """

    def test_an_all_video_protocol_estimates(self):
        protocol = _build_protocol(
            [
                _make_step(name=f'V{i}', acquire='video', video_config={'duration': 5})
                for i in range(3)
            ]
        )
        assert protocol.estimate_write_mb(sizing=SIZING, video_fps=VIDEO_FPS) > 0

    def test_a_video_step_with_no_rate_is_refused_not_sized(self):
        protocol = _build_protocol(
            [_make_step(name='V0', acquire='video', video_config={'duration': 5})]
        )
        with pytest.raises(ValueError, match='without the rate it records at'):
            protocol.estimate_write_mb(sizing=SIZING, video_fps=None)

    def test_an_empty_protocol_estimates_zero(self):
        protocol = _build_protocol([_make_step(name='A1_BF')])
        protocol.delete_step(0)
        assert protocol.estimate_write_mb(sizing=SIZING, video_fps=None) == 0.0

    def test_an_execution_copy_estimates(self):
        # copy_for_execution's object is what the run loop actually holds, and
        # it is the shape a cached total broke: an __init__-only attribute is
        # absent there.
        protocol = _build_protocol(_mixed_steps())
        assert protocol.copy_for_execution().estimate_write_mb(
            sizing=SIZING, video_fps=VIDEO_FPS
        ) == protocol.estimate_write_mb(sizing=SIZING, video_fps=VIDEO_FPS)


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
        before = protocol.estimate_write_mb(sizing=SIZING, video_fps=VIDEO_FPS)

        # The stored frame, edited in place as an in-place writer edits it.
        steps = protocol._config['steps']
        steps.at[1, 'Acquire'] = 'video'
        steps.at[1, 'Video Config'] = {'duration': 600}

        after = protocol.estimate_write_mb(sizing=SIZING, video_fps=VIDEO_FPS)
        assert after > before
        assert protocol.num_steps == 2, 'the step count is unchanged; only the size moved'

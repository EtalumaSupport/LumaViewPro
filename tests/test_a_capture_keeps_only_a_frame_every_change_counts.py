# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture keeps only a frame the one judgment counts for every change.

The capture drains until each change can be met by its source's rule --
a credit-rule change by its frame count, a clock-rule change once the host's
clock passes the instant a frame must start after -- and then keeps the
first frame ``FrameValidity.judge`` counts for every change it does not
exclude, taking the next otherwise. The camera's clock is mapped once per
cycle, before its drain, and that mapping is what the drain, the judge and
the frame's record read. A grab whose bound passes with no frame counting
raises ``no_fresh_frame``, naming the change's source, and its words say
the change as a person says it.
"""

import dataclasses

import pytest

from drivers import host_clock
from drivers.exceptions import CameraReadFailedError
from modules.exceptions import CaptureFailedError
from modules.frame_validity import FrameRule, FrameValidity, Rule

AUTO_GAIN = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 100.0,
    'min_exposure_ms': 0.1,
}

_PROVENANCE = {
    'run_id': 'run-1',
    'host': 'bench-pc',
    'os': 'Windows 11',
    'lvp_version': '4.0.0',
    'plugin_version': '1.0.90',
    'measured_date': '2026-10-20',
}


def _clock_rule_for(scope, source: str, delay_us: float) -> None:
    """``source`` runs the clock rule at ``delay_us`` for the changes recorded from now on."""
    frame_validity = scope.imaging.frame_validity
    rules = dict(frame_validity.credit_rules())
    rules[source] = FrameRule(
        rule=Rule.CLOCK, count=rules[source].count, delay_us=delay_us, provenance=_PROVENANCE
    )
    frame_validity.use_rules(rules)


def _judgment(record, source: str):
    return next(j for j in record.judgments if j.change.source == source)


def test_the_clock_is_mapped_before_the_frame_the_capture_returns(sim_scope):
    """The record's mapping was latched after the grab, so the drain could not
    wait on it; it is now taken at the top of the cycle, before the drain and
    the grab, and the returned frame arrived after it."""
    assert sim_scope.capabilities.camera_has_clock_latch, 'precondition: the simulator latches'
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)

    record = sim_scope.imaging.capture_still(accept_dark=True).record
    gain = _judgment(record, 'gain')
    arrived_ns = gain.change.written_ns + gain.after_write_ms * 1e6
    assert record.clock_mapping.host_ns < arrived_ns


def test_a_latch_failing_on_a_later_cycle_leaves_the_arm_re_armed(sim_scope, monkeypatch):
    """A standing auto-gain arm is locked on the first cycle, which then runs
    again; a latch failing at the top of that cycle fails the capture with
    its own error and puts the arm back, so the live view keeps adjusting."""
    sim_scope.camera.set_auto_gain(True, AUTO_GAIN)
    driver = sim_scope._camera_driver
    latches = []
    real_latch = driver.latch_clock

    def latch_fails_the_second_time():
        latches.append(None)
        if len(latches) == 2:
            raise CameraReadFailedError('TimestampLatch failed: scripted')
        real_latch()

    monkeypatch.setattr(driver, 'latch_clock', latch_fails_the_second_time)

    with pytest.raises(CameraReadFailedError):
        sim_scope.imaging.capture_and_wait(accept_dark=True)

    assert len(latches) == 2, 'precondition: the lock re-ran the cycle'
    # A lock takes the arm that stands; with none it locks nothing.
    assert sim_scope.camera.lock_auto_gain().state is not None, 'the arm was not re-armed'


def test_a_clock_rule_change_is_waited_out_on_the_cameras_clock(sim_scope):
    """Under the clock rule the drain waits until a frame can start after the
    change plus its delay, whatever frames were counted toward it, and the
    frame returned started after that instant."""
    delay_us = 300_000.0
    _clock_rule_for(sim_scope, 'gain', delay_us)
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)
    written_ns = sim_scope.imaging.recorded_changes()['gain'].written_ns

    gain = _judgment(sim_scope.imaging.capture_still(accept_dark=True).record, 'gain')
    assert gain.change.rule.rule is Rule.CLOCK
    assert gain.counts
    assert gain.frame_start_ticks > gain.qualifies_after_ticks
    assert host_clock.steady_ns() - written_ns >= delay_us * 1000


def test_a_capture_no_frame_counts_for_raises_naming_the_change(sim_scope, monkeypatch):
    """A frame the judge does not count is passed over, and when the grab's
    bound passes with none counting the capture raises ``no_fresh_frame``
    naming the change's source, its words saying the change as a person
    does, never returning the old frame."""
    frame_validity = sim_scope.imaging.frame_validity
    sim_scope.motion.home('Z')
    sim_scope.motion.move_z(sim_scope.motion.get_current_position('Z') + 50.0)
    judgments = frame_validity.judgments

    # a stand-in by design: a frame every count and settle says is fresh, yet
    # judged stale, is a fault the simulator cannot yet produce (step 8d-2's
    # apply delay produces it).
    def stale_for_the_move(frame, mapping):
        return tuple(
            dataclasses.replace(j, counts=False) if j.change.source == 'z_move' else j
            for j in judgments(frame, mapping)
        )

    monkeypatch.setattr(frame_validity, 'judgments', stale_for_the_move)

    with pytest.raises(CaptureFailedError) as raised:
        sim_scope.imaging.capture_and_wait(accept_dark=True)

    assert (raised.value.reason, raised.value.source) == ('no_fresh_frame', 'z_move')
    assert 'the focus move' in str(raised.value)


def test_a_frame_an_excluded_change_does_not_count_for_is_kept(sim_scope, monkeypatch):
    """A capture that excludes a source keeps a frame that change does not
    count for: the frame can predate it, as the exclusion says."""
    frame_validity = sim_scope.imaging.frame_validity
    sim_scope.motion.home('Z')
    sim_scope.motion.move_z(sim_scope.motion.get_current_position('Z') + 50.0)
    judgments = frame_validity.judgments

    # a stand-in by design, as above.
    def stale_for_the_move(frame, mapping):
        return tuple(
            dataclasses.replace(j, counts=False) if j.change.source == 'z_move' else j
            for j in judgments(frame, mapping)
        )

    monkeypatch.setattr(frame_validity, 'judgments', stale_for_the_move)

    assert (
        sim_scope.imaging.capture_and_wait(accept_dark=True, exclude_sources=('z_move',))
        is not None
    )


def test_a_capture_can_fail_for_every_source_in_a_persons_words():
    # A source with no words would turn the capture's no_fresh_frame into a
    # TypeError at the raise.
    unspoken = []
    for source in FrameValidity.SKIP_FRAMES:
        try:
            CaptureFailedError('no_fresh_frame', source=source)
        except TypeError:
            unspoken.append(source)
    assert unspoken == []

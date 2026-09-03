# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a composite run refuses, and what a refusal leaves behind.

``test_composite_run_e2e`` proves the happy path: a click that should
produce a merged file does. This file is the other half -- the clicks
that must NOT produce anything. A composite is the one run kind a user
fires by hand from a live scope, so every refusal here is something a
person will actually do: click it twice, click it during a scan, click
it with the camera unplugged, click it with one channel armed.

The interesting property is not that these raise. It is that they raise
having changed NOTHING. A refusal that half-committed -- a claim still
held, a run directory already made, an LED left on, a rival's run
aborted -- turns a mistyped click into lost work, and the second click
that would recover from it gets refused for the wreckage of the first.
So each test here asserts the refusal AND the absence of a footprint.

The bench list phrases that footprint as "neither leaves cosmetics
changed", which is a GUI sentence: buttons, spinners, the capture
controls staying usable. Headless there are no cosmetics, so the
analogue asserted here is the state those cosmetics render -- the
refusal leaves no activity claim held, no run directory created, and no
hardware touched, and the very next attempt is refused for the ORIGINAL
reason rather than for a run that never started.
"""

import pathlib
import threading

import pytest

from modules.exceptions import ProtocolRunRefusedError
from tests.test_composite_run_e2e import (
    composite_session,  # noqa: F401  -- imported as a fixture, used by name
    headless_settings,
    open_composite_session,
    single_run_dir,
)


def _run_dirs(tmp_path):
    """Every run directory currently under the run root.

    ``single_run_dir`` answers for the runs that happened; this answers
    for the ones that must not have, which is the whole subject here.
    """
    return [p for p in tmp_path.iterdir() if p.is_dir()]


class _StepGate:
    """Park a live run between two steps so a rival can be attempted.

    Polling ``is_protocol_running`` from the test thread would race a run
    that finishes first -- on simulated hardware a two-channel composite
    is over in well under a second. Holding the run inside a callback it
    already fires makes the mid-run window as wide as the test needs it,
    without the test reaching into the engine to create one.

    ``update_step_number`` is the callback used because it is purely
    observational: unlike ``go_to_step`` it does not REPLACE the move the
    step runner would otherwise make, so the parked run still completes
    normally once released.
    """

    def __init__(self):
        self.reached = threading.Event()
        self.release = threading.Event()

    def callbacks(self):
        return {'update_step_number': self._on_step}

    def _on_step(self, _step_number):
        self.reached.set()
        self.release.wait(timeout=60)


def _plain_scan_protocol(session):
    """The steps a composite would capture, for a NON-composite run.

    ``run_single_scan`` needs a Protocol, and production assembly already
    builds one from the same settings snapshot the composite uses. Borrow
    it rather than hand-rolling a DataFrame: what makes this a rival is
    the run MODE and the claim it takes, not which positions it visits.
    """
    import modules.config_helpers as config_helpers

    input_config = config_helpers.get_composite_capture_config_from_settings(
        session.settings,
        session.objective_helper,
        position=session.get_current_plate_position(),
    )
    return session.scope.protocols.create_protocol(input_config=input_config)


def _record_led_calls(monkeypatch, led):
    """Record every LED-on call, then let the real driver do it.

    Observation only -- the driver still runs. A stub would answer for a
    board that was never asked to light, which is the very thing under
    test.
    """
    calls = []

    for name in ('led_on', 'led_on_fast'):
        real = getattr(led, name)

        def _observed(*args, _real=real, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return _real(*args, **kwargs)

        monkeypatch.setattr(led, name, _observed)

    return calls


class TestARivalRunRefusesTheComposite:
    """Both directions of the two-runs-at-once collision."""

    def test_a_composite_during_a_composite_is_refused_and_the_first_still_merges(
        self, composite_session
    ):
        # The second click must be refused BY the engine's typed refusal,
        # not by whatever the caller happens to check first -- and the
        # refusal must cost the live run nothing. A gate that aborted the
        # incumbent instead of refusing the newcomer would lose a run the
        # user never asked to stop, and would still look like "the second
        # click did not run".
        session, runner, tmp_path = composite_session
        gate = _StepGate()

        outcome = runner.start_composite(
            sequence_name='rival_incumbent',
            parent_dir=str(tmp_path),
            callbacks=gate.callbacks(),
        )
        try:
            assert gate.reached.wait(timeout=60), 'the first composite never reached a step'
            assert session.is_protocol_running, 'the parked run does not hold the claim'
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                runner.start_composite(sequence_name='rival_second', parent_dir=str(tmp_path))
        finally:
            gate.release.set()

        assert refusal.value.reason == 'already_running', (
            f'a composite clicked during a composite was refused for '
            f'{refusal.value.reason!r}, not for the run already holding the scope'
        )

        settled = outcome.wait(timeout_s=120)
        assert settled is not None and settled.merged, (
            f'the refused second click cost the first run its merge: {settled}'
        )
        assert pathlib.Path(settled.artifact_path).exists()
        # One run started, so exactly one run directory: the refusal must
        # not have carved out a home for a run that never happened.
        assert single_run_dir(tmp_path)

    def test_a_composite_during_a_plain_scan_is_refused(self, composite_session):
        # The rival here is a SINGLE_SCAN, not another composite: the
        # claim is held by the run subsystem regardless of run kind, and a
        # gate keyed to "a composite is running" rather than to the claim
        # would let a composite start on top of a scan and interleave two
        # runs' LED and stage commands.
        session, runner, tmp_path = composite_session
        gate = _StepGate()

        runner.run_single_scan(
            _plain_scan_protocol(session),
            sequence_name='scan_incumbent',
            parent_dir=str(tmp_path),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            callbacks=gate.callbacks(),
        )
        try:
            assert gate.reached.wait(timeout=60), 'the scan never reached a step'
            assert session.is_protocol_running, 'the parked scan does not hold the claim'
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                runner.start_composite(sequence_name='composite_second', parent_dir=str(tmp_path))
        finally:
            gate.release.set()

        assert refusal.value.reason == 'already_running', (
            f'a composite clicked during a scan was refused for '
            f'{refusal.value.reason!r}, not for the run already holding the scope'
        )
        assert runner.wait_for_completion(timeout=120), 'the scan never completed'
        assert not session.is_protocol_running, 'the scan finished still holding the claim'
        # The scan's own directory, and only it.
        assert single_run_dir(tmp_path)


class TestADisconnectedCameraRefusesTheComposite:
    """The camera-off click: refused loudly, and recoverable."""

    def test_a_disconnected_camera_is_refused_before_anything_is_committed(self, composite_session):
        # A composite with no camera cannot produce a frame, so the only
        # acceptable outcome is a refusal at the boundary. The failure
        # this catches is the run committing anyway and failing later on
        # the protocol thread, where an L2 caller sees a "completed" run
        # with nothing in it.
        session, runner, tmp_path = composite_session

        assert session.scope._camera_driver.disconnect(), (
            'the camera was not connected to begin with'
        )

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            runner.start_composite(sequence_name='camera_off', parent_dir=str(tmp_path))

        assert refusal.value.reason == 'hardware_disconnected', (
            f'a composite with no camera was refused for {refusal.value.reason!r}'
        )
        assert not session.is_protocol_running, 'the refused run left the activity claim held'
        assert _run_dirs(tmp_path) == [], (
            f'the refused run created a run directory: {_run_dirs(tmp_path)}'
        )

    def test_the_second_attempt_is_refused_for_the_camera_not_for_a_run(self, composite_session):
        # THE test of "nothing was committed". If the failed start left
        # the run flag or the claim set, the next click reports
        # 'already_running' -- a phantom run the user can neither see nor
        # stop, and every later click refused for it. The reason staying
        # 'hardware_disconnected' is what proves the first refusal
        # released everything it touched.
        session, runner, tmp_path = composite_session
        session.scope._camera_driver.disconnect()

        with pytest.raises(ProtocolRunRefusedError):
            runner.start_composite(sequence_name='camera_off_1', parent_dir=str(tmp_path))
        with pytest.raises(ProtocolRunRefusedError) as second:
            runner.start_composite(sequence_name='camera_off_2', parent_dir=str(tmp_path))

        assert second.value.reason == 'hardware_disconnected', (
            f'the second click was refused for {second.value.reason!r} -- the first '
            f'refusal left run state behind, so the camera is no longer the reason'
        )
        assert _run_dirs(tmp_path) == [], f'two refusals left directories: {_run_dirs(tmp_path)}'

    def test_a_reconnected_camera_runs_the_composite(self, composite_session):
        # The refusal has to be a pause, not a dead end: plug the camera
        # back in and the same click works. A refusal that poisoned the
        # runner would show up here rather than above.
        session, runner, tmp_path = composite_session
        session.scope._camera_driver.disconnect()

        with pytest.raises(ProtocolRunRefusedError):
            runner.start_composite(sequence_name='camera_off', parent_dir=str(tmp_path))

        session.scope._camera_driver.connect()
        session.scope.imaging.start_streaming()

        artifact = pathlib.Path(
            runner.run_composite(sequence_name='after', parent_dir=str(tmp_path))
        )
        assert artifact.exists(), 'the composite produced no file after the camera came back'


class TestFewerThanTwoChannels:
    """One channel and none: the same refusal, before the run exists."""

    @pytest.mark.parametrize('acquiring', [('BF',), ()])
    def test_fewer_than_two_channels_is_refused_before_any_hardware_is_touched(
        self, tmp_path, monkeypatch, acquiring
    ):
        # Zero channels is the case worth parametrizing: an empty channel
        # set also makes an empty PROTOCOL, so a composite that assembled
        # first and let the engine judge it would refuse with
        # 'empty_protocol' -- true, but useless. The user set no channels;
        # "add a step" is not the action that fixes it. The channel guard
        # refusing FIRST is what makes both cases say the same, actionable
        # thing.
        #
        # The LED assertion is the "nothing was touched" half: the guard
        # runs before the protocol is assembled, so a refusal that had
        # already lit a channel would mean the assembly ran ahead of its
        # own precondition and left the board illuminated with no run to
        # turn it off.
        with open_composite_session(headless_settings(tmp_path, acquiring=acquiring)) as (
            session,
            runner,
        ):
            led_calls = _record_led_calls(monkeypatch, session.scope._led_driver)

            with pytest.raises(ProtocolRunRefusedError) as refusal:
                runner.start_composite(sequence_name='too_few', parent_dir=str(tmp_path))

            assert refusal.value.reason == 'composite_needs_two_channels', (
                f'{len(acquiring)} acquiring channel(s) refused for '
                f'{refusal.value.reason!r}, not for the channel count'
            )
            assert led_calls == [], f'the refused composite lit the board: {led_calls}'
            assert not session.is_protocol_running, 'the refused run left the activity claim held'
            assert _run_dirs(tmp_path) == [], (
                f'the refused run created a run directory: {_run_dirs(tmp_path)}'
            )

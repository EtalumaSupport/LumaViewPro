# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a composite run does when a capture, or the merge, does not work.

``test_composite_run_e2e`` proves the happy path reaches a readable file.
The failures are the other half of the same contract, and they are the
half a caller actually has to branch on: a run where one channel came
back black must not report the same thing as a run where every channel
landed. Three behaviours are pinned here.

A capture fails when the camera hands back a frame with no signal in it
while the LEDs were commanded lit. That check is armed from the
COMMANDED illumination alone (``imaging.py``, ``expected_lit =
bool(live_lit_pairs(...))``), so it governs EVERY layer that maps to an
LED channel -- transmitted BF exactly as much as Blue. It is not a
fluorescence-only gate. The simulated camera's black test pattern is the
one lever this file uses to trip it; nothing here mocks the engine, the
runner, the session, or a driver.

That the check is right on real hardware is a photometric question this
file cannot answer. Whether a capped epi channel's genuinely dim frame
clears the dark floor on a real scope has to be measured on a bench with
a real sample; the simulator's black frame only proves the wiring from a
signal-free frame to a failed capture to the run's disposition of it.

A rejected capture leaves NO image file: the writer records a
``capture_failed`` row in the run's ``protocol_record.tsv`` and returns
before ``save_image``. So the merge sees a directory with one frame per
SUCCESSFUL channel, and the per-channel detail lives in the record and
the log rather than in the outcome the caller waits on.
"""

import contextlib
import csv
import pathlib

import pytest

from modules.exceptions import CaptureError
from modules.image_mode import OUTPUT_FORMAT_JPG, OUTPUT_FORMAT_TIFF
from tests.test_composite_run_e2e import (
    headless_settings,
    open_composite_session,
    single_run_dir,
)

# The invariant middle of the substitution notice
# config_helpers.get_composite_image_capture_config_from_settings emits --
# the two format names around it are interpolated. Matching the fixed span
# rather than the whole line keeps the pair of tests below symmetrical:
# the same needle proves the line present in one case and absent in the
# other, so a reworded message cannot quietly turn the absence assertion
# into one that passes for the wrong reason.
_COERCION_LOG_FRAGMENT = 'cannot be merged; this run saves'

# The layer whose steps are made to fail. Blue is never first in the
# release layer order (BF, PC, DF, Blue, Green, Red, Lumi), so a run that
# fails only Blue always has a successful step before it -- which is what
# makes the strike counter's reset, not just its increment, observable.
_FAILING = 'Blue'


@contextlib.contextmanager
def _info_lines():
    """Collect what production logs at INFO while the block runs.

    ``caplog`` cannot see this: the suite replaces the whole ``lvp_logger``
    module with a MagicMock in ``conftest.install_mock_deps``, so
    ``logger.info`` never reaches the logging machinery pytest attaches
    to. Wrapping the one call the assertion is about is observation only
    -- nothing about the run changes -- and it is the same instrument
    ``test_composite_run_config`` already uses for this exact line.
    """
    import lvp_logger

    lines = []
    original = lvp_logger.logger.info
    lvp_logger.logger.info = lambda msg, *a, **kw: lines.append(str(msg))
    try:
        yield lines
    finally:
        lvp_logger.logger.info = original


def _fail_these_channels(camera, step_colors, failing):
    """Callbacks that black out the camera for *failing* channels' steps.

    ``update_step_number`` fires synchronously on the protocol thread
    before the step it names is positioned and lit (headless has no UI
    dispatcher, so ``schedule_ui`` calls straight through), which makes it
    the one hook that can arm a per-channel condition ahead of that
    channel's capture. Step 1 never gets the call, so the pattern the
    session opens with is what step 1 sees.

    The step number is 1-based; *step_colors* is the run's layer order.
    """

    def update_step_number(step):
        color = step_colors[step - 1]
        if color in failing:
            camera.set_test_pattern(True, 'black')
        else:
            camera.set_test_pattern(False)

    return {'update_step_number': update_step_number}


def _record_rows(run_dir):
    """The run's execution-record rows, as dicts keyed by column name.

    The record is a TSV behind three preamble lines (a file header, a
    version, and the protocol path); the fourth line is the column row.
    """
    record = pathlib.Path(run_dir) / 'protocol_record.tsv'
    assert record.exists(), f'the run wrote no execution record in {run_dir}'
    with open(record, newline='') as fp:
        rows = list(csv.reader(fp, delimiter='\t'))
    columns = rows[3]
    # strict: a row that does not fit the column header is a malformed
    # record, and silently truncating it would hide exactly the missing
    # failure row these tests read the file to find.
    return [dict(zip(columns, row, strict=True)) for row in rows[4:]]


def _failed_step_names(run_dir):
    """Step names the record marks as having captured nothing."""
    return [
        row['Step Name'] for row in _record_rows(run_dir) if row['Filename'] == 'capture_failed'
    ]


class TestAChannelThatCapturesNothing:
    """One channel comes back black; what the caller is told about it."""

    def test_two_channels_with_one_failure_leave_nothing_to_merge(self, tmp_path):
        # Two channels is the minimum a merge can consume, so losing one
        # leaves a group of a single frame -- which the post-processor
        # skips at source rather than writing a one-channel "composite".
        # The caller therefore gets a resolved, not-merged outcome; the
        # channel that failed is named in the run record, not in the
        # outcome, because the outcome answers "is there an artifact".
        settings = headless_settings(tmp_path, acquiring=('BF', _FAILING))
        with open_composite_session(settings) as (_session, runner):
            outcome = runner.start_composite(
                sequence_name='one_of_two',
                parent_dir=str(tmp_path),
                callbacks=_fail_these_channels(
                    _session.scope._camera_driver, ('BF', _FAILING), {_FAILING}
                ),
            )
            settled = outcome.wait(timeout_s=120)

            assert settled is not None, 'the run never settled its outcome'
            assert not settled.merged, (
                f'a run that captured one of two channels reported a merge: {settled}'
            )
            assert settled.artifact_path is None, (
                f'a failed merge handed back a path: {settled.artifact_path}'
            )

            run_dir = single_run_dir(tmp_path)
            assert any(_FAILING in name for name in _failed_step_names(run_dir)), (
                f'the run record does not name {_FAILING} as the failed '
                f'channel: {_record_rows(run_dir)}'
            )
            # Asserted last so the facts above are checked either way. The
            # code is the post-processor's own: 'no_data' is what it returns
            # when no image group is usable, and the runner relays it as is.
            # The runner's 'merge_failed' is only the fallback for a result
            # that carries no reason at all, so a caller seeing it here
            # would mean the specific code was lost on the way.
            assert settled.reason == 'no_data', (
                f'the outcome named the failure {settled.reason!r}; the reason '
                f'is what a REST or SDK caller maps to a response, so it is '
                f'part of the contract, not prose'
            )

    def test_a_rejected_capture_leaves_no_image_behind(self, tmp_path):
        # A file for a channel that captured nothing would be worse than
        # no file: the merge reads its inputs off disk, so a black frame
        # on disk is a black channel in the composite rather than a
        # refusal the caller can see.
        settings = headless_settings(tmp_path, acquiring=('BF', _FAILING))
        with open_composite_session(settings) as (_session, runner):
            runner.start_composite(
                sequence_name='no_file',
                parent_dir=str(tmp_path),
                callbacks=_fail_these_channels(
                    _session.scope._camera_driver, ('BF', _FAILING), {_FAILING}
                ),
            ).wait(timeout_s=120)

            run_dir = single_run_dir(tmp_path)
            frames = sorted(p.name for p in run_dir.glob('*.tiff'))
            assert not any(_FAILING in name for name in frames), (
                f'the failed channel left a frame on disk: {frames}'
            )
            assert any('BF' in name for name in frames), (
                f'the channel that DID capture left no frame either, so this '
                f'run failed for some other reason: {frames}'
            )

    def test_the_same_failure_reaches_an_l2_caller_as_a_typed_raise(self, tmp_path):
        # run_composite returns a path, so it has no way to hand back a
        # not-merged outcome; a caller that got None or '' would have to
        # guess whether the run aborted, the merge failed, or the wait
        # expired. The typed error carries the machine-readable reason.
        settings = headless_settings(tmp_path, acquiring=('BF', _FAILING))
        with open_composite_session(settings) as (_session, runner):
            with pytest.raises(CaptureError) as excinfo:
                runner.run_composite(
                    sequence_name='typed_raise',
                    parent_dir=str(tmp_path),
                    callbacks=_fail_these_channels(
                        _session.scope._camera_driver, ('BF', _FAILING), {_FAILING}
                    ),
                )

            assert excinfo.value.reason == 'no_data', (
                f'the raise carried reason {excinfo.value.reason!r}, not the '
                f"post-processor's code for a merge with no usable group"
            )

    def test_three_channels_with_one_failure_still_merge_the_other_two(self, tmp_path):
        # The true partial merge: two survivors are still a mergeable
        # group, so the run must produce a real composite AND still say
        # which channel is missing from it. A run that merged silently
        # here would hand a user a two-channel image they asked three
        # channels for.
        step_colors = ('BF', _FAILING, 'Green')
        settings = headless_settings(tmp_path, acquiring=step_colors)
        with open_composite_session(settings) as (_session, runner):
            outcome = runner.start_composite(
                sequence_name='two_of_three',
                parent_dir=str(tmp_path),
                callbacks=_fail_these_channels(
                    _session.scope._camera_driver, step_colors, {_FAILING}
                ),
            )
            settled = outcome.wait(timeout_s=120)

            assert settled is not None and settled.merged, (
                f'two surviving channels are a mergeable group, but the run '
                f'produced nothing: {settled}'
            )
            assert pathlib.Path(settled.artifact_path).exists(), (
                f'the merge reported {settled.artifact_path}, which is not there'
            )

            run_dir = single_run_dir(tmp_path)
            assert any(_FAILING in name for name in _failed_step_names(run_dir)), (
                f'a composite was produced without the record saying which '
                f'channel is absent from it: {_record_rows(run_dir)}'
            )


class TestTheThreeStrikeFatalAbort:
    """Three consecutive failed captures stop the run outright."""

    def test_three_consecutive_failures_abort_before_the_last_channel(self, tmp_path):
        # The cap exists so a scope whose camera has stopped delivering
        # does not grind through every remaining step producing nothing.
        # Four acquiring channels with the camera black for the whole run
        # is the shape that separates "the run stopped" from "the run
        # finished badly": the fourth channel must never be reached.
        step_colors = ('BF', 'Blue', 'Green', 'Red')
        settings = headless_settings(tmp_path, acquiring=step_colors)
        with open_composite_session(settings) as (session, runner):
            session.scope._camera_driver.set_test_pattern(True, 'black')

            outcome = runner.start_composite(sequence_name='fatal', parent_dir=str(tmp_path))
            settled = outcome.wait(timeout_s=120)

            assert settled is not None, 'the aborted run never settled its outcome'
            assert not settled.merged, f'an aborted run reported a merge: {settled}'
            assert settled.reason == 'aborted', (
                f'the outcome named the ending {settled.reason!r}; a caller '
                f'cannot tell an abort from a failed merge that way'
            )

            run_dir = single_run_dir(tmp_path)
            assert not list(run_dir.glob('*.tiff')), (
                'a run in which every capture was rejected wrote image files'
            )
            names = [row['Step Name'] for row in _record_rows(run_dir)]
            assert not any(step_colors[-1] in name for name in names), (
                f'the run reached its fourth channel, so it did not stop at '
                f'the third strike: {names}'
            )

    def test_a_fatal_abort_releases_the_run(self, tmp_path):
        # An abort that left the activity claim held would refuse every
        # later run and recording, so the run has to be fully unwound by
        # the time the outcome resolves -- the same settling contract the
        # successful path has.
        settings = headless_settings(tmp_path, acquiring=('BF', 'Blue', 'Green', 'Red'))
        with open_composite_session(settings) as (session, runner):
            session.scope._camera_driver.set_test_pattern(True, 'black')

            runner.start_composite(sequence_name='released', parent_dir=str(tmp_path)).wait(
                timeout_s=120
            )

            assert not session.is_protocol_running, (
                'the aborted run still held the activity claim after settling'
            )

    def test_a_fatal_abort_darkens_every_led_including_one_it_did_not_light(self, tmp_path):
        # A composite hands the illumination back the way it found it
        # (start_composite passes leds_state_at_end='return_to_original'),
        # so a channel the user had lit before the run normally comes
        # back on at the end. A FATAL abort is the exception: the run died
        # on a fault, and leaving the sample illuminated by a scope nobody
        # is driving is the outcome that policy must not produce.
        settings = headless_settings(tmp_path, acquiring=('BF', 'Blue', 'Green', 'Red'))
        with open_composite_session(settings) as (session, runner):
            led_board = session.scope._led_driver
            # PC is an LED layer this run does not acquire, so it can only
            # be lit at the end by the return-to-original restore.
            session.scope.illumination.led_on('PC', 30.0)
            assert led_board.read_led_current(led_board.color2ch('PC')) > 0, (
                'the pre-run channel never lit, so the restore this test '
                'exists to override was never armed'
            )

            session.scope._camera_driver.set_test_pattern(True, 'black')
            runner.start_composite(sequence_name='dark', parent_dir=str(tmp_path)).wait(
                timeout_s=120
            )

            lit = {
                channel: led_board.read_led_current(channel)
                for channel in led_board.available_channels()
                if led_board.read_led_current(channel)
            }
            assert not lit, f'a fatal abort left LED channels driven: {lit}'


class TestTheFormatTheCompositeFollows:
    """The composite follows the SEQUENCED preference, or says why not."""

    def test_a_live_jpg_preference_does_not_change_the_composite(self, tmp_path):
        # The live format is what the retired GUI worker read, and reading
        # it is what made a user's composite silently change format when
        # they picked JPG for the live view. The run kind reads the
        # sequenced preference, so TIFF stays TIFF -- and because nothing
        # was substituted, there is nothing to log: the override line
        # fires only in the coercion branch.
        settings = headless_settings(
            tmp_path,
            sequenced_format=OUTPUT_FORMAT_TIFF,
            live_format=OUTPUT_FORMAT_JPG,
        )
        with open_composite_session(settings) as (_session, runner):
            with _info_lines() as logged:
                artifact = pathlib.Path(
                    runner.run_composite(sequence_name='live_jpg', parent_dir=str(tmp_path))
                )

            run_dir = single_run_dir(tmp_path)
            assert not list(run_dir.glob('*.jpg')), (
                f'the composite run saved JPGs from the LIVE preference: '
                f'{sorted(p.name for p in run_dir.iterdir())}'
            )
            assert sorted(p.name for p in run_dir.glob('*.tiff')), (
                'the run saved no TIFF frames at all'
            )
            assert artifact.name.endswith('.tiff') and not artifact.name.endswith('.ome.tiff'), (
                f'a TIFF-preference run produced {artifact.name}'
            )
            assert not [line for line in logged if _COERCION_LOG_FRAGMENT in line], (
                f'a run whose format was NOT substituted logged an override, '
                f'so the line no longer distinguishes the two cases: {logged}'
            )

    def test_an_unmergeable_sequenced_preference_coerces_and_says_so(self, tmp_path):
        # JPG cannot be read back by the merge, so the run has to save
        # something else. Substituting silently leaves a user staring at
        # OME-TIFFs they never asked for with nothing to explain it, so
        # the log line is part of the behaviour, not a debugging aid.
        settings = headless_settings(tmp_path, sequenced_format=OUTPUT_FORMAT_JPG)
        with open_composite_session(settings) as (_session, runner):
            with _info_lines() as logged:
                artifact = pathlib.Path(
                    runner.run_composite(sequence_name='coerced', parent_dir=str(tmp_path))
                )

            assert artifact.name.endswith('.ome.tiff'), (
                f'an unmergeable preference produced {artifact.name} rather '
                f'than the OME-TIFF the merge can read'
            )
            assert [
                line
                for line in logged
                if _COERCION_LOG_FRAGMENT in line and OUTPUT_FORMAT_JPG in line
            ], (
                f'the format was substituted with no override naming the rejected preference: {logged}'
            )

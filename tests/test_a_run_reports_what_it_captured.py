# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A run's ending says what the instrument did.

Driven end to end on the simulated scope: a real session, runner, run
loop, writer and file lane; only the camera driver's grab is replaced
where a test needs a capture to produce no frame.
"""

import contextlib
import csv
import pathlib
import time

import modules.common_utils as common_utils
from modules.exceptions import RunImagesNotSavedError, RunIncompleteError

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_composite_run_failures import _info_lines

WAIT_S = 60.0


def _run(runner, run_parent, steps, callbacks=None):
    pending = runner.run_single_scan(
        protocol=_protocol(steps),
        parent_dir=str(run_parent),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        callbacks=callbacks or {},
    )
    return pending.wait(timeout_s=WAIT_S)


def _two_steps():
    return [_step('C1', 0, x=20.0, gain=1.0), _step('C2', 1, x=30.0, gain=1.0)]


class TestANormalRunLogsNoCrash:
    def test_the_late_cleanup_passes_name_no_crash(self, tmp_path):
        # Cleanup is asked three times on every run; the safety net's pass
        # carries a 'run_loop_crashed' ending it never uses. Naming that
        # ending in the log recorded a crash on every normal run.
        with (
            _info_lines() as info,
            open_composite_session(headless_settings(tmp_path)) as (
                _session,
                runner,
            ),
        ):
            outcome = _run(runner, tmp_path / 'runs', _two_steps())
        assert outcome.status == 'completed', outcome
        crash_lines = [line for line in info if 'crash' in line.lower()]
        assert crash_lines == [], crash_lines
        assert any('no longer live' in line for line in info), (
            'the late passes stopped logging at all; this test no longer sees them'
        )


def _record_rows(run_parent):
    records = list(pathlib.Path(run_parent).rglob('protocol_record.tsv'))
    assert len(records) == 1, records
    with open(records[0], newline='') as fp:
        return [row[0] for row in list(csv.reader(fp, delimiter='\t'))[4:]]


def _no_frame(timeout_s):
    return (False, None, 0)


class TestARunWhoseCameraIsGoneEndsAtOnce:
    def test_the_first_failure_after_a_removal_ends_the_run_failed(self, tmp_path):
        # The bench case: an unplugged camera, a run too short for three
        # strikes. The driver had already declared the camera removed, so
        # the first failed capture ends the run, and in the word that
        # refuses the next run.
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            camera = session.scope._camera_driver
            imaging = session.scope.imaging
            # Every camera write cleanup makes after the removal. The
            # simulated camera accepts a write whatever its latch says, so
            # the writes themselves are what is counted, not their reports.
            writes_after_removal = []
            for setter in ('_set_gain_db_impl', '_set_exposure_ms_impl'):
                real = getattr(imaging, setter)

                def counted(value, *a, _real=real, _name=setter, **kw):
                    if camera.is_device_removed():
                        writes_after_removal.append((_name, value))
                    return _real(value, *a, **kw)

                setattr(imaging, setter, counted)

            def unplugged(timeout_s):
                camera._mark_disconnected()
                return _no_frame(timeout_s)

            camera.grab_new_capture = unplugged
            steps = [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(3)]
            outcome = _run(runner, run_parent, steps)
            rows = _record_rows(run_parent)
        assert (outcome.status, outcome.reason) == ('failed', 'hardware_disconnected'), outcome
        assert rows == ['capture_failed'], (
            f'the run went on capturing after its camera was removed: {rows}'
        )
        # Cleanup's camera restore writes nothing to a removed camera; each
        # write would report the camera absent again after the ending did.
        assert writes_after_removal == [], writes_after_removal


@contextlib.contextmanager
def _reports_of(outcome_type):
    """Every outcome of *outcome_type* reported while the block runs."""
    from modules.notification_center import notifications

    reported = []
    report = notifications.report_outcome

    def counted(outcome, *a, **kw):
        if isinstance(outcome, outcome_type):
            reported.append(outcome)
        return report(outcome, *a, **kw)

    notifications.report_outcome = counted
    try:
        yield reported
    finally:
        notifications.report_outcome = report


class TestARunWithFailedCapturesEndsIncomplete:
    def test_a_run_whose_every_capture_failed_is_not_completed(self, tmp_path):
        # The bench's two lines, as a test: two steps, no frame from either,
        # and the run said 'completed'.
        seen = []
        with (
            _reports_of(RunIncompleteError) as reported,
            open_composite_session(headless_settings(tmp_path)) as (session, runner),
        ):
            session.scope._camera_driver.grab_new_capture = _no_frame
            outcome = _run(
                runner,
                tmp_path / 'runs',
                _two_steps(),
                callbacks={'run_complete': lambda **kw: seen.append(kw['status'])},
            )
        assert (outcome.status, outcome.reason) == ('incomplete', 'captures_failed'), outcome
        assert (outcome.captures.asked, outcome.captures.captured) == (2, 0), outcome.captures
        assert [failed.step_name for failed in outcome.captures.failed] == ['C1', 'C2']
        assert seen == ['incomplete'], 'run_complete told its subscribers a different ending'
        assert len(reported) == 1, reported

    def test_a_success_between_failures_does_not_hide_them(self, tmp_path):
        # A success resets the three-in-a-row abort, so a run failing every
        # other capture never stops; it must still not end 'completed'.
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            camera = session.scope._camera_driver
            real_grab = camera.grab_new_capture

            def update_step_number(step):
                # 1-based, and step 1 never gets the call: armed below.
                camera.grab_new_capture = _no_frame if step % 2 == 1 else real_grab

            camera.grab_new_capture = _no_frame
            steps = [_step(f'C{i}', i, x=20.0 + i, gain=1.0) for i in range(8)]
            outcome = _run(
                runner,
                tmp_path / 'runs',
                steps,
                callbacks={'update_step_number': update_step_number},
            )
        assert outcome.status == 'incomplete', outcome
        assert (outcome.captures.asked, outcome.captures.captured) == (8, 4), outcome.captures
        assert [failed.step_name for failed in outcome.captures.failed] == [
            'C0',
            'C2',
            'C4',
            'C6',
        ]

    def test_a_run_that_captured_everything_is_completed_and_says_so(self, tmp_path):
        with (
            _reports_of(RunIncompleteError) as reported,
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
        ):
            outcome = _run(runner, tmp_path / 'runs', _two_steps())
        assert outcome.status == 'completed', outcome
        assert (outcome.captures.asked, outcome.captures.captured) == (2, 2), outcome.captures
        assert outcome.captures.failed == ()
        assert reported == []

    def test_a_run_that_saves_no_images_asks_for_none(self, tmp_path):
        # An autofocus scan saves nothing; it cannot fall short of captures
        # it was never asked for.
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            pending = runner.run_single_scan(
                protocol=_protocol(_two_steps()),
                parent_dir=str(tmp_path / 'runs'),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                enable_image_saving=False,
            )
            outcome = pending.wait(timeout_s=WAIT_S)
        assert outcome.status == 'completed', outcome
        assert outcome.captures.asked == 0, outcome.captures


class TestACompositeMissingAChannel:
    def test_it_merges_the_rest_and_says_which_is_missing(self, tmp_path):
        from tests.test_composite_run_failures import _FAILING, _fail_these_channels

        step_colors = ('BF', _FAILING, 'Green')
        settings = headless_settings(tmp_path, acquiring=step_colors)
        with (
            _reports_of(RunIncompleteError) as reported,
            open_composite_session(settings) as (session, runner),
        ):
            outcome = runner.run_composite(
                sequence_name='two_of_three',
                parent_dir=str(tmp_path),
                callbacks=_fail_these_channels(
                    session.scope._camera_driver, step_colors, {_FAILING}
                ),
            )
        assert outcome.merged and pathlib.Path(outcome.artifact_path).exists(), outcome
        assert outcome.status == 'incomplete', outcome
        assert len(outcome.captures.failed) == 1, outcome.captures
        assert _FAILING in outcome.captures.failed[0].step_name, outcome.captures
        assert len(reported) == 1, reported


def _files_complete(seen):
    deadline = time.monotonic() + WAIT_S
    while not seen and time.monotonic() < deadline:
        time.sleep(0.02)
    return seen


class TestTheFileCountCountsImages:
    def test_a_capture_that_produced_nothing_is_not_a_file(self, tmp_path):
        # The bench line: "2 written, 0 not written" for a run with no image
        # on disk. A failed capture's record row is a write, not an image.
        files = []
        with (
            _info_lines() as info,
            _reports_of(RunImagesNotSavedError) as lost,
            open_composite_session(headless_settings(tmp_path)) as (session, runner),
        ):
            session.scope._camera_driver.grab_new_capture = _no_frame
            _run(
                runner,
                tmp_path / 'runs',
                _two_steps(),
                callbacks={'files_complete': lambda **kw: files.append(kw['files'])},
            )
            _files_complete(files)
        assert files == ['written'], files
        assert any('0 written, 0 not written' in line for line in info), [
            line for line in info if 'files are' in line
        ]
        assert lost == [], 'no image was lost on the way to disk; none was captured'

    def test_an_image_refused_for_disk_space_is_not_written(self, tmp_path, monkeypatch):
        # The disk floor refused the image and returned as if it had saved
        # it, so the run counted it written.
        files = []
        with (
            _reports_of(RunImagesNotSavedError) as lost,
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
        ):
            # The run loop binds its own name for the check at import; this
            # replaces only the per-write floor the writer reads.
            import modules.protocol_run_loop  # noqa: F401

            monkeypatch.setattr(common_utils, 'check_disk_space_ok', lambda path, mb: (False, 1.0))
            outcome = _run(
                runner,
                tmp_path / 'runs',
                _two_steps(),
                callbacks={'files_complete': lambda **kw: files.append(kw['files'])},
            )
            _files_complete(files)
            batch = runner._executor.write_batch()
        assert (outcome.status, outcome.reason) == ('failed', 'disk_space_critical'), outcome
        assert files == ['incomplete'], files
        assert batch.written == 0 and batch.not_written >= 1, (batch.written, batch.not_written)
        assert batch.not_written_reason == 'write_batch_disk_full'
        assert len(lost) == 1 and lost[0].reason == 'write_batch_disk_full', lost

    def test_the_disk_floor_is_told_once(self, tmp_path, monkeypatch):
        # The floor's own popup says the run stopped to protect the data.
        # The image it refused and the record row never written are that
        # same cause, and showed as two more popups after it.
        from modules.notification_center import notifications

        shown = []
        notify = notifications.notify

        def _shown(severity, category, title, message, **kw):
            delivered = notify(severity, category, title, message, **kw)
            if delivered:
                shown.append(title)
            return delivered

        files = []
        with (
            _reports_of(RunImagesNotSavedError) as lost,
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
        ):
            import modules.protocol_run_loop  # noqa: F401

            monkeypatch.setattr(common_utils, 'check_disk_space_ok', lambda path, mb: (False, 1.0))
            monkeypatch.setattr(notifications, 'notify', _shown)
            outcome = _run(
                runner,
                tmp_path / 'runs',
                _two_steps(),
                callbacks={'files_complete': lambda **kw: files.append(kw['files'])},
            )
            _files_complete(files)
        assert (outcome.status, outcome.reason) == ('failed', 'disk_space_critical'), outcome
        assert files == ['incomplete'], files
        assert len(lost) == 1, 'the lost image is still reported, to the log'
        assert shown == ['Disk Space Critical'], shown

    def test_a_video_whose_file_did_not_finish_is_not_written(self):
        # A video step writes its file on its own lane, outside the batch;
        # when the file does not finish, the run's image is still missing.
        from unittest.mock import MagicMock

        from modules.protocol_image_writer import RunWriteBatch

        batch = RunWriteBatch(MagicMock())
        batch.count_not_written('video_unfinished', 'The video V1')
        outcomes = []
        batch.close(outcomes.append)
        assert outcomes == ['incomplete']
        assert (batch.written, batch.not_written) == (0, 1)
        assert batch.not_written_reason == 'write_batch_video_unfinished'


class TestARunWhoseCleanupFailedSaysWhich:
    def test_the_outcome_names_the_step_and_the_person_is_told_once(self, tmp_path):
        # A REST caller waiting on the run was told 'completed' while the
        # scope was not put back; only the GUI's popup said so.
        from modules.exceptions import RunCleanupFailedError

        with (
            _reports_of(RunCleanupFailedError) as reported,
            open_composite_session(headless_settings(tmp_path)) as (session, runner),
        ):

            def unreachable(snapshot):
                raise RuntimeError('the camera lane is gone')

            session.scope.imaging.restore_camera_state = unreachable
            outcome = _run(runner, tmp_path / 'runs', _two_steps())
        assert outcome.status == 'completed', outcome
        assert outcome.cleanup_failures == ('Restore camera gain/exposure',), outcome
        assert len(reported) == 1, reported

    def test_a_clean_cleanup_names_none(self, tmp_path):
        with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
            outcome = _run(runner, tmp_path / 'runs', _two_steps())
        assert outcome.cleanup_failures == (), outcome

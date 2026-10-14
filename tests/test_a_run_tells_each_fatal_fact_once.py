# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's fatal fact is told once, and a fact met after the run settled is not told as its ending.

The fatal funnel reported a fresh failure on every entry, so a disk floor
met by four queued writes was four fatal reports, and a floor met after the
run had settled 'completed' was a fatal "Aborting protocol" popup for a run
that completed -- with the LEDs commanded off after the run let go of the
scope. A run-loop crash was latched and shown to nobody. Now the run's latch
says what each fact is told as: the first fatal fact is shown, after a Stop
too; another behind it is logged once; one met again is not told again; and
once cleanup has closed the latch, nothing is. The files report reads which
fault the person was shown.
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock

import modules.common_utils as common_utils
from modules.exceptions import RecordIncompleteError, RunImagesNotSavedError
from modules.run_outcome import ENDING_TITLES, EndingLatch, FatalTold, RunEnding
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_audit_fixes import _bare_protocol_writer
from tests.test_composite_run_e2e import headless_settings, open_composite_session

OUTCOMES = 'LVP.outcomes'
WAIT_S = 60.0
STOPPED = RunEnding('aborted', 'stopped', 'Stopped')


def _fatal(reason: str) -> RunEnding:
    return RunEnding('failed', reason, f'the {reason} fact')


def _failed_posts(centre_posts):
    return [(n.title, n.shown) for n in centre_posts if n.reason in ENDING_TITLES]


def _logged(caplog, reason):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == OUTCOMES and f'RunFailedError ({reason})' in r.getMessage()
    ]


# --- What the latch tells ------------------------------------------------------


class TestTheLatchSaysWhatEachFactIsTold:
    def test_the_first_fatal_is_shown_and_ends_the_run(self):
        latch = EndingLatch()

        assert latch.record_fatal(_fatal('position_lost')) is FatalTold.SHOW
        assert latch.get().reason == 'position_lost'
        assert latch.shown_fatal == 'position_lost'

    def test_another_fatal_is_logged_once_and_the_same_one_again_not_at_all(self):
        latch = EndingLatch()
        latch.record_fatal(_fatal('position_lost'))

        told = [
            latch.record_fatal(_fatal(reason))
            for reason in ('disk_space_critical', 'disk_space_critical', 'position_lost')
        ]

        assert told == [FatalTold.LOG, FatalTold.TOLD, FatalTold.TOLD]
        assert latch.get().reason == 'position_lost', 'the first fact stays the ending'

    def test_a_fatal_after_a_stop_is_shown_and_the_stop_stays_the_ending(self):
        latch = EndingLatch()
        latch.set_if_unset(STOPPED)

        assert latch.record_fatal(_fatal('video_writer_died')) is FatalTold.SHOW
        assert latch.record_fatal(_fatal('disk_space_critical')) is FatalTold.LOG
        assert latch.get() is STOPPED
        assert latch.shown_fatal == 'video_writer_died'

    def test_a_fatal_after_cleanup_closed_the_latch_is_not_told(self):
        latch = EndingLatch()
        completed = RunEnding('completed', 'completed', 'done')

        assert latch.close(completed) is completed
        assert latch.record_fatal(_fatal('disk_space_critical')) is FatalTold.SETTLED
        assert latch.close(STOPPED) is completed, 'the ending is final once closed'
        assert latch.shown_fatal is None

    def test_a_fatal_kept_before_cleanup_closes_is_the_ending_close_returns(self):
        latch = EndingLatch()
        latch.record_fatal(_fatal('file_writer_stalled'))

        assert latch.close(RunEnding('completed', 'completed', 'done')).reason == (
            'file_writer_stalled'
        )

    def test_two_faults_racing_behind_a_stop_are_shown_once(self):
        latch = EndingLatch()
        latch.set_if_unset(STOPPED)
        start = threading.Barrier(8)
        told = []

        def race(reason):
            start.wait()
            told.append(latch.record_fatal(_fatal(reason)))

        reasons = ['disk_space_critical', 'video_writer_died'] * 4
        threads = [threading.Thread(target=race, args=(r,)) for r in reasons]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert told.count(FatalTold.SHOW) == 1, told
        assert told.count(FatalTold.LOG) == 1, told


# --- What the funnel does with it ------------------------------------------------


def _writer():
    # A stand-in by design: the subject is the funnel; the scope is the
    # collaborator it darkens, counted here.
    latch = EndingLatch()
    fatal = threading.Event()
    writer = _bare_protocol_writer(ending=latch, fatal_abort_event=fatal)
    return writer, latch, fatal, writer._scope.illumination.force_off


class TestTheFunnelTellsAsTheLatchSays:
    def test_a_floor_met_by_four_queued_writes_is_shown_once_and_logged_once(
        self, centre_posts, caplog
    ):
        writer, _latch, _fatal, force_off = _writer()

        with caplog.at_level(logging.ERROR, logger=OUTCOMES):
            for _ in range(4):
                writer._abort_run_fatal('disk_space_critical', 'FileIO', 'Only 1 MB free.')

        assert _failed_posts(centre_posts) == [('Disk Space Critical', True)]
        assert len(_logged(caplog, 'disk_space_critical')) == 1
        assert force_off.call_count == 4, 'every entry before the run settles darkens'

    def test_a_stop_then_a_fault_shows_the_fault_once_and_darkens(self, centre_posts):
        writer, latch, fatal, force_off = _writer()
        latch.set_if_unset(STOPPED)

        writer._abort_run_fatal('video_writer_died', 'Recording', 'the writer died')

        assert _failed_posts(centre_posts) == [(ENDING_TITLES['video_writer_died'], True)]
        assert fatal.is_set() and force_off.call_count == 1
        assert latch.get() is STOPPED

    def test_two_faults_show_the_first_and_log_the_second_once(self, centre_posts, caplog):
        writer, _latch, _fatal, _force_off = _writer()

        with caplog.at_level(logging.ERROR, logger=OUTCOMES):
            writer._abort_run_fatal('run_loop_crashed', 'Protocol', 'the loop died')
            writer._abort_run_fatal('disk_space_critical', 'FileIO', 'Only 1 MB free.')
            writer._abort_run_fatal('disk_space_critical', 'FileIO', 'Only 1 MB free.')

        assert _failed_posts(centre_posts) == [(ENDING_TITLES['run_loop_crashed'], True)]
        assert len(_logged(caplog, 'disk_space_critical')) == 1

    def test_a_fault_after_the_run_settled_is_neither_told_nor_darkened(self, centre_posts, caplog):
        writer, latch, _fatal, force_off = _writer()
        latch.close(RunEnding('completed', 'completed', 'done'))

        with caplog.at_level(logging.ERROR, logger=OUTCOMES):
            writer._abort_run_fatal('disk_space_critical', 'FileIO', 'Only 1 MB free.')

        assert _failed_posts(centre_posts) == []
        assert _logged(caplog, 'disk_space_critical') == []
        assert force_off.call_count == 0, 'the run let go of the scope; its LEDs are not its'

    def test_the_strike_ceiling_is_told_as_a_fault_is_without_the_darkening(self, centre_posts):
        writer, latch, fatal, force_off = _writer()

        writer._end_run_failed('consecutive_scan_failures', 'Protocol', 'failed 3 times in a row')

        assert _failed_posts(centre_posts) == [(ENDING_TITLES['consecutive_scan_failures'], True)]
        assert latch.get().reason == 'consecutive_scan_failures'
        assert not fatal.is_set() and force_off.call_count == 0


# --- On the simulated scope --------------------------------------------------------


def test_a_floor_met_after_the_run_settled_is_told_once_as_lost_images(
    tmp_path, monkeypatch, centre_posts, caplog
):
    """The disk answers slowly, as a dying disk's might: the run queues its
    writes and settles 'completed' before the first one meets the floor."""
    import modules.lumascope_api.illumination as illumination
    import modules.protocol_run_loop  # noqa: F401 -- binds the run's own floor check first

    settled_at = []
    darkened_at = []
    real_force_off = illumination.IlluminationAPI.force_off

    def counted_force_off(self, *args, **kwargs):
        darkened_at.append(time.monotonic())
        return real_force_off(self, *args, **kwargs)

    def slow_floor(path, mb):
        time.sleep(0.5)
        return (False, 1.0)

    monkeypatch.setattr(illumination.IlluminationAPI, 'force_off', counted_force_off)
    with (
        caplog.at_level(logging.DEBUG),
        open_composite_session(headless_settings(tmp_path)) as (_session, runner),
    ):
        monkeypatch.setattr(common_utils, 'check_disk_space_ok', slow_floor)
        start = len(centre_posts)
        steps = [_step(f'C{i}', i, x=20.0 + 5 * i, gain=1.0) for i in range(3)]
        run = runner.start_single_scan(protocol=_protocol(steps), parent_dir=str(tmp_path / 'runs'))
        outcome = run.result(timeout=WAIT_S)
        settled_at.append(time.monotonic())
        files = run.files.result(timeout=WAIT_S)

    posts = [(n.title, n.shown) for n in centre_posts[start:]]
    assert (outcome.status, outcome.reason) == ('completed', 'completed'), outcome
    assert files.outcome == 'incomplete' and files.not_written_reason == 'write_batch_disk_full'
    assert posts.count((RunImagesNotSavedError.title, True)) == 1, posts
    assert ('Disk Space Critical', True) not in posts, posts
    assert not [p for p in posts if p[0] == RecordIncompleteError.title and p[1]], posts
    assert not [r for r in caplog.records if '[Task:write_capture]' in r.getMessage()]
    assert not [t for t in darkened_at if t > settled_at[0]], 'the LEDs were commanded after'


def test_a_crashed_run_loop_is_shown_once(tmp_path, monkeypatch, centre_posts):
    """It was latched and logged, and shown to nobody."""
    import modules.protocol_run_loop as prl

    def crashes(self, run):
        raise RuntimeError('the loop died here')

    # A stand-in by design: no simulated fault escapes the loop's own handlers.
    monkeypatch.setattr(prl.ProtocolRunLoop, '_run_loop_inner', crashes)
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        start = len(centre_posts)
        run = runner.start_single_scan(
            protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
        )
        outcome = run.result(timeout=WAIT_S)
        run.files.result(timeout=WAIT_S)

    assert (outcome.status, outcome.reason) == ('failed', 'run_loop_crashed'), outcome
    shown = [n.title for n in centre_posts[start:] if n.shown and n.fatal]
    assert shown == [ENDING_TITLES['run_loop_crashed']], shown


# --- Cleanup reads one ending -------------------------------------------------------


def test_a_cleanup_that_raised_before_its_read_settles_the_fault_not_the_callers_word(
    monkeypatch,
):
    """The files and the outcome read one ending: the outcome took the
    caller's word while the files report read the fault."""
    from tests.guards.test_run_ending import _cleanup_stub

    import modules.sequenced_capture_runner as scr

    stub = _cleanup_stub(latched=_fatal('disk_space_critical'), forced_dark=True)
    # The video drain cleanup waits on raises before the read.
    stub._image_writer = MagicMock(video_busy=True)
    stub._image_writer.wait_for_video_drains.side_effect = OSError('the share is gone')

    try:
        scr.SequencedCaptureRunner._cleanup_inner(stub, STOPPED, stub._last_run(), [])
    except OSError:
        pass

    (settled,) = stub._settle_run_outcome.call_args.args
    assert (settled.status, settled.reason) == ('failed', 'disk_space_critical')


def test_files_lost_to_a_fault_shown_behind_a_stop_are_not_shown_again(centre_posts):
    """The files report said "already told" only when the run's ending was
    the fault; behind a Stop it was shown a second time."""
    from tests.guards.test_run_ending import _cleanup_stub

    import modules.sequenced_capture_runner as scr
    from modules.protocol_image_writer import RunWriteBatch

    stub = _cleanup_stub(latched=STOPPED)
    stub._ending.record_fatal(_fatal('disk_space_critical'))
    stub._disable_saving_artifacts = True
    stub._on_run_idle = None
    batch = RunWriteBatch(MagicMock())
    batch.count_not_written('video_unfinished', 'The video V1')

    files_written = scr.SequencedCaptureRunner._close_run_writes(
        stub, batch, STOPPED, stub._last_run()
    )
    batch.when_complete(files_written)

    assert [n.title for n in centre_posts if n.shown] == []

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build does not start while the scope's own work needs the host.

A build runs on the post-processing lane, a thread of the same process, and
shares the cores, the interpreter lock and the disk with whatever holds the
scope -- a run, a recording, a diagnostic, a home -- and with a finished
run's file writes (#812 measured a folder build halving the capture rate).
So `PostProcessingAPI` refuses a build whose turn comes while something
holds the scope (``'exclusive_activity_running'``, naming the holder) or a
run's files are still being written (``'files_writing'``, or
``'files_writing_stalled'`` with the writer's recovery), for every caller, in
the words a run's start is refused in. It is asked as the build starts, on
the lane: a build queued behind another is refused if a hold began while it
waited.

Nor does a build that writes beside its folder start when that folder's drive
has less free than the floor a run starts above
(``'disk_space_insufficient'``): a video's size is not known before it is
encoded, and one that fills the drive is lost with the time it took; a drive
whose free space cannot be read is refused as a save location that cannot be
used (``'capture_location_unusable'``). The folder is read as there first, so
a missing one is refused as missing, whatever else would refuse the build.

Driven through the Session on the simulated scope.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from _lumascope.modules import common_utils, live_work
from _lumascope.modules.caller import Caller
from _lumascope.modules.cell_count_method import CellCountMethod
from _lumascope.modules.exceptions import (
    FileWriterStalledError,
    Holder,
    LiveFolderPathRefusedError,
    PostProcessingRefusedError,
    RefusalCause,
)
from _lumascope.modules.post_processing_api import Stitch
from _lumascope.modules.sequential_io_executor import IOTask
from _lumascope.modules.scope_session import ScopeSession
from tests.disk_fakes import free_space_unreadable
from tests.protocol_drives import run_identity
from tests.settings_fixtures import complete_settings
from tests.test_run_refusal_contract import _a_closed_batch_still_writing

QUEUED_WAIT_S = 10
FLOOR_BYTES = common_utils.MIN_REQUIRED_DISK_MB * 1024**2


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def test_a_build_asked_while_the_scope_is_held_is_refused_for_its_state(session, tmp_path):
    with session.diagnostic_claim(), pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.stitch(tmp_path)

    assert refused.value.reason == 'exclusive_activity_running'
    assert refused.value.cause is RefusalCause.STATE
    assert refused.value.holder is Holder.DIAGNOSTIC
    assert refused.value.title == 'Microscope Busy'
    assert 'diagnostic' in str(refused.value).lower()


def test_a_build_asked_while_a_run_holds_names_the_run_and_who_started_it(session, tmp_path):
    # The run's claim taken as a run takes it, with no run behind it: a
    # simulated run on the fast tier ends before a build could be asked.
    held = session._activity_claim.try_claim('protocol', run=run_identity(Caller.REST))
    try:
        with pytest.raises(PostProcessingRefusedError) as refused:
            session.post_processing.stitch(tmp_path)
    finally:
        held.release()

    assert refused.value.reason == 'exclusive_activity_running'
    assert (refused.value.holder, refused.value.holder_trigger) == (Holder.PROTOCOL, Caller.REST)


def test_a_build_asked_while_a_runs_files_are_written_is_refused(session, tmp_path):
    # A stand-in by design: a closed batch whose write never runs is the
    # simulator's only drain that stays open; the session's close waits on
    # it, so the runner gets its own batch back.
    runner = session._sequenced_capture_runner
    held, runner._write_batch = runner._write_batch, _a_closed_batch_still_writing()
    try:
        assert session.live_work.has(live_work.WorkKind.RUN_FILES)
        with pytest.raises(PostProcessingRefusedError) as refused:
            session.post_processing.zproject(tmp_path)
    finally:
        runner._write_batch = held

    assert refused.value.reason == 'files_writing'
    assert refused.value.remedy is None


def test_a_build_asked_while_a_runs_writer_is_stalled_is_given_its_recovery(session, tmp_path):
    # A stand-in by design, as above, with its one write stuck past the stall.
    runner = session._sequenced_capture_runner
    stalled = _a_closed_batch_still_writing(stuck_write='frame_0001.tiff')
    held, runner._write_batch = runner._write_batch, stalled
    try:
        with pytest.raises(PostProcessingRefusedError) as refused:
            session.post_processing.zproject(tmp_path)
    finally:
        runner._write_batch = held

    assert refused.value.reason == 'files_writing_stalled'
    assert refused.value.remedy == FileWriterStalledError.recovery(1)
    assert refused.value.title == 'File Writer Stalled'
    assert 'frame_0001.tiff' in str(refused.value)
    assert 'Recover the file writer before building' in str(refused.value)


def test_a_build_queued_before_a_hold_is_refused_when_its_turn_comes(session, tmp_path):
    lane = session.post_processing.lane
    running = threading.Event()
    release = threading.Event()

    def block():
        running.set()
        release.wait(QUEUED_WAIT_S)

    lane.submit(IOTask(action=block), 'test.blocker')
    outcome = {}

    def ask():
        try:
            session.post_processing.stitch(tmp_path)
        except Exception as e:
            outcome['raised'] = e

    caller = threading.Thread(target=ask, name='queued_build')
    try:
        # Off the queue and on the worker before the build is asked, so the
        # one task queued is the build: a worker that woke late left the
        # blocker queued beside it, or alone before it.
        assert running.wait(QUEUED_WAIT_S), 'the blocker never reached the lane'
        caller.start()
        # The lane's queue is the one signal that the build is waiting its turn.
        deadline = time.monotonic() + QUEUED_WAIT_S
        while lane.queue_size() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lane.queue_size() == 1, 'the build never queued behind the blocker'

        with session.diagnostic_claim():
            release.set()
            caller.join(QUEUED_WAIT_S)
    finally:
        # A blocker left waiting holds the lane, and the session's close
        # waits on the lane: a failure here would hang the teardown.
        release.set()

    assert not caller.is_alive()
    assert isinstance(outcome.get('raised'), PostProcessingRefusedError), outcome
    assert outcome['raised'].reason == 'exclusive_activity_running'


def _the_drive_has(monkeypatch, free_bytes: int) -> None:
    # The one probe's own seam (``tests/test_disk_space_helper.py``): a real
    # drive with this little free is not one the suite can make on every host.
    monkeypatch.setattr(
        common_utils.psutil,
        'disk_usage',
        lambda path: SimpleNamespace(total=10**12, used=10**11, free=free_bytes),
    )


def _a_folder_with_an_image(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    tifffile.imwrite(images / 'frame.tif', np.full((64, 64), 100, dtype=np.uint16))
    return images


def _the_builds_that_write_beside_their_folder(session, images):
    pp = session.post_processing
    return {
        'stitch': lambda: pp.stitch(images),
        'zproject': lambda: pp.zproject(images),
        'composite': lambda: pp.composite(images),
        'video': lambda: pp.video(images),
        'build': lambda: pp.build(images, stitch=Stitch()),
        'enhance': lambda: pp.enhance(images / 'frame.tif'),
        'count_cells': lambda: pp.count_cells(images, method=CellCountMethod()),
    }


def test_a_build_on_a_drive_below_the_floor_is_refused_before_it_writes(
    session, tmp_path, monkeypatch
):
    images = _a_folder_with_an_image(tmp_path)
    builds = _the_builds_that_write_beside_their_folder(session, images)
    before = sorted(tmp_path.rglob('*'))
    _the_drive_has(monkeypatch, FLOOR_BYTES - 1)

    wrong = []
    for member, build in builds.items():
        try:
            build()
        except PostProcessingRefusedError as refused:
            if refused.reason != 'disk_space_insufficient':
                wrong.append(f'{member}: refused {refused.reason!r}')
            elif refused.title != 'Not Enough Disk Space' or str(images) not in str(refused):
                wrong.append(f'{member}: told {refused.title!r}: {refused}')
        else:
            wrong.append(f'{member}: was not refused')

    assert wrong == [], '\n'.join(wrong)
    assert sorted(tmp_path.rglob('*')) == before


def test_a_build_on_a_drive_that_cannot_be_read_is_refused_before_it_writes(
    session, tmp_path, monkeypatch
):
    images = _a_folder_with_an_image(tmp_path)
    before = sorted(tmp_path.rglob('*'))
    free_space_unreadable(monkeypatch)

    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.stitch(images)

    assert refused.value.reason == 'capture_location_unusable'
    assert refused.value.title == 'Save Location Unusable'
    assert sorted(tmp_path.rglob('*')) == before


def test_a_build_on_a_drive_at_the_floor_starts(session, tmp_path, monkeypatch):
    images = _a_folder_with_an_image(tmp_path)
    _the_drive_has(monkeypatch, FLOOR_BYTES)

    # Past the door, a folder of one still holds no recording to make a video of.
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.video(images)

    assert refused.value.reason != 'disk_space_insufficient'


def test_a_graph_is_drawn_whatever_the_results_drive_has_free(session, tmp_path, monkeypatch):
    # Its image goes to the path it is given, which may be on another drive.
    results = tmp_path / 'results.csv'
    results.write_text('file,count,mean_area\na.tif,3,1.5\nb.tif,5,2.5\n')
    _the_drive_has(monkeypatch, 0)

    drawn = session.post_processing.graph(results, tmp_path / 'g.png', x='count', y='mean_area')

    assert drawn.path.is_file()


def test_a_missing_folder_is_refused_as_missing_while_the_scope_is_held(session, tmp_path):
    with session.diagnostic_claim(), pytest.raises(LiveFolderPathRefusedError) as refused:
        session.post_processing.stitch(tmp_path / 'not_there')

    assert refused.value.reason == 'path_not_found'

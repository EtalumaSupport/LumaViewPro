"""A capture into a live folder that is not there is refused, naming the folder.

The capture-location owner (`src/_lumascope/modules/path_utils.py`) treats a missing live
folder as an unplugged drive or a stale path: it refuses rather than building
a fresh tree on whatever is mounted there. Two writers did not ask it. A
manual still created the whole tree itself, so a drive that was gone became a
bare OSError, shown as the generic "did not complete" sentence; a recording
probed free disk space on the missing folder first, so the same drive became
an untyped FileNotFoundError ahead of the recording's own refusal.
"""

import os
import pathlib

import pytest

from _lumascope.modules import path_utils
from _lumascope.modules.exceptions import (
    LiveFolderPathRefusedError,
    RecordingRefusedError,
    RefusalCause,
)
from _lumascope.modules.notification_center import OutcomeKind, outcome_of
from tests.test_manual_capture_overlay_save import (  # noqa: F401 -- the fixture is used by name
    LAYER,
    capture_ctx,
)
from tests.test_manual_recording_controller import make_controller
from _lumascope.modules.channel_record import release_colors

# The permission bits are what the owner reads; Windows keeps no such bits on
# a folder, and root reads and writes through them.
_permission_bits_decide = pytest.mark.skipif(
    os.name == 'nt' or os.geteuid() == 0, reason='needs POSIX permission bits that bind'
)


@pytest.fixture
def locked():
    """Set a folder's mode for the test, and give every folder back its own after."""
    taken = []

    def lock(folder: pathlib.Path, mode: int) -> None:
        taken.append((folder, folder.stat().st_mode))
        os.chmod(folder, mode)

    yield lock
    for folder, mode in reversed(taken):
        os.chmod(folder, mode)


class TestTheLocationOwnerAnswers:
    def test_an_existing_folder_is_the_location(self, tmp_path):
        assert path_utils.require_capture_location(str(tmp_path)) == tmp_path

    def test_a_missing_folder_is_refused_by_name(self, tmp_path):
        gone = tmp_path / 'unplugged' / 'capture'
        with pytest.raises(path_utils.CaptureLocationError, match=str(gone)) as refused:
            path_utils.require_capture_location(gone)
        # An unplugged drive comes back: the same capture may succeed later.
        outcome = outcome_of(refused.value)
        assert (outcome.kind, outcome.cause, outcome.reason) == (
            OutcomeKind.REFUSAL,
            RefusalCause.STATE,
            'capture_location_unusable',
        )

    def test_a_file_is_not_a_location(self, tmp_path):
        a_file = tmp_path / 'capture'
        a_file.write_text('')
        with pytest.raises(path_utils.CaptureLocationError, match=str(a_file)):
            path_utils.require_capture_location(a_file)

    @_permission_bits_decide
    def test_a_folder_under_one_that_cannot_be_read_is_refused_by_name(self, tmp_path, locked):
        # The look itself fails here: is_dir raises PermissionError.
        live = tmp_path / 'share' / 'capture'
        live.mkdir(parents=True)
        locked(live.parent, 0o000)
        with pytest.raises(path_utils.CaptureLocationError, match=str(live)) as refused:
            path_utils.require_capture_location(live)
        assert refused.value.reason == 'capture_location_unusable'

    @_permission_bits_decide
    def test_a_folder_that_cannot_be_written_into_is_refused_by_name(self, tmp_path, locked):
        live = tmp_path / 'capture'
        live.mkdir()
        locked(live, 0o500)
        with pytest.raises(path_utils.CaptureLocationError, match=str(live)) as refused:
            path_utils.require_capture_location(live)
        assert refused.value.reason == 'capture_location_unusable'


def test_a_manual_still_into_a_missing_folder_is_refused_before_the_grab(capture_ctx):
    from _lumascope.modules.manual_capture import ManualCaptureController
    from tests.settings_fixtures import complete_settings

    gone = pathlib.Path(capture_ctx.tmp_path) / 'unplugged' / 'capture'
    settings = complete_settings(
        live_folder=str(gone), separate_folder_per_channel=False, image_mode='8bit'
    )
    settings[LAYER].update({'exposure_ms': 100, 'sum': 1, 'illumination_ma': 0})
    capture = ManualCaptureController(
        on_saved=lambda folder: None,
        scope=capture_ctx.scope,
        settings_snapshot=lambda: settings,
        shown_channel=lambda: LAYER,
        engineering_mode=lambda: False,
        channel_colors=release_colors,
    )

    future = capture.capture(bullseye=False, crosshairs=False)
    with pytest.raises(LiveFolderPathRefusedError) as refused:
        future.result(timeout=10)

    assert refused.value.reason == 'capture_location_unusable'
    assert str(gone) in str(refused.value)
    assert not gone.exists()
    capture_ctx.scope.imaging.capture_holding_channel_on_lane.assert_not_called()


def test_a_recording_into_a_missing_folder_is_refused_before_the_disk_probe(
    tmp_path, post_processing
):
    gone = tmp_path / 'unplugged' / 'capture'
    controller, scope, _ = make_controller(gone, post_processing=post_processing)

    with pytest.raises(RecordingRefusedError) as refused:
        controller.start()

    assert refused.value.reason == 'capture_location_unusable'
    assert str(gone) in str(refused.value)
    assert scope.imaging.listener is None
    assert not gone.exists()

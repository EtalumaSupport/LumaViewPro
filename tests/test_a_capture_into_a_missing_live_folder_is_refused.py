"""A capture into a live folder that is not there is refused, naming the folder.

The capture-location owner (`modules/path_utils.py`) treats a missing live
folder as an unplugged drive or a stale path: it refuses rather than building
a fresh tree on whatever is mounted there. Two writers did not ask it. A
manual still created the whole tree itself, so a drive that was gone became a
bare OSError, shown as the generic "did not complete" sentence; a recording
probed free disk space on the missing folder first, so the same drive became
an untyped FileNotFoundError ahead of the recording's own refusal.
"""

import pathlib

import pytest

from modules import path_utils
from modules.exceptions import CaptureError, RecordingRefusedError
from tests.test_manual_capture_overlay_save import (  # noqa: F401 -- the fixture is used by name
    LAYER,
    capture_ctx,
)
from tests.test_manual_recording_controller import make_controller


class TestTheLocationOwnerAnswers:
    def test_an_existing_folder_is_the_location(self, tmp_path):
        assert path_utils.require_capture_location(str(tmp_path)) == tmp_path

    def test_a_missing_folder_is_refused_by_name(self, tmp_path):
        gone = tmp_path / 'unplugged' / 'capture'
        with pytest.raises(path_utils.CaptureLocationError, match=str(gone)):
            path_utils.require_capture_location(gone)

    def test_a_file_is_not_a_location(self, tmp_path):
        a_file = tmp_path / 'capture'
        a_file.write_text('')
        with pytest.raises(path_utils.CaptureLocationError, match=str(a_file)):
            path_utils.require_capture_location(a_file)


def test_a_manual_still_into_a_missing_folder_is_refused_before_the_grab(capture_ctx):
    from modules.manual_capture import ManualCaptureController
    from tests.settings_fixtures import complete_settings

    gone = pathlib.Path(capture_ctx.tmp_path) / 'unplugged' / 'capture'
    settings = complete_settings(
        live_folder=str(gone), separate_folder_per_channel=False, image_mode='8bit'
    )
    settings[LAYER].update({'exposure_ms': 100, 'sum': 1, 'illumination_ma': 0})
    capture = ManualCaptureController(
        scope=capture_ctx.scope, settings_snapshot=lambda: settings, engineering_mode=lambda: False
    )

    future = capture.capture(layer=LAYER, false_color_on=False, bullseye=False, crosshairs=False)
    with pytest.raises(CaptureError) as refused:
        future.result(timeout=10)

    assert refused.value.reason == 'capture_location_unusable'
    assert str(gone) in str(refused.value)
    assert not gone.exists()
    capture_ctx.scope.imaging._capture_and_wait_impl.assert_not_called()


def test_a_recording_into_a_missing_folder_is_refused_before_the_disk_probe(tmp_path):
    gone = tmp_path / 'unplugged' / 'capture'
    controller, scope, _ = make_controller(gone)

    with pytest.raises(RecordingRefusedError) as refused:
        controller.start()

    assert refused.value.reason == 'capture_location_unusable'
    assert str(gone) in str(refused.value)
    assert scope.imaging.listener is None
    assert not gone.exists()

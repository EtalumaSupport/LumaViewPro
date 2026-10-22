# Copyright Etaluma, Inc.
"""Regression test: the per-step CAPTURE DIAG line reads no camera.

The #610 capture diagnostic compares the step's intended gain/exposure
against the camera's. Its reads once fired on every protocol step, even
with the line dropped, because the f-string arguments evaluate before
logger.debug decides; they were then gated on debug. The line now logs the
values the camera holds, so no step reads the camera for it, debug or not.

Driven behaviorally: capture() runs once with debug on (a stub logger
controls isEnabledFor), counting the fresh camera reads.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from tests.protocol_drives import lent_run_claim
from tests.frame_records import frame_record, plate
from modules.protocol_image_writer import RunWriteBatch
from modules.image_mode import ImageCaptureConfig
from modules.protocol_image_writer import ProtocolImageWriter
from modules.run_events import RunEvents


from modules.run_outcome import EndingLatch
from modules.layer_record import release_colors


def _drive_capture(monkeypatch, debug_enabled):
    writer = ProtocolImageWriter(
        scope=MagicMock(),
        events=RunEvents(),
        aborted=threading.Event(),
        write_batch=RunWriteBatch(MagicMock()),
        abort_fn=lambda: None,
        end_run_fatally=lambda reason, message: None,
        fatal_abort_event=threading.Event(),
        ending=EndingLatch(),
        execution_record=None,
        leds_off_fn=lambda: None,
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        timestamp_overlay=True,
        video_max_fps=0,
        engineering_mode=False,
        run_claim=lent_run_claim(),
        labware=plate(),
        to_plate=None,
        captures_asked=1,
        channel_colors=release_colors(),
    )
    scope = writer._scope
    # The objective the frame is taken with, read at capture.
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
    scope.capabilities.has_turret = False
    scope.led_connected = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
    # A capture hands on its frame's record; a stub's would answer every fact
    # it carries (blown, dark) with a truthy stand-in. A run's capture
    # always measures the frame's saturation.
    scope.imaging.last_capture_info = {'frame_record': frame_record(saturated_fraction=0.0)}

    def quiet(*args, **kwargs):
        return None

    monkeypatch.setattr(
        'modules.protocol_image_writer.logger',
        SimpleNamespace(
            isEnabledFor=lambda level: debug_enabled,
            debug=quiet,
            info=quiet,
            warning=quiet,
            error=quiet,
            exception=quiet,
            critical=quiet,
        ),
    )
    protocol = MagicMock()
    protocol.capture_root.return_value = ''
    writer.capture(
        save_folder='/tmp',
        step={
            'Name': 'stepA',
            'Label': '',
            'Acquire': 'image',
            'Auto_Gain': False,
            'Color': 'BF',
            'Gain': 2.0,
            'Exposure': 10.0,
            'Objective': '4x',
            'Well': 'A1',
            'Z-Slice': 0,
            'Tile': '',
            'Illumination': 50.0,
            'False_Color': False,
        },
        output_format='TIFF',
        protocol=protocol,
        enable_image_saving=True,
    )
    return scope.camera


def test_the_diagnostic_reads_no_camera_with_debug_on(monkeypatch):
    camera = _drive_capture(monkeypatch, debug_enabled=True)
    assert camera.read_gain_db.call_count == 0, 'the diagnostic read the gain from the camera'
    assert camera.read_exposure_ms.call_count == 0, (
        'the diagnostic read the exposure from the camera'
    )

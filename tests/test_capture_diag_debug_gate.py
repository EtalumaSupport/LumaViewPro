# Copyright Etaluma, Inc.
"""Regression test: the per-step CAPTURE DIAG camera reads are debug-gated.

The #610 capture diagnostic compares the step's intended gain/exposure
against the camera's ACTUAL (live) values, so the reads must stay live --
but they were firing on every protocol step even in normal operation,
because the f-string arguments evaluate before logger.debug decides to drop
the line. The two live SDK reads are now gated on debug being enabled.

Driven behaviorally: capture() runs once with debug off and once with debug
on (a stub logger controls isEnabledFor), counting the live driver reads.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from modules.protocol_callbacks import ProtocolCallbacks
from modules.protocol_image_writer import ProtocolImageWriter


def _drive_capture(monkeypatch, debug_enabled):
    writer = ProtocolImageWriter(
        scope=MagicMock(),
        callbacks=ProtocolCallbacks(),
        aborted=threading.Event(),
        file_io_executor=MagicMock(),
        abort_fn=lambda: None,
        execution_record=None,
        leds_off_fn=lambda: None,
        led_on_fn=lambda **kw: None,
        is_run_in_progress_fn=lambda: True,
    )
    scope = writer._scope
    scope.motion.has_turret.return_value = False
    scope.led_connected = False
    scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)

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
        image_capture_config={'capture_depth': 8},
        enable_image_saving=True,
    )
    return scope.imaging


def test_camera_reads_skipped_when_debug_disabled(monkeypatch):
    imaging = _drive_capture(monkeypatch, debug_enabled=False)
    assert imaging.get_gain.call_count == 0, (
        'the diagnostic live gain read must not run when debug is off'
    )
    assert imaging.get_exposure_time.call_count == 0, (
        'the diagnostic live exposure read must not run when debug is off'
    )


def test_camera_reads_run_when_debug_enabled(monkeypatch):
    imaging = _drive_capture(monkeypatch, debug_enabled=True)
    assert imaging.get_gain.call_count == 1, (
        'with debug on, the diagnostic must read the live gain '
        '(comparing intended vs actual is the whole point)'
    )
    assert imaging.get_exposure_time.call_count == 1, (
        'with debug on, the diagnostic must read the live exposure'
    )

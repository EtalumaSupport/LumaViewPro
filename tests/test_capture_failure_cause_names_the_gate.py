# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture the chunk gate rejects is reported as that, not as a dead
camera.

Bug shape: the capture returned None for a rejected frame exactly as it
did for an inactive camera, and the protocol writer's cause ladder fell
through to "camera inactive or not grabbing" for both. A field bundle in
which every auto-gain capture had been rejected read as a camera that
delivered nothing, and the investigation started on the wrong hardware.
The capture now names the rejected source in its evidence record and the
writer's ladder reads it before falling through.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from tests import ast_seams
from tests.test_auto_gain_lock import _build

WRITER = 'modules/protocol_image_writer.py'
IMAGING = 'modules/lumascope_api/imaging.py'


def test_rejected_capture_names_the_source():
    """A stale exposure target with no auto-gain arm: the gate rejects the
    frame and the evidence record says which source never matched."""
    imaging, cam = _build(ae_lands_on_ms=62.0)
    imaging._set_exposure_ms_impl(100.0)
    cam._exposure_us = 62000.0  # the camera moved on its own; the target did not
    assert imaging._capture_and_wait_impl(timeout_s=1.0) is None
    info = imaging.last_capture_info
    assert info['chunk_rejected'] == 'exposure'
    assert not info.get('deadline_expired') and not info.get('drain_failed')


def test_cause_ladder_reads_the_rejection_before_the_fallback():
    """One ladder for every reader of a failed capture's evidence record:
    the protocol writer and the manual path name the same cause."""
    ladder = ast_seams.find_def(IMAGING, 'capture_failure_cause')
    assert ladder is not None
    source = ast.unparse(ladder)
    rejected = source.index("info.get('chunk_rejected')")
    fallback = source.index("'camera inactive or not grabbing'")
    assert rejected < fallback, 'the rejection branch must come before the inactive fallback'
    capture = ast_seams.find_def(WRITER, 'capture', class_name='ProtocolImageWriter')
    assert 'capture_failure_cause' in ast.unparse(capture)
    assert "'camera inactive or not grabbing'" not in ast.unparse(capture)


def test_manual_capture_that_saves_nothing_tells_the_user_why(tmp_path):
    """A manual capture the gate rejects saved nothing and said nothing: the
    save returned None and the button re-enabled, while the composite path
    notified for the identical rejection. On a field unit three manual
    captures in a row were rejected against a stale target with no sign to
    the user. The notice comes from the API's save, naming the cause the
    evidence record carries, so the GUI and a REST caller see the same
    failure."""
    from modules import image_save

    imaging, cam = _build(ae_lands_on_ms=62.0)
    imaging._set_exposure_ms_impl(100.0)
    cam._exposure_us = 62000.0
    with patch('modules.image_save.notifications') as notifications:
        out = image_save.save_live_image(
            imaging._scope,
            save_folder=tmp_path,
            file_root='live_',
            timeout_s=1.0,
            channel='BF',
            false_color_on=False,
            save_encoding='raw',
        )
    assert out is None
    assert notifications.error.call_count == 1
    message = notifications.error.call_args.args[2]
    assert 'exposure' in message and 'never matched' in message

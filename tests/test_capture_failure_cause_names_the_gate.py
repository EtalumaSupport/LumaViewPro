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

from tests import ast_seams
from tests.test_auto_gain_lock import _build

WRITER = 'modules/protocol_image_writer.py'


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


def test_writer_ladder_reads_the_rejection_before_the_fallback():
    capture = ast_seams.find_def(WRITER, 'capture', class_name='ProtocolImageWriter')
    assert capture is not None
    source = ast.unparse(capture)
    rejected = source.index("info.get('chunk_rejected')")
    fallback = source.index("'camera inactive or not grabbing'")
    assert rejected < fallback, 'the rejection branch must come before the inactive fallback'

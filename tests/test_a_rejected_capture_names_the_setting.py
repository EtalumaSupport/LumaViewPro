# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture the chunk gate rejects is reported as that, not as a dead
camera.

Bug shape: the capture failed a rejected frame exactly as it did an
inactive camera, and the protocol writer's cause fell through to "camera
inactive or not grabbing" for both. A field bundle in which every auto-gain
capture had been rejected read as a camera that delivered nothing, and the
investigation started on the wrong hardware. The capture now raises
``chunk_rejected`` naming the setting no frame matched.
"""

from __future__ import annotations

import pytest

from modules.exceptions import CaptureFailedError
from tests.test_auto_gain_lock import _build


def test_rejected_capture_names_the_setting():
    """A stale exposure target with no auto-gain arm: the gate rejects every
    frame and the failure names the setting that never matched."""
    imaging, cam = _build(ae_lands_on_ms=62.0)
    imaging._set_exposure_ms_impl(100.0)
    cam._exposure_us = 62000.0  # the camera moved on its own; the target did not

    with pytest.raises(CaptureFailedError) as failed:
        imaging._capture_and_wait_impl(timeout_s=1.0)

    assert (failed.value.reason, failed.value.setting) == ('chunk_rejected', 'exposure')
    assert 'exposure' in str(failed.value)

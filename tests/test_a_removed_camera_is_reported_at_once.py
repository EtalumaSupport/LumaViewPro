# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A grab that fails because the camera was removed is reported at once.

A removal sets the driver's removal latch immediately and releases the SDK
handle later, off-thread, so ``active`` stays truthy in between. A grab
failure in that window is a removal, and ``get_image`` says so instead of
retrying for its whole timeout.
"""

import time

from modules.notification_center import Severity, notifications
from tests.scope_fakes import build_scope


def test_a_grab_after_the_removal_latch_returns_at_once_and_says_so():
    scope = build_scope(simulate=True)
    scope.imaging.start_streaming()
    time.sleep(0.3)
    driver = scope._camera_driver

    posted = []

    def _listen(n):
        posted.append(n.title)

    notifications.add_listener(_listen, min_severity=Severity.DEBUG)
    try:
        driver._mark_disconnected()
        assert driver.active, 'the handle is released later, by disconnect()'

        t0 = time.monotonic()
        image = scope.imaging.get_image()
        elapsed = time.monotonic() - t0
    finally:
        notifications.remove_listener(_listen)

    assert image is None
    assert elapsed < 1.0, f'get_image retried for {elapsed:.2f}s after the removal'
    assert 'Camera Disconnected' in posted

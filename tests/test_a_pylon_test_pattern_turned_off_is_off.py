# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler's test pattern turned off is off, not Black.

`Camera.set_test_pattern` writes the `TestPattern` node: the named entry
when enabled, `'Off'` when not. A daA3840 offers Off / Black / White /
Testimage1-3 (bench-read, camera P24); Pylon's own body wrote the pattern
whatever `enabled` said, so turning the pattern off turned Black on.
"""

from tests.camera_fakes import WriteRecorderNode, bare_pylon_camera


def test_a_pattern_turned_on_and_then_off_leaves_the_node_off():
    cam = bare_pylon_camera()
    cam.active.TestPattern = WriteRecorderNode()

    cam.set_test_pattern(True, 'Black')
    cam.set_test_pattern(False)

    assert cam.active.TestPattern.writes == ['Black', 'Off']

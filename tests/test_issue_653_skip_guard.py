"""Regression for #653 residual: OnImagesSkipped must guard on device-removed.

On a USB camera disconnect the SDK fires a burst of OnImagesSkipped
callbacks during teardown. Those frames were dropped by the removal, not by
LatestImageOnly grab-strategy pressure, so logging them is misleading noise
(Rule 20) -- the bench saw ~16 stray "frames discarded" lines after the
device was already marked removed. OnImageGrabbed already early-returns on
self._parent._device_removed; OnImagesSkipped must do the same.

ImageHandler subclasses the pypylon SDK handler (a MagicMock base under
conftest), so it can't be instantiated in unit tests -- the codebase tests
these callbacks by source assertion (cf. test_audit_fixes OnImageGrabbed
tests). Same approach here.
"""

import ast
import pathlib

PYLON_SRC = pathlib.Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py'


def _function_body(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f'{name} not found in {PYLON_SRC}')


def test_on_images_skipped_guards_on_device_removed():
    body = _function_body(PYLON_SRC.read_text(encoding='utf-8'), 'OnImagesSkipped')
    assert '_device_removed' in body, (
        'OnImagesSkipped must early-return on self._parent._device_removed so '
        'teardown skip-bursts are not logged as LatestImageOnly drops (#653)'
    )
    # The guard must precede the info log so removal-time skips are silenced.
    guard_pos = body.index('_device_removed')
    log_pos = body.index('OnImagesSkipped:')  # the info-log message literal
    assert guard_pos < log_pos, 'the device-removed guard must come before the skip log'

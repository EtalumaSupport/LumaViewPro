# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A removed camera is torn down by one implementation, the base Camera's.

Each driver learns of a removal its own way (an SDK callback, a presence
probe, silence on the wire), but what follows is the same for all of them:
disconnect() on a thread of its own, once, so the thread that noticed can
return. Two copies of that teardown had already drifted: one re-armed after
it ran and one did not, and one logged a failed teardown at DEBUG, out of
sight.
"""

import ast
import logging
import pathlib
import time
from unittest.mock import MagicMock

from tests.camera_fakes import bare_ids_camera
from tests.log_capture import capture_module_log

REPO = pathlib.Path(__file__).resolve().parent.parent


def _wait_until(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_only_the_base_camera_defines_the_removal_teardown():
    definers = []
    for path in sorted((REPO / 'drivers').rglob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_schedule_async_teardown':
                definers.append(path.relative_to(REPO).as_posix())
    assert definers == ['drivers/camera.py']


def test_a_teardown_that_fails_is_logged_as_a_warning(monkeypatch):
    import drivers.camera as camera_module

    # The camera log, which reaches camera.log and the errors log.
    records = capture_module_log(monkeypatch, camera_module)
    monkeypatch.setattr(camera_module, '_cam_log', camera_module.logger)
    cam = bare_ids_camera()
    cam.disconnect = MagicMock(side_effect=RuntimeError('the handle is already gone'))

    cam._schedule_async_teardown()

    assert _wait_until(lambda: any(r.levelno == logging.WARNING for r in records))
    warning = next(r for r in records if r.levelno == logging.WARNING)
    assert 'the handle is already gone' in warning.getMessage()


def test_a_camera_built_by_its_own_constructor_can_be_torn_down():
    # A driver that seeds none of the teardown's state itself still gets it
    # from the base, so a real removal never meets a missing latch.
    from drivers.simulated_camera import SimulatedCamera

    cam = SimulatedCamera()
    cam.disconnect = MagicMock()

    cam._schedule_async_teardown()

    assert _wait_until(lambda: cam.disconnect.call_count == 1)


def test_the_teardown_can_run_again_once_it_has_finished():
    cam = bare_ids_camera()
    cam.disconnect = MagicMock()

    cam._schedule_async_teardown()
    assert _wait_until(lambda: cam.disconnect.call_count == 1)
    assert _wait_until(lambda: cam._async_teardown_started is False)

    cam._schedule_async_teardown()
    assert _wait_until(lambda: cam.disconnect.call_count == 2)

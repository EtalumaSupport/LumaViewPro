# Copyright Etaluma, Inc.
"""A camera driver marks its camera removed only on definite evidence it is gone.

A removal has no reset short of a reconnect: frames stop, every later camera
command is refused, a run ends. So it is decided only where the evidence is
definite -- an SDK's removal callback, a grab result saying the device is not
found, a consecutive-failure count, a released handle -- and never from one
caught exception, which the Pylon setters and grab path once did on their first
RuntimeException (#726). Three rules over every driver:

- no ``_mark_disconnected()`` or ``_handle_device_lost()`` call inside an
  ``except`` body: the bug's own shape, removal decided from a caught error;
- the ``_mark_disconnected()`` calls are the ones listed below, each with its
  evidence, so a new site is added here, with its evidence, in the commit
  that adds it;
- the removal flag is set only by ``Camera._mark_disconnected``, which takes
  the state lock and logs the edge.
"""

import ast
from collections import Counter

from tests.ast_seams import iter_package_modules

_LATCHES = frozenset({'_mark_disconnected', '_handle_device_lost'})

# (driver file, class, method) -> calls, each with the evidence it acts on.
_DEFINITE_SITES = {
    # The SDK's removal callback.
    ('drivers/pyloncamera.py', '_CameraRemovalHandler', 'OnCameraDeviceRemoved'): 1,
    # A grab result of DEVICE_NOT_FOUND; a callback finding the handle released.
    ('drivers/pyloncamera.py', 'ImageHandler', 'OnImageGrabbed'): 2,
    # MAX_CONSECUTIVE_FAILURES failed grabs in a row.
    ('drivers/pyloncamera.py', '_PylonImageGrabWorker', '_count_failed_grab'): 1,
    # Re-asserts a removal already latched, or a handle already released.
    ('drivers/pyloncamera.py', 'PylonCamera', 'is_connected'): 2,
    # The handle already released.
    ('drivers/idscamera.py', 'IDSCamera', 'is_connected'): 1,
    # IDS's one removal owner: DeviceLost, the presence probe, the stall ceilings.
    ('drivers/idscamera.py', 'IDSCamera', '_handle_device_lost'): 1,
    # The FX2 grab loop's end of the stream: an unplug the bus confirmed,
    # or a stream silent past the ceiling, as IDS's stall ceiling.
    ('drivers/fx2driver.py', 'FX2Camera', '_on_stream_ended'): 1,
    # The simulated camera leaving its bus.
    ('drivers/simulated_camera.py', 'SimulatedCamera', 'leave_bus'): 1,
}


def _walk(path, node, cls, fn, in_except):
    for child in ast.iter_child_nodes(node):
        c, f, e = cls, fn, in_except
        if isinstance(child, ast.ClassDef):
            c, f, e = child.name, None, False
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f, e = child.name, False
        elif isinstance(child, ast.ExceptHandler):
            e = True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in _LATCHES
        ):
            yield path, c, f, child.func.attr, e
        yield from _walk(path, child, c, f, e)


def _calls(packages=('drivers',)):
    """Yield (path, class, method, called name, inside an except body)."""
    for path, tree in iter_package_modules(packages):
        yield from _walk(path, tree, None, None, False)


def test_no_removal_is_decided_inside_an_except():
    offenders = [
        f'{path} {cls}.{fn}: {name}() inside an except body'
        for path, cls, fn, name, in_except in _calls()
        if in_except
    ]
    assert not offenders, '\n'.join(offenders)


def test_every_removal_site_is_listed_with_its_evidence():
    found = Counter(
        (path, cls, fn) for path, cls, fn, name, _ in _calls() if name == '_mark_disconnected'
    )
    found.pop(('drivers/camera.py', 'Camera', '_mark_disconnected'), None)
    assert dict(found) == _DEFINITE_SITES


def test_only_the_helper_sets_the_removal_flag():
    offenders = []
    for path, tree in iter_package_modules(('drivers',)):
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (path, fn.name) == ('drivers/camera.py', '_mark_disconnected'):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                sets_flag = any(
                    isinstance(t, ast.Attribute) and t.attr == '_device_removed'
                    for t in node.targets
                )
                if sets_flag and isinstance(node.value, ast.Constant) and node.value.value is True:
                    offenders.append(f'{path}:{node.lineno} {fn.name}')
    assert not offenders, '\n'.join(offenders)

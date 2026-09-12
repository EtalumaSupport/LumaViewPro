# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera-executor task may not call a camera setter that re-dispatches.

Bug
---
``ProtocolSettings``-style handlers hand work to the camera worker by putting
an ``IOTask`` on ``camera_executor``. The binning selector put the PUBLIC
``imaging.set_binning_size`` there. That setter routes through
``_dispatch_camera``, which enqueues onto the camera lane and blocks on the
future -- so a task already running on that lane waited for a queue it was
itself holding. Every binning apply died on the 30s geometry timeout without
reaching the camera, the failure callback rewrote the selector text, that
re-entered the handler, and one user selection became an endless cycle of
30-second timeouts. Nothing was wrong with the driver and nothing said so:
the popup blamed the camera.

The sibling pixel-format apply already binds ``_set_pixel_format_impl`` and
carries a comment about the same hazard, so the pattern was known and this
one site missed it.

Test approach
-------------
Both halves are derived from the source, not listed here, so the guard cannot
drift: the re-dispatching setters are whichever ImagingAPI methods call
``_dispatch_camera``, and the call sites are every ``camera_executor.put``
in ``ui/``. Listing either by hand would leave a new setter or a new call
site unguarded, which is exactly how this one arrived.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
IMAGING_SRC = REPO / 'modules' / 'lumascope_api' / 'imaging.py'


def _redispatching_setters() -> set[str]:
    """ImagingAPI methods that enqueue onto the camera lane and wait."""
    tree = ast.parse(IMAGING_SRC.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != '_dispatch_camera':
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and ast.unparse(call.func).endswith(
                    '_dispatch_camera'
                ):
                    names.add(node.name)
                    break
    return names


def _camera_task_actions() -> list[tuple[str, int, str]]:
    """(file, line, action expression) for every camera_executor.put(IOTask(...))."""
    found = []
    for path in sorted((REPO / 'ui').glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not ast.unparse(node.func).endswith('camera_executor.put'):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, ast.Call) and ast.unparse(arg.func).endswith('IOTask'):
                    action = next(
                        (ast.unparse(k.value) for k in arg.keywords if k.arg == 'action'), None
                    )
                    if action is None and arg.args:
                        action = ast.unparse(arg.args[0])
                    if action:
                        found.append((str(path.relative_to(REPO)), node.lineno, action))
    return found


def test_the_setter_census_is_not_empty():
    # A guard whose census silently collapses to nothing passes forever. If
    # the dispatch helper is renamed, this fails instead of going quiet.
    setters = _redispatching_setters()
    assert 'set_binning_size' in setters, (
        f'expected the binning setter among the re-dispatching ones; got {sorted(setters)}'
    )


def test_the_call_site_census_is_not_empty():
    sites = _camera_task_actions()
    assert sites, 'found no camera_executor.put(IOTask(...)) sites -- the walk broke'


def test_no_camera_task_binds_a_redispatching_setter():
    setters = _redispatching_setters()
    offenders = [
        f'{path}:{line} action={action} -- bind the _impl; this runs ON the camera lane'
        for path, line, action in _camera_task_actions()
        if action.rsplit('.', 1)[-1] in setters
    ]
    assert offenders == [], '\n'.join(offenders)

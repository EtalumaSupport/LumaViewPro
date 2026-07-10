# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: camera-driver warning/error/exception sites route to
the camera log (not the parent lvp_logger).

Background -- `drivers/pyloncamera.py`, `drivers/camera.py`, and
`drivers/idscamera.py` all `from lvp_logger import logger`. That bare
`logger` resolves to the lvp_logger module logger, whose handlers
include the main log + errors log but NOT `camera_file_handler`. The
dedicated `_cam_log` alias (`lvp_logger.camera_logger`, name
`LVP.camera`) propagates to camera.log + errors.log.

Before this fix, ~180 `logger.warning/error/exception` call sites in
the three driver files routed camera failures away from camera.log,
defeating the dedicated log's purpose. The canonical exemplar is
`modules/autofocus_runner.py:24` which uses
`_af_log = logging.getLogger('LVP.autofocus')`.

This test is AST-based against the source files so it survives
whitespace / argument changes and runs without hardware. It catches
any future reintroduction of bare `logger.warning|error|exception(`
in the three driver files.

`logger.info(` and `logger.debug(` are intentionally NOT covered --
state transitions belong in the main log per the lvp_logger.py
docstring + Rule 5.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_FILES = (
    REPO_ROOT / 'drivers' / 'pyloncamera.py',
    REPO_ROOT / 'drivers' / 'camera.py',
    REPO_ROOT / 'drivers' / 'idscamera.py',
)
BANNED_LEVELS = ('warning', 'error', 'exception')


def _find_banned_logger_calls(source: str) -> list[tuple[int, str]]:
    """Return list of (line_number, level) for every bare `logger.<level>(...)`
    call in the source where level is in BANNED_LEVELS.

    Bare means the receiver identifier is exactly `logger` (not
    `self.logger`, not `something_logger`). Matches the actual
    `from lvp_logger import logger` import shape.
    """
    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        receiver = func.value
        if not isinstance(receiver, ast.Name):
            continue
        if receiver.id != 'logger':
            continue
        if func.attr in BANNED_LEVELS:
            hits.append((node.lineno, func.attr))
    return hits


@pytest.mark.parametrize('driver_path', DRIVER_FILES, ids=lambda p: p.name)
def test_camera_driver_routes_warning_error_exception_to_cam_log(driver_path):
    """No bare `logger.warning|error|exception(` sites may remain in the
    camera driver files. They must route through `_cam_log` so failures
    land in camera.log (plus errors.log via the shared handler)."""
    source = driver_path.read_text()
    hits = _find_banned_logger_calls(source)
    assert hits == [], (
        f'{driver_path.name} contains {len(hits)} bare logger.<level>(...) '
        f'sites that should route through _cam_log instead: {hits[:10]}'
    )


def test_camera_driver_modules_define_cam_log_alias():
    """Each driver module must bind `_cam_log` at module scope so the
    routed call sites resolve. Defensive against a future commit
    removing the alias without removing its users."""
    import importlib

    for module_name in ('drivers.pyloncamera', 'drivers.camera', 'drivers.idscamera'):
        module = importlib.import_module(module_name)
        assert getattr(module, '_cam_log', None) is not None, (
            f'{module_name} is missing the _cam_log alias but contains '
            f'_cam_log.* call sites. Add the alias to keep the routing intact.'
        )


def test_camera_cam_log_import_fallback_binds_logger_not_none():
    """camera.py's module-level `try: import camera_logger / except ImportError`
    must fall back to the general `logger`, never None.

    The `_cam_log.warning|error|exception` call sites in camera.py are
    UNGUARDED (no `if _cam_log is not None`), so a None fallback turns every
    one into an AttributeError the moment the dedicated camera logger is
    unavailable. Source-based (not a module reload) so it stays deterministic
    and free of reload side-effects on the shared Camera class.
    """
    source = (REPO_ROOT / 'drivers' / 'camera.py').read_text()
    tree = ast.parse(source)

    fallback_values = []
    for node in tree.body:  # module scope only -- ignore the guarded local re-import
        if not isinstance(node, ast.Try):
            continue
        imports_cam_log = any(
            isinstance(stmt, ast.ImportFrom)
            and any(alias.asname == '_cam_log' for alias in stmt.names)
            for stmt in node.body
        )
        if not imports_cam_log:
            continue
        for handler in node.handlers:
            for stmt in handler.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == '_cam_log' for t in stmt.targets
                ):
                    fallback_values.append(stmt.value)

    assert fallback_values, 'no module-level `_cam_log` ImportError fallback found in camera.py'
    for value in fallback_values:
        assert not (isinstance(value, ast.Constant) and value.value is None), (
            'camera.py binds `_cam_log = None` on ImportError; the unguarded '
            'call sites would raise AttributeError. Fall back to `logger`.'
        )
        assert isinstance(value, ast.Name) and value.id == 'logger', (
            'the `_cam_log` ImportError fallback must bind the general `logger`'
        )

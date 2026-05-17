# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Structural guard: no production code references the pre-Wave-7
driver attribute names on a Lumascope instance.

Wave 7 Phase 1 renamed driver instance attributes:
    scope.motion  -> scope._motion_driver  (sub-API name reserved for MotionAPI)
    scope.led     -> scope._led_driver     (sub-API name reserved for IlluminationAPI)
    scope.camera  -> scope._camera_driver  (sub-API name reserved for ImagingAPI)

The mechanical rename in LVP 1dd7baf claimed 259 sites, but two
post-merge bugfix sessions surfaced live-code misses:
    _lumascope.py:622  -- hasattr(self, 'camera') string literal
    main_display.py:304-305, 690-692  -- self.scope.camera in record_init
                                        and recording-finalize manifest builder

Each miss caused a user-visible failure on real hardware. This test
walks every .py file under modules/ and ui/ (the production layers
above drivers) and asserts that NO Attribute node with the form
`<chain>.scope.{camera,led,motion}` exists, where `scope` is either a
Name or the final segment of an attribute chain. AST parsing skips
comments and string literals automatically, so docstring mentions of
the OLD API (e.g. in scope_capabilities.py module docstring) don't
flag.

Tests are deliberately excluded -- test fixtures legitimately assign
to `scope.led = object()` etc. to construct controlled scope state,
and that pattern won't crash production callers.

When Wave 7 Phase 2-7 eventually moves bodies into ImagingAPI /
MotionAPI / IlluminationAPI, the right callers transition to
`scope.imaging.X` / `scope.motion.X` / `scope.illumination.X`. This
test stays valid throughout -- the renamed underscore attributes are
the only legitimate way for production code to reach the driver
during the transition.
"""

import ast
import pathlib

BANNED_ATTRS = frozenset({'camera', 'led'})

# Production code roots; tests/ deliberately excluded (see docstring).
PROD_ROOTS = ('modules', 'ui')

# Repo root inferred from this file's location (LumaViewPro/tests/).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _iter_prod_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root_name in PROD_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        files.extend(sorted(p for p in root.rglob('*.py')))
    return files


def _chain_ends_in_scope(node: ast.AST) -> bool:
    """True iff the value chain of an Attribute node terminates in a
    Name('scope') OR an Attribute(..., attr='scope'). Catches
    `scope.camera`, `self.scope.camera`, `lumaview.scope.camera`,
    `_app_ctx.ctx.scope.camera`, etc."""
    if isinstance(node, ast.Name):
        return node.id == 'scope'
    if isinstance(node, ast.Attribute):
        return node.attr == 'scope'
    return False


def _find_banned_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in BANNED_ATTRS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_legacy_scope_driver_accesses_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f"{path}: read failed: {e}")
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{path}: parse failed: {e}")
            continue
        for lineno, attr in _find_banned_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f"{rel}:{lineno}: scope.{attr} -- use scope._{attr}_driver "
                f"(post-Wave-7 rename). See test_wave7_rename_complete.py."
            )
    assert not failures, (
        "Wave 7 rename incomplete -- production code still uses the "
        "pre-rename driver attribute names:\n  "
        + "\n  ".join(failures)
    )


# Methods that live ONLY on the MotionAPI sub-API. Production callers
# must reach them via `scope.motion.<name>(...)`, never `scope.<name>(...)`.
# Derived by diffing dir(scope.motion) against dir(scope) on a
# `Lumascope(simulate=True)` instance; hardcoded here so the test is
# pure-AST (no simulator instantiation at collection time).
MOTION_ONLY_METHODS = frozenset({
    'add_position_listener', 'get_actual_position', 'get_axes_config',
    'get_axis_limits', 'get_axis_state', 'get_current_position',
    'get_home_status', 'get_limit_switch_status',
    'get_limit_switch_status_all_axes', 'get_overshoot',
    'get_reference_status', 'get_target_position', 'get_target_status',
    'get_turret_position_for_objective_id', 'has_homed', 'has_thomed',
    'has_turret', 'home', 'init_axes', 'is_any_axis_moving',
    'is_current_turret_position_objective_set', 'is_moving',
    'move_absolute_async', 'move_absolute_position', 'move_absolute_sync',
    'move_home_async', 'move_relative_async', 'move_relative_position',
    'refresh_position_cache', 'remove_position_listener',
    'safe_turret_move', 'set_acceleration_limit',
    'set_motor_precision_mode', 'stop_motion', 'thome', 'tmove',
    'wait_until_finished_moving', 'xycenter', 'zhome',
})


def _find_motion_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<motion_only_method>` accesses.

    The chain-ending logic mirrors _chain_ends_in_scope so we catch
    `scope.zhome`, `self._scope.zhome`, `p._scope.zhome`,
    `_app_ctx.ctx.scope.zhome`, etc.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in MOTION_ONLY_METHODS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_motion_method_calls_on_bare_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f"{path}: read failed: {e}")
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{path}: parse failed: {e}")
            continue
        for lineno, attr in _find_motion_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f"{rel}:{lineno}: scope.{attr} -- use scope.motion.{attr}"
            )
    assert not failures, (
        "Motion methods reached on bare scope -- production code must "
        "go through scope.motion.<method>:\n  "
        + "\n  ".join(failures)
    )

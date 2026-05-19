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

import pytest

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


# Methods that live ONLY on the IlluminationAPI sub-API. Production
# callers must reach them via `scope.illumination.<name>(...)`, never
# `scope.<name>(...)`. Derived by diffing dir(scope.illumination)
# against dir(scope) on a `Lumascope(simulate=True)` instance;
# hardcoded here so the test is pure-AST (no simulator instantiation
# at collection time).
ILLUMINATION_ONLY_METHODS = frozenset({
    'add_led_listener', 'ch2color', 'color2ch', 'get_led_ma',
    'get_led_state', 'get_led_states', 'get_led_status', 'led_enabled',
    'led_illumination', 'led_off', 'led_off_async', 'led_off_fast',
    'led_on', 'led_on_async', 'led_on_fast', 'led_on_sync',
    'led_states', 'leds_disable', 'leds_enable', 'leds_off',
    'leds_off_async', 'leds_off_fast', 'leds_off_owned',
    'leds_off_sync', 'remove_led_listener', 'restore_led_state',
    'save_led_state', 'wait_until_led_on',
})


def _find_illumination_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<illumination_only_method>` accesses."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in ILLUMINATION_ONLY_METHODS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_illumination_method_calls_on_bare_scope_in_production():
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
        for lineno, attr in _find_illumination_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f"{rel}:{lineno}: scope.{attr} -- use scope.illumination.{attr}"
            )
    assert not failures, (
        "Illumination methods reached on bare scope -- production code "
        "must go through scope.illumination.<method>:\n  "
        + "\n  ".join(failures)
    )


# Inside-Lumascope guard: catches self.<sub-API-method> calls within
# _lumascope.py itself. The scope-chain guards above target callers in
# OTHER files; this one targets Lumascope's own methods reaching for
# sub-API methods via bare self. Both classes of miss have the same
# fix shape (route through the sub-API attribute), but the bare-scope
# guard never fired on inside-Lumascope calls because the chain
# terminates in `self`, not `scope`. Issue #670 (beta12 DOA) was the
# motivating incident: Lumascope.initialize() called self.leds_off()
# which was retired in Phase 3f.

_LUMASCOPE_PATH = _REPO_ROOT / 'modules' / 'lumascope_api' / '_lumascope.py'


def _find_self_method_accesses(tree: ast.AST, banned: frozenset[str]) -> list[tuple[int, str]]:
    """Find `self.<banned_method>` accesses (bare self, not self.foo.bar)."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in banned:
            continue
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            hits.append((node.lineno, node.attr))
    return hits


def test_no_self_illumination_calls_in_lumascope():
    """Lumascope's own methods must not reach illumination-only methods
    via bare `self.X` -- they belong on `self.illumination.X` after
    Phase 3f forwarder retirement."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, ILLUMINATION_ONLY_METHODS)
    failures = [
        f"_lumascope.py:{lineno}: self.{attr} -- use self.illumination.{attr}"
        for lineno, attr in hits
    ]
    assert not failures, (
        "Lumascope reached illumination-only methods via bare self -- "
        "migrate to self.illumination.<method>:\n  "
        + "\n  ".join(failures)
    )


def test_no_self_motion_calls_in_lumascope():
    """Same shape for motion methods retired in Phase 2f -- Lumascope
    must reach them via `self.motion.X`, not bare `self.X`."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, MOTION_ONLY_METHODS)
    failures = [
        f"_lumascope.py:{lineno}: self.{attr} -- use self.motion.{attr}"
        for lineno, attr in hits
    ]
    assert not failures, (
        "Lumascope reached motion-only methods via bare self -- "
        "migrate to self.motion.<method>:\n  "
        + "\n  ".join(failures)
    )

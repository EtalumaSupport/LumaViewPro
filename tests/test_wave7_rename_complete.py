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
    """True iff the value chain terminates in `scope` or `_scope`.

    The leading-underscore form is the private-handle convention used
    inside modules that hold the Lumascope reference (e.g.
    `self._scope`, `p._scope`). Catches `scope.camera`,
    `self.scope.camera`, `lumaview.scope.camera`,
    `_app_ctx.ctx.scope.camera`, `self._scope.camera`, etc.

    Without the `_scope` case, every reach site that uses the private
    handle silently bypasses the guard. All sub-API migration guards in
    this file (motion, illumination, imaging, diagnostics, runtime_state,
    image_save, diagnostic_facade, compute_focus_score) call through this
    helper, so the hole would cascade to all of them."""
    if isinstance(node, ast.Name):
        return node.id in ('scope', '_scope')
    if isinstance(node, ast.Attribute):
        return node.attr in ('scope', '_scope')
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
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_banned_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f'{rel}:{lineno}: scope.{attr} -- use scope._{attr}_driver '
                f'(post-Wave-7 rename). See test_wave7_rename_complete.py.'
            )
    assert not failures, (
        'Wave 7 rename incomplete -- production code still uses the '
        'pre-rename driver attribute names:\n  ' + '\n  '.join(failures)
    )


# Methods that live ONLY on the MotionAPI sub-API. Production callers
# must reach them via `scope.motion.<name>(...)`, never `scope.<name>(...)`.
# Derived by diffing dir(scope.motion) against dir(scope) on a
# `Lumascope(simulate=True)` instance; hardcoded here so the test is
# pure-AST (no simulator instantiation at collection time).
MOTION_ONLY_METHODS = frozenset(
    {
        'add_position_listener',
        'get_actual_position',
        'get_axes_config',
        'get_axis_limits',
        'get_axis_state',
        'get_current_position',
        'get_home_status',
        'get_limit_switch_status',
        'get_limit_switch_status_all_axes',
        'get_overshoot',
        'get_reference_status',
        'get_target_position',
        'get_target_status',
        'get_turret_position_for_objective_id',
        'has_homed',
        'has_thomed',
        'has_turret',
        'home',
        'init_axes',
        'is_any_axis_moving',
        'is_current_turret_position_objective_set',
        'is_moving',
        'move_absolute_async',
        'move_absolute_position',
        'move_home_async',
        'move_relative_async',
        'move_relative_position',
        'refresh_position_cache',
        'remove_position_listener',
        'safe_turret_move',
        'set_acceleration_limit',
        'set_precision_mode',
        'stop_motion',
        'thome',
        'tmove',
        'wait_until_finished_moving',
        'zhome',
    }
)


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
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_motion_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(f'{rel}:{lineno}: scope.{attr} -- use scope.motion.{attr}')
    assert not failures, (
        'Motion methods reached on bare scope -- production code must '
        'go through scope.motion.<method>:\n  ' + '\n  '.join(failures)
    )


# Methods that live ONLY on the IlluminationAPI sub-API. Production
# callers must reach them via `scope.illumination.<name>(...)`, never
# `scope.<name>(...)`. Derived by diffing dir(scope.illumination)
# against dir(scope) on a `Lumascope(simulate=True)` instance;
# hardcoded here so the test is pure-AST (no simulator instantiation
# at collection time).
ILLUMINATION_ONLY_METHODS = frozenset(
    {
        'add_led_listener',
        'ch2color',
        'color2ch',
        'get_led_ma',
        'get_led_state',
        'get_led_states',
        'get_led_status',
        'led_enabled',
        'led_illumination',
        'led_off',
        'led_off_async',
        'led_on',
        'led_on_async',
        'led_states',
        'leds_disable',
        'leds_enable',
        'leds_off',
        'leds_off_async',
        'leds_off_owned',
        'remove_led_listener',
        'restore_led_state',
        'save_led_state',
        'wait_until_led_on',
    }
)


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
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_illumination_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(f'{rel}:{lineno}: scope.{attr} -- use scope.illumination.{attr}')
    assert not failures, (
        'Illumination methods reached on bare scope -- production code '
        'must go through scope.illumination.<method>:\n  ' + '\n  '.join(failures)
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
        f'_lumascope.py:{lineno}: self.{attr} -- use self.illumination.{attr}'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached illumination-only methods via bare self -- '
        'migrate to self.illumination.<method>:\n  ' + '\n  '.join(failures)
    )


def test_no_self_motion_calls_in_lumascope():
    """Same shape for motion methods retired in Phase 2f -- Lumascope
    must reach them via `self.motion.X`, not bare `self.X`."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, MOTION_ONLY_METHODS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- use self.motion.{attr}' for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached motion-only methods via bare self -- '
        'migrate to self.motion.<method>:\n  ' + '\n  '.join(failures)
    )


# Methods that will live ONLY on the ImagingAPI sub-API after the
# Wave 7 Phase 4 imaging body relocation completes. Hardcoded from the
# Phase 4a inventory (docs/WAVE7_PHASE_4_PLAN.md section 11.6) -- 71
# forwarders on the imaging.py facade as of 2026-05-19. Same shape as
# MOTION_ONLY_METHODS / ILLUMINATION_ONLY_METHODS.
#
# Names track the CURRENT method surface. Phase 4d.5e retired the
# `register_frame_callback` / `unregister_frame_callback` deprecated
# aliases that 4d.5b had kept as forwarders; only the new names
# (`add_frame_listener` / `remove_frame_listener`) remain.
IMAGING_ONLY_METHODS = frozenset(
    {
        'add_camera_listener',
        'add_frame_listener',
        'apply_layer_camera_settings',
        'auto_gain_once',
        'autofocus_return',
        'active_cached',
        'exposure_ms_cached',
        'frame_size_cached',
        'gain_cached',
        'max_exposure_cached',
        'max_gain_cached',
        'min_frame_size_cached',
        'pixel_format_cached',
        'capture_and_wait',
        'capture_return',
        'count_frame',
        'frame_is_valid',
        'frames_until_valid',
        'get_available_binning_sizes',
        'get_binning_size',
        'get_exposure_time',
        'get_gain',
        'get_height',
        'get_image',
        'get_image_from_buffer',
        'get_max_height',
        'get_max_width',
        'get_supported_pixel_formats',
        'get_width',
        'is_capturing',
        'is_focusing',
        'log_camera_temps',
        'remove_camera_listener',
        'remove_frame_listener',
        'restore_camera_state',
        'save_camera_state',
        'scale_bar_config',
        'scale_bar_enabled',
        'set_acquisition_stop_mode',
        'set_auto_exposure_time',
        'set_auto_gain',
        'set_bandwidth_reserve_mode',
        'set_binning_size',
        'set_device_link_throughput_limit',
        'set_exposure_time',
        'set_frame_size',
        'set_gain',
        'set_gev_inter_packet_delay',
        'set_gev_packet_size',
        'set_max_acquisition_frame_rate',
        'set_max_transfer_size',
        'set_num_max_queued_urbs',
        'set_pixel_format',
        'set_scale_bar',
        'start_camera_temp_logging',
        'stop_camera_temp_logging',
        'suppress_value_warnings',
        'update_auto_gain_target_brightness',
        'update_camera_config',
    }
)


def _find_imaging_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<imaging_only_method>` accesses.

    Mirrors the motion / illumination chain check above -- catches
    `scope.set_gain`, `self.scope.set_gain`, `p.scope.set_gain`, etc.
    Does NOT catch `self._scope.set_gain` (sub-API back-reference
    pattern) -- those are addressed by the bulk word-boundary sed in
    Phase 4e production migration; carried-forward Phase 3 limitation.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in IMAGING_ONLY_METHODS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_imaging_method_calls_on_bare_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_imaging_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(f'{rel}:{lineno}: scope.{attr} -- use scope.imaging.{attr}')
    assert not failures, (
        'Imaging methods reached on bare scope -- production code must '
        'go through scope.imaging.<method>:\n  ' + '\n  '.join(failures)
    )


def test_no_self_imaging_calls_in_lumascope():
    """Lumascope's own methods must not reach imaging-only methods via
    bare `self.X` -- they belong on `self.imaging.X` after Phase 4f
    forwarder retirement."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, IMAGING_ONLY_METHODS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- use self.imaging.{attr}' for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached imaging-only methods via bare self -- '
        'migrate to self.imaging.<method>:\n  ' + '\n  '.join(failures)
    )


# Methods that will live ONLY on the DiagnosticsAPI sub-API after the
# Wave 7 Phase 5 diagnostics body relocation completes. Hardcoded from
# Phase 5 plan section 3.1 -- 7 stateless probes on the diagnostics.py
# facade as of 2026-05-19. Same shape as IMAGING_ONLY_METHODS /
# MOTION_ONLY_METHODS / ILLUMINATION_ONLY_METHODS.
DIAGNOSTICS_ONLY_METHODS = frozenset(
    {
        'get_camera_temperatures',
        'get_camera_diagnostic_info',
        'run_camera_bandwidth_test',
        'run_grab_lifecycle_benchmark',
        'run_pylon_diagnostic_probe',
        'send_diagnostic_command',
        'send_diagnostic_command_multiline',
    }
)


def _find_diagnostics_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<diagnostics_only_method>` accesses.

    Mirrors the imaging / motion / illumination chain checks above.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in DIAGNOSTICS_ONLY_METHODS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_diagnostics_method_calls_on_bare_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_diagnostics_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(f'{rel}:{lineno}: scope.{attr} -- use scope.diagnostics.{attr}')
    assert not failures, (
        'Diagnostics methods reached on bare scope -- production code '
        'must go through scope.diagnostics.<method>:\n  ' + '\n  '.join(failures)
    )


def test_no_self_diagnostics_calls_in_lumascope():
    """Lumascope's own methods must not reach diagnostics-only methods
    via bare `self.X` -- they belong on `self.diagnostics.X` after
    Phase 5c body relocation moved the 3 inside-class sibling calls
    into DiagnosticsAPI (they now resolve intra-DiagnosticsAPI)."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, DIAGNOSTICS_ONLY_METHODS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- use self.diagnostics.{attr}'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached diagnostics-only methods via bare self -- '
        'migrate to self.diagnostics.<method>:\n  ' + '\n  '.join(failures)
    )


# Phase 6 (image_save extraction) retires 6 instance methods + 5 *_static
# duplicates from Lumascope, replacing them with free functions in
# `modules/image_save.py`. See docs/WAVE7_PHASE_6_PLAN.md. Three guards
# stage the migration:
#   1. Bare-scope guard -- no `scope.<image_save_method>(` in production.
#      Flips at 6e (production caller migration).
#   2. Static-method guard -- no `Lumascope.<X>_static(` anywhere.
#      Flips at 6c (static chain retirement).
#   3. Inside-class self-guard -- no `self.<image_save_method>` in
#      _lumascope.py. Flips at 6c (instance bodies move to module;
#      wrappers are thin forwarders that don't self-call).
IMAGE_SAVE_METHODS = frozenset(
    {
        'save_image',
        'save_live_image',
        'get_next_save_path',
        'generate_image_save_path',
        'generate_image_metadata',
        'prepare_image_for_saving',
    }
)

IMAGE_SAVE_STATIC_METHODS = frozenset(
    {
        'save_image_static',
        'get_next_save_path_static',
        'generate_image_save_path_static',
        'generate_image_metadata_static',
        'prepare_image_for_saving_static',
    }
)


def _find_image_save_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<image_save_method>` accesses.

    Mirrors the imaging / motion / illumination / diagnostics chain
    checks above.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in IMAGE_SAVE_METHODS:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def _find_lumascope_static_method_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `Lumascope.<static_method>` attribute accesses.

    Catches the class-method-style calls used to invoke the *_static
    duplicates (e.g., `Lumascope.save_image_static(...)`). These have
    no chain-ends-in-scope analog -- the base is the class name itself.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in IMAGE_SAVE_STATIC_METHODS:
            continue
        if isinstance(node.value, ast.Name) and node.value.id == 'Lumascope':
            hits.append((node.lineno, node.attr))
    return hits


def test_no_image_save_method_calls_on_bare_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_image_save_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f'{rel}:{lineno}: scope.{attr} -- '
                f'use `from modules.image_save import {attr}; {attr}(scope, ...)`'
            )
    assert not failures, (
        'image_save methods reached on bare scope -- production code '
        'must import the free functions from modules.image_save:\n  ' + '\n  '.join(failures)
    )


def test_no_lumascope_class_static_method_calls():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_lumascope_static_method_accesses(tree):
            rel = path.relative_to(_REPO_ROOT)
            bare_name = attr.removesuffix('_static')
            failures.append(
                f'{rel}:{lineno}: Lumascope.{attr} retired -- '
                f'use `from modules.image_save import {bare_name}`'
            )
    assert not failures, (
        'Lumascope.*_static methods called -- the static chain is '
        'retired in Phase 6c; use the free functions in '
        'modules.image_save:\n  ' + '\n  '.join(failures)
    )


def test_no_self_image_save_calls_in_lumascope():
    """Lumascope's own methods must not reach image_save methods via
    bare `self.X` -- after Phase 6c the bodies live in
    `modules.image_save` and the Lumascope wrappers are thin forwarders
    that call the free function directly with `self` as the scope arg."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, IMAGE_SAVE_METHODS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- '
        f'use `from modules.image_save import {attr}; {attr}(self, ...)`'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached image_save methods via bare self -- the '
        'bodies live in modules.image_save after Phase 6c; call the '
        'free function with self as the scope arg:\n  ' + '\n  '.join(failures)
    )


# Phase 7 (composition root cleanup) -- see docs/WAVE7_PHASE_7_PLAN.md.
# Two distinct migrations stage four guards:
#
#   1. Six diagnostic facade getters RELOCATE from Lumascope to
#      DiagnosticsAPI. Bare-scope guard + inside-class self.X guard,
#      both `xfail strict=True` until their respective flip stages.
#      Bare-scope flips at 7e (production caller migration); inside-
#      class flips at 7c (get_system_info body relocates with its 3
#      self.X self-calls per WAVE7_PHASE_7_PLAN sec.9 #5).
#
#   2. compute_focus_score RETIRES outright (no replacement on a
#      sub-API; callers migrate to
#      modules.autofocus_functions.focus_function directly). Bare-
#      scope guard `xfail strict=True` until 7e; inside-class guard
#      passes plain today (the wrapper has zero internal self-callers
#      verified at 7a) and stays green through 7f retirement.
#
# The DIAGNOSTICS_ONLY_METHODS frozenset above covers the 7 Phase 5
# diagnostic probes (camera temperatures, bandwidth tests, etc.).
# DIAGNOSTIC_FACADE_GETTERS below covers the 6 thin facade getters
# Phase 5 deliberately left on Lumascope -- Phase 7 finishes the
# migration. Kept separate so the two phases' guard staging stays
# obvious.
DIAGNOSTIC_FACADE_GETTERS = frozenset(
    {
        'get_motor_info',
        'get_led_info',
        'get_camera_info',
        'get_camera_profile_info',
        'get_system_info',
        'get_microscope_model',
    }
)

COMPUTE_FOCUS_SCORE_RETIRED = frozenset({'compute_focus_score'})


def _find_chain_method_accesses(tree: ast.AST, banned: frozenset[str]) -> list[tuple[int, str]]:
    """Find `<chain ending in scope>.<banned_method>` accesses.

    Parameterized variant of the per-sub-API finders above. Same shape
    as _find_imaging_method_accesses / _find_diagnostics_method_accesses
    / etc. but takes the frozenset as an argument.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in banned:
            continue
        if _chain_ends_in_scope(node.value):
            hits.append((node.lineno, node.attr))
    return hits


def test_no_diagnostic_facade_getter_calls_on_bare_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_chain_method_accesses(tree, DIAGNOSTIC_FACADE_GETTERS):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(f'{rel}:{lineno}: scope.{attr} -- use scope.diagnostics.{attr}')
    assert not failures, (
        'Diagnostic facade getters reached on bare scope -- production '
        'code must go through scope.diagnostics.<getter> after Phase 7e:\n  '
        + '\n  '.join(failures)
    )


def test_no_self_diagnostic_facade_getter_calls_in_lumascope():
    """Lumascope's own methods must not reach diagnostic facade getters
    via bare `self.X` -- get_system_info's 3 self-calls migrate naturally
    when its body relocates to DiagnosticsAPI at Phase 7c."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, DIAGNOSTIC_FACADE_GETTERS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- use self.diagnostics.{attr}'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached diagnostic facade getters via bare self -- '
        'migrate to self.diagnostics.<getter>:\n  ' + '\n  '.join(failures)
    )


def test_no_compute_focus_score_calls_on_scope_in_production():
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_chain_method_accesses(tree, COMPUTE_FOCUS_SCORE_RETIRED):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f'{rel}:{lineno}: scope.{attr} retired -- '
                f'use `from modules import autofocus_functions; '
                f'autofocus_functions.focus_function(image=..., '
                f'skip_score_logging=True)`'
            )
    assert not failures, (
        'compute_focus_score reached on scope -- the wrapper retires in '
        'Phase 7f; callers must use modules.autofocus_functions.'
        'focus_function directly:\n  ' + '\n  '.join(failures)
    )


def test_no_self_compute_focus_score_calls_in_lumascope():
    """Lumascope's own methods must not reach compute_focus_score via
    bare `self.X`. Today there are zero internal callers (verified at
    Phase 7a); this guard locks the invariant through 7f retirement so
    a stray new self.compute_focus_score call doesn't sneak in before
    the wrapper deletes."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, COMPUTE_FOCUS_SCORE_RETIRED)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- the wrapper retires '
        f'in Phase 7f; do not add new internal callers'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached compute_focus_score via bare self -- the '
        'wrapper retires in Phase 7f and has no internal callers:\n  ' + '\n  '.join(failures)
    )


# Phase 8 (RuntimeState population) -- see docs/WAVE7_PHASE_8_PLAN.md.
# 12 settings-host methods relocate from Lumascope to RuntimeState.
# Two guards staged xfail(strict=True) at 8b; the bare-scope guard flips
# at 8e (production callers migrated), the inside-class guard flips at
# 8d (the 4 self.set_X sites in initialize() update with the body move).
#
# Today's counts (from 8a inventory in WAVE7_PHASE_8_PLAN.md):
#   - bare-scope: 11 production sites across image_save.py / scope_session.py
#     / protocol_image_writer.py / 4 ui/ files
#   - inside-class self: 4 sites in _lumascope.py::initialize (lines
#     622, 624, 625, 628)
RUNTIME_STATE_ONLY_METHODS = frozenset(
    {
        'set_labware',
        'get_labware',
        'set_objective',
        'get_current_objective_id',
        'get_objective_info',
        'get_available_objectives',
        'get_current_objective',
        'set_turret_config',
        'get_turret_config',
        'set_stage_offset',
        'get_stage_offset',
        'get_well_label',
    }
)


def test_no_runtime_state_method_calls_on_bare_scope_in_production():
    """All 12 settings-host methods belong on scope.runtime_state by 8f.
    Production callers via `scope.X` / `ctx.scope.X` / `lumaview.scope.X`
    must migrate to the `.runtime_state.` chain at 8e."""
    failures: list[str] = []
    for path in _iter_prod_files():
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            failures.append(f'{path}: read failed: {e}')
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            failures.append(f'{path}: parse failed: {e}')
            continue
        for lineno, attr in _find_chain_method_accesses(tree, RUNTIME_STATE_ONLY_METHODS):
            rel = path.relative_to(_REPO_ROOT)
            failures.append(
                f'{rel}:{lineno}: scope.{attr} -- migrate to scope.runtime_state.{attr}'
            )
    assert not failures, (
        'Settings-host methods reached on bare scope -- migrate to '
        'scope.runtime_state per Phase 8e:\n  ' + '\n  '.join(failures)
    )


def test_no_self_runtime_state_calls_in_lumascope():
    """Lumascope-internal self.X calls for the 12 settings-host methods
    must reach via self.runtime_state. The 4 initialize() sites were
    migrated in 8d; this guard pins that the method bodies (now
    forwarders) and any future composition-root code don't reach back
    onto bare self.X for these methods."""
    source = _LUMASCOPE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(_LUMASCOPE_PATH))
    hits = _find_self_method_accesses(tree, RUNTIME_STATE_ONLY_METHODS)
    failures = [
        f'_lumascope.py:{lineno}: self.{attr} -- migrate to self.runtime_state.{attr} (Phase 8d)'
        for lineno, attr in hits
    ]
    assert not failures, (
        'Lumascope reached settings-host methods via bare self -- '
        'migrate to self.runtime_state per Phase 8d:\n  ' + '\n  '.join(failures)
    )

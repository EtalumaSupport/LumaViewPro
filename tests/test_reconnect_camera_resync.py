"""Regression: reconnect resyncs the whole per-camera UI surface, uniformly.

Reconnecting to a different camera_type must refresh every per-camera UI cap
and gate from the NEW camera -- not a drifting subset. Two confirmed gaps this
pins:

  - The gain-slider cap (ctx.max_gain) was refreshed only in load_settings, so
    reconnecting from a higher-cap camera to a lower one (e.g. LS850 -> LS620)
    left the gain slider over-ranged -- the user could drag gain past the
    usable range and black out the image.
  - reconnect re-applied settings only for a hardcoded 'BF', so a non-BF open
    layer's controls (e.g. gain/exposure sliders disabled while the prior
    camera's auto-gain was on) were never refreshed against the new camera.

The fix:
  - config_helpers.camera_max_exposure_for_ui / camera_max_gain_for_ui are the
    single UI-facing cap resolvers: the live camera's cap, or the documented
    no-camera default (camera_max_* are None by design then; #616). Both
    load_settings and reconnect resolve through them so the fallback can't be
    applied two different ways.
  - ImageSettings.sync_camera_capability_ranges groups the per-layer setters
    (exposure + gain ranges + autogain gate) AND clamp_layer_settings_to_caps,
    which reconciles each layer's stored gain_db/exp_ms down to the new caps
    (the blackout fix, matching load_settings); _init_ui (connect) and reconnect
    call the SAME grouping.
  - reconnect refreshes ctx.max_* then re-applies the VISIBLE layer
    (ImageSettings.open_or_default_layer), not a hardcoded channel.

The UI modules touch Kivy widgets and cannot be imported under the test mocks
(see test_ids_native_roi_sync_binning), so the wiring is pinned with AST
guards; the pure cap resolvers are exercised directly.
"""

import ast
import pathlib
from types import SimpleNamespace

from modules.config_helpers import (
    DEFAULT_MAX_EXPOSURE_MS,
    DEFAULT_MAX_GAIN_DB,
    camera_max_exposure_for_ui,
    camera_max_gain_for_ui,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
MS_PATH = REPO_ROOT / 'ui' / 'microscope_settings.py'


def _method_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def _name_calls(method: ast.FunctionDef, name: str):
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def _attr_calls(method: ast.FunctionDef, attr: str):
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


class TestCapResolvers:
    """The single UI-facing cap resolvers apply the documented #616 fallback."""

    def test_exposure_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(camera_max_exposure=500.0)
        assert camera_max_exposure_for_ui(imaging) == 500.0

    def test_exposure_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(camera_max_exposure=None)
        assert camera_max_exposure_for_ui(imaging) == DEFAULT_MAX_EXPOSURE_MS

    def test_gain_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(camera_max_gain=24.0)
        assert camera_max_gain_for_ui(imaging) == 24.0

    def test_gain_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(camera_max_gain=None)
        assert camera_max_gain_for_ui(imaging) == DEFAULT_MAX_GAIN_DB


class TestSyncGrouping:
    """sync_camera_capability_ranges groups all three per-layer setters."""

    def test_grouping_calls_all_setters_and_clamp(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')
        for setter in (
            'set_layer_exposure_ranges',
            'set_layer_gain_ranges',
            'set_layer_autogain_support',
            'clamp_layer_settings_to_caps',
        ):
            assert _attr_calls(method, setter), f'sync_camera_capability_ranges must call {setter}.'

    def test_clamp_reconciles_stored_gain_and_exposure_to_caps(self):
        # The blackout fix: a stored gain_db/exp_ms above the new camera's cap
        # must be brought down to the cap (and persisted) for every layer, so a
        # downshift reconnect can't push an over-cap value that blacks out.
        method = _method_node(IMAGE_SETTINGS_PATH, 'clamp_layer_settings_to_caps')
        clamped = {
            t.slice.value
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exp_ms')
        }
        assert clamped == {'gain_db', 'exp_ms'}, (
            'clamp_layer_settings_to_caps must reconcile both stored gain_db and '
            'exp_ms down to the camera caps.'
        )

    def test_init_ui_uses_the_grouping(self):
        method = _method_node(IMAGE_SETTINGS_PATH, '_init_ui')
        assert _attr_calls(method, 'sync_camera_capability_ranges'), (
            'ImageSettings._init_ui must resync via sync_camera_capability_ranges.'
        )

    def test_open_or_default_layer_reuses_helper_and_defaults_to_bf(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'open_or_default_layer')
        # Delegates the open-layer scan to the shared guarded helper, and
        # defaults to BF when none is open.
        assert _attr_calls(method, 'get_opened_layer') or any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'get_opened_layer'
            for n in ast.walk(method)
        ), 'open_or_default_layer must reuse common_utils.get_opened_layer.'
        returns_bf = any(
            isinstance(n, ast.Return)
            and isinstance(n.value, ast.Constant)
            and n.value.value == 'BF'
            for n in ast.walk(method)
        )
        assert returns_bf, 'open_or_default_layer must default to BF when no layer is open.'


class TestReconnectResync:
    """reconnect refreshes caps, regroups the setters, and re-applies the open layer."""

    def test_reconnect_refreshes_both_caps_via_resolvers(self):
        method = _method_node(MS_PATH, 'reconnect')
        assert _name_calls(method, 'camera_max_exposure_for_ui'), (
            'reconnect must refresh ctx.max_exposure from the new camera.'
        )
        assert _name_calls(method, 'camera_max_gain_for_ui'), (
            'reconnect must refresh ctx.max_gain from the new camera (blackout fix).'
        )

    def test_reconnect_uses_the_grouped_resync(self):
        method = _method_node(MS_PATH, 'reconnect')
        assert _attr_calls(method, 'sync_camera_capability_ranges'), (
            'reconnect must resync the per-layer surface via the same grouping as connect.'
        )

    def test_reconnect_reapplies_visible_layer_not_hardcoded(self):
        method = _method_node(MS_PATH, 'reconnect')
        assert _attr_calls(method, 'open_or_default_layer'), (
            'reconnect must re-apply the VISIBLE layer (open_or_default_layer), '
            'not a hardcoded channel.'
        )

    def test_load_settings_uses_the_cap_resolvers(self):
        # The de-fragmentation: load_settings resolves caps through the same
        # helpers as reconnect, not its own inline `or DEFAULT`.
        method = _method_node(MS_PATH, 'load_settings')
        assert _name_calls(method, 'camera_max_exposure_for_ui'), (
            'load_settings must resolve the exposure cap via camera_max_exposure_for_ui.'
        )
        assert _name_calls(method, 'camera_max_gain_for_ui'), (
            'load_settings must resolve the gain cap via camera_max_gain_for_ui.'
        )

    def test_load_settings_delegates_clamp_not_inline(self):
        # De-dup: load_settings reconciles over-cap values via the single
        # clamp_layer_settings_to_caps owner, not a duplicate inline clamp.
        method = _method_node(MS_PATH, 'load_settings')
        assert _attr_calls(method, 'clamp_layer_settings_to_caps'), (
            'load_settings must delegate over-cap reconciliation to clamp_layer_settings_to_caps.'
        )
        inline = [
            t
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exp_ms')
        ]
        assert not inline, (
            'load_settings must not carry an inline gain_db/exp_ms clamp-persist '
            '(it duplicates clamp_layer_settings_to_caps).'
        )

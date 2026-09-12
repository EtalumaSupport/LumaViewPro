"""Regression: the per-camera UI surface resyncs from the camera, uniformly.

Every per-camera UI cap and gate must come from the attached camera -- not
a drifting subset. Two confirmed gaps this pins:

  - The gain-slider cap (ctx.max_gain) was refreshed on only one of two
    bring-up paths, so a swap from a higher-cap camera to a lower one
    (e.g. LS850 -> LS620) left the gain slider over-ranged -- the user
    could drag gain past the usable range and black out the image.
  - A bring-up path re-applied settings only for a hardcoded 'BF', so a
    non-BF open layer's controls (e.g. gain/exposure sliders disabled while
    the prior camera's auto-gain was on) were never refreshed.

The fix:
  - config_helpers.camera_max_exposure_for_ui / camera_max_gain_for_ui are the
    single UI-facing cap resolvers: the live camera's cap, or the documented
    no-camera default (camera_max_* are None by design then; #616).
    load_settings resolves through them so the fallback can't be applied
    two different ways.
  - ImageSettings.sync_camera_capability_ranges groups the per-layer setters
    (exposure + gain ranges + autogain gate) AND clamp_layer_settings_to_caps,
    which reconciles each layer's stored gain_db/exposure_ms down to the new caps
    (the blackout fix, matching load_settings); _init_ui (connect) calls that
    grouping. A scope swap (the auto-reconnect item) re-runs it and re-applies
    the VISIBLE layer (ImageSettings.open_or_default_layer), never a hardcoded
    channel -- its acceptance list lives with that item.

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
        imaging = SimpleNamespace(max_exposure_ms_cached=500.0)
        assert camera_max_exposure_for_ui(imaging) == 500.0

    def test_exposure_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(max_exposure_ms_cached=None)
        assert camera_max_exposure_for_ui(imaging) == DEFAULT_MAX_EXPOSURE_MS

    def test_gain_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(max_gain_db_cached=24.0)
        assert camera_max_gain_for_ui(imaging) == 24.0

    def test_gain_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(max_gain_db_cached=None)
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
        # The blackout fix: a stored gain_db/exposure_ms above the new camera's cap
        # must be brought down to the cap (and persisted) for every layer, so a
        # downshift swap can't push an over-cap value that blacks out.
        method = _method_node(IMAGE_SETTINGS_PATH, 'clamp_layer_settings_to_caps')
        clamped = {
            t.slice.value
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        }
        assert clamped == {'gain_db', 'exposure_ms'}, (
            'clamp_layer_settings_to_caps must reconcile both stored gain_db and '
            'exposure_ms down to the camera caps.'
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


class TestLoadSettingsResync:
    """load_settings resolves the caps and the clamp through the shared owners."""

    def test_load_settings_uses_the_cap_resolvers(self):
        # The de-fragmentation: load_settings resolves caps through the same
        # helpers, not its own inline `or DEFAULT`.
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
            and t.slice.value in ('gain_db', 'exposure_ms')
        ]
        assert not inline, (
            'load_settings must not carry an inline gain_db/exposure_ms clamp-persist '
            '(it duplicates clamp_layer_settings_to_caps).'
        )


class TestCapabilitySyncDoesNotImpersonateTheUser:
    """The app's own bound-application must not read back as a user drag.

    Narrowing a slider's max makes Kivy clamp its value, which fires on_value
    into LayerControl's handler. By the time sync_camera_capability_ranges
    runs, load_settings has already cleared the per-layer _initializing flag
    (it clears it in sync_widgets_from_settings' finally), so the handler is
    live: it rewrote settings[layer][...] with the bound and emitted a SLIDER
    record crediting the user. Measured in the simulator with nobody touching
    the app -- a stored BF illumination of 500 became 50 in current.json, and
    a stored DF exposure of 500 became 200, each with a matching SLIDER line.

    clamp_layer_settings_to_caps must stay outside the flag: it reconciles a
    value the hardware cannot honor, which is a real settings change.
    """

    def _sync_method(self):
        return _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')

    def test_setters_run_under_the_initializing_flag(self):
        method = self._sync_method()
        try_nodes = [n for n in ast.walk(method) if isinstance(n, ast.Try)]
        assert try_nodes, (
            'sync_camera_capability_ranges must run its setters inside try/finally '
            'so the _initializing flag is cleared even if a setter raises.'
        )
        guarded = {
            n.func.attr
            for try_node in try_nodes
            for n in ast.walk(try_node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr.startswith('set_layer_')
        }
        for setter in (
            'set_layer_exposure_ranges',
            'set_layer_gain_ranges',
            'set_layer_illumination_ranges',
            'set_layer_autogain_support',
        ):
            assert setter in guarded, (
                f'{setter} must run inside the _initializing guard -- outside it, '
                'the Kivy max-clamp reaches the layer handler and is recorded as a drag.'
            )

    def test_the_flag_is_set_and_cleared(self):
        method = self._sync_method()
        writes = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == '_initializing'
        ]
        values = {n.value.value for n in writes if isinstance(n.value, ast.Constant)}
        assert values == {True, False}, (
            'sync_camera_capability_ranges must both set and clear _initializing; '
            f'found {values or "no writes"}.'
        )
        finallys = [n for n in ast.walk(method) if isinstance(n, ast.Try) and n.finalbody]
        cleared_in_finally = any(
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant)
            and n.value.value is False
            for try_node in finallys
            for stmt in try_node.finalbody
            for n in ast.walk(stmt)
        )
        assert cleared_in_finally, 'The flag must be cleared in a finally, not on the happy path.'

    def test_clamp_stays_outside_the_guard(self):
        method = self._sync_method()
        in_try = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'clamp_layer_settings_to_caps'
            for try_node in [x for x in ast.walk(method) if isinstance(x, ast.Try)]
            for n in ast.walk(try_node)
        )
        assert not in_try, (
            'clamp_layer_settings_to_caps must stay OUTSIDE the _initializing guard -- '
            'it reconciles a value the hardware cannot honor, which is a real change.'
        )
        assert _attr_calls(method, 'clamp_layer_settings_to_caps'), (
            'sync_camera_capability_ranges must still run the clamp.'
        )

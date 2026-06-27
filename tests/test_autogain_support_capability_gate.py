"""Regression: the Auto Gain/Exp control is hidden on cameras without hardware AG/AE.

The "Auto Gain/Exp" checkbox drives the camera's HARDWARE auto-gain/exposure
(``imaging.set_auto_gain`` -> ``driver.auto_gain``); ``set_auto_gain`` itself
already guards its settle path on ``profile.has_auto_gain``. On a camera whose
profile reports neither hardware AG nor AE (IDS U3-34Lx, FX2 LS620) the checkbox
is a no-op, so it must not be shown.

The fix:
  - ``config_ui_getters.camera_autogain_supported()`` is the single gate: True
    when the live camera reports hardware auto-gain OR auto-exposure (the one
    control drives both). It resolves through ``_live_capabilities()``, which
    reads ``ctx.lumaview.scope`` -- the reference ``reconnect()`` rebuilds --
    not the build-time ``ctx.scope`` registry field. The sibling
    ``firmware_stim_supported()`` shares that accessor, so the stale-scope bug
    can't live on in one gate but not the other. Fails safe to True (show) when
    no capability surface exists yet.
  - ``LayerControl.camera_autogain_support`` is an orthogonal runtime gate (like
    ``show_camera_controls``), AND-ed with the per-layer static
    ``autogain_support`` in the kv so Lumi stays hidden regardless. Set on every
    layer at both capability-sync points: ``ImageSettings._init_ui`` (connect)
    and ``MicroscopeSettings.reconnect`` (scope-change / reconnect).
  - The effective enable is derived NON-DESTRUCTIVELY at the consumption point
    (``LayerControl.effective_auto_gain`` = saved preference AND capability),
    used by ``apply_settings`` for the slider-disable / camera enable. The
    persisted ``auto_gain`` is never mutated, so a capable camera's saved
    preference survives a swap to an AG-less body and back.

The UI modules touch Kivy widgets and cannot be imported under the test mocks
(see test_ids_native_roi_sync_binning), so the wiring is pinned structurally
with AST guards; the gate logic + capability mapping are exercised directly.
"""

import ast
import pathlib

from unittest.mock import MagicMock

import modules.app_context as _app_ctx
from modules.config_ui_getters import camera_autogain_supported
from drivers.camera_profiles import lookup_profile
from modules.lumascope_api import Lumascope
from modules.scope_capabilities import ScopeCapabilities

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_PATH = REPO_ROOT / 'ui' / 'layer_control.py'
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
MS_PATH = REPO_ROOT / 'ui' / 'microscope_settings.py'
KV_PATH = REPO_ROOT / 'ui' / 'lumaviewpro.kv'
CONFIG_GETTERS_PATH = REPO_ROOT / 'modules' / 'config_ui_getters.py'


def _method_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def _calls_named(method: ast.FunctionDef, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


def _ctx_reporting(auto_gain=False, auto_exposure=False):
    """A ctx whose LIVE scope (ctx.lumaview.scope) reports the given caps."""
    ctx = MagicMock()
    caps = ctx.lumaview.scope.capabilities
    caps.camera_supports_auto_gain = auto_gain
    caps.camera_supports_auto_exposure = auto_exposure
    return ctx


class TestAutogainGetter:
    """camera_autogain_supported() -- the single visibility gate."""

    def test_true_when_hardware_auto_gain(self, monkeypatch):
        monkeypatch.setattr(_app_ctx, 'ctx', _ctx_reporting(auto_gain=True))
        assert camera_autogain_supported() is True

    def test_true_when_only_auto_exposure(self, monkeypatch):
        # The single control drives BOTH; either capability keeps it visible.
        monkeypatch.setattr(_app_ctx, 'ctx', _ctx_reporting(auto_exposure=True))
        assert camera_autogain_supported() is True

    def test_false_when_neither(self, monkeypatch):
        # IDS / LS620 shape: no hardware AG and no hardware AE -> hide.
        monkeypatch.setattr(_app_ctx, 'ctx', _ctx_reporting())
        assert camera_autogain_supported() is False

    def test_reads_live_scope_not_stale_registry(self, monkeypatch):
        # The gate must read ctx.lumaview.scope (reconnect rebuilds it), not the
        # ctx.scope registry field reconnect never refreshes. A stale ctx.scope
        # reporting True must NOT keep the control visible on an AG-less camera.
        ctx = _ctx_reporting()  # live scope: no AG/AE
        ctx.scope.capabilities.camera_supports_auto_gain = True  # stale registry
        ctx.scope.capabilities.camera_supports_auto_exposure = True
        monkeypatch.setattr(_app_ctx, 'ctx', ctx)
        assert camera_autogain_supported() is False

    def test_fails_safe_to_shown_when_no_capabilities(self, monkeypatch):
        ctx = MagicMock()
        ctx.lumaview.scope.capabilities = None
        monkeypatch.setattr(_app_ctx, 'ctx', ctx)
        assert camera_autogain_supported() is True


class TestCapabilityMapping:
    """The capability flags reflect the per-camera hardware reality (real path)."""

    def _caps_for(self, model_name):
        from types import SimpleNamespace

        motion = MagicMock()
        motion.detect_present_axes.return_value = ('X', 'Y', 'Z')
        motion.get_microscope_model.return_value = 'TEST-MODEL'
        motion.motorconfig = None
        led = MagicMock()
        led.available_channels.return_value = ('Blue', 'Green', 'Red')
        led.available_colors.return_value = ('Blue', 'Green', 'Red')
        led.supports_firmware_stim.return_value = False
        camera = SimpleNamespace(
            profile=lookup_profile(model_name),
            get_max_frame_size=lambda: {'width': 1024, 'height': 768},
        )
        return ScopeCapabilities.from_drivers(motion=motion, led=led, camera=camera)

    def test_ids_reports_no_hardware_autogain(self):
        caps = self._caps_for('U3-34LxXCP-M')
        assert caps.camera_supports_auto_gain is False
        assert caps.camera_supports_auto_exposure is False

    def test_fx2_ls620_reports_no_hardware_autogain(self):
        caps = self._caps_for('LS620')
        assert caps.camera_supports_auto_gain is False
        assert caps.camera_supports_auto_exposure is False

    def test_pylon_reports_hardware_autogain(self):
        assert self._caps_for('daA3840-45um').camera_supports_auto_gain is True

    def test_simulated_scope_reports_hardware_autogain(self):
        # Real-path anchor for the mocked getter tests: the sim camera has AG.
        assert Lumascope(simulate=True).capabilities.camera_supports_auto_gain is True


class TestUiWiring:
    """Pin the production wiring so the gate cannot silently regress."""

    def test_layer_control_defines_camera_autogain_property(self):
        body = LAYER_CONTROL_PATH.read_text(encoding='utf-8')
        assert 'camera_autogain_support = BooleanProperty' in body, (
            'LayerControl must declare the camera_autogain_support gate property.'
        )

    def test_kv_visibility_ands_in_camera_gate(self):
        # Substring, not the whole expression -- a harmless reorder/rewrap of the
        # AND terms must not fail CI; what matters is the gate is present.
        body = KV_PATH.read_text(encoding='utf-8')
        assert 'root.camera_autogain_support' in body, (
            'The Auto Gain/Exp visibility must AND in root.camera_autogain_support.'
        )

    def test_set_layer_autogain_support_uses_the_single_gate(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'set_layer_autogain_support')
        name_calls = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'camera_autogain_supported'
        ]
        assert name_calls, (
            'set_layer_autogain_support must resolve visibility via the single '
            'camera_autogain_supported() gate.'
        )
        body = ast.dump(method)
        assert 'camera_autogain_support' in body, (
            'set_layer_autogain_support must set layer_obj.camera_autogain_support.'
        )

    def test_set_layer_autogain_support_does_not_mutate_persisted_setting(self):
        # Non-destructive: the gate must NOT write back settings[layer]['auto_gain']
        # (the writeback destroyed a capable camera's saved preference on swap-back).
        method = _method_node(IMAGE_SETTINGS_PATH, 'set_layer_autogain_support')
        writebacks = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == 'auto_gain'
                for t in node.targets
            )
        ]
        assert not writebacks, (
            "set_layer_autogain_support must NOT write back settings[layer]['auto_gain'] "
            '-- the effective enable is derived non-destructively at apply_settings so '
            'a capable camera preference survives a swap to an AG-less body.'
        )

    def test_effective_auto_gain_gates_preference_on_capability(self):
        # The non-destructive derivation: effective = saved preference AND the
        # camera capability gate, computed on read.
        method = _method_node(LAYER_CONTROL_PATH, 'effective_auto_gain')
        body = ast.dump(method)
        assert 'camera_autogain_support' in body and 'auto_gain' in body, (
            'effective_auto_gain must AND the saved auto_gain preference with '
            'camera_autogain_support.'
        )

    def test_apply_settings_uses_effective_auto_gain(self):
        method = _method_node(LAYER_CONTROL_PATH, 'apply_settings')
        assert _calls_named(method, 'effective_auto_gain'), (
            'apply_settings must derive the slider-disable / camera enable from '
            'effective_auto_gain(), not the raw stored auto_gain.'
        )

    def test_both_gates_share_live_capabilities_accessor(self):
        # Cluster fix: firmware_stim_supported and camera_autogain_supported must
        # both resolve through _live_capabilities() (ctx.lumaview.scope), so the
        # stale-ctx.scope twin bug can't reappear in one gate but not the other.
        for fn in ('firmware_stim_supported', 'camera_autogain_supported'):
            method = _method_node(CONFIG_GETTERS_PATH, fn)
            assert _calls_named(method, '_live_capabilities') or any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == '_live_capabilities'
                for n in ast.walk(method)
            ), f'{fn} must resolve capabilities via _live_capabilities().'

    def test_init_ui_applies_autogain_support_on_connect(self):
        method = _method_node(IMAGE_SETTINGS_PATH, '_init_ui')
        assert _calls_named(method, 'set_layer_autogain_support'), (
            'ImageSettings._init_ui must apply the autogain gate on connect.'
        )

    def test_reconnect_applies_autogain_support_on_reconnect(self):
        method = _method_node(MS_PATH, 'reconnect')
        assert _calls_named(method, 'set_layer_autogain_support'), (
            'MicroscopeSettings.reconnect must re-apply the autogain gate on '
            'scope-change / reconnect.'
        )

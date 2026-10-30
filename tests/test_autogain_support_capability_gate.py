"""Regression: the Auto Gain/Exp control is hidden on cameras without hardware auto-gain.

The "Auto Gain/Exp" checkbox drives the camera's HARDWARE auto-gain
(``imaging.set_auto_gain`` -> ``driver.auto_gain``), which ``CameraAPI``
refuses on a camera without it (IDS U3-34Lx, FX2 LS620); auto-gain and
auto-exposure come together on every shipped camera. So the control is
shown only where the scope answers ``capabilities.camera_supports_auto_gain``.

  - ``ImageSettings.set_layer_autogain_support`` asks the scope and sets
    ``LayerControl.camera_autogain_support`` on every layer, an orthogonal
    runtime gate AND-ed in the kv with the per-layer static
    ``autogain_support``, so Lumi stays hidden regardless. It runs in the
    capability-sync grouping ``ImageSettings._init_ui`` runs on connect.
  - The effective enable is the API's answer, read NON-DESTRUCTIVELY at the
    consumption point (``LayerControl.effective_auto_gain`` reads
    ``imaging.applied_auto_gain_for``), used by ``apply_settings`` for the
    slider-disable / camera enable. The persisted ``auto_gain`` is never
    mutated, so a capable camera's saved preference survives a swap to an
    AG-less body and back.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from drivers.camera import Camera
from drivers.camera_profiles import lookup_profile
from tests.scope_fakes import build_scope, capabilities_of
from ui.image_settings import ImageSettings

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_PATH = REPO_ROOT / 'ui' / 'layer_control.py'
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
MS_PATH = REPO_ROOT / 'ui' / 'microscope_settings.py'
KV_PATH = REPO_ROOT / 'ui' / 'lumaviewpro.kv'


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


class TestTheControlFollowsTheScopesAnswer:
    """set_layer_autogain_support shows the control on every layer exactly where the scope's camera has auto-gain."""

    @pytest.mark.parametrize(('model', 'shown'), [('LS850T', True), ('LS620', False)])
    def test_each_layer_shows_the_control_where_the_camera_has_auto_gain(
        self, monkeypatch, model, shown
    ):
        scope = build_scope(simulate=True, sim_model=model, register_atexit=False)
        try:
            controls = {
                layer: SimpleNamespace(camera_autogain_support=None)
                for layer in common_utils.get_layers()
            }
            panel = ImageSettings.__new__(ImageSettings)
            panel._drawers = {
                layer: SimpleNamespace(ids={'control': control})
                for layer, control in controls.items()
            }
            monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(scope=scope))

            panel.set_layer_autogain_support()

            assert {c.camera_autogain_support for c in controls.values()} == {shown}
        finally:
            scope.disconnect()


class TestCapabilityMapping:
    """The capability flags reflect the per-camera hardware reality (real path)."""

    def _caps_for(self, model_name):
        from types import SimpleNamespace

        camera = SimpleNamespace(
            model_name=model_name,
            device_serial=None,
            device_firmware=None,
            active=None,
            get_binning_size=lambda: 1,
            timestamp_tick_frequency_hz=None,
            profile=lookup_profile(model_name),
            get_max_frame_size=lambda: {'width': 1024, 'height': 768},
            # The base driver's depth rule, which a real driver inherits.
            significant_bits_for_format=lambda name: Camera.format_significant_bits(name, 16),
            # The base driver's SDK label (none), which a real driver inherits.
            get_sdk_info=lambda: Camera.get_sdk_info(None),
        )
        return capabilities_of(camera)

    def test_ids_reports_no_hardware_autogain(self):
        caps = self._caps_for('U3-34LxXCP-M')
        assert caps.camera_supports_auto_gain is False
        assert caps.camera_supports_auto_exposure is False

    def test_fx2_ls620_reports_no_hardware_autogain(self):
        caps = self._caps_for('MT9P031-LS620')  # the name FX2Camera sets
        assert caps.camera_supports_auto_gain is False
        assert caps.camera_supports_auto_exposure is False

    def test_pylon_reports_hardware_autogain(self):
        assert self._caps_for('daA3840-45um').camera_supports_auto_gain is True

    def test_simulated_scope_reports_hardware_autogain(self):
        # Real-path anchor for the mocked getter tests: the sim camera has AG.
        assert build_scope(simulate=True).capabilities.camera_supports_auto_gain is True


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

    def test_effective_auto_gain_reads_the_api_answer(self):
        # The non-destructive read: the API decides what the saved preference
        # becomes on this camera; the GUI shows it and decides nothing.
        method = _method_node(LAYER_CONTROL_PATH, 'effective_auto_gain')
        body = ast.dump(method)
        assert _calls_named(method, 'applied_auto_gain_for') and 'auto_gain' in body, (
            'effective_auto_gain must show imaging.applied_auto_gain_for for the saved '
            'auto_gain preference.'
        )
        assert 'camera_autogain_support' not in body, (
            'effective_auto_gain must not decide from the visibility flag -- that '
            "decision is the API's."
        )

    def test_the_shown_layer_uses_effective_auto_gain(self):
        method = _method_node(LAYER_CONTROL_PATH, 'show_as_shown')
        assert _calls_named(method, 'effective_auto_gain'), (
            'show_as_shown must derive the box (and so the slider-disable) from '
            'effective_auto_gain(), not the raw stored auto_gain.'
        )

    def test_init_ui_applies_autogain_support_on_connect(self):
        # _init_ui applies the gate via the sync_camera_capability_ranges
        # grouping (which calls set_layer_autogain_support -- see
        # test_reconnect_camera_resync).
        method = _method_node(IMAGE_SETTINGS_PATH, '_init_ui')
        assert _calls_named(method, 'sync_camera_capability_ranges'), (
            'ImageSettings._init_ui must apply the autogain gate (via the '
            'sync_camera_capability_ranges grouping) on connect.'
        )

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the AG/Exp toggle <-> settings sync in apply_settings.

Bug
---
LayerControl's auto_gain CheckBox has no Kivy binding from
settings[layer]['auto_gain']. The .kv ships the CheckBox with active=False
as a static default and only binds on_release to update_auto_gain. So when
LVP starts with settings loaded from disk that have auto_gain=True for some
layer, the CheckBox stays at False while apply_settings reads True from
settings and applies it to the camera. The user sees toggle OFF while the
camera is actually running AG -- the symptom reported on GitHub issue #655
("Green AG/Exp was active with the toggle OFF. Clicking it on and off made
it work.").

Fix (apply_settings sync)
-------------------------
At the start of the auto_gain block in apply_settings, sync the CheckBox
.active and the dependent slider .disabled flags to settings before the
IOTask is queued. Programmatic .active = bool fires no on_release in
Kivy (CheckBox in the .kv only binds on_release, not on_active), so this
does not re-enter the update_auto_gain callback path.

Test approach
-------------
Source-level structural locks via AST extraction. Mirror the
TestPylonDeviceNotFoundClassification pattern in test_audit_fixes.py.
Behavioral test of the sync would require a full Kivy CheckBox + slider
mock harness (LayerControl(BoxLayout) is MagicMock'd in the test env),
which is heavier than the bug warrants; the source-level lock catches a
future cleanup that drops the sync.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'


def _method_body(class_name: str, method_name: str) -> str:
    """Extract the source text of a class method as a string."""
    source = LAYER_CONTROL_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    text = ast.get_source_segment(source, child)
                    if text is None:
                        raise AssertionError(
                            f'could not extract source for {class_name}.{method_name}'
                        )
                    return text
    raise AssertionError(f'{class_name}.{method_name} not found in {LAYER_CONTROL_SRC}')


class TestApplySettingsSyncsAutoGainCheckbox:
    """apply_settings must sync the auto_gain CheckBox active flag and the
    dependent slider disabled flags to settings[layer]['auto_gain'] before
    queuing the camera-settings IOTask.

    Without this sync, a settings-vs-UI divergence persists across the
    apply_settings call: the camera AG state ends up matching settings
    (correct) but the toggle UI continues to show the stale .kv default
    of False. The user has to click the toggle on-then-off to force the
    update_auto_gain_cb path to re-write settings and re-fire
    apply_settings, at which point the toggle visibly reflects state.

    The sync is gated by ``if not protocol_running_global.is_set()`` --
    same guard as the rest of the auto_gain block. During protocols the
    layer's AG state is managed by protocol_step_runner with
    ignore_auto_gain=True; the UI-vs-settings sync is irrelevant there.
    """

    def test_apply_settings_syncs_checkbox_active(self):
        """The auto_gain block must contain the CheckBox.active <- settings
        sync. Structural check; a future cleanup that drops the sync
        line would reintroduce the #655 toggle/camera divergence."""
        body = _method_body('LayerControl', 'apply_settings')
        assert "self.ids['auto_gain'].active = auto_gain_enabled" in body, (
            'LayerControl.apply_settings must sync the auto_gain CheckBox '
            ".active flag to settings[layer]['auto_gain'] inside the "
            'non-protocol auto_gain block. See class docstring for the '
            '#655 divergence this protects against.'
        )

    def test_apply_settings_syncs_slider_disabled(self):
        """The auto_gain block must also sync the gain/exposure slider +
        text-input .disabled flags so the UI's editable state reflects
        AG-vs-manual mode. Without this, after a settings-driven AG
        toggle change the user might still see editable sliders for
        values the camera is actively overriding (or vice versa)."""
        body = _method_body('LayerControl', 'apply_settings')
        # Pattern: for slider_item in ('gain_slider', 'gain_text', ...)
        # Loose check (any iteration over those four ids assigning disabled).
        assert 'gain_slider' in body, 'apply_settings auto_gain sync must reference gain_slider'
        assert 'gain_text' in body, 'apply_settings auto_gain sync must reference gain_text'
        assert 'exp_slider' in body, 'apply_settings auto_gain sync must reference exp_slider'
        assert 'exp_text' in body, 'apply_settings auto_gain sync must reference exp_text'
        assert '.disabled = auto_gain_enabled' in body, (
            'apply_settings auto_gain sync must set .disabled = '
            'auto_gain_enabled on the gain/exposure slider + text widgets.'
        )

    def test_apply_settings_sync_precedes_iotask_queue(self):
        """The CheckBox + slider sync must precede the apply_layer_camera_settings
        IOTask queue so the UI reflects the new state by the time the
        camera command lands. The opposite order would leave a brief
        window where the camera state has changed but the UI lags."""
        body = _method_body('LayerControl', 'apply_settings')
        sync_idx = body.find("self.ids['auto_gain'].active = auto_gain_enabled")
        queue_idx = body.find('apply_layer_camera_settings')
        assert sync_idx >= 0, (
            'sync line missing (precondition test_apply_settings_syncs_checkbox_active)'
        )
        assert queue_idx >= 0, 'apply_layer_camera_settings call missing (precondition)'
        assert sync_idx < queue_idx, (
            'CheckBox sync must precede the apply_layer_camera_settings '
            'IOTask queue. Reverse ordering leaves a window where the '
            "camera state has changed but the UI hasn't caught up."
        )

    def test_apply_settings_sync_inside_non_protocol_guard(self):
        """The sync code must be inside the ``if not protocol_running_global.is_set():``
        block (same guard as the rest of the auto_gain handling).
        During protocols, protocol_step_runner manages AG state with
        ignore_auto_gain=True; syncing the toggle UI from settings
        during a protocol-driven layer switch would be incorrect."""
        body = _method_body('LayerControl', 'apply_settings')
        # Find the guard line and the sync line; assert sync comes after the guard.
        guard_idx = body.find('if not protocol_running_global.is_set():')
        sync_idx = body.find("self.ids['auto_gain'].active = auto_gain_enabled")
        assert guard_idx >= 0, 'protocol_running guard not found (precondition)'
        assert sync_idx >= 0, 'sync line not found (precondition)'
        assert guard_idx < sync_idx, (
            'auto_gain CheckBox sync must be INSIDE the '
            '`if not protocol_running_global.is_set():` block, not '
            'outside it. Syncing during a protocol-driven layer change '
            "would fight protocol_step_runner's AG management."
        )

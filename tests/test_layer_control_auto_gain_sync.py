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
.active to settings before the IOTask is queued. Programmatic
.active = bool fires no on_release in Kivy (CheckBox in the .kv only
binds on_release, not on_active), so this does not re-enter the
update_auto_gain callback path. The gain/exposure widgets' enabled
state follows the box through ONE kv rule,
``disabled: app.run_lockout or auto_gain.active``: an imperative
``.disabled`` write from Python was erased at every run boundary, because
the widgets' ``disabled: app.run_lockout`` rule re-fires when the lockout
clears and overwrites whatever Python last wrote.

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

import pytest

from tests.ast_seams import find_def


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'
# pin-justified: the kv is declarative source with no headless seam; the
# per-widget event binds and disabled expressions below are the contract.
KV_LINES = (REPO / 'ui' / 'lumaviewpro.kv').read_text().splitlines()


def _kv_indent(line: str) -> int:
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _kv_widget_block(widget_id: str) -> list[str]:
    """Every property line of the widget carrying ``id: <widget_id>``.

    The properties sit at one indent under the widget's class line; the
    block runs from the first property to the last, in both directions from
    the id line, so a bind written above the id is seen too.
    """
    marker = f'id: {widget_id}'
    idx = next(
        i
        for i, line in enumerate(KV_LINES)
        if line.strip() == marker or line.strip().startswith(marker + ' ')
    )
    depth = _kv_indent(KV_LINES[idx])

    def _same_depth(i: int) -> bool:
        line = KV_LINES[i]
        return not line.strip() or _kv_indent(line) >= depth

    start = idx
    while start > 0 and _same_depth(start - 1):
        start -= 1
    end = idx
    while end + 1 < len(KV_LINES) and _same_depth(end + 1):
        end += 1
    return [line.strip() for line in KV_LINES[start : end + 1] if line.strip()]


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
    """apply_settings must sync the auto_gain CheckBox active flag to
    settings[layer]['auto_gain'] before queuing the camera-settings IOTask;
    the gain/exposure widgets' disabled state follows the box in the kv.

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

    @pytest.mark.parametrize('widget_id', ['gain_slider', 'gain_text', 'exp_slider', 'exp_text'])
    def test_gain_exposure_widgets_follow_the_auto_gain_box_in_kv(self, widget_id):
        """The kv rule is the single owner of the four widgets' enabled
        state: the run lockout OR the auto-gain box. A Python-side
        `.disabled` write was clobbered at every run boundary (the rule
        re-fires on the lockout edge), which left the sliders editable
        under a live auto-gain after a run."""
        block = _kv_widget_block(widget_id)
        assert 'disabled: app.run_lockout or auto_gain.active' in block, block

    def test_no_imperative_disabled_write_for_the_gain_exposure_widgets(self):
        """No Python writer competes with the kv rule."""
        source = LAYER_CONTROL_SRC.read_text()
        for pattern in ('.disabled = auto_gain_enabled', '.disabled = state'):
            assert pattern not in source, (
                f'{pattern!r} in ui/layer_control.py: the kv rule owns the '
                "gain/exposure widgets' disabled state"
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
        guard_idx = body.find('if not ctx.session.run_lockout:')
        sync_idx = body.find("self.ids['auto_gain'].active = auto_gain_enabled")
        assert guard_idx >= 0, 'protocol_running guard not found (precondition)'
        assert sync_idx >= 0, 'sync line not found (precondition)'
        assert guard_idx < sync_idx, (
            'auto_gain CheckBox sync must be INSIDE the '
            '`if not ctx.session.run_lockout:` block, not '
            'outside it. Syncing during a protocol-driven layer change '
            "would fight protocol_step_runner's AG management."
        )


class TestSetStepStateWidgetWiring:
    """What makes ``LayerControl.set_step_state`` a pure widget setter is
    the kv wiring, not the ``_initializing`` flag: of the widgets it sets,
    only the illumination / gain / exposure sliders bind ``on_value`` (and
    each handler opens with the ``_initializing`` guard); the other sliders
    bind ``on_release``, which ModSlider dispatches only from a touch-up or
    the wheel, and a CheckBox's ``on_release`` never fires from a
    programmatic ``.active`` write. A slider re-bound to ``on_value`` would
    turn every step display into a settings write again."""

    @pytest.mark.parametrize(
        'slider_id',
        [
            'sum_slider',
            'video_duration_slider',
            'stim_ill_slider',
            'stim_freq_slider',
            'stim_pulse_width_slider',
            'stim_pulse_count_slider',
        ],
    )
    def test_release_sliders_bind_on_release_not_on_value(self, slider_id):
        block = _kv_widget_block(slider_id)
        assert any(line.startswith('on_release:') for line in block), block
        assert not any(line.startswith('on_value:') for line in block), block

    @pytest.mark.parametrize('handler', ['ill_slider', 'gain_slider', 'exp_slider'])
    def test_value_handlers_open_with_the_initializing_guard(self, handler):
        """Before the guard, only a plain name assignment or an
        early-return ``if`` may appear -- no settings write, no call."""
        fn = find_def('ui/layer_control.py', handler, class_name='LayerControl')
        assert fn is not None, handler

        def _is_guard(stmt):
            return (
                isinstance(stmt, ast.If)
                and ast.unparse(stmt.test) == 'self._initializing'
                and len(stmt.body) == 1
                and isinstance(stmt.body[0], ast.Return)
            )

        guard_idx = next((i for i, stmt in enumerate(fn.body) if _is_guard(stmt)), None)
        assert guard_idx is not None, f'{handler} has no `if self._initializing: return`'
        for stmt in fn.body[:guard_idx]:
            harmless = (
                (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
                or (
                    isinstance(stmt, ast.Assign)
                    and all(isinstance(t, ast.Name) for t in stmt.targets)
                )
                or (
                    isinstance(stmt, ast.If)
                    and len(stmt.body) == 1
                    and isinstance(stmt.body[0], ast.Return)
                )
            )
            assert harmless, f'{handler} acts before its _initializing guard: {ast.unparse(stmt)}'

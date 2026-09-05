# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run displays each step in the layer panel but never writes it
into the user's live layer settings; the panel re-syncs from settings at run
end.

The defect: the run's step executor navigates to every step through the
GUI's ``go_to_step`` callback, which wrote the step's nine values into the
user's live layer settings on every step, and the per-layer widget setter
wrote two more (video and stim config). After a run the user's live
configuration was the protocol's last step per colour: a stored auto-gain
preference flipped off by an AG-off step, an illumination the user never
set, a focus the run chose. With the run's cleanup re-arming the camera
from its own snapshot, the panel and the camera then disagreed until the
first slider touch, and the panel won.

Contract under test:

- ``go_to_step`` writes the layer's settings ONLY for manual navigation
  (``called_from_protocol=False``); a protocol-cycle invocation displays the
  step and leaves the settings alone.
- ``LayerControl.set_step_state`` is a pure widget setter: it touches no
  settings.
- At run end the cleanup schedules ``sync_layer_widgets`` exactly once,
  outside the autofocus-restore loop and its empty-snapshot gate, and that
  callback re-syncs every layer's widgets from its stored settings through
  the one settings-to-widgets implementation (the startup loop, extracted).
"""

from __future__ import annotations

import ast
import copy
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.ast_seams import find_def, parse_module
from tests.test_issue_733_step_nav_preview_led_button import (  # noqa: F401
    GREEN_LAYER_SETTINGS,
    stepnav_env,
)


# A step whose every field differs from GREEN_LAYER_SETTINGS, so a write of
# any key is visible as a change.
def _make_step():
    return {
        'X': 10.0,
        'Y': 20.0,
        'Z': 7.5,
        'Color': 'Green',
        'Auto_Focus': True,
        'False_Color': False,
        'Illumination': 350.0,
        'Gain': 12.5,
        'Auto_Gain': True,
        'Exposure': 40.0,
        'Sum': 4,
        'Acquire': 'video',
        'Objective': 'obj1',
        'Video Config': {'duration': 30},
        'Stim_Config': {
            'Green': {
                'enabled': True,
                'illumination_ma': 200,
                'frequency': 2,
                'pulse_width': 20,
                'pulse_count': 3,
            },
        },
    }


# The settings key each step column lands in under manual navigation.
STEP_TO_SETTINGS = {
    'Auto_Focus': 'autofocus',
    'False_Color': 'false_color',
    'Illumination': 'illumination_ma',
    'Gain': 'gain_db',
    'Auto_Gain': 'auto_gain',
    'Exposure': 'exposure_ms',
    'Sum': 'sum',
    'Acquire': 'acquire',
    'Z': 'focus',
    'Video Config': 'video_config',
}


def _go_to_step(step, *, called_from_protocol, include_move=True):
    import ui.step_navigation as step_navigation

    protocol = SimpleNamespace(
        num_steps=MagicMock(return_value=1),
        step=MagicMock(return_value=step),
    )
    step_navigation.go_to_step(
        protocol,
        step_idx=0,
        include_move=include_move,
        called_from_protocol=called_from_protocol,
    )


class TestRunNavigationLeavesLayerSettingsAlone:
    def test_run_navigation_leaves_layer_settings_alone(self, stepnav_env):
        """A protocol-cycle invocation leaves the layer's settings exactly
        as they were. The display still follows: the widget setter is
        called once with the step -- the green half of this test, so a
        future change cannot silence the display to satisfy the red half."""
        env = stepnav_env
        env.ctx.session.run_lockout = True
        before = copy.deepcopy(env.ctx.settings['Green'])
        step = _make_step()

        _go_to_step(step, called_from_protocol=True)

        assert env.ctx.settings['Green'] == before, (
            "a protocol run must not write the step into the user's layer settings; "
            f'changed: {sorted(k for k in before if env.ctx.settings["Green"].get(k) != before[k])}'
        )
        assert env.layer_obj.set_step_state.call_count == 1
        assert env.layer_obj.set_step_state.call_args.args[0] is step

    def test_run_navigation_without_move_neither_writes_nor_displays(self, stepnav_env):
        """include_move=False skips the whole step block, the display update
        included. Recorded so a future caller cannot lose the display
        silently by flipping that flag."""
        env = stepnav_env
        env.ctx.session.run_lockout = True
        before = copy.deepcopy(env.ctx.settings['Green'])

        _go_to_step(_make_step(), called_from_protocol=True, include_move=False)

        assert env.ctx.settings['Green'] == before
        assert env.layer_obj.set_step_state.call_count == 0


class TestManualNavigationLoadsTheStepIntoTheLayer:
    def test_manual_navigation_loads_the_step_into_the_layer(self, stepnav_env, monkeypatch):
        """Manual navigation writes all eleven keys, the two config dicts as
        deep copies, and the write lands BEFORE the manual-nav outcome is
        applied (its apply_settings reads the settings)."""
        env = stepnav_env
        seen_at_outcome = {}

        def _record_outcome(**kwargs):
            seen_at_outcome.update(copy.deepcopy(kwargs['settings']['Green']))

        monkeypatch.setattr('ui.step_navigation._apply_manual_nav_outcome', _record_outcome)
        step = _make_step()

        _go_to_step(step, called_from_protocol=False)

        green = env.ctx.settings['Green']
        for column, key in STEP_TO_SETTINGS.items():
            assert green[key] == step[column], f"{key} did not take the step's {column}"
        assert green['stim_config'] == step['Stim_Config']['Green']
        assert seen_at_outcome, 'the manual-nav outcome must be applied'
        assert seen_at_outcome == green, 'the settings write must precede the outcome'

        # Deep copies: the step's dicts are the protocol's; mutating them
        # afterwards must not reach into the user's settings.
        step['Video Config']['duration'] = 999
        step['Stim_Config']['Green']['illumination_ma'] = 999
        assert green['video_config']['duration'] == 30
        assert green['stim_config']['illumination_ma'] == 200


# Every widget id set_step_state writes, with the attribute it writes.
_STEP_WIDGETS = {
    'autofocus': 'active',
    'false_color': 'active',
    'ill_text': 'text',
    'ill_slider': 'value',
    'gain_text': 'text',
    'gain_slider': 'value',
    'auto_gain': 'active',
    'exp_text': 'text',
    'exp_slider': 'value',
    'sum_text': 'text',
    'sum_slider': 'value',
    'video_duration_text': 'text',
    'video_duration_slider': 'value',
    'stim_enable_btn': 'active',
    'stim_disable_btn': 'active',
    'stim_ill_text': 'text',
    'stim_ill_slider': 'value',
    'stim_freq_text': 'text',
    'stim_freq_slider': 'value',
    'stim_pulse_width_text': 'text',
    'stim_pulse_width_slider': 'value',
    'stim_pulse_count_text': 'text',
    'stim_pulse_count_slider': 'value',
    'acquire_video': 'active',
    'acquire_image': 'active',
    'acquire_none': 'active',
}


class _Widget:
    def __init__(self):
        self.text = None
        self.value = None
        self.active = None
        self.state = None
        self.visible = None
        self.opacity = None
        self.max = None


class _NoSettingsCtx:
    """A ctx whose settings cannot be read: a pure widget setter never asks."""

    settings_lock = threading.Lock()

    @property
    def settings(self):
        raise AssertionError('set_step_state must not touch ctx.settings')


class _LayerStand:
    """A LayerControl stand-in: the widgets, the layer name, the capability
    flag, and no-op visibility; the methods under test are called unbound."""

    def __init__(self, layer, widget_ids, camera_autogain_support=True):
        from ui.layer_control import LayerControl

        self.ids = {name: _Widget() for name in widget_ids}
        self.layer = layer
        self.camera_autogain_support = camera_autogain_support
        self.show_stim_controls = None
        self._initializing = False
        self.visibility_calls = 0
        self.effective_auto_gain = LayerControl.effective_auto_gain.__get__(self)

    def update_stim_controls_visibility(self):
        self.visibility_calls += 1


class TestSetStepStateIsAPureWidgetSetter:
    def test_set_step_state_is_a_pure_widget_setter(self, monkeypatch):
        from ui.layer_control import LayerControl

        monkeypatch.setattr('modules.app_context.ctx', _NoSettingsCtx())
        stand = _LayerStand('Green', _STEP_WIDGETS)
        step = _make_step()

        LayerControl.set_step_state(stand, step)

        ids = stand.ids
        stim = step['Stim_Config']['Green']
        expected = {
            'autofocus': True,
            'false_color': False,
            'ill_text': '350.0',
            'ill_slider': 350.0,
            'gain_text': '12.5',
            'gain_slider': 12.5,
            'auto_gain': True,
            'exp_text': '40.0',
            'exp_slider': 40.0,
            'sum_text': '4',
            'sum_slider': 4,
            'video_duration_text': '30',
            'video_duration_slider': 30.0,
            'stim_enable_btn': True,
            'stim_disable_btn': False,
            'stim_ill_text': str(stim['illumination_ma']),
            'stim_ill_slider': float(stim['illumination_ma']),
            'stim_freq_text': str(stim['frequency']),
            'stim_freq_slider': float(stim['frequency']),
            'stim_pulse_width_text': str(stim['pulse_width']),
            'stim_pulse_width_slider': float(stim['pulse_width']),
            'stim_pulse_count_text': str(stim['pulse_count']),
            'stim_pulse_count_slider': int(stim['pulse_count']),
            'acquire_video': True,
            'acquire_image': False,
            'acquire_none': False,
        }
        actual = {name: getattr(ids[name], attr) for name, attr in _STEP_WIDGETS.items()}
        assert actual == expected
        assert stand.visibility_calls == 1
        assert stand._initializing is False

    def test_auto_gain_checkbox_is_gated_on_camera_support(self, monkeypatch):
        """A camera whose Auto Gain control is hidden never shows the box
        ticked: the kv greys the gain/exposure widgets off that box, and
        with the control hidden there would be no way to un-grey them."""
        from ui.layer_control import LayerControl

        monkeypatch.setattr('modules.app_context.ctx', _NoSettingsCtx())
        stand = _LayerStand('Green', _STEP_WIDGETS, camera_autogain_support=False)

        LayerControl.set_step_state(stand, {'Auto_Gain': True})

        assert stand.ids['auto_gain'].active is False


# Every widget the startup settings-to-widgets loop sets.
_SETTINGS_WIDGETS = (
    *_STEP_WIDGETS,
    'composite_threshold_slider',
    'stim_ill_box',
    'stim_pulse_count_box',
    'stim_freq_box',
    'stim_pulse_width_box',
)


class _CountingLock:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *exc):
        return False


def _layer_settings(**overrides):
    base = copy.deepcopy(GREEN_LAYER_SETTINGS)
    base['composite_brightness_threshold'] = 60
    base.update(overrides)
    return base


class TestRunEndResyncsEveryLayerOnceFromSettings:
    def test_run_end_callback_syncs_every_layer_once(self, monkeypatch):
        import modules.common_utils as common_utils
        import ui.layer_control as layer_control
        from ui.ui_helpers import sync_layer_widgets_from_settings

        layers = {layer: MagicMock() for layer in common_utils.get_layers()}
        ctx = SimpleNamespace(
            image_settings=SimpleNamespace(layer_lookup=lambda layer: layers[layer]),
        )
        monkeypatch.setattr('modules.app_context.ctx', ctx)

        sync_layer_widgets_from_settings()

        for layer, layer_obj in layers.items():
            assert layer_obj.sync_widgets_from_settings.call_count == 1, layer
        # The per-widget re-sync helpers the one implementation replaced.
        for retired in ('init_autofocus', 'init_acquire', 'sync_camera_widgets_from_settings'):
            assert not hasattr(layer_control.LayerControl, retired), retired

    @pytest.mark.parametrize('layer', ['BF', 'Green'])
    def test_sync_widgets_from_settings_sets_every_widget(self, monkeypatch, layer):
        """BF ships with a None stim_config; Green carries one. Both sync
        without raising, every widget carries the settings value, the
        auto_gain box carries the effective auto-gain, and the settings
        were read once under the settings lock."""
        from ui.layer_control import LayerControl

        stim = {
            'enabled': True,
            'illumination_ma': 120,
            'frequency': 3,
            'pulse_width': 15,
            'pulse_count': 2,
        }
        settings = {
            # A None stim_config is representable and ships that way.
            'BF': _layer_settings(auto_gain=True, acquire='image', sum=2, stim_config=None),
            'Green': _layer_settings(
                auto_gain=False, acquire=None, stim_config=stim, video_config={'duration': 12}
            ),
        }
        lock = _CountingLock()
        monkeypatch.setattr(
            'modules.app_context.ctx', SimpleNamespace(settings=settings, settings_lock=lock)
        )
        stand = _LayerStand(layer, _SETTINGS_WIDGETS)
        stand._initializing = True

        LayerControl.sync_widgets_from_settings(stand)

        layer_settings = settings[layer]
        ids = stand.ids
        assert lock.entries == 1
        assert stand._initializing is False
        assert ids['ill_slider'].value == layer_settings['illumination_ma']
        assert ids['gain_slider'].value == layer_settings['gain_db']
        assert ids['exp_slider'].value == layer_settings['exposure_ms']
        assert ids['false_color'].active == layer_settings['false_color']
        assert ids['sum_slider'].value == layer_settings['sum']
        assert ids['video_duration_text'].text == str(layer_settings['video_config']['duration'])
        assert ids['video_duration_slider'].value == layer_settings['video_config']['duration']
        assert ids['autofocus'].active == layer_settings['autofocus']
        assert ids['auto_gain'].active == bool(layer_settings['auto_gain'])
        if layer == 'BF':
            assert ids['acquire_image'].active is True
            assert ids['composite_threshold_slider'].value is None, 'BF has no composite threshold'
            assert ids['stim_enable_btn'].active is None, 'no stim_config: stim widgets untouched'
            assert stand.visibility_calls == 0
        else:
            assert ids['acquire_none'].active is True
            assert ids['composite_threshold_slider'].value == 60
            assert ids['stim_enable_btn'].active is True
            assert ids['stim_disable_btn'].active is False
            assert ids['stim_ill_text'].text == '120'
            assert ids['stim_ill_slider'].value == 120.0
            assert ids['stim_freq_slider'].value == 3.0
            assert ids['stim_pulse_width_slider'].value == 15.0
            assert ids['stim_pulse_count_slider'].value == 2
            for box in (
                'stim_ill_box',
                'stim_pulse_count_box',
                'stim_freq_box',
                'stim_pulse_width_box',
            ):
                assert ids[box].visible is False and ids[box].opacity == 0
            assert stand.visibility_calls == 1

    def test_cleanup_schedules_the_sync_once_outside_the_autofocus_restore(self):
        """The schedule sits outside every loop and outside the
        empty-snapshot gate: one call per run, whether or not any autofocus
        state was restored."""
        fn = find_def('modules/protocol_cleanup.py', 'run_cleanup')
        assert fn is not None
        parents = {}
        for node in ast.walk(fn):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        sites = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Attribute) and node.attr == 'sync_layer_widgets'
        ]
        assert sites, 'run_cleanup must schedule callbacks.sync_layer_widgets'
        for site in sites:
            node = site
            while node in parents:
                node = parents[node]
                assert not isinstance(node, (ast.For, ast.While)), (
                    'the sync must be scheduled once, not per restored layer'
                )
                if isinstance(node, ast.If):
                    assert 'autofocus_snapshot.states' not in ast.unparse(node.test), (
                        'the sync must not be gated on the autofocus snapshot'
                    )

    def test_callbacks_carry_sync_layer_widgets_not_reset_autofocus_btns(self):
        tree = parse_module('modules/protocol_callbacks.py')
        fields = {
            stmt.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == 'ProtocolCallbacks'
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        assert 'sync_layer_widgets' in fields
        assert 'reset_autofocus_btns' not in fields

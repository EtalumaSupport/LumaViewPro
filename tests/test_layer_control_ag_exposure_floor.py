# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the AG-feedback exposure floor in update_auto_gain_cb.

Bug
---
When auto-gain runs on a bright transmitted sample (BF/PC/DF), the Pylon
camera can drive ExposureTime to its physical minimum (~30 us = 0.030 ms
on common sensors). update_auto_gain_cb then reads that value via
get_exposure_time() and writes it to settings[layer]['exp_ms'] without an
appropriate floor:

- Fluorescence + luminescence already get a 1.0 ms floor via
  FLUORESCENCE_MIN_EXPOSURE_MS in the original conditional.
- Transmitted (BF/PC/DF) had NO conditional floor; the slider's .kv
  default min is 0.01 ms, so np.clip(0.030, 0.01, max) returned 0.030.
- The 0.030 ms write then fires set_exposure_time's <0.1 ms
  "Value should be in milliseconds" WARNING on every subsequent
  apply_settings (visible in the beta tester's beta9 logs as recurring spam).

Fix
---
Add TRANSMITTED_MIN_EXPOSURE_MS = 0.1 (matching set_exposure_time's
internal warning gate) and apply it via an else branch on the existing
get_image_layers conditional. Live AG output to the camera is untouched
(the floor applies only to the settings write-back).

Test approach
-------------
- Structural lock (source-level): the conditional has BOTH branches.
- Behavioral (AST-extract + exec pattern from
  test_layer_control_type_consistency.py): simulate the AG-off callback
  with sub-floor exp values; assert settings store the floored value,
  not the raw value.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import modules.common_utils as real_common_utils


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'


# ---------------------------------------------------------------------------
# Source-level structural lock
# ---------------------------------------------------------------------------


def _method_body(method_name: str) -> str:
    source = LAYER_CONTROL_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'LayerControl':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    text = ast.get_source_segment(source, child)
                    if text is None:
                        raise AssertionError(
                            f'could not extract source for LayerControl.{method_name}'
                        )
                    return text
    raise AssertionError(f'LayerControl.{method_name} not found')


class TestExposureFloorSourceStructure:
    """Source-level lock on the AG-feedback floor logic in
    update_auto_gain_cb."""

    def test_transmitted_min_constant_defined(self):
        """TRANSMITTED_MIN_EXPOSURE_MS must be a module-level constant.
        Bare-number 0.1 floors scattered through code violate Rule 27."""
        src = LAYER_CONTROL_SRC.read_text()
        assert 'TRANSMITTED_MIN_EXPOSURE_MS' in src, (
            'TRANSMITTED_MIN_EXPOSURE_MS must be defined at module scope. '
            'See class docstring for the bug it gates.'
        )
        # Ensure it's a numeric assignment (not a typo / stub).
        for line in src.splitlines():
            line = line.strip()
            if line.startswith('TRANSMITTED_MIN_EXPOSURE_MS'):
                assert '=' in line and '0.1' in line, (
                    f'TRANSMITTED_MIN_EXPOSURE_MS assignment must be 0.1, '
                    f"got: {line!r}. The value matches set_exposure_time's "
                    f'internal <0.1ms warning gate; changing it changes '
                    f'which AG-feedback values fire the warning.'
                )
                return
        raise AssertionError('TRANSMITTED_MIN_EXPOSURE_MS assignment not found')

    def test_floor_conditional_covers_both_classes(self):
        """update_auto_gain_cb must apply BOTH FLUORESCENCE_MIN_EXPOSURE_MS
        and TRANSMITTED_MIN_EXPOSURE_MS to the AG-feedback exp value.
        A missing else branch reintroduces the BF AG -> 0.03 ms ->
        warning-spam path."""
        body = _method_body('update_auto_gain_cb')
        assert 'FLUORESCENCE_MIN_EXPOSURE_MS' in body, (
            'update_auto_gain_cb must reference FLUORESCENCE_MIN_EXPOSURE_MS '
            'in the AG-feedback floor (fluorescence + luminescence branch).'
        )
        assert 'TRANSMITTED_MIN_EXPOSURE_MS' in body, (
            'update_auto_gain_cb must reference TRANSMITTED_MIN_EXPOSURE_MS '
            'in the AG-feedback floor (transmitted else branch). See class '
            'docstring for the BF/PC/DF bug this catches.'
        )


# ---------------------------------------------------------------------------
# Behavioral test: AST-extract + exec pattern
# ---------------------------------------------------------------------------


def _extract_method_source(class_name: str, method_name: str) -> str:
    source = LAYER_CONTROL_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.unparse(child)
    raise AssertionError(f'{class_name}.{method_name} not found in source')


def _compile_cb():
    """Compile update_auto_gain_cb into a standalone callable."""
    fn_src = _extract_method_source('LayerControl', 'update_auto_gain_cb')
    # Stub common_utils.get_image_layers to return the fluorescence + lumi
    # list; the validity predicates are the REAL ones so the stub cannot
    # drift from production's definition of a usable reading.
    common_utils_stub = SimpleNamespace(
        get_image_layers=lambda: ['Blue', 'Green', 'Red', 'Lumi'],
        is_valid_gain_db=real_common_utils.is_valid_gain_db,
        is_valid_exposure_ms=real_common_utils.is_valid_exposure_ms,
    )
    app_ctx_stub = SimpleNamespace(ctx=SimpleNamespace(settings={}))
    ns = {
        'np': np,
        'logger': MagicMock(),
        'common_utils': common_utils_stub,
        '_app_ctx': app_ctx_stub,
        # Constants the floor references -- must match production values.
        'FLUORESCENCE_MIN_EXPOSURE_MS': 1.0,
        'TRANSMITTED_MIN_EXPOSURE_MS': 0.1,
    }
    exec(compile(fn_src, '<layer_control::update_auto_gain_cb>', 'exec'), ns)
    return ns['update_auto_gain_cb'], app_ctx_stub


def _make_fake_layer(layer: str, slider_min: float, slider_max: float = 1000.0):
    """Fake `self` for LayerControl.update_auto_gain_cb.

    Reflects the post-AG-off state: toggle is 'normal' (state=False), so
    the (not init) and (not state) branch fires that writes settings.
    """
    fake = SimpleNamespace()
    fake.layer = layer

    fake.ids = {}
    fake.ids['auto_gain'] = MagicMock()
    fake.ids['auto_gain'].state = 'normal'  # toggle up = AG off
    fake.ids['exp_slider'] = MagicMock()
    fake.ids['exp_slider'].min = slider_min
    fake.ids['exp_slider'].max = slider_max
    fake.ids['gain_slider'] = MagicMock()
    fake.ids['gain_slider'].value = 0
    fake.ids['gain_slider'].min = 0
    fake.ids['gain_slider'].max = 48
    fake.ids['gain_text'] = MagicMock()
    fake.ids['gain_text'].text = '0'
    fake.ids['exp_text'] = MagicMock()
    fake.ids['exp_text'].text = '0'
    fake.apply_settings = MagicMock()
    return fake


class TestExposureFloorBehavior:
    """Behavioral verification that AG-feedback writes are floored before
    landing in settings[layer]['exp_ms']."""

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 0.1),  # Pylon ExposureTime.Min for ace 2 etc.
            (0.05, 0.1),  # below threshold
            (0.099, 0.1),  # just below threshold
            (0.1, 0.1),  # at threshold (still allowed)
            (5.0, 5.0),  # above threshold -> passes through
        ],
    )
    def test_bf_ag_feedback_floored_to_transmitted_min(self, raw_exp_ms, expected_floor):
        """For BF (transmitted), AG-feedback exp values < 0.1 ms must be
        floored to 0.1 before being written to settings. Without this,
        the next apply_settings fires the set_exposure_time(<0.1ms)
        WARNING on every layer switch."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'BF': {'exp_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        fake = _make_fake_layer('BF', slider_min=0.01)  # .kv default for transmitted

        # AG-off callback: init=False, state=False (read from toggle), gain, exp.
        cb(fake, result=(False, False, 0.0, raw_exp_ms))

        stored = app_ctx_stub.ctx.settings['BF']['exp_ms']
        assert stored == expected_floor, (
            f'BF AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms, got {stored}ms. See class docstring '
            f'for the WARNING-spam bug this floor prevents.'
        )

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 1.0),  # camera minimum -> 1ms fluorescence floor
            (0.5, 1.0),  # below fluo floor
            (0.999, 1.0),  # just below fluo floor
            (1.0, 1.0),  # at fluo floor
            (15.0, 15.0),  # above floor -> passes through
        ],
    )
    def test_blue_ag_feedback_floored_to_fluorescence_min(self, raw_exp_ms, expected_floor):
        """For Blue (fluorescence), AG-feedback exp values < 1 ms must be
        floored to 1.0 (FLUORESCENCE_MIN_EXPOSURE_MS). Pre-existing
        behavior; this test locks it against accidental regression
        during the transmitted-floor refactor."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'Blue': {'exp_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        fake = _make_fake_layer('Blue', slider_min=1.0)  # set_layer_exposure_ranges value

        cb(fake, result=(False, False, 0.0, raw_exp_ms))

        stored = app_ctx_stub.ctx.settings['Blue']['exp_ms']
        assert stored == expected_floor, (
            f'Blue AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms (FLUORESCENCE_MIN_EXPOSURE_MS), '
            f'got {stored}ms.'
        )

    def test_pc_uses_transmitted_floor(self):
        """PC is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'PC': {'exp_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        fake = _make_fake_layer('PC', slider_min=0.01)
        cb(fake, result=(False, False, 0.0, 0.050))
        assert app_ctx_stub.ctx.settings['PC']['exp_ms'] == 0.1

    def test_df_uses_transmitted_floor(self):
        """DF is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'DF': {'exp_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        fake = _make_fake_layer('DF', slider_min=0.01)
        cb(fake, result=(False, False, 0.0, 0.050))
        assert app_ctx_stub.ctx.settings['DF']['exp_ms'] == 0.1

    def test_lumi_uses_fluorescence_floor(self):
        """Lumi (luminescence) is in get_image_layers; must use the 1.0
        floor like fluorescence, not the 0.1 transmitted floor."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'Lumi': {'exp_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        fake = _make_fake_layer('Lumi', slider_min=1.0)
        cb(fake, result=(False, False, 0.0, 0.5))
        assert app_ctx_stub.ctx.settings['Lumi']['exp_ms'] == 1.0

    def test_unknown_exposure_keeps_previous_settings(self):
        """A non-physical exposure reading (nothing was ever successfully
        read from the camera) must not overwrite the layer's stored
        exposure -- the previous value is the best truth available."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'BF': {'exp_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
        fake = _make_fake_layer('BF', slider_min=0.01)
        cb(fake, result=(False, False, 3.0, 0.0))
        assert app_ctx_stub.ctx.settings['BF']['exp_ms'] == 42.0
        # The valid gain reading in the same callback still lands.
        assert app_ctx_stub.ctx.settings['BF']['gain_db'] == 3.0

    def test_unknown_gain_keeps_previous_settings(self):
        """A non-physical gain reading must not overwrite the layer's
        stored gain, while a valid exposure in the same callback still
        floors and lands."""
        cb, app_ctx_stub = _compile_cb()
        app_ctx_stub.ctx.settings = {'BF': {'exp_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
        fake = _make_fake_layer('BF', slider_min=0.01)
        cb(fake, result=(False, False, -1.0, 5.0))
        assert app_ctx_stub.ctx.settings['BF']['gain_db'] == 7.0
        assert app_ctx_stub.ctx.settings['BF']['exp_ms'] == 5.0

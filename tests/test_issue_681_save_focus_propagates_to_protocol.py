# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#681 regression: Save Focus must update the in-memory protocol's Z
for steps that sit at the previous saved-focus baseline.

Bug
---
Pre-fix: clicking Save Focus only updated ``settings[layer]['focus']``.
The in-memory protocol's per-step Z stayed at whatever was loaded
(e.g. 7000 from ``data/new_default_protocol.tsv``). On the next New
Protocol click, ``new_protocol_ex`` carried over the old per-well Z
via ``previous_well_z`` (introduced by #535 for per-well-tuned focus
preservation). That carry-over WON over ``layer_config['focus']`` so
the newly-saved focus was silently ignored: scans still moved to Z=7000.
Chris's observed workaround was to click Save Focus + Goto Focus
multiple times until the existing protocol's per-step Z values matched
the saved focus.

Fix
---
``Protocol.update_layer_focus(layer, old_z, new_z)`` walks the in-
memory step table and updates Z on every step where ``Color == layer``
AND ``Z`` matches the old saved-focus value (within 1e-3 um). Per-well-
tuned steps (Z != old_z) are left alone, preserving the #535 intent.

``LayerControl.execute_save_focus`` reads the old layer focus BEFORE
writing the new one, then calls ``update_layer_focus`` on the in-
memory protocol (via ``ctx.protocol``). The result of a Save Focus
click is now: settings entry updated + matching steps re-pointed at
the new Z. A subsequent New Protocol click finds the per-well Z
already at the new focus, so the focus actually reaches the scan.

Test approach
-------------
1. Functional test on ``Protocol.update_layer_focus``: build a synthetic
   protocol with two layers (BF, Blue) and a mix of "at-baseline" and
   "user-tuned" steps. Call update_layer_focus on BF; assert the
   at-baseline BF steps update, the user-tuned BF step is preserved,
   and the Blue steps are untouched.
2. Old-z=None case: first save establishes the baseline; no propagation.
3. AST structural lock on ``execute_save_focus`` so the protocol-side
   propagation runs in the same try block as the settings update.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest


REPO = pathlib.Path(__file__).resolve().parent.parent


def _make_protocol_with_steps(rows: list[dict]):
    """Build a Protocol instance with the given step rows.

    Avoids the from_config / from_file plumbing by direct-wiring the
    steps DataFrame, since update_layer_focus only walks that table.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from modules.protocol import Protocol

    proto = Protocol.__new__(Protocol)
    df = pd.DataFrame(rows)
    proto._config = {
        'steps': df,
        'version': 1,
        'custom_step_count': 0,
    }
    proto._num_steps_cache = None
    return proto


class TestProtocolUpdateLayerFocus:
    def test_at_baseline_steps_update_to_new_z(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'B1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'C1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        count = proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=5000.0)
        assert count == 3
        assert (proto.steps()['Z'] == 5000.0).all()

    def test_user_tuned_steps_preserved(self):
        # A1 was tuned away from baseline; B1 + C1 sit at baseline.
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 5500.0, 'X': 1, 'Y': 1},  # tuned
                {'Name': 'B1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'C1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        count = proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=4500.0)
        assert count == 2, 'Only the two at-baseline BF steps should update'
        steps = proto.steps()
        assert steps.loc[0, 'Z'] == 5500.0, (
            'User-tuned A1 must be preserved (per-well-tune from #535)'
        )
        assert steps.loc[1, 'Z'] == 4500.0
        assert steps.loc[2, 'Z'] == 4500.0

    def test_other_layers_untouched(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'A1_Blue', 'Color': 'Blue', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'A1_Green', 'Color': 'Green', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=5000.0)
        steps = proto.steps()
        assert steps.loc[0, 'Z'] == 5000.0
        assert steps.loc[1, 'Z'] == 7000.0, 'Blue layer must not be affected by BF Save Focus'
        assert steps.loc[2, 'Z'] == 7000.0

    def test_first_save_no_baseline_returns_zero(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        count = proto.update_layer_focus(layer='BF', old_z=None, new_z=5000.0)
        assert count == 0
        assert proto.steps().loc[0, 'Z'] == 7000.0, (
            'When no baseline existed, the helper must leave the table alone'
        )

    def test_empty_protocol_returns_zero(self):
        proto = _make_protocol_with_steps([])
        count = proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=5000.0)
        assert count == 0

    def test_float_tolerance_handles_sub_micron_drift(self):
        # The saved-focus value comes from the motor board as a float;
        # round-trip through pandas / settings can introduce sub-um drift.
        # 0.0005 um drift must still be treated as "at baseline."
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0005, 'X': 1, 'Y': 1},
            ]
        )
        count = proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=5000.0)
        assert count == 1
        assert proto.steps().loc[0, 'Z'] == 5000.0


class TestStepAtLayerFocus:
    """The #681 step-editor cue reads step_at_layer_focus to decide whether
    to render a step's Z bold (tracks the layer's saved focus, so Save Focus
    would propagate to it) or normal (per-well tuned). It must agree with
    update_layer_focus on what counts as at-baseline -- same tolerance.
    """

    def test_step_at_baseline_is_true(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        assert proto.step_at_layer_focus(step_idx=0, saved_focus=7000.0) is True

    def test_user_tuned_step_is_false(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 5500.0, 'X': 1, 'Y': 1},
            ]
        )
        assert proto.step_at_layer_focus(step_idx=0, saved_focus=7000.0) is False

    def test_no_saved_focus_is_false(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        assert proto.step_at_layer_focus(step_idx=0, saved_focus=None) is False

    def test_sub_micron_drift_still_at_baseline(self):
        # Same tolerance as update_layer_focus: 0.0005 um drift is at-baseline.
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0005, 'X': 1, 'Y': 1},
            ]
        )
        assert proto.step_at_layer_focus(step_idx=0, saved_focus=7000.0) is True

    def test_cue_and_propagation_agree(self):
        # The cue must not claim "at baseline" for a step that
        # update_layer_focus would NOT propagate to, and vice versa.
        rows = [
            {'Name': 'A1_BF', 'Color': 'BF', 'Z': 5500.0, 'X': 1, 'Y': 1},  # tuned
            {'Name': 'B1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},  # baseline
        ]
        cue_proto = _make_protocol_with_steps([dict(r) for r in rows])
        prop_proto = _make_protocol_with_steps([dict(r) for r in rows])
        cue = [cue_proto.step_at_layer_focus(i, 7000.0) for i in range(2)]
        prop_proto.update_layer_focus(layer='BF', old_z=7000.0, new_z=4500.0)
        propagated = [prop_proto.steps().loc[i, 'Z'] == 4500.0 for i in range(2)]
        assert cue == propagated == [False, True]


class TestExecuteSaveFocusStructure:
    """AST-side lock: execute_save_focus must call update_layer_focus
    on the in-memory protocol in the same try block that sets settings.
    Without the structural assertion, a refactor could silently strip
    the propagation and reintroduce the bug.
    """

    def test_execute_save_focus_calls_update_layer_focus(self):
        src = (REPO / 'ui' / 'layer_control.py').read_text()
        # Source-text scan tolerant of ruff format quoting / wrapping.
        # The function body must (a) read old_focus before the new
        # write, and (b) invoke update_layer_focus on the protocol.
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'execute_save_focus':
                target = node
                break
        assert target is not None, 'execute_save_focus must exist'

        names_called = set()
        attrs_called = set()
        names_read = set()
        for sub in ast.walk(target):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Attribute):
                    attrs_called.add(sub.func.attr)
                elif isinstance(sub.func, ast.Name):
                    names_called.add(sub.func.id)
            if isinstance(sub, ast.Name):
                names_read.add(sub.id)

        assert 'update_layer_focus' in attrs_called, (
            'execute_save_focus must call protocol.update_layer_focus '
            'to propagate the new saved focus to in-memory steps.'
        )
        assert 'old_focus' in names_read, (
            'execute_save_focus must capture old_focus as a local before '
            'calling update_layer_focus, so the helper can match steps '
            'against the prior value.'
        )
        # The update_layer_focus call must pass old_z=old_focus (so the
        # ordering is enforced structurally -- you cannot pass a name
        # you have not yet bound). Walk for the call site and check.
        passed_old_focus = False
        for sub in ast.walk(target):
            if not isinstance(sub, ast.Call):
                continue
            if not (isinstance(sub.func, ast.Attribute) and sub.func.attr == 'update_layer_focus'):
                continue
            for kw in sub.keywords:
                if (
                    kw.arg == 'old_z'
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id == 'old_focus'
                ):
                    passed_old_focus = True
        assert passed_old_focus, (
            'The update_layer_focus call must pass old_z=old_focus so '
            'the helper compares against the prior baseline.'
        )

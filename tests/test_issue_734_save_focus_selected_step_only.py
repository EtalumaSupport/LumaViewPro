# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#734 regression: Save Focus writes ONLY the selected step, never siblings.

Bug class: intent inferred from float equality. Save Focus propagated the
new Z to every step of the layer whose Z matched the previous saved focus
(within 1e-3 um). Because both protocol builders create every step of a
layer at the identical layer focus, equality was the DEFAULT state, not
evidence the user wanted the step updated -- and step navigation re-synced
the layer focus to the viewed step's Z, so the "baseline" always matched
the current step. Net effect (customer log): tuning and saving three steps
one after another collapsed all three to the last saved value.

Fix: the equality inference is deleted outright (no update_layer_focus, no
tolerance constant). Save Focus writes the layer default plus the Z of the
step selected AT CLICK TIME, and nothing else. Sibling steps can only be
changed through the explicit apply-to-all-steps-in-channel action.

Where it is held: the selected-step-only write is ``ScopeSession.save_focus``,
pinned on the simulated scope by ``test_saving_a_focus_is_an_api_capability``.
This file keeps the bulk action's contract and the tripwire on the deleted
inference.
"""

from __future__ import annotations

import pathlib

import pandas as pd


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'
PROTOCOL_SRC = REPO / 'modules' / 'protocol.py'
SESSION_SRC = REPO / 'modules' / 'scope_session.py'
PROTOCOLS_API_SRC = REPO / 'modules' / 'lumascope_api' / 'protocols.py'


def _make_protocol_with_steps(rows: list[dict]):
    """Build a Protocol instance by direct-wiring the steps DataFrame."""
    import sys

    sys.path.insert(0, str(REPO))
    from modules.protocol import Protocol

    proto = Protocol.__new__(Protocol)
    proto._config = {
        'steps': pd.DataFrame(rows),
        'version': 1,
        'custom_step_count': 0,
    }
    proto._num_steps_cache = None
    return proto


class TestApplyFocusAllLayerSteps:
    """The explicit bulk action: every step of the channel, unconditionally."""

    def test_sets_whole_channel_and_returns_count(self):
        proto = _make_protocol_with_steps(
            [
                {'Name': 'A1_BF', 'Color': 'BF', 'Z': 7000.0, 'X': 1, 'Y': 1},
                {'Name': 'B1_BF', 'Color': 'BF', 'Z': 6500.0, 'X': 1, 'Y': 1},
                {'Name': 'A1_Blue', 'Color': 'Blue', 'Z': 7000.0, 'X': 1, 'Y': 1},
            ]
        )
        count = proto.apply_focus_all_layer_steps(layer='BF', z=5000.0)
        assert count == 2
        steps = proto.steps()
        assert steps.loc[0, 'Z'] == 5000.0
        assert steps.loc[1, 'Z'] == 5000.0, 'per-well-tuned step included -- unconditional'
        assert steps.loc[2, 'Z'] == 7000.0, 'other channels untouched'

    def test_empty_protocol_returns_zero(self):
        proto = _make_protocol_with_steps([])
        assert proto.apply_focus_all_layer_steps(layer='BF', z=5000.0) == 0

    def test_takes_no_previous_z_parameter(self):
        # The inference has no representation left: the bulk write cannot
        # be given a baseline to match against.
        import inspect
        import sys

        sys.path.insert(0, str(REPO))
        from modules.protocol import Protocol

        params = list(inspect.signature(Protocol.apply_focus_all_layer_steps).parameters)
        assert params == ['self', 'layer', 'z']


class TestNoBaselineEqualityInferenceRemains:
    """Absence lock: the equality-inference machinery must stay deleted.

    This is a tripwire, not the guard itself -- the guard is that no
    propagation code exists and the bulk action takes no old-Z parameter,
    so the inference has no representation left. The tripwire catches the
    names coming back under refactoring pressure.
    """

    def test_inference_names_absent_from_protocol_and_layer_control(self):
        for path in (PROTOCOL_SRC, LAYER_CONTROL_SRC, SESSION_SRC, PROTOCOLS_API_SRC):
            src = path.read_text()
            for name in (
                'update_layer_focus',
                'step_at_layer_focus',
                'FOCUS_BASELINE_TOLERANCE_UM',
            ):
                assert name not in src, (
                    f'{name} found in {path.name}: baseline-equality '
                    'propagation must not be reintroduced'
                )

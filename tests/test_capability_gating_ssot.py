# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Capability gating has one store: the drivers.

"Does this scope have focus / XY / turret?" used to be answered twice --
once from `scope.capabilities` (derived from the motion driver) and once
from `data/scopes.json` keyed by `settings['microscope']`, which a user
edits in Advanced Settings while the app runs. The UI gated on the
second. Selecting a model the hardware was not therefore produced an XY
stage control on a scope with no stage, and a protocol then imaged one
physical position while labelling every file with a different well.

These tests pin the single store: the UI asks the drivers, and the
drivers answer from the hardware that is actually attached.
"""

import ast
import pathlib

from drivers.null_motorboard import NullMotionBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from modules.scope_capabilities import ScopeCapabilities

REPO = pathlib.Path(__file__).resolve().parent.parent

# The scopes.json keys that duplicate a driver-derived capability. `Layers`
# is deliberately absent: no capability describes which illumination
# modalities a scope exposes, so that gating legitimately still reads the
# config file.
DUPLICATED_FLAGS = ('XYStage', 'Turret', 'Focus')


def _caps_for(motion):
    return ScopeCapabilities.from_drivers(motion=motion, led=None, camera=None)


class TestUiGatesOnTheDriver:
    """The UI reads capabilities; it must not re-derive them from the
    configured scope model."""

    def test_scope_ui_apply_path_reads_no_model_keyed_capability(self):
        tree = ast.parse((REPO / 'ui' / 'microscope_settings.py').read_text())
        func = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'set_ui_features_for_scope'
        )
        src = ast.unparse(func)
        for flag in DUPLICATED_FLAGS:
            assert f"'{flag}'" not in src, (
                f"set_ui_features_for_scope reads scopes.json '{flag}'; the scope "
                'model is user-editable at runtime, so gating on it lets the UI '
                'advertise hardware the driver does not have. Read '
                'scope.capabilities instead.'
            )
        assert 'capabilities' in src, (
            'set_ui_features_for_scope must gate the controls on scope.capabilities'
        )

    def test_no_ui_file_gates_on_a_model_keyed_capability(self):
        # Pass 2/3 of the cluster: the same shape lived in protocol_settings
        # (labware validation) and, as a mirror, in stage and the session.
        for path in sorted((REPO / 'ui').glob('*.py')):
            src = path.read_text()
            for flag in DUPLICATED_FLAGS:
                assert f"scope_config['{flag}']" not in src, (
                    f'{path.name} gates on scopes.json {flag!r}; ask the drivers'
                )

    def test_the_stale_capability_mirrors_are_gone(self):
        # Each was a copy of the XY fact written from the scope model. A
        # mirror needing manual sync is how the wrong value reached the
        # crosshair; deriving at read makes the stale state unconstructible.
        stage_src = (REPO / 'ui' / 'stage.py').read_text()
        assert 'set_xy_stage_capability' not in stage_src
        assert 'self._has_xy_stage' not in stage_src
        session_src = (REPO / 'modules' / 'scope_session.py').read_text()
        assert 'xystage_configured' not in session_src

    def test_reconnect_regates_the_ui(self):
        # Control visibility now comes from the drivers, so a reconnect onto
        # different hardware leaves the previous scope's controls on screen
        # unless the apply path runs again. There is no second store to
        # paper over it.
        tree = ast.parse((REPO / 'ui' / 'microscope_settings.py').read_text())
        func = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'reconnect'
        )
        assert 'reconfigure_for_scope' in ast.unparse(func), (
            'reconnect must re-gate the UI against the newly attached scope'
        )


class TestCapabilitiesComeFromTheHardware:
    def test_stage_less_scope_reports_no_xy(self):
        caps = _caps_for(SimulatedMotorBoard(model='Lumi'))
        assert caps.has_focus is True
        assert caps.has_xy_stage is False
        assert caps.has_turret is False

    def test_full_scope_reports_every_axis(self):
        caps = _caps_for(SimulatedMotorBoard(model='LS850T'))
        assert caps.has_focus is True
        assert caps.has_xy_stage is True
        assert caps.has_turret is True

    def test_disconnected_scope_offers_nothing(self):
        # Ruled 2026-08-27: with no motion board the app shows no motion
        # controls at all. Controls that drive nothing are the defect this
        # work removes, so an empty surface here is the honest answer and
        # must not be "restored" later as a bug fix.
        caps = _caps_for(NullMotionBoard())
        assert caps.axes == ()
        assert caps.has_focus is False
        assert caps.has_xy_stage is False
        assert caps.has_turret is False

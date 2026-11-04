# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The stage map shows the crosshair again once the position is known.

A home puts every axis in HOMING, and the map's redraw then hides the
crosshair because the position is unknown. When the home ends where the
crosshair was last drawn -- a Home pressed at the centre ends at the centre
-- the redraw that follows must still draw it: the crosshair it would skip
redrawing is no longer on screen.

The redraw's skip compares the position against what it last drew, so these
drive it over a simulated session and stop it at the first step past the
skip (the scope's labware lookup): reaching that step is "drawn", returning
before it is "skipped".
"""

from types import SimpleNamespace

import pytest

import ui.stage as stage_module
from tests.scope_fakes import home_sim_scope
from ui.stage import Stage

HOME = (0.0, 0.0)


class _DrawnError(Exception):
    """The redraw went past its skip and into the drawing."""


def _refuse_to_draw():
    raise _DrawnError


@pytest.fixture
def stage(gui_ctx, monkeypatch):
    """The stage map over an unhomed simulated session; the crosshair was last
    drawn where a home lands, the simulated stage's origin."""
    monkeypatch.setattr(gui_ctx.scope.runtime_state, 'get_labware', _refuse_to_draw)
    built = Stage.__new__(Stage)
    built._protocol_step_redraw = False
    built._prev_x_target, built._prev_y_target = HOME
    built._prev_x_current, built._prev_y_current = HOME
    return built


def _home(scope):
    home_sim_scope(scope)
    motion = scope.motion
    landed = [motion.get_target_positions()[a] for a in 'XY']
    landed += [motion.get_current_position(a) for a in 'XY']
    assert landed == [*HOME, *HOME], 'the home lands where the crosshair was last drawn'


def test_an_unchanged_known_position_is_not_redrawn(stage, gui_ctx):
    _home(gui_ctx.scope)

    assert stage.draw_labware_io_calculations() is None


def test_after_the_position_was_unknown_the_same_position_is_drawn(stage, gui_ctx):
    assert gui_ctx.scope.motion.axes_without_position(), 'the scope comes up unhomed'
    with pytest.raises(_DrawnError):  # no position: the redraw hides the crosshair
        stage.draw_labware_io_calculations()
    _home(gui_ctx.scope)
    with pytest.raises(_DrawnError):
        stage.draw_labware_io_calculations()


class _Plate:
    name = 'plate'

    def __init__(self):
        self.spacing_x_mm = 9
        self.spacing_y_mm = 9
        self.well_diameter_mm = 6
        self.width_mm = 127.76
        self.height_mm = 85.48


def test_a_homed_stage_whose_position_the_api_no_longer_answers_draws_none(
    stage, gui_ctx, monkeypatch
):
    """The close swaps in the null board between the redraw's axes_without_position() and
    its position reads: the reads answer None, and the whole redraw runs,
    hiding the crosshair, rather than raising on the Kivy clock."""
    scope = gui_ctx.scope
    _home(scope)
    # A stand-in by design: the simulator has no fault that lands the close's
    # board swap between two reads of one redraw, so the reads answer as the
    # null board does.
    monkeypatch.setattr(scope.motion, 'get_target_positions', lambda: {})
    monkeypatch.setattr(scope.motion, 'get_current_position', lambda axis: None)
    monkeypatch.setattr(scope.runtime_state, 'get_labware', lambda: _Plate())
    monkeypatch.setattr(stage_module, 'Clock', SimpleNamespace(schedule_once=lambda *a, **k: None))
    stage._cached_labware_name = 'plate'
    stage.width = stage.height = 100
    stage.x = stage.y = 0

    stage.draw_labware_io_calculations()


def test_the_crosshair_is_gated_on_the_scopes_xy_capability():
    """The plate graphic shows no crosshair on a scope with no XY stage --
    there is no live XY position to indicate (#710). What this pins is where
    the gate's truth comes from: the driver capability, read live. A gate fed
    from a copy is how a scope-model change once put a crosshair on a
    stage-less scope."""
    import ast

    from tests.ast_seams import parse_module

    tree = parse_module('ui/stage.py')
    # The crosshair sits inside an outer `if position_available:`, so every
    # enclosing If also contains the body markers -- collect them all and
    # require that ONE of the nested gates is the capability check.
    gates = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and '_crosshair_h_line' in '\n'.join(ast.unparse(s) for s in node.body)
        and 'h_line_points' in '\n'.join(ast.unparse(s) for s in node.body)
    ]
    assert gates, 'the per-frame crosshair update is no longer a gated If'
    # The gate may sit where the position is first decided available.
    gates += [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'position_available'
            for s in node.body
        )
    ]
    assert any('_xy_stage_present()' in gate for gate in gates), (
        f'the crosshair gate must derive from the scope XY capability; found {gates!r}'
    )

    helper = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == '_xy_stage_present'
    )
    assert 'capabilities.has_xy_stage' in ast.unparse(helper), (
        '_xy_stage_present must read the capability off the live scope, so a '
        'reconnect onto different hardware cannot leave a stale answer behind'
    )

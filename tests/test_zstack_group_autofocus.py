# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack is focused once per group and spans the range the user set.

Bug history
-----------
Bench 2026-09-03, LS850T: a Green layer with autofocus on and a 3-slice
5 um z-stack ran a sweep at EVERY slice. Each slice was moved to its
nominal Z, autofocused from there, and captured at the sweep's result --
so all three landed within 0.5 um of each other while the files kept
their _Z0_/_Z1_/_Z2_ names. The stack was not a stack, and nothing warned.

Root: the expansion copied the layer's Auto_Focus onto every slice, and
nothing told the slices they were a group whose focus was already found.

Two roles had to be separated to fix it, and they cannot be the same
slice:

* WHICH slice sweeps -- necessarily the group's first, because slices are
  acquired in order and a correction found at the middle slice arrives
  after the earlier ones have already been captured.
* WHICH slice the offsets are measured from -- the slice at the layer
  focus, because that is the plane the stack was built around. Measuring
  from the first slice instead would put the found focus at the BOTTOM of
  the stack rather than at its reference plane.

Auto_Focus on a slice now marks the second; the sweep fires at the first.

The placement writes the STORED steps frame deliberately: each step's move
is issued from that frame, so a correction applied anywhere else moves no
stage at all.
"""

from __future__ import annotations

import ast
import pathlib

from tests.test_protocol_roundtrip import _build_protocol, _make_step


_WIDE_Z = {'Z': {'limits': {'min': 0.0, 'max': 100_000.0}}}
# center reference at Z=5000, range 10, step 5 -> 4995 / 5000 / 5005
_ZSTACK = {'range': 10.0, 'step_size': 5.0, 'z_reference': 'center'}
_LAYER_FOCUS = 5000.0

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNNER_SRC = REPO / 'modules' / 'protocol_step_runner.py'


def _stacked(auto_focus: bool = True):
    proto = _build_protocol(
        [_make_step(name='A1_Green', z=_LAYER_FOCUS, z_slice=-1, auto_focus=auto_focus)]
    )
    proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=_WIDE_Z)
    return proto


def test_exactly_one_slice_per_group_carries_the_focus_flag():
    steps = _stacked().steps()
    assert len(steps) == 3
    assert list(steps['Auto_Focus']).count(True) == 1, list(steps['Auto_Focus'])


def test_the_flagged_slice_is_the_one_at_the_layer_focus():
    """Not slice 0 -- that would put the found focus at the stack's bottom."""
    steps = _stacked().steps()
    flagged = steps[steps['Auto_Focus']]
    assert len(flagged) == 1
    assert flagged.iloc[0]['Z'] == _LAYER_FOCUS


def test_a_layer_without_autofocus_flags_nothing():
    steps = _stacked(auto_focus=False).steps()
    assert list(steps['Auto_Focus']) == [False, False, False]


def test_the_anchor_answers_only_at_the_groups_first_slice():
    proto = _stacked()
    first, middle, last = list(proto.steps().index)
    # The reference plane is the middle slice; the sweep fires at the first.
    assert proto.zstack_group_focus_anchor(step_idx=first) == middle
    assert proto.zstack_group_focus_anchor(step_idx=middle) is None
    assert proto.zstack_group_focus_anchor(step_idx=last) is None


def test_placing_the_group_spans_the_configured_range_around_the_found_focus():
    proto = _stacked()
    anchor = proto.zstack_group_focus_anchor(step_idx=proto.steps().index[0])
    found = 5015.0  # 15 um above where the layer thought focus was

    moved = proto.apply_zstack_group_focus(reference_step_idx=anchor, z=found)

    assert moved == 3
    # Read back the STORED frame -- this is what the stage move is issued from.
    assert proto.steps()['Z'].tolist() == [5010.0, 5015.0, 5020.0]


def test_placing_the_group_preserves_spacing_for_every_reference_mode():
    for reference, expected in (
        ('center', [5010.0, 5015.0, 5020.0]),
        ('bottom', [5015.0, 5020.0, 5025.0]),
        ('top', [5005.0, 5010.0, 5015.0]),
    ):
        proto = _build_protocol(
            [_make_step(name='A1_Green', z=_LAYER_FOCUS, z_slice=-1, auto_focus=True)]
        )
        proto.apply_zstacking(
            zstack_params={**_ZSTACK, 'z_reference': reference}, axes_config=_WIDE_Z
        )
        anchor = proto.zstack_group_focus_anchor(step_idx=proto.steps().index[0])
        proto.apply_zstack_group_focus(reference_step_idx=anchor, z=5015.0)
        assert proto.steps()['Z'].tolist() == expected, reference


def test_a_second_scan_finding_the_same_focus_leaves_the_group_where_it_is():
    """A repeated scan must not compound the correction."""
    proto = _stacked()
    anchor = proto.zstack_group_focus_anchor(step_idx=proto.steps().index[0])

    proto.apply_zstack_group_focus(reference_step_idx=anchor, z=5015.0)
    after_first = proto.steps()['Z'].tolist()
    proto.apply_zstack_group_focus(reference_step_idx=anchor, z=5015.0)

    assert proto.steps()['Z'].tolist() == after_first


def test_the_runner_places_the_group_instead_of_the_single_step():
    """Wiring guard: a correction the capture path never reads moves nothing.

    The stage move is issued from the stored frame, so an earlier attempt
    that adjusted the run's own step row changed no position at all while
    still stamping the shifted Z into the file. Pin that the runner reaches
    the protocol's group placement.
    """
    tree = ast.parse(RUNNER_SRC.read_text(encoding='utf-8'))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert 'apply_zstack_group_focus' in called
    assert 'zstack_group_focus_anchor' in called


def _drive_group_scan(found_z: float, max_ticks: int = 400):
    """Run a real 3-slice group through scan_iterate, recording the Z moves.

    The suite's other z-stack tests are protocol-level: they pin the placement
    arithmetic and the stored frame, both of which were already correct while
    the stack still came out one step high. Nothing pinned the ORDER moves are
    issued in relative to the sweep, which is where the defect lived, so this
    drives the real step runner and watches the motor.

    The io executor runs its tasks inline, so the moves travel the production
    path -- protocol_put, the future, the wait -- and land on the mocked
    motion impl, which records every commanded position in order.
    """
    from concurrent.futures import Future
    from unittest.mock import MagicMock

    from modules.sequential_io_executor import PROTOCOL_ENQUEUED
    from tests.protocol_drives import protocol_step, scan_ready_runner

    class _InlineIOExecutor:
        def protocol_put(self, task, return_future=False):
            task.action(**task.kwargs)
            if return_future:
                fut = Future()
                fut.set_result(None)
                return fut
            return PROTOCOL_ENQUEUED

    proto = _stacked()
    runner = scan_ready_runner(protocol_step())
    runner._protocol = proto
    runner._io_executor = _InlineIOExecutor()
    runner._coordinate_transformer = MagicMock()
    runner._coordinate_transformer.plate_to_stage.return_value = (1.0, 2.0)
    runner._wellplate_loader = MagicMock()
    runner._scope.motion.is_moving.return_value = False

    # The state the runner is in on the poll after a sweep has resolved: the
    # future is done and already consumed, and AFE holds the found focus.
    runner._autofocus_runner.best_focus_position.return_value = found_z
    runner._autofocus_runner.complete.return_value = True
    runner._af_future = MagicMock()
    runner._af_future.done.return_value = True
    runner._af_future.exception.return_value = None
    runner._af_result_consumed = True

    for _ in range(max_ticks):
        if not runner._scan_in_progress.is_set():
            break
        runner._step_executor.scan_iterate()

    z_moves = [
        call.kwargs['position']
        for call in runner._scope.motion._move_absolute_impl.call_args_list
        if call.kwargs.get('axis') == 'Z'
    ]
    return proto, z_moves


def test_the_group_is_captured_on_the_ladder_its_placement_defines():
    """The sweeping slice is moved to its placed Z before it is captured.

    Its move was issued at the end of the previous step, before the sweep ran,
    so without a corrective move it captures wherever autofocus parked the
    stage -- the found focus lands at the stack's bottom and the whole ladder
    sits one step high.
    """
    found = 5015.0
    proto, z_moves = _drive_group_scan(found_z=found)

    assert proto.steps()['Z'].tolist() == [5010.0, 5015.0, 5020.0]
    # The sweeping slice's corrective move comes first, then each later slice
    # is moved from the placed frame as usual. Exactly one move per slice: the
    # placement is idempotent but its move is not, and re-issuing it on every
    # settle poll would starve the step of its capture.
    assert z_moves == [5010.0, 5015.0, 5020.0]


def test_the_found_focus_lands_on_the_reference_plane_not_the_stack_floor():
    """The whole point of the ladder: the focus is the MIDDLE slice."""
    found = 4980.0
    _, z_moves = _drive_group_scan(found_z=found)

    assert z_moves[0] == found - 5.0
    assert z_moves[1] == found
    assert z_moves[2] == found + 5.0

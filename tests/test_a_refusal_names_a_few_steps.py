# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal that names steps names a few of them, in the protocol's order.

The z-stack refusal listed every step it found, sorted as text. On a 96-well
protocol that was 96 names, taller than the popup, so the count the message
opened with ("192 of the 1248 z-slices ...") scrolled out of sight, and A10
read before A1. The camera-range refusal had the same shape, one line per
step. Each now names the first five in the order the protocol holds them,
then how many more.

A build that succeeds says what it made, so a log tells how many steps an
Apply produced, whoever asked for it.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import modules.protocol as protocol_module
from modules.exceptions import ProtocolRunRefusedError
from modules.labware_loader import WellPlateLoader
from modules.objectives_loader import ObjectiveLoader
from tests.test_protocol_execution import _make_multi_step_protocol, scope  # noqa: F401
from tests.test_step_label_ssot import _build_protocol, _labeled_step
from tests.test_zstack_group_identity import _WIDE_XY

# Text order and protocol order differ: sorted, 'W10' comes before 'W2'.
_LABELS = ['W2', 'W10', 'W1', 'W11', 'W3', 'W12', 'W4', 'W13']
_STACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}  # 4950..5050


def test_a_short_list_is_named_whole():
    from modules.common_utils import first_few

    assert first_few(['a', 'b', 'c'], separator=', ') == 'a, b, c'
    assert first_few(['a', 'b', 'c', 'd', 'e'], separator=', ') == 'a, b, c, d, e'


def test_a_long_list_names_five_in_the_given_order_then_the_rest():
    from modules.common_utils import first_few

    items = [f'item{n}' for n in (9, 3, 7, 1, 5, 8, 2)]
    assert first_few(items, separator=', ') == 'item9, item3, item7, item1, item5, and 2 more'
    assert first_few(items, separator='\n').endswith('item5\nand 2 more')


def _stack_refusal(proto):
    with (
        patch('modules.protocol.notifications.report_outcome'),
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        proto.apply_zstacking(
            zstack_params=_STACK, axes_config={'Z': {'limits': {'min': 4960.0, 'max': 10_000.0}}}
        )
    return str(refusal.value)


def test_a_stack_refusal_names_five_steps_in_protocol_order():
    proto = _build_protocol([_labeled_step(label=label, z=5000.0) for label in _LABELS])
    names = list(proto.steps()['Name'])

    message = _stack_refusal(proto)

    assert message.startswith('8 of the 48 z-slices fall outside')
    assert f'(steps: {", ".join(names[:5])}, and 3 more)' in message
    assert names[5] not in message


def test_a_camera_range_refusal_names_five_steps_then_the_rest(scope):
    maximum = scope.imaging.max_gain_db_cached
    proto = _make_multi_step_protocol(
        [{'name': f'Step {n}', 'gain_db': maximum + 10.0, 'well': f'A{n}'} for n in range(1, 9)]
    )

    with pytest.raises(ProtocolRunRefusedError) as refusal:
        scope.protocols.refuse_camera_values_out_of_range(proto.steps())

    lines = str(refusal.value).split('\n')
    assert [line.split('"')[1] for line in lines[:5]] == [f'Step {n}' for n in range(1, 6)]
    assert lines[5] == 'and 3 more'
    assert 'Step 6' not in str(refusal.value)


@pytest.fixture
def protocol_log(monkeypatch, caplog):
    # The suite's lvp_logger is a stand-in that records nothing.
    monkeypatch.setattr(protocol_module, 'logger', logging.getLogger('LVP.test.protocol'))
    caplog.set_level(logging.INFO, logger='LVP.test.protocol')
    return caplog


def test_a_stack_that_is_built_says_how_many_steps_it_made(protocol_log):
    proto = _build_protocol([_labeled_step(z=5000.0)])
    proto.apply_zstacking(
        zstack_params=_STACK, axes_config={'Z': {'limits': {'min': 0.0, 'max': 10_000.0}}}
    )

    assert [r.getMessage() for r in protocol_log.records] == [
        '[Protocol] Z-stack applied (range 100.0 um, step 20.0 um): 1 -> 6 steps'
    ]


def test_a_grid_that_is_built_says_how_many_steps_it_made(protocol_log, scale_capabilities):
    proto = _build_protocol([_labeled_step(x=60.0, y=40.0)])
    proto.apply_tiling(
        tiling='2x2',
        frame_dimensions={'width': 1900, 'height': 1900},
        binning_size=1,
        curr_step_idx=0,
        axes_config=_WIDE_XY,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        stage_offset={'x': 0, 'y': 0},
        overlap_percent=0.0,
        capabilities=scale_capabilities,
        objective_helper=ObjectiveLoader(),
    )

    assert [r.getMessage() for r in protocol_log.records] == [
        '[Protocol] Tile grid 2x2 applied: 1 -> 4 steps'
    ]

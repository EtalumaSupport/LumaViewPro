# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""At every place a protocol visits, its channels run in the drawers' order, however it was built.

The drawers' order is the release catalogue's (``common_utils.get_channels``:
BF, PC, DF, then the 405 nm LED's channel before the 488 nm's). New, a tiled
New, a z-stacked New and Add each put a place's channels in that order; Z
orders only a stack's slices, every channel at each slice. Each channel
here is saved at its own focus, the lower ones last in the drawers' order,
so an order taken from Z would run them reversed; the 405 and 488 pair is
the one an earlier release ran the other way round.
"""

from __future__ import annotations

import pytest

import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.config_helpers as config_helpers
from tests.stock_channels import LED0, LED1
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

# Highest focus first in the drawers' order, so a sort by Z reverses them.
_FOCUS_UM = {'BF': 8000.0, LED0: 7990.0, LED1: 7980.0}


def _acquiring(session):
    """BF and the stock 405 and 488 channels acquire, each at its own saved focus."""
    for layer in config_helpers.get_channel_configs(session.settings):
        session.set_channel_settings(layer, acquire=None)
        session.set_channel_settings(layer, autofocus=False)
    for layer, focus in _FOCUS_UM.items():
        session.set_channel_settings(layer, acquire='image')
        session.set_channel_settings(layer, focus_z_um=focus)


def _stacked_ten_um_by_five(session):
    """A stack 10 um deep in 5 um steps, centred on each step's Z."""
    session.update_settings('zstack.range', 10.0)
    session.update_settings('zstack.step_size', 5.0)
    session.update_settings('zstack.position', 'center')


def _drawer_order() -> list[str]:
    return [c for c in common_utils.get_channels() if c in _FOCUS_UM]


def _channels_by_visit(steps) -> dict[tuple, list[str]]:
    """Each place and slice, in the order the run reaches it, with its channels in run order."""
    visits: dict[tuple, list[str]] = {}
    for row in steps.to_dict('records'):
        visits.setdefault((row['X'], row['Y'], row['Z-Slice']), []).append(row['channel'])
    return visits


def test_the_drawers_put_405_before_488():
    # The premise the tests below stand on: the pair an earlier release ran
    # 488 first is 405 first in the drawers, so the old order is told apart.
    assert _drawer_order() == ['BF', LED0, LED1]


@pytest.mark.parametrize(
    ('tiling', 'use_zstacking'),
    [('1x1', False), ('2x2', False), ('1x1', True)],
    ids=['new', 'tiled_new', 'zstacked_new'],
)
def test_every_new_runs_a_places_channels_in_drawer_order(session, tiling, use_zstacking):
    _acquiring(session)
    _stacked_ten_um_by_five(session)

    protocol = session.new_protocol(tiling=tiling, use_zstacking=use_zstacking)

    visits = _channels_by_visit(protocol.steps())
    wrong = {visit: channels for visit, channels in visits.items() if channels != _drawer_order()}
    assert not wrong, (
        f'{len(wrong)} of {len(visits)} visits out of drawer order, e.g. {next(iter(wrong.items()))}'
    )


def test_a_stack_runs_slice_by_slice_from_the_lowest(session):
    _acquiring(session)
    _stacked_ten_um_by_five(session)

    protocol = session.new_protocol(use_zstacking=True)

    steps = protocol.steps()
    first_place = steps[(steps['X'] == steps['X'].iloc[0]) & (steps['Y'] == steps['Y'].iloc[0])]
    assert list(first_place['Z-Slice']) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_new_and_add_run_a_places_channels_in_one_order(session):
    _acquiring(session)
    protocol = session.new_protocol()
    built = protocol.steps()
    new_order = list(built[built['Well'] == built['Well'].iloc[0]]['channel'])

    names = session.add_step(protocol)

    by_name = protocol.steps().set_index('Name')['channel']
    added = [by_name[name] for name in names]
    assert added == new_order == _drawer_order()

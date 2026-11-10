# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""At every place a protocol visits, its channels run in step order, however it was built.

Step order is the release's ``step_channel_order`` (the catalogue's
``step_order``: BF, PC, DF, then Green's LED before Blue's). New, a tiled
New, a z-stacked New and Add each put a place's channels in that order; Z
orders only a stack's slices, every channel at each slice. Each channel
here is saved at its own focus, the lower ones last in step order, so an
order taken from Z would run them reversed, and two of them (the stock
Blue and Green LEDs) are in opposite order on the drawers and in step
order.
"""

from __future__ import annotations

import pytest

import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.config_helpers as config_helpers
from tests.stock_channels import LED0, LED1
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

# Highest focus first in step order, so a sort by Z reverses them.
_FOCUS_UM = {'BF': 8000.0, LED1: 7990.0, LED0: 7980.0}


def _acquiring(session):
    """BF and the stock Blue and Green channels acquire, each at its own saved focus."""
    for layer in config_helpers.get_channel_configs(session.settings):
        session.settings[layer]['acquire'] = None
        session.settings[layer]['autofocus'] = False
    for layer, focus in _FOCUS_UM.items():
        session.settings[layer]['acquire'] = 'image'
        session.save_channel_focus(layer, focus)


def _step_order(session) -> list[str]:
    return [c for c in session.settings['step_channel_order'] if c in _FOCUS_UM]


def _channels_by_visit(steps) -> dict[tuple, list[str]]:
    """Each place and slice, in the order the run reaches it, with its channels in run order."""
    visits: dict[tuple, list[str]] = {}
    for row in steps.to_dict('records'):
        visits.setdefault((row['X'], row['Y'], row['Z-Slice']), []).append(row['channel'])
    return visits


def test_the_step_order_is_not_the_drawer_order_for_these_channels(session):
    # The premise the tests below stand on: these channels' drawer order and
    # step order disagree, so a build in drawer order is told apart.
    drawers = [c for c in common_utils.get_channels() if c in _FOCUS_UM]

    assert drawers != _step_order(session)


@pytest.mark.parametrize(
    ('tiling', 'use_zstacking'),
    [('1x1', False), ('2x2', False), ('1x1', True)],
    ids=['new', 'tiled_new', 'zstacked_new'],
)
def test_every_new_runs_a_places_channels_in_step_order(session, tiling, use_zstacking):
    _acquiring(session)
    session.settings['zstack'] = {'range': 10.0, 'step_size': 5.0, 'position': 'center'}

    protocol = session.new_protocol(tiling=tiling, use_zstacking=use_zstacking)

    visits = _channels_by_visit(protocol.steps())
    wrong = {
        visit: channels for visit, channels in visits.items() if channels != _step_order(session)
    }
    assert not wrong, (
        f'{len(wrong)} of {len(visits)} visits out of step order, e.g. {next(iter(wrong.items()))}'
    )


def test_a_stack_runs_slice_by_slice_from_the_lowest(session):
    _acquiring(session)
    session.settings['zstack'] = {'range': 10.0, 'step_size': 5.0, 'position': 'center'}

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
    assert added == new_order == _step_order(session)

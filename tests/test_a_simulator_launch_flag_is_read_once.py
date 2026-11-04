# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulator launch flag (``--sim-camera-stall``, ``--sim-file-stall``,
``--sim-focal-spread``) is read once, beside ``--simulate``.

``take_sim_launch_value`` gives the flag's value, or None when it is not
given, and takes every copy of it out of ``argv`` whether it refuses or not,
since Kivy exits on a flag it does not know. A flag given twice, or without
``--simulate``, is refused.
"""

from __future__ import annotations

import pytest

from modules.app_environment import take_sim_launch_value


def test_the_value_is_given_and_the_flag_leaves_argv():
    argv = ['lumaviewpro.py', '--sim-focal-spread=25', '--sim-walk=w.json']

    assert take_sim_launch_value(argv, 'sim-focal-spread', simulate=True) == '25'
    assert argv == ['lumaviewpro.py', '--sim-walk=w.json']


def test_a_flag_not_given_is_none_and_argv_is_untouched():
    argv = ['lumaviewpro.py', '--sim-camera-stall=30,20']

    assert take_sim_launch_value(argv, 'sim-focal-spread', simulate=True) is None
    assert argv == ['lumaviewpro.py', '--sim-camera-stall=30,20']


@pytest.mark.parametrize(
    ('argv', 'simulate', 'why'),
    [
        (['--sim-focal-spread=25', '--sim-focal-spread=10'], True, 'give --sim-focal-spread once'),
        (['--sim-focal-spread=25'], False, '--sim-focal-spread needs --simulate'),
    ],
)
def test_a_repeated_flag_or_one_without_simulate_is_refused_and_still_removed(argv, simulate, why):
    # parametrize-justified: the two refusals, each with its own reason
    with pytest.raises(ValueError, match=why):
        take_sim_launch_value(argv, 'sim-focal-spread', simulate=simulate)
    assert argv == []

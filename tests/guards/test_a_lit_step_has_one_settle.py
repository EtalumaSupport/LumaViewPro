# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lit step's LED settle is frame validity's alone.

``ProtocolStepRunner._apply_step_light`` lights the step's channel through
STEP_LIGHT, a confirm-on transition: it returns once the board reports the
channel on. How long after that a frame counts is frame validity's rule
(``FrameValidity`` credits only frames that arrive after the LED write), so
the step's grab already waits out the LED's settle. A fixed sleep in the
light path is a second owner of that settle: it delays the grab call, not the
frame count, so it can neither shorten nor lengthen the wait the rule holds,
and it costs every lit step of every run.

The path is a run's middle, reachable only through a whole protocol run, and
the sleep's cost (0-1 ms a step, overlapped by the frame count) is below
what a timed test can see, so the structural fact is pinned here: the light
path calls no sleep.
"""

import ast

from tests.ast_seams import find_def


def test_the_step_light_holds_no_fixed_settle_of_its_own():
    light = find_def(
        'src/_lumascope/modules/protocol_step_runner.py',
        '_apply_step_light',
        class_name='ProtocolStepRunner',
    )
    assert light is not None, 'ProtocolStepRunner._apply_step_light is gone; re-pin the settle'
    sleeps = [
        ast.unparse(node)
        for node in ast.walk(light)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'sleep'
    ]
    assert sleeps == [], f'the step light sleeps a settle frame validity owns: {sleeps}'

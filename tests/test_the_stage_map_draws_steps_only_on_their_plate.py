# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The stage map draws a protocol's steps only on the plate they are stated against.

The map draws the scope's plate, and a click on it moves through that plate.
A protocol's step positions are stated against the protocol's own plate, and
an API caller can put a protocol on a plate the scope is not on; drawn on the
scope's plate, those positions land in the wrong wells. The map was handed a
bare frame of positions, with no plate, so it could not tell. It is now
handed the protocol, and draws its steps only when the plates match.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from ui.stage import Stage


CACHED = object()


def _protocol(plate):
    steps = pd.DataFrame({'X': [10.0, 20.0], 'Y': [5.0, 5.0], 'Z': [0.0, 0.0]})
    return types.SimpleNamespace(steps=lambda: steps, labware_id=lambda: plate)


@pytest.fixture
def stage(gui_ctx):
    # The template's plate, 96 well, is the scope's.
    built = Stage.__new__(Stage)
    built._protocol_step_locations_show = True
    return built


def _with_the_drawing_cached(stage):
    """The markers already drawn for these steps on the scope's plate."""
    stage._step_locations_fbo = CACHED


def test_steps_on_the_scopes_plate_are_drawn(stage):
    stage.set_protocol_steps(_protocol('96 well microplate'))
    _with_the_drawing_cached(stage)

    assert stage.create_step_locations_fbo() is CACHED


def test_steps_on_another_plate_are_not_drawn_on_this_one(stage):
    stage.set_protocol_steps(_protocol('6 well microplate'))
    _with_the_drawing_cached(stage)

    assert stage.create_step_locations_fbo() is None


def test_new_steps_are_drawn_anew_and_the_toggle_reuses_the_drawing(stage):
    # The drawing is reused until the steps it shows change: their one writer,
    # set_protocol_steps, clears it. Turning the markers off and on changes
    # nothing the drawing holds.
    stage.set_protocol_steps(_protocol('96 well microplate'))
    _with_the_drawing_cached(stage)

    stage.show_protocol_steps(False)
    stage.show_protocol_steps(True)
    assert stage.create_step_locations_fbo() is CACHED

    stage.width, stage.height = 200, 100  # the fixture's stage was never laid out
    stage.set_protocol_steps(_protocol('96 well microplate'))
    assert stage.create_step_locations_fbo() is not CACHED

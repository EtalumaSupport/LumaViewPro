# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped template's layer gains are within reach of every camera.

A run refuses a step whose gain is outside the camera's range, and a new step
copies its layer's gain. The template once shipped Lumi at 48 dB, Red at 30
and Green at 20, so a fresh install on a camera with a smaller ceiling had its
first Lumi run refused. Every layer's default must fit the smallest ceiling a
shipped camera advertises.

The real cameras advertise their ceilings at connect, so no static profile
holds them; the bound is named here with its source. It once came from the
static ceilings in `_PROFILES`, of which the only one was the simulator's
invented 20 dB -- the guard held by accident, and a simulator that matches
the daA3840 (48 dB) would have let a 45 dB layer through.
"""

import json
import pathlib

import pytest

SETTINGS = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'settings.json'
LAYERS = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')

# The smallest gain ceiling a shipped camera advertises: the IDS U3-34L0XCP's
# 30 dB, pinned by its driver's test
# (test_ids_driver.py, `total_max_db == pytest.approx(30.0, ...)`). The others:
# the Basler bodies 48 dB, the FX2's MT9P031 42.1 dB.
SMALLEST_SHIPPED_CEILING_DB = 30.0


@pytest.mark.parametrize('layer', LAYERS)
def test_the_layer_gain_fits_the_smallest_camera(layer):
    gain = json.loads(SETTINGS.read_text())[layer]['gain_db']
    assert 0.0 <= gain <= SMALLEST_SHIPPED_CEILING_DB, (
        f"{layer} ships at {gain} dB, outside the smallest shipped camera's range"
    )


@pytest.mark.parametrize('layer', ('Green', 'Red', 'Lumi'))
def test_the_fluorescence_layers_ship_at_10_db(layer):
    """The fit alone admits Green at 30 dB, at the smallest ceiling; the
    template ships the three at 10 dB, well inside every camera's range."""
    assert json.loads(SETTINGS.read_text())[layer]['gain_db'] == 10.0

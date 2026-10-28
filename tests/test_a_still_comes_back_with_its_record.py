# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A script grabs a still and gets it back with its record, its wait bounded.

``capture_and_wait`` hands back an array; its record had to be read
afterwards from ``last_capture_info``, a one-slot store the next capture
replaces, and its depth from a second slot. ``capture_still`` -- the
Session's door, and the scope's beneath it -- hands back one
``CapturedStill``: the pixels, the frame's own ``FrameRecord``, and the
depth that record gives, whatever is captured after it.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from modules.exceptions import ArgumentRefusedError
from modules.lumascope_api.camera import AutoGainConvergence
from modules.lumascope_api.frame_record import CapturedStill
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0

# Distinct from the simulated camera's default (10 ms, 1 dB) and from each
# other, so a record of the wrong layer, or of no layer, cannot pass.
_BF = {'exposure_ms': 4.0, 'gain_db': 3.0, 'auto_gain': False}
_BLUE = {'exposure_ms': 30.0, 'gain_db': 10.0, 'auto_gain': False}


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(BF=_BF, **{LED0: _BLUE}), simulate=True)
    home_sim_scope(built.scope)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


class TestTheStillIsHandedBackWithItsRecord:
    def test_a_still_of_a_channel_is_taken_at_that_layers_stored_settings(self, session):
        still = session.capture_still(LED0)

        assert isinstance(still, CapturedStill)
        assert still.record.held_channel == LED0
        assert round(still.record.exposure_ms, 3) == _BLUE['exposure_ms']
        assert round(still.record.gain_db, 3) == _BLUE['gain_db']

    def test_a_still_with_no_channel_is_taken_with_the_camera_as_set(self, session):
        session.scope.camera.set_gain_db(7.0)

        still = session.capture_still()

        assert still.record.held_channel is None
        assert round(still.record.gain_db, 3) == 7.0

    def test_its_pixels_cannot_be_changed(self, session):
        still = session.capture_still('BF')

        with pytest.raises(ValueError):
            still.image[0, 0] = 0

    def test_its_depth_is_its_own_after_another_capture(self, session):
        """A sum's depth is the bits it can reach, and stays that after a
        single-frame capture replaces the scope's last-capture slots."""
        summed = session.capture_still('BF', force_to_8bit=False, sum_count=4)
        session.scope.imaging.capture_and_wait(force_to_8bit=False)

        bits = summed.record.frame_significant_bits
        assert summed.record.frames_summed == 4
        assert summed.significant_bits == (4 * ((1 << bits) - 1)).bit_length()
        assert summed.full_scale == 4 * ((1 << bits) - 1)
        assert int(summed.image.max()) <= summed.full_scale

    def test_an_8_bit_rendering_is_8_bits(self, session):
        still = session.capture_still('BF', force_to_8bit=True)

        assert still.image.dtype == np.uint8
        assert (still.significant_bits, still.full_scale) == (8, 255)

    def test_two_stills_compare_by_identity(self, session):
        still = session.capture_still('BF')

        assert still == still
        assert still != session.capture_still('BF')

    def test_a_locked_auto_gain_states_where_it_landed(self, session):
        session.set_channel_auto_gain('BF', True)

        still = session.capture_still('BF')

        assert isinstance(still.record.auto_gain, AutoGainConvergence)

    def test_a_still_with_no_auto_gain_states_none(self, session):
        assert session.capture_still('BF').record.auto_gain is None


class TestTheStillIsRefusedBeforeAnythingIsCaptured:
    def test_a_name_that_is_no_layer_is_refused(self, session):
        before = session.scope.camera.get_held_layer()

        with pytest.raises(ArgumentRefusedError) as refused:
            session.capture_still('Not a layer')

        assert refused.value.reason == 'channel_unknown'
        assert session.scope.camera.get_held_layer() == before

    def test_a_wait_no_number_bounds_is_refused(self, session):
        with pytest.raises(ArgumentRefusedError) as refused:
            session.capture_still('BF', timeout_s=math.nan)

        assert refused.value.reason == 'not_a_number'

# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The public members that transmitted beside the lanes now go through them.

Acceleration, the LED restore, the raw diagnostic channel, the fan duty, LED
engineering mode, streaming start and stop, the grab benchmark and the Pylon
probe each wrote to a board or the camera on the caller's thread, so a REST
or script call landed in the middle of a run or a diagnostic with nothing to
refuse it. Each is now one task on its device's lane: refused to anyone not
acting under the holder's taking, and run on the lane's worker for the
holder. The camera temperature read goes on the camera lane too -- it sets a
selector that must not interleave with another camera write -- but a hold
does not refuse it, so the temperature log keeps running through a run.
"""

from unittest.mock import patch

import pytest

from modules import sequential_io_executor


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
    )
    try:
        yield s
    finally:
        s.shutdown()


def _lane_now():
    return getattr(sequential_io_executor._lane_worker, 'executor', None)


def _spy(sc, slot, method):
    """Wrap one driver method so each call records the lane it ran on."""
    driver = getattr(sc, slot)
    real = getattr(driver, method)
    lanes = []

    def _record(*a, **k):
        lanes.append(_lane_now())
        return real(*a, **k)

    return patch.object(driver, method, side_effect=_record), lanes


class TestTheTemperatureRead:
    @pytest.mark.parametrize('kind', ['diagnostic', 'protocol'])
    def test_a_hold_does_not_refuse_it_and_it_runs_on_the_camera_lane(self, sim_session, kind):
        sc = sim_session.scope
        spy, lanes = _spy(sc, '_camera_driver', 'get_all_temperatures')
        held = sim_session.activity_claim.try_claim(kind)
        try:
            with spy:
                temps = sc.diagnostics.get_camera_temperatures_degc()
        finally:
            held.release()
        assert temps, 'the temperature read was refused or returned nothing during a hold'
        assert lanes == [sim_session.camera_executor]

    def test_it_passes_a_runs_protocol_fence(self, sim_session):
        sc = sim_session.scope
        held = sim_session.activity_claim.try_claim('protocol')
        sim_session.camera_executor.protocol_start(held)
        try:
            assert sc.diagnostics.get_camera_temperatures_degc()
        finally:
            sim_session.camera_executor.protocol_end()
            held.release()

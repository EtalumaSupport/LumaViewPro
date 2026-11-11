# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LED selftest leaves every LED dark, able to light, and recorded so.

The v2 LED firmware's SELFTEST ramps each channel up to 500 mA, switches it
off, a channel the API had lit included, and ends with every output
disabled, which no later light enables. Sent raw by the support report, it
left the API believing a dark LED lit, and every later light dark.
``DiagnosticsAPI.run_led_selftest`` enables the outputs and writes the
all-off through the LED owner after it.
"""

import functools

import pytest

import _lumascope.drivers.sim_wire.backend as sim_backend
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import FirmwareDiagnostics
from tests.settings_fixtures import complete_settings


def _lit(illumination) -> list[int]:
    return [r.address for r in illumination.get_led_states() if r.state is not LedState.DARK]


def test_the_support_reports_selftest_leaves_no_led_believed_lit(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850'), simulate=True
    )
    try:
        illumination = session.scope.illumination
        illumination.led_on('BF', 10)
        assert _lit(illumination)
        FirmwareDiagnostics(scope=session.scope).run_led_selftest()
        assert _lit(illumination) == []
    finally:
        session.shutdown()


@pytest.mark.firmware_tier
def test_after_a_selftest_an_led_lights_again():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sim_backend, 'LedBoardSpec', functools.partial(sim_backend.LedBoardSpec, oracle=True)
        )
        session = ScopeSession.create(
            complete_settings(simulator_tier='firmware', microscope='LS850T'),
            simulate=True,
            warn_pre_release=False,
        )
    try:
        illumination = session.scope.illumination
        board = session.scope._led_driver._backend.led_board
        illumination.led_on_by_address(0, 10)
        # A stand-in by design: the field firmware the simulator runs has no
        # SELFTEST, so the v2 selftest's end, every output disabled behind a
        # light the API holds, is written to the board directly.
        session.scope._led_driver.leds_disable()

        session.scope.diagnostics.run_led_selftest(timeout_s=1)

        assert _lit(illumination) == []
        illumination.led_on_by_address(0, 10)
        enabled, powered, code = board.state('DAC')[0]
        assert enabled and powered and code > 0, board.state('DAC')[0]
    finally:
        session.shutdown()

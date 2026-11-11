# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope can be built with a camera that has a conversion gain mode.

The stock dart has no switchable conversion gain; an IDS on firmware 3.95
and some Basler bodies do. ``SimulatedScope(conversion_gain_mode=True)``
(GUI ``--sim-camera=hcg``) builds the simulated camera with the mode, so
the Advanced Settings toggle and a run in each mode are shown without such
a camera. The default simulated camera has none, which
``test_camera_write_authority.py`` pins through its refusal.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


def test_a_session_built_with_the_mode_turns_it_on_and_off_through_the_api(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
        simulated=SimulatedScope(conversion_gain_mode=True),
        warn_pre_release=False,
    )
    try:
        scope = session.scope
        assert scope.capabilities.camera_supports_conversion_gain_mode is True
        held = {}
        for enabled in (True, False):
            scope.camera.set_high_conversion_gain(enabled)
            # The camera's own mode, so the write is shown to reach it, not
            # only the API's record of it.
            held[enabled] = (
                scope.camera.high_conversion_gain,
                scope._camera_driver._conversion_gain_mode,
            )
        assert held == {True: (True, 'High'), False: (False, 'Low')}
    finally:
        session.shutdown()


def test_a_real_scope_refuses_the_mode():
    with pytest.raises(ArgumentRefusedError) as refused:
        build_scope(simulate=False, sim_conversion_gain_mode=True)
    assert (refused.value.reason, refused.value.argument) == (
        'needs_simulated_scope',
        'sim_conversion_gain_mode',
    )


def test_a_scope_simulated_with_an_fx2_refuses_the_mode():
    # The FX2 is the camera on that model, and it has no conversion gain.
    with pytest.raises(ArgumentRefusedError) as refused:
        build_scope(simulate=True, sim_model='LS620', sim_conversion_gain_mode=True)
    assert (refused.value.reason, refused.value.argument) == (
        'needs_simulated_camera',
        'sim_conversion_gain_mode',
    )

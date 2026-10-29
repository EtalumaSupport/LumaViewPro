# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack asked for and not configured is refused, not quietly flattened.

The protocol build (``ProtocolsAPI.create_protocol``, behind New, REST and
``start_zstack``) used to treat ``use_zstacking=True`` with a zero
range or a zero step size as a request for a single plane: it logged a
warning and built one step per position. The caller asked for a stack and
got a photograph, and because the result was non-empty it sailed past the
empty-protocol refusal and ran to a reported success.

That is the defaulting shape the cardinal rule names -- a missing value
papered over with a plausible wrong result. The builder now refuses.

From the GUI this case is unreachable today: the z-stack starter runs its
own zero-range check before the builder is ever called. The live case is
the headless one, where ``ProtocolRunner.start_zstack()`` passes
``use_zstacking=True`` with no guard of its own, which is why the second
class here drives the refusal through that entry point rather than
through a widget.

The zero-extent count itself is pinned separately, on ``ZStackConfig``;
this is the layer above, where the degrade branch decided not to build a
``ZStackConfig`` at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import os

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.labware_loader import WellPlateLoader
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_start_zstack_entry_point import _POSITION, _zstack_settings
from tests.objective_records import objective


# Both halves of "not configured": the count needs a range to sweep and a
# step to sweep it in, and either one at zero makes the stack impossible.
# Parametrised rather than written twice so a fix that guards only one of
# them cannot pass.
UNCONFIGURED = [
    pytest.param({'range': 0.0, 'step_size': 5.0}, id='no-range'),
    pytest.param({'range': 20.0, 'step_size': 0.0}, id='no-step-size'),
]


def _standalone_config(zstack: dict, *, use_zstacking: bool = True) -> dict:
    """The config a z-stack caller builds, with the stack's shape overridden."""
    import modules.config_helpers as config_helpers

    settings = _zstack_settings()
    settings['zstack'] = {**settings['zstack'], **zstack}

    objective_helper = MagicMock()
    objective_helper.get_objective_info.return_value = objective(magnification=10)
    wellplate_loader = WellPlateLoader()

    return config_helpers.get_standalone_capture_config_from_settings(
        settings,
        objective_helper,
        wellplate_loader,
        layer='BF',
        position=_POSITION,
        position_name='ZStack',
        autofocus=False,
        use_zstacking=use_zstacking,
        stim_config={},
    )


@pytest.fixture
def sim_scope(tmp_path):
    """The simulated scope, composed into a session: the protocol build
    stamps turret slots from the session's settings."""
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield session.scope
    finally:
        session.shutdown()


class TestTheBuilderRefuses:
    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_an_unconfigured_stack_is_refused(self, zstack, sim_scope):
        with pytest.raises(ProtocolRunRefusedError) as refusal:
            sim_scope.protocols.create_protocol(input_config=_standalone_config(zstack))

        assert refusal.value.reason == 'zstack_not_configured', (
            'the reason code is the contract a REST or SDK caller branches on'
        )

    def test_a_configured_stack_still_builds(self, sim_scope):
        """The guard must not refuse the stacks that are fine."""
        protocol = sim_scope.protocols.create_protocol(
            input_config=_standalone_config({'range': 20.0, 'step_size': 5.0})
        )

        assert protocol.num_steps() > 1, 'a configured stack is more than one plane'

    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_a_caller_not_asking_for_a_stack_is_unaffected(self, zstack, sim_scope):
        """use_zstacking=False is the ordinary single-plane request.

        Zero range and zero step size are what every non-stack config
        carries, so a guard keyed on the values alone would refuse most of
        the protocols in the program.
        """
        protocol = sim_scope.protocols.create_protocol(
            input_config=_standalone_config(zstack, use_zstacking=False)
        )

        assert protocol.num_steps() >= 1


class TestTheRefusalReachesTheUser:
    """The builder raises; the boundary its caller sits behind tells.

    The builder decides nothing about who is told: a REST caller is
    answered and shown nothing, and a person's press is shown the refusal
    once by the GUI's boundary, the one place every GUI caller's report
    goes through, so the telling cannot drift per caller.
    """

    def _build(self, sim_scope):
        sim_scope.protocols.create_protocol(
            input_config=_standalone_config({'range': 0.0, 'step_size': 5.0})
        )

    def test_the_builder_raises_and_posts_nothing(self, sim_scope, centre_posts):
        with pytest.raises(ProtocolRunRefusedError) as refused:
            self._build(sim_scope)

        assert refused.value.reason == 'zstack_not_configured'
        assert list(centre_posts) == []

    def test_a_press_is_told_once_even_with_a_run_in_flight(
        self, sim_scope, centre_posts, unattended_run
    ):
        """Solicited is the difference between told and silently dropped.

        An unsolicited notification is suppressed while a run is running,
        which is exactly when a user is most likely to be clicking.
        """
        from ui.ui_helpers import run_reported

        run_reported(lambda: self._build(sim_scope), None, 'APPLY_ZSTACK')

        posted = [n for n in centre_posts if n.severity == Severity.WARNING]
        assert [(n.reason, n.shown, n.solicited) for n in posted] == [
            ('zstack_not_configured', True, True)
        ], 'a refusal answers something the caller just asked for'


class TestTheHeadlessCallerGetsIt:
    """The reachable case: ProtocolRunner passes use_zstacking=True unguarded."""

    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_start_zstack_refuses_instead_of_capturing_one_plane(self, zstack, sim_session):
        from tests.scope_fakes import home_sim_scope

        home_sim_scope(sim_session.scope)
        for key, value in zstack.items():
            sim_session.update_settings(f'zstack.{key}', value)
        live_folder = sim_session.settings['live_folder']
        runner = sim_session.create_protocol_runner()

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            runner.start_zstack(channel='BF')

        assert refusal.value.reason == 'zstack_not_configured'
        # The refusal arrives before the run is prepared, not after it commits.
        assert not sim_session.live_work.work
        assert list(os.scandir(live_folder)) == [], 'a refused stack made a run folder'

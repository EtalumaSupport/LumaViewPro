# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A z-stack asked for and not configured is refused, not quietly flattened.

``Protocol.from_config`` used to treat ``use_zstacking=True`` with a zero
range or a zero step size as a request for a single plane: it logged a
warning and built one step per position. The caller asked for a stack and
got a photograph, and because the result was non-empty it sailed past the
empty-protocol refusal and ran to a reported success.

That is the defaulting shape the cardinal rule names -- a missing value
papered over with a plausible wrong result. The builder now refuses.

From the GUI this case is unreachable today: the z-stack starter runs its
own zero-range check before the builder is ever called. The live case is
the headless one, where ``ProtocolRunner.run_zstack()`` passes
``use_zstacking=True`` with no guard of its own, which is why the second
class here drives the refusal through that entry point rather than
through a widget.

The zero-extent count itself is pinned separately, on ``ZStackConfig``;
this is the layer above, where the degrade branch decided not to build a
``ZStackConfig`` at all.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.protocol import Protocol
from tests.test_run_zstack_entry_point import _POSITION, _zstack_settings


_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TILING = _REPO_ROOT / 'data' / 'tiling.json'


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
    objective_helper.get_objective_info.return_value = {'magnification': 10}
    wellplate_loader = MagicMock()
    wellplate_loader.get_plate_list.return_value = ['96 well microplate']

    return config_helpers.get_standalone_capture_config_from_settings(
        settings,
        objective_helper,
        wellplate_loader,
        layer='BF',
        position=dict(_POSITION),
        position_name='ZStack',
        autofocus=False,
        use_zstacking=use_zstacking,
        stim_config={},
    )


@pytest.fixture
def sim_scope():
    from modules.lumascope_api import Lumascope

    scope = Lumascope(simulate=True)
    scope.protocols.register_source_path(_REPO_ROOT)
    try:
        yield scope
    finally:
        scope.disconnect()


class TestTheBuilderRefuses:
    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_an_unconfigured_stack_is_refused(self, zstack, sim_scope):
        with pytest.raises(ProtocolRunRefusedError) as refusal:
            Protocol.from_config(
                input_config=_standalone_config(zstack),
                tiling_configs_file_loc=_TILING,
                capabilities=sim_scope.capabilities,
            )

        assert refusal.value.reason == 'zstack_not_configured', (
            'the reason code is the contract a REST or SDK caller branches on'
        )

    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_it_is_refused_rather_than_flattened(self, zstack, sim_scope):
        """The defect's signature: a non-empty protocol of single planes.

        Worth asserting separately from the raise -- a fix that refused
        somewhere else while still building the flat protocol first would
        satisfy the test above and leave the wrong result in memory.
        """
        with pytest.raises(ProtocolRunRefusedError):
            Protocol.from_config(
                input_config=_standalone_config(zstack),
                tiling_configs_file_loc=_TILING,
                capabilities=sim_scope.capabilities,
            )

    def test_a_configured_stack_still_builds(self, sim_scope):
        """The guard must not refuse the stacks that are fine."""
        protocol = Protocol.from_config(
            input_config=_standalone_config({'range': 20.0, 'step_size': 5.0}),
            tiling_configs_file_loc=_TILING,
            capabilities=sim_scope.capabilities,
        )

        assert protocol.num_steps() > 1, 'a configured stack is more than one plane'

    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_a_caller_not_asking_for_a_stack_is_unaffected(self, zstack, sim_scope):
        """use_zstacking=False is the ordinary single-plane request.

        Zero range and zero step size are what every non-stack config
        carries, so a guard keyed on the values alone would refuse most of
        the protocols in the program.
        """
        protocol = Protocol.from_config(
            input_config=_standalone_config(zstack, use_zstacking=False),
            tiling_configs_file_loc=_TILING,
            capabilities=sim_scope.capabilities,
        )

        assert protocol.num_steps() >= 1


class TestTheRefusalReachesTheUser:
    """The builder raises AND tells, because nothing downstream will.

    The runner's refusal funnel logs and notifies before it raises, so its
    callers re-notify nothing. The builder runs before any run exists, so
    no funnel sees this one: if it only raised, a GUI caller would have to
    tell the user itself, and the telling would drift per caller -- which
    is the Rule 2 failure the whole refusal work is undoing.
    """

    def test_it_posts_exactly_one_notification(self, monkeypatch, sim_scope):
        import modules.notification_center as nc

        posted: list[dict] = []
        monkeypatch.setattr(nc.notifications, 'warning', lambda *a, **kw: posted.append(kw))

        with pytest.raises(ProtocolRunRefusedError):
            Protocol.from_config(
                input_config=_standalone_config({'range': 0.0, 'step_size': 5.0}),
                tiling_configs_file_loc=_TILING,
                capabilities=sim_scope.capabilities,
            )

        assert len(posted) == 1, f'the refusal must reach the user exactly once: {posted}'

    def test_the_notification_survives_a_run_in_flight(self, monkeypatch, sim_scope):
        """Solicited is the difference between told and silently dropped.

        An unsolicited notification is suppressed while a run is running,
        which is exactly when a user is most likely to be clicking.
        """
        import modules.notification_center as nc

        posted: list[dict] = []
        monkeypatch.setattr(nc.notifications, 'warning', lambda *a, **kw: posted.append(kw))

        with pytest.raises(ProtocolRunRefusedError):
            Protocol.from_config(
                input_config=_standalone_config({'range': 0.0, 'step_size': 5.0}),
                tiling_configs_file_loc=_TILING,
                capabilities=sim_scope.capabilities,
            )

        assert posted[0].get('solicited') is True, (
            'a refusal answers something the caller just asked for'
        )


class TestTheHeadlessCallerGetsIt:
    """The reachable case: ProtocolRunner passes use_zstacking=True unguarded."""

    @pytest.mark.parametrize('zstack', UNCONFIGURED)
    def test_run_zstack_refuses_instead_of_capturing_one_plane(self, zstack, sim_scope, tmp_path):
        from modules.protocol_runner import ProtocolRunner

        settings = _zstack_settings()
        settings['zstack'] = {**settings['zstack'], **zstack}
        settings['live_folder'] = str(tmp_path)

        session = MagicMock()
        session.settings = settings
        session.scope = sim_scope
        session.get_current_plate_position.return_value = dict(_POSITION)
        session.objective_helper.get_objective_info.return_value = {'magnification': 10}
        session.wellplate_loader.get_plate_list.return_value = ['96 well microplate']
        runner = ProtocolRunner(session)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            runner.run_zstack(layer='BF')

        assert refusal.value.reason == 'zstack_not_configured'
        assert not runner._executor.prepare.called, (
            'the refusal must arrive before the run is prepared, not after it commits'
        )

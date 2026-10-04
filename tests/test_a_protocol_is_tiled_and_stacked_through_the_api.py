# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol is tiled and z-stacked through the Session, judged on the axes the scope has.

The protocol panel gathered the tiling inputs itself -- the scope's plate,
the stage offset, the overlap, the frame, the binning -- and the travel
limits from ``scope.motion.get_axes_config()``, then ordered the steps with
a second call. A script or REST caller had none of it. The motion board
answers X and Y limits on a Z-only scope too, so a tile grid on one was
built against limits no motor can reach, and the "no X/Y motor" refusal
never fired. Now the Session takes only what a person chooses, the API
judges against the axes in ``capabilities.axes`` -- the read the run gate
makes -- and each build leaves the steps in the order a run visits them.
"""

from __future__ import annotations

import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ConfigError, ProtocolRunRefusedError
from modules.notification_center import notifications
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture


@pytest.fixture
def z_only_session(tmp_path):
    built = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS820'), simulate=True
    )
    assert built.scope.capabilities.axes == ('Z',)
    home_sim_scope(built.scope)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def reported(monkeypatch):
    reports = []
    real = notifications.report_outcome

    def spy(exc, **kw):
        if type(exc).__name__ == 'ProtocolStepsInvalidNotice':
            reports.append(kw['solicited'])
        return real(exc, **kw)

    monkeypatch.setattr(notifications, 'report_outcome', spy)
    return reports


def _bf_steps_at(session, xs_um):
    """One BF step at each X in ``xs_um``, in that order, mid-travel in Y and Z."""
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'
    session.settings['BF']['focus'] = 5000.0
    session.scope.motion.move_absolute('Y', 40000, wait_until_complete=True)
    protocol = session.create_empty_protocol()
    for x_um in xs_um:
        session.scope.motion.move_absolute('X', x_um, wait_until_complete=True)
        session.add_step(protocol, after_step=protocol.num_steps() - 1)
    return protocol


def _rows(protocol):
    return protocol.steps()[['X', 'Y', 'Z', 'Tile', 'Z-Slice']].to_dict('records')


def _ordered(protocol):
    """What ``optimize_step_ordering`` would make of the protocol's steps."""
    other = protocol.copy_for_execution()
    other.optimize_step_ordering()
    return _rows(other)


class TestTiling:
    def test_every_step_becomes_the_grid_in_the_order_a_run_visits(self, session):
        # Two steps at one X with another between them: built step by step,
        # the tiles of the two would not be next to each other.
        protocol = _bf_steps_at(session, (10000, 20000, 10000))

        session.apply_tiling(protocol, '2x2')

        assert protocol.num_steps() == 12
        assert _rows(protocol) == _ordered(protocol)

    @pytest.mark.parametrize(
        'store',
        [
            pytest.param(lambda s: s.__setitem__('tiling_overlap_percent', 25.0), id='overlap'),
            pytest.param(lambda s: s['binning'].__setitem__('size', '2x2'), id='binning'),
            pytest.param(lambda s: s['frame'].__setitem__('width', 600), id='frame'),
        ],
    )
    def test_the_tile_spacing_is_the_stored_frame_binning_and_overlap(self, session, store):
        def span():
            protocol = _bf_steps_at(session, (40000,))
            session.apply_tiling(protocol, '2x2')
            xs = protocol.steps()['X']
            return xs.max() - xs.min()

        session.settings['tiling_overlap_percent'] = 0.0
        before = span()
        store(session.settings)

        assert span() != before

    def test_the_tiles_are_judged_on_the_protocols_plate(self, session, monkeypatch):
        protocol = _bf_steps_at(session, (40000,))
        session.set_protocol_labware(protocol, '6 well microplate')
        assert session.settings['protocol']['labware'] != protocol.labware()
        loader = session.scope.wellplate_loader
        asked = []
        real = loader.get_plate

        def spy(plate_key):
            asked.append(plate_key)
            return real(plate_key=plate_key)

        monkeypatch.setattr(loader, 'get_plate', spy)

        session.apply_tiling(protocol, '2x2')

        assert asked == [protocol.labware()]

    def test_a_z_only_scope_is_refused_for_no_x_y_motor(self, z_only_session):
        protocol = _bf_steps_at_center(z_only_session)
        before = _rows(protocol)

        with pytest.raises(ProtocolRunRefusedError, match='no motor for X, Y') as refused:
            z_only_session.apply_tiling(protocol, '2x2')

        assert refused.value.reason == 'positions_unreachable'
        assert _rows(protocol) == before

    def test_a_grid_not_on_offer_is_refused_and_nothing_changes(self, session):
        protocol = _bf_steps_at(session, (10000,))
        before = _rows(protocol)

        with pytest.raises(ProtocolRunRefusedError):
            session.apply_tiling(protocol, '17x17')

        assert _rows(protocol) == before

    def test_a_build_leaving_a_step_the_run_refuses_is_noticed_once(self, session, reported):
        protocol = _bf_steps_at(session, (10000,))
        # A value the run gate refuses but the frame's type admits, as a
        # loaded file can hold.
        protocol._config['steps'].at[0, 'Exposure'] = 0.0

        session.apply_tiling(protocol, '2x2')

        assert reported == [True]


def _bf_steps_at_center(session):
    """One BF step where a stage-less scope's steps are: wherever the stage is."""
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'
    session.settings['BF']['focus'] = 5000.0
    protocol = session.create_empty_protocol()
    session.add_step(protocol, after_step=-1)
    return protocol


class TestZStacking:
    def test_every_step_becomes_a_stack_in_the_order_a_run_visits(self, session):
        protocol = _bf_steps_at(session, (10000, 20000, 10000))

        session.apply_zstacking(protocol, range_um=20.0, step_size_um=5.0, z_reference='center')

        assert protocol.num_steps() == 15
        assert _rows(protocol) == _ordered(protocol)

    @pytest.mark.parametrize(('range_um', 'step_size_um'), [(0.0, 5.0), (20.0, 0.0)])
    def test_no_extent_is_refused_and_nothing_changes(self, session, range_um, step_size_um):
        protocol = _bf_steps_at(session, (10000,))
        before = _rows(protocol)

        with pytest.raises(ProtocolRunRefusedError) as refused:
            session.apply_zstacking(
                protocol, range_um=range_um, step_size_um=step_size_um, z_reference='center'
            )

        assert refused.value.reason == 'zstack_not_configured'
        assert _rows(protocol) == before

    def test_an_unknown_reference_is_refused_and_nothing_changes(self, session):
        protocol = _bf_steps_at(session, (10000,))
        before = _rows(protocol)

        with pytest.raises(ConfigError, match='middle'):
            session.apply_zstacking(protocol, range_um=20.0, step_size_um=5.0, z_reference='middle')

        assert _rows(protocol) == before

    def test_a_build_leaving_a_step_the_run_refuses_is_noticed_once(self, session, reported):
        protocol = _bf_steps_at(session, (10000,))
        protocol._config['steps'].at[0, 'Exposure'] = 0.0

        session.apply_zstacking(protocol, range_um=20.0, step_size_um=5.0, z_reference='center')

        assert reported == [True]

    def test_a_z_only_scope_stacks(self, z_only_session):
        protocol = _bf_steps_at_center(z_only_session)

        z_only_session.apply_zstacking(
            protocol, range_um=20.0, step_size_um=5.0, z_reference='bottom'
        )

        assert protocol.num_steps() == 5


def test_the_motion_api_has_one_read_of_the_travel(session):
    # The limits are get_axis_limits'; a second door answered them as the
    # driver's microstep config.
    assert not hasattr(session.scope.motion, 'get_axes_config')

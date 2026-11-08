# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated scope comes up as the unit whose motorconfig.json it is handed.

What is pinned here (``SimulatedScope.unit_config``): on the simulated motor
firmware the scope comes up as the unit does on a bench, under the name its
board reports, on the axes the config marks present, with its optics and
travel, and with the channels bring-up resolves for it (an OEM unit's own
name has the stock optics module's, a config's ``Channels`` block its own).
Anywhere the firmware does not run, and for a config the simulator cannot
boot, it is refused by reason before anything starts.

The configs are the field firmware's own ``motorconfig.json`` (the one the
simulator boots by default) with a unit's facts written over it, as an OEM
unit's file is the same file with its own.
"""

import copy
import json

import pytest

from _lumascope.drivers.sim_wire import backend as sim_backend
from _lumascope.modules.channel_record import (
    STOCK_OPTICS_MODEL,
    IdentitySource,
    MotorBoardFamily,
    load_scope_models,
)
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.lumascope_api.simulation import SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

OEM_NAME = 'ACME-1'
FIELD_CONFIG = json.loads(sim_backend._MOTORCONFIG_BASE.read_text())


def unit_config(axes: str, **facts) -> dict:
    """The field motorconfig.json of an OEM unit named ``OEM_NAME`` on ``axes``,
    with a focal length and a Z travel no catalogue model has."""
    config = copy.deepcopy(FIELD_CONFIG)
    config['Microscope'] = OEM_NAME
    config['Axis Present'] = {axis: int(axis in axes) for axis in 'XYZT'}
    config['Optics'] = {'LensFocalLength': 61.5, 'PixelSize': 2.0}
    config['Axis Travel Limit']['Z'] = 11.5
    config.update(facts)
    return config


def _session(tier: str, unit: object, microscope: str = 'LS850') -> ScopeSession:
    return ScopeSession.create(
        complete_settings(simulator_tier=tier, microscope=microscope),
        simulate=True,
        simulated=SimulatedScope(unit_config=unit),
        warn_pre_release=False,
    )


@pytest.fixture
def unit_session():
    sessions = []

    def _make(unit, microscope='LS850'):
        session = _session('firmware', unit, microscope)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


@pytest.mark.firmware_tier
class TestTheScopeIsTheUnit:
    def test_an_oem_unit_comes_up_under_its_name_on_its_axes_with_the_stock_channels(
        self, unit_session
    ):
        # Z only, under a stored XY model: the axes are the unit's, not the
        # stored model's, and a field config carries no channel block.
        scope = unit_session(unit_config('Z')).scope
        assert set(scope.capabilities.axes) == {'Z'}
        identity = scope.channel_identity
        assert (identity.model, identity.source) == (OEM_NAME, IdentitySource.STOCK)
        stock = ScopeSession.create(
            complete_settings(microscope=STOCK_OPTICS_MODEL), simulate=True, warn_pre_release=False
        )
        try:
            assert [c.key_name for c in identity.channels] == [
                c.key_name for c in stock.scope.channel_identity.channels
            ]
        finally:
            stock.shutdown()
        assert scope.capabilities.lens_focal_length_mm == 61.5
        assert scope.motion.get_axis_limits('Z').max_um == 11500.0

    def test_a_units_channel_block_is_its_channels(self, unit_session):
        block = {'BF': {'led_channel': 3, 'display_name': 'Brightfield'}}
        scope = unit_session(unit_config('XYZ', Channels=block, Filterset='Custom')).scope
        assert set(scope.capabilities.axes) == {'X', 'Y', 'Z'}
        identity = scope.channel_identity
        assert (identity.source, identity.filterset) == (IdentitySource.MOTORCONFIG, 'Custom')
        assert [c.key_name for c in identity.channels] == ['BF']


def _refusal(tier: str, unit: object, microscope: str = 'LS850') -> ArgumentRefusedError:
    with pytest.raises(ArgumentRefusedError) as refused:
        _session(tier, unit, microscope)
    assert refused.value.argument.startswith('sim_unit_config')
    return refused.value


class TestWhatTheFirmwareCannotBootIsRefused:
    def test_the_fast_tier_does_not_run_it(self):
        refused = _refusal('fast', unit_config('Z'))
        assert refused.reason == 'needs_simulated_firmware'
        assert refused.board.startswith('the fast simulated motor board')

    def test_a_machine_with_no_runtime_says_so(self, monkeypatch):
        # a stand-in by design: a machine with no MicroPython runtime is a
        # state of the host, which no simulated part can present.
        monkeypatch.setattr(sim_backend, 'runtime_missing', lambda dialect: 'no runtime here')
        refused = _refusal('firmware', unit_config('Z'))
        assert refused.reason == 'needs_simulated_firmware'
        assert 'no MicroPython runtime for the others (no runtime here)' in refused.board

    def test_a_model_with_no_el0940_motor_board_does_not_run_it(self):
        other = sorted(
            name
            for name, entry in load_scope_models().items()
            if entry.motor_board != MotorBoardFamily.EL_0940
        )
        refused = _refusal('firmware', unit_config('Z'), microscope=other[0])
        assert refused.reason == 'needs_simulated_firmware'

    @pytest.mark.firmware_tier
    def test_a_config_without_a_key_the_simulator_boots_it_by(self):
        unit = unit_config('Z')
        del unit['IniFiles']
        refused = _refusal('firmware', unit)
        assert (refused.reason, refused.argument) == ('missing_key', "sim_unit_config['IniFiles']")

    @pytest.mark.firmware_tier
    def test_a_config_marking_no_axis_present(self):
        refused = _refusal('firmware', unit_config(''))
        assert (refused.reason, refused.argument) == (
            'wrong_kind',
            "sim_unit_config['Axis Present']",
        )

    @pytest.mark.firmware_tier
    def test_a_register_table_the_firmware_does_not_carry(self):
        unit = unit_config('Z')
        unit['IniFiles']['ZT'] = 'ztmotorconfig2.ini'
        refused = _refusal('firmware', unit)
        assert (refused.reason, refused.value) == ('not_offered', 'ztmotorconfig2.ini')
        assert refused.offered == tuple(sorted(sim_backend.register_tables('field')))

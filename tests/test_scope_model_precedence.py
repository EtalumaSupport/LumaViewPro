# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's model: the hardware's word wins, and a simulated scope has one.

Two facts, one root. A simulated scope's motor board took the module-global
settings' model and ignored the ``configured_model`` its constructor was
handed, so the driver and the selection disagreed from construction on the
sim. And the bring-up read the STORED model into the axis configuration
without asking the hardware, so a unit whose file named the wrong model
configured for the wrong axes -- the GUI corrected that in its own copy of
the bring-up, and a headless host never did.
"""

import logging

import pytest

import _lumascope.modules.lumascope_api._lumascope as lumascope_module
from _lumascope.drivers.null_ledboard import NullLEDBoard
from _lumascope.drivers.null_motorboard import NullMotionBoard
from _lumascope.modules.exceptions import ScopeModelUnknownError
from _lumascope.modules.lumascope_api.simulation import SimulatedModelAnswer, SimulatedScope
from _lumascope.modules.scope_session import ScopeSession
from tests.log_capture import records_of
from tests.settings_fixtures import complete_settings, installation_of_its_own
from tests.scope_fakes import build_scope, simulated_camera_on_a_real_scope


def _sim_scope(**kwargs):
    return build_scope(
        simulate=True,
        register_atexit=False,
        warn_pre_release=False,
        **kwargs,
    )


@pytest.fixture
def session_log(caplog):
    """What the session module has logged so far in the test."""
    import _lumascope.modules.scope_session as scope_session

    caplog.set_level(logging.DEBUG)
    return lambda: records_of(caplog, scope_session)


class TestASimulatedScopeReportsItsDeclaredModel:
    def test_configured_model_outranks_the_templates(self):
        # The shipped template declares LS850.
        scope = _sim_scope(configured_model='LS850T')
        try:
            assert scope.diagnostics.motor_info().model == 'LS850T'
        finally:
            scope.disconnect()

    def test_sim_model_still_outranks_configured_model(self):
        scope = _sim_scope(sim_model='LS850T', configured_model='LS850')
        try:
            assert scope.diagnostics.motor_info().model == 'LS850T'
        finally:
            scope.disconnect()

    def test_the_template_headless_session_has_no_turret(self, tmp_path):
        # The shipped template declares LS850. The session used to report
        # LS850T -- and a turret axis -- because the sim ignored the
        # declaration; now the declaration is what the scope is.
        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            assert session.settings['microscope'] == 'LS850'
            assert session.scope.diagnostics.motor_info().model == 'LS850'
            assert session.scope.capabilities.has_turret is False
        finally:
            session.shutdown()
            session.scope.disconnect()


class TestTheBringUpRunsAsTheReportedModel:
    """Bring-up configures for the model the motor board reports, and the
    stored selection stands: it is the model chosen for a scope that does
    not report its own."""

    def _created(self, tmp_path, *, sim_model=None, sim_model_answer=None, **settings_overrides):
        return ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), **settings_overrides),
            simulate=True,
            simulated=SimulatedScope(model=sim_model, model_answer=sim_model_answer),
            warn_pre_release=False,
        )

    def test_a_catalogued_reported_model_decides_the_turret_and_the_stored_one_stands(
        self, tmp_path, session_log
    ):
        # The file says LS850 (no turret) with a stored objective; the board
        # is an LS850T. Whether the objective is stored or derived from the
        # slot depends on the model having a turret, so the scope is
        # configured as the board's model, while the stored selection is
        # kept for a scope that does not report its own.
        session = self._created(
            tmp_path,
            sim_model='LS850T',
            microscope='LS850',
            objective_id='20x Oly',
            turret_objectives={1: '10x Oly', 2: None, 3: None, 4: None},
        )
        try:
            assert session.scope.channel_identity.model == 'LS850T'
            assert session.settings['microscope'] == 'LS850'
            assert session.scope.capabilities.has_turret, (
                "the turret decided against the stored model, not the board's"
            )
            # Derived from a slot no turret command has named yet, so
            # unknown -- never the stored turretless selection.
            assert session.scope.runtime_state.get_current_objective_id() is None
            assert any('and runs as it' in r.getMessage() for r in session_log())
        finally:
            session.shutdown()

    def test_a_model_outside_the_catalogue_leaves_the_stored_one(self, tmp_path, session_log):
        session = self._created(
            tmp_path, sim_model_answer=SimulatedModelAnswer('LS999'), microscope='LS850T'
        )
        try:
            assert session.settings['microscope'] == 'LS850T'
            assert any('not in the catalogue' in r.getMessage() for r in session_log())
        finally:
            session.shutdown()

    def test_no_reported_model_leaves_the_stored_one(self, tmp_path):
        session = self._created(
            tmp_path, sim_model_answer=SimulatedModelAnswer(None), microscope='LS850T'
        )
        try:
            assert session.scope.diagnostics.motor_info().model is None
            assert session.settings['microscope'] == 'LS850T'
        finally:
            session.shutdown()

    def test_a_matching_report_writes_nothing(self, tmp_path, session_log):
        # Green before and after the change by construction; kept so the
        # silent case stays silent.
        session = self._created(tmp_path, microscope='LS850T')
        try:
            assert session.settings['microscope'] == 'LS850T'
            assert not any('scope reports model' in r.getMessage() for r in session_log())
        finally:
            session.shutdown()

    def test_a_board_reporting_a_model_no_catalogue_lists_describes_the_scope(self, tmp_path):
        # An OEM board names itself (its branding), which no catalogue lists;
        # its own enabled axes describe the scope, so an unlisted stored
        # model is not refused under it. Green before the change too: it
        # guards the refusal below from reaching a board that spoke.
        session = self._created(
            tmp_path,
            sim_model='LS850',
            sim_model_answer=SimulatedModelAnswer('OEM-200'),
            microscope='LS999',
        )
        try:
            assert session.scope.diagnostics.motor_info().model == 'OEM-200'
            assert session.scope.capabilities.axes == ('X', 'Y', 'Z')
            assert session.settings['microscope'] == 'LS999'
        finally:
            session.shutdown()

    def test_a_listed_reported_model_runs_over_an_unlisted_stored_one(self, tmp_path):
        # The board's word is taken before the stored model is judged, so an
        # unlisted one is not refused under a board that names its model;
        # it is left as it is, as every stored selection is.
        session = self._created(tmp_path, sim_model='LS850', microscope='LS999')
        try:
            assert session.scope.channel_identity.model == 'LS850'
            assert session.settings['microscope'] == 'LS999'
        finally:
            session.shutdown()

    def test_a_declared_model_survives_a_simulated_create(self, tmp_path):
        # A regression guard, green before and after: the sim reports the
        # declaration, so step 0 has nothing to correct.
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
        )
        try:
            assert session.settings['microscope'] == 'LS850'
        finally:
            session.shutdown()
            session.scope.disconnect()


def test_with_no_board_to_say_what_the_scope_is_an_unlisted_stored_model_is_refused(monkeypatch):
    """A real scope whose motor board reports no model (none is plugged in: an
    FX2 Classic, a manual scope) is described by its stored model alone, so
    one the catalogue does not list is refused at bring-up, as the simulator
    already refuses it. That the refused bring-up's threads end is
    ``test_a_refused_create_leaves_nothing_running.py``'s.
    """
    # a stand-in by design: the registries' port scans, which the suite never
    # runs; the boards are production's null drivers, the camera the simulated one.
    monkeypatch.setattr(
        lumascope_module.motor_registry,
        'create_with_fallback',
        lambda name='auto', **_kwargs: (NullMotionBoard(), None),
    )
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **_kwargs: (NullLEDBoard(), None),
    )
    simulated_camera_on_a_real_scope(monkeypatch)

    with pytest.raises(ScopeModelUnknownError) as refused:
        ScopeSession.create(
            complete_settings(microscope='LS999'),
            source_path=str(installation_of_its_own()),
            simulate=False,
            warn_pre_release=False,
        )
    assert refused.value.model == 'LS999'

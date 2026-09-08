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

import modules.lumascope_api as lumascope_api
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _sim_scope(**kwargs):
    return lumascope_api.Lumascope(
        simulate=True,
        register_atexit=False,
        register_metrics=False,
        warn_pre_release=False,
        **kwargs,
    )


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def session_log(monkeypatch):
    """The session module's log records. The suite replaces the LVP logger
    with a mock, so the module's logger is swapped for a private real one
    with a capturing handler for the test's duration."""
    import modules.scope_session as scope_session

    private = logging.getLogger('test.scope_model_precedence.session')
    private.propagate = False
    private.setLevel(logging.DEBUG)
    handler = _CaptureHandler()
    private.addHandler(handler)
    monkeypatch.setattr(scope_session, 'logger', private)
    try:
        yield handler.records
    finally:
        private.removeHandler(handler)


class TestASimulatedScopeReportsItsDeclaredModel:
    def test_configured_model_outranks_the_module_global(self, monkeypatch):
        import modules.settings_init as settings_init

        monkeypatch.setattr(settings_init, 'settings', {'microscope': 'LS850T'})
        scope = _sim_scope(configured_model='LS850')
        try:
            assert scope.diagnostics.get_microscope_model() == 'LS850'
        finally:
            scope.disconnect()

    def test_sim_model_still_outranks_configured_model(self):
        scope = _sim_scope(sim_model='LS850T', configured_model='LS850')
        try:
            assert scope.diagnostics.get_microscope_model() == 'LS850T'
        finally:
            scope.disconnect()

    def test_the_template_headless_session_has_no_turret(self, tmp_path):
        # The shipped template declares LS850. The session used to report
        # LS850T -- and a turret axis -- because the sim ignored the
        # declaration; now the declaration is what the scope is.
        session = ScopeSession.create_headless(
            settings=complete_settings(live_folder=str(tmp_path))
        )
        try:
            assert session.settings['microscope'] == 'LS850'
            assert session.scope.diagnostics.get_microscope_model() == 'LS850'
            assert session.scope.capabilities.has_turret is False
        finally:
            session.shutdown()
            session.scope.disconnect()


class TestTheBringUpAdoptsTheReportedModel:
    """configure_scope() step 0: the model the motor board reports lands in
    the session's settings before anything reads the stored one."""

    def _caller_scope_session(self, tmp_path, scope, **settings_overrides):
        raw = complete_settings(live_folder=str(tmp_path), **settings_overrides)
        return ScopeSession.create(settings=raw, scope=scope, warn_pre_release=False)

    def test_a_catalogued_model_is_written_before_the_slot_one_adoption(
        self, tmp_path, session_log
    ):
        # The file says LS850 (no turret) with an objective assigned to
        # slot 1; the board says LS850T. Adopting slot 1 depends on the
        # model having a turret, so the write must precede that read.
        scope = _sim_scope(sim_model='LS850T')
        session = self._caller_scope_session(
            tmp_path,
            scope,
            microscope='LS850',
            objective_id='20x Oly',
            turret_objectives={1: '10x Oly', 2: None, 3: None, 4: None},
        )
        try:
            session.configure_scope()
            assert session.settings['microscope'] == 'LS850T'
            assert session.settings['objective_id'] == '10x Oly', (
                'slot 1 adopted against the ADOPTED model'
            )
            assert any('the hardware wins' in r.getMessage() for r in session_log)
        finally:
            session.shutdown()
            scope.disconnect()

    def test_a_model_outside_the_catalogue_leaves_the_stored_one(
        self, tmp_path, session_log, monkeypatch
    ):
        scope = _sim_scope(sim_model='LS850T')
        session = self._caller_scope_session(tmp_path, scope, microscope='LS850T')
        monkeypatch.setattr(scope.diagnostics, 'get_microscope_model', lambda: 'LS999')
        try:
            session.configure_scope()
            assert session.settings['microscope'] == 'LS850T'
            assert any('not in the catalogue' in r.getMessage() for r in session_log)
        finally:
            session.shutdown()
            scope.disconnect()

    def test_no_reported_model_leaves_the_stored_one(self, tmp_path, monkeypatch):
        scope = _sim_scope(sim_model='LS850T')
        session = self._caller_scope_session(tmp_path, scope, microscope='LS850T')
        monkeypatch.setattr(scope.diagnostics, 'get_microscope_model', lambda: None)
        try:
            session.configure_scope()
            assert session.settings['microscope'] == 'LS850T'
        finally:
            session.shutdown()
            scope.disconnect()

    def test_a_matching_report_writes_nothing(self, tmp_path, session_log):
        # Green before and after the change by construction; kept so the
        # silent case stays silent.
        scope = _sim_scope(sim_model='LS850T')
        session = self._caller_scope_session(tmp_path, scope, microscope='LS850T')
        try:
            session.configure_scope()
            assert session.settings['microscope'] == 'LS850T'
            assert not any('scope reports model' in r.getMessage() for r in session_log)
        finally:
            session.shutdown()
            scope.disconnect()

    def test_a_declared_model_survives_create_headless(self, tmp_path):
        # A regression guard, green before and after: the sim reports the
        # declaration, so step 0 has nothing to correct.
        session = ScopeSession.create_headless(
            settings=complete_settings(live_folder=str(tmp_path), microscope='LS850')
        )
        try:
            assert session.settings['microscope'] == 'LS850'
        finally:
            session.shutdown()
            session.scope.disconnect()

    def test_the_catalogue_refusal_precedes_any_write_to_the_callers_dict(
        self, tmp_path, monkeypatch
    ):
        from modules import layer_record
        from modules.exceptions import ConfigError

        scope = _sim_scope(sim_model='LS850T')
        raw = complete_settings(live_folder=str(tmp_path), microscope='LS850')
        raw['turret_objectives'] = {'1': '10x Oly', '2': None, '3': None, '4': None}
        session = ScopeSession.create(settings=raw, scope=scope, warn_pre_release=False)
        monkeypatch.setattr(
            layer_record,
            'load_scope_models',
            lambda *a, **k: (_ for _ in ()).throw(ConfigError('no Models')),
        )
        try:
            with pytest.raises(ConfigError):
                session.configure_scope()
            assert session.settings['microscope'] == 'LS850', 'nothing was written past the refusal'
            assert list(session.settings['turret_objectives']) == ['1', '2', '3', '4'], (
                'the slot keys were not normalized past the refusal'
            )
        finally:
            session.shutdown()
            scope.disconnect()

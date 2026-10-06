# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's identity carries its model, and a model change applies at the next start.

An FX2 scope (LS560, LS620, LS720) cannot report its own model, so the
operator's selection is its identity. The resolver settled on a model and
threw it away: the capabilities asked the motor board, which on an FX2 scope
is the null board and answers nothing, and a saved image recorded the same
empty answer. So the layers came from the selected model while the
capabilities and every file said there was none. And a selection in Advanced
Settings re-resolved the layers of the running scope while its capabilities,
fixed at construction, stayed with the old model -- one scope, two models.

The model lives on the identity, every reader takes it from there, and a new
selection is saved for the next start without touching the running scope.
"""

import pytest

from modules import layer_record
from modules.exceptions import Refusal
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.frame_records import frame_record, plate


def _resolve(**kwargs):
    args = {
        'board_block': None,
        'board_config_read_ok': True,
        'motor_model': None,
        'configured_model': None,
        'models': layer_record.load_scope_models(),
        'catalogue': layer_record.release_catalogue(),
    }
    args.update(kwargs)
    return layer_record.resolve_layer_identity(**args)


class TestEveryRungCarriesTheModel:
    def test_the_override_rung(self):
        assert _resolve(motor_model='LS850T', override_model='LS560').model == 'LS560'

    def test_the_board_block_rung_carries_the_boards_model(self):
        block = {'Layers': [], 'Filterset': ''}
        assert _resolve(board_block=block, motor_model='LS850T').model == 'LS850T'

    def test_the_catalogue_rung_carries_the_selection_when_the_board_reports_none(self):
        identity = _resolve(configured_model='LS620')
        assert identity.source == 'scopes'
        assert identity.model == 'LS620'

    def test_a_board_model_the_catalogue_lacks_is_carried_with_no_layers(self):
        identity = _resolve(motor_model='LS9999', configured_model='LS560')
        assert identity.model == 'LS9999'
        assert identity.layers == ()
        assert identity.source == 'unresolved'

    def test_no_model_at_all_is_the_shared_unresolved_snapshot(self):
        identity = _resolve()
        assert identity is layer_record.UNRESOLVED
        assert identity.model is None


@pytest.fixture
def ls620_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS620'), simulate=True
    )
    yield session
    session.shutdown()
    session.scope.disconnect()


class TestTheRunningScopeHasOneModel:
    def test_the_capabilities_carry_the_selected_model_of_an_fx2_scope(self, ls620_session):
        # The null motor board reports no model; the selection is the identity.
        assert ls620_session.scope.diagnostics.get_motor_info()['model'] is None
        assert ls620_session.scope.capabilities.model == 'LS620'

    def test_a_saved_image_records_the_identitys_model(self, ls620_session):
        from modules import image_save

        scope = ls620_session.scope
        metadata = image_save.generate_image_metadata(
            scope,
            'BF',
            None,
            None,
            None,
            objective_id='20x Oly',
            frame_record=frame_record(),
            labware=plate(),
            well_label=None,
        )
        assert metadata['microscope'] == 'LS620'
        assert metadata['microscope_model'] == 'LS620'
        assert metadata['instrument']['model'] == 'LS620'


class TestAModelSelectionAppliesAtTheNextStart:
    def test_the_selection_is_saved_and_the_running_scope_is_unchanged(self, ls620_session):
        scope = ls620_session.scope
        identity = scope.layer_identity
        capabilities = scope.capabilities

        ls620_session.select_model('LS560')

        assert ls620_session.settings['microscope'] == 'LS560'
        assert scope.layer_identity is identity
        assert scope.capabilities is capabilities
        assert scope.capabilities.model == 'LS620'

    def test_a_model_the_catalogue_lacks_is_refused_and_nothing_is_saved(self, ls620_session):
        from modules.exceptions import ScopeModelUnknownError

        with pytest.raises(ScopeModelUnknownError) as refused:
            ls620_session.select_model('LS9999')
        assert isinstance(refused.value, Refusal)
        assert 'LS9999' in str(refused.value)
        assert ls620_session.settings['microscope'] == 'LS620'

    def test_the_declared_turret_answer_is_the_running_scopes(self, ls620_session):
        # With no motor board to ask, the turret question reads the declared
        # model -- the one the scope is running as, not a selection saved
        # for the next start.
        ls620_session.select_model('LS850T')
        assert ls620_session.scope.motor_connected is False
        assert ls620_session.scope_has_turret() is False


class TestNothingReselectsARunningScope:
    def test_a_running_scope_cannot_be_re_resolved_as_a_selected_model(self, ls620_session):
        # Re-resolving the layers for a new selection is what split one scope
        # into two models; the override stays for tests.
        with pytest.raises(TypeError):
            ls620_session.scope.refresh_layer_identity(configured_model='LS560')

    def test_the_session_is_the_only_writer_of_the_stored_model(self):
        from tests.ast_seams import writers_of_settings_keys

        assert writers_of_settings_keys({'microscope'}) == {
            ('modules/scope_session.py', 'ScopeSession.configure_scope'),
            ('modules/scope_session.py', 'ScopeSession.select_model'),
        }

    def test_no_production_code_re_resolves_the_identity(self):
        import ast

        from tests.ast_seams import production_modules

        callers = sorted(
            rel_path
            for rel_path, tree in production_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'refresh_layer_identity'
        )
        assert callers == []

    def test_the_model_label_shows_the_running_model_not_the_selection(self):
        import ast

        from tests.ast_seams import find_def

        fn = find_def(
            'ui/microscope_settings.py',
            'set_ui_features_for_scope',
            class_name='MicroscopeSettings',
        )
        assert fn is not None
        keys = {
            node.slice.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
        }
        assert 'microscope' not in keys

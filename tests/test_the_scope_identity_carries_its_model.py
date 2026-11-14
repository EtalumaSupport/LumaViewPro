# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's identity carries its model, and a model change applies at the next start.

An FX2 scope (LS460, LS560, LS620, LS720) cannot report its own model, so the
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

import ast

import pytest

from _lumascope.modules import channel_record
from _lumascope.modules.exceptions import Refusal
from tests.scope_fakes import refuse_a_report
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.frame_records import frame_record, plate


def _resolve(**kwargs):
    args = {
        'board_block': None,
        'board_config_read_ok': True,
        'motor_model': None,
        'configured_model': None,
        'models': channel_record.load_scope_models(),
        'catalogue': channel_record.release_channels(),
        'report': refuse_a_report,
    }
    args.update(kwargs)
    return channel_record.resolve_channel_identity(**args)


class TestEveryRungCarriesTheModel:
    def test_the_board_block_rung_carries_the_boards_model(self):
        block = {'Channels': {}, 'Filterset': ''}
        assert _resolve(board_block=block, motor_model='LS850T').model == 'LS850T'

    def test_the_catalogue_rung_carries_the_selection_when_the_board_reports_none(self):
        identity = _resolve(configured_model='LS620')
        assert identity.source == 'scopes'
        assert identity.model == 'LS620'

    def test_a_board_model_the_catalogue_lacks_is_carried_with_the_stock_layers(self):
        # An OEM unit: its board names a model no catalogue lists, and its
        # optics are the stock module's.
        identity = _resolve(motor_model='LS9999', configured_model='LS560')
        stock = channel_record.load_scope_models()[channel_record.STOCK_OPTICS_MODEL]
        assert identity.model == 'LS9999'
        assert identity.channels == stock.channels
        assert identity.filterset == stock.filterset
        assert identity.source == 'stock'

    def test_no_model_at_all_is_the_shared_unresolved_snapshot(self):
        identity = _resolve()
        assert identity is channel_record.UNRESOLVED
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
    def test_the_identity_carries_the_selected_model_of_an_fx2_scope(self, ls620_session):
        # The null motor board reports no model; the selection is the identity.
        assert ls620_session.scope.diagnostics.motor_info().model is None
        assert ls620_session.scope.channel_identity.model == 'LS620'

    def test_a_saved_image_records_the_identitys_model(self, ls620_session):
        from _lumascope.modules import image_save

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
            fallback_focus=None,
        )
        assert metadata['instrument']['model'] == 'LS620'


class TestAModelSelectionAppliesAtTheNextStart:
    def test_the_selection_is_saved_and_the_running_scope_is_unchanged(self, ls620_session):
        scope = ls620_session.scope
        identity = scope.channel_identity
        capabilities = scope.capabilities

        ls620_session.select_model('LS560')

        assert ls620_session.settings['microscope'] == 'LS560'
        assert scope.channel_identity is identity
        assert scope.capabilities is capabilities
        assert scope.channel_identity.model == 'LS620'

    def test_a_model_the_catalogue_lacks_is_refused_and_nothing_is_saved(self, ls620_session):
        from _lumascope.modules.exceptions import ScopeModelUnknownError

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
        assert ls620_session.scope.capabilities.has_turret is False


class TestNothingReselectsARunningScope:
    def test_a_scopes_channels_are_resolved_only_when_it_is_built(self):
        # Re-resolving the layers of a running scope is what split one scope
        # into two models: its capabilities are fixed when it is built, so
        # its channel identity is written there and nowhere else.
        from tests.ast_seams import production_modules, walk_defs

        writers = set()
        for rel_path, tree in production_modules():
            for qualname, fn in walk_defs(tree.body):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                        targets = [node.target]
                    else:
                        continue
                    if any(
                        isinstance(t, ast.Attribute) and t.attr == 'channel_identity'
                        for t in targets
                    ):
                        writers.add((rel_path, qualname))

        assert writers == {
            ('src/_lumascope/modules/lumascope_api/_lumascope.py', 'Lumascope.__init__')
        }

    def test_the_selection_is_the_only_writer_of_the_stored_model(self):
        # Bring-up does not write it: the stored model is the selection, kept
        # for a scope that does not report its own, never the board's model.
        from tests.ast_seams import writers_of_settings_keys

        assert writers_of_settings_keys({'microscope'}) == {
            ('src/_lumascope/modules/scope_session.py', 'ScopeSession.select_model'),
        }

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


@pytest.mark.parametrize('sim_session', [{'microscope': 'Lumi'}], indirect=True)
def test_a_lumi_offers_darkfield_and_lights_its_led(sim_session):
    """A Lumi's illuminator carries the darkfield ring, so the Lumi model
    offers DF and lights its LED at the board's DF address, as the LS820's."""
    from _lumascope.modules.lumascope_api.illumination import LedState

    scope = sim_session.scope
    assert 'DF' in [record.key_name for record in scope.channel_identity.channels]

    scope.illumination.led_on('DF', 10)

    (df,) = [reading for reading in scope.illumination.led_states() if reading.name == 'DF']
    assert (df.address, df.state) == (5, LedState.LIT)

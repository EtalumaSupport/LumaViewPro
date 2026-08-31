# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Adding a layer's IDENTITY is one catalogue row and one model row.

The receipt: with an eighth layer present in the (fixture) release
catalogue and a model's row list, identity resolution, the layer
vocabulary, protocol Color validation, and saved-image metadata all
accept it with ZERO code edits. Every place that would have needed a
hand edit before -- the seven-name literals, the enum-derived validation
set, the metadata vocabulary gate -- now derives from the catalogue.

(The settings-block and GUI-accordion shares of adding a layer are
deliberately not covered: they belong to later waves, and this receipt
pins exactly the identity subset.)
"""

import json

import pytest

import modules.common_utils as common_utils
import modules.layer_record as layer_record
from modules.image_save import generate_image_metadata
from modules.protocol import Protocol

EIGHT_KEYS = ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi', 'NIR']


@pytest.fixture
def eighth_layer_release(tmp_path, monkeypatch):
    """A release whose catalogue and LS850T model carry an eighth layer."""
    data = {
        'LayerOrder': EIGHT_KEYS,
        'Models': {
            'LS850T': {
                'Layers': [
                    {
                        'key_name': 'BF',
                        'display_name': 'BF',
                        'led_channel': 3,
                        'excitation_nm': None,
                    },
                    {
                        'key_name': 'NIR',
                        'display_name': 'NIR',
                        'led_channel': 6,
                        'excitation_nm': 780,
                    },
                ]
            }
        },
    }
    path = tmp_path / 'scopes_fixture.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(
        layer_record,
        '_CATALOGUE_CACHE',
        layer_record.load_layer_catalogue(layer_record.load_scopes_data(str(path))),
    )
    return str(path)


def test_identity_resolves_the_eighth_layer(eighth_layer_release):
    identity = layer_record.resolve_layer_identity(
        board_block=None,
        board_config_read_ok=True,
        motor_model='LS850T',
        configured_model=None,
        data_file=eighth_layer_release,
    )
    nir = identity.find('NIR')
    assert nir is not None
    assert nir.id == 7
    assert nir.led_channel == (6,)
    assert nir.excitation_nm == pytest.approx(780.0)


def test_vocabulary_and_validation_accept_the_eighth_layer(eighth_layer_release):
    assert 'NIR' in common_utils.get_layers()
    assert 'NIR' in Protocol.valid_colors()


def test_metadata_accepts_the_eighth_layer(eighth_layer_release, sim_scope):
    from modules.labware_loader import WellPlateLoader

    loader = WellPlateLoader()
    sim_scope.runtime_state.set_objective('20x Oly')
    sim_scope.runtime_state.set_labware(loader.get_plate('96 well microplate'))
    sim_scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
    metadata = generate_image_metadata(sim_scope, channel='NIR', x=0, y=0, z=0)
    assert metadata['channel'] == 'NIR'


def test_shipped_categories_partition_the_shipped_catalogue():
    """On the REAL release data the category functions cover the whole
    catalogue with no overlap -- the partition holds by construction
    because every function filters the same derived list."""
    layers = common_utils.get_layers()
    union = (
        common_utils.get_transmitted_layers()
        + common_utils.get_fluorescence_layers()
        + common_utils.get_luminescence_layers()
    )
    assert sorted(union) == sorted(layers)
    assert len(union) == len(set(union))

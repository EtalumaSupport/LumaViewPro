# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The image-settings panel shows a drawer for each layer the unit has, drawn from its record.

``ImageSettings.show_drawers`` takes the unit's identity: each layer it holds
is shown in catalogue order, titled by its record's label beside a swatch of
its display colour, its controls drawn in that colour lightened, with
illumination where its row drives an LED and summing for luminescence; each
layer it lacks is hidden and stops being acquired. The identities are the
real ones, resolved from the installation's ``scopes.json`` for three models
that differ in which layers they carry.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.layer_record as layer_record
from tests.scope_fakes import refuse_a_report
from ui.image_settings import ImageSettings

_uids = itertools.count()


class _Drawer:
    """A drawer's state show_drawers writes and resort_accordion reads."""

    def __init__(self, layer):
        self.layer = layer
        self.uid = next(_uids)
        self.parent = None
        self.collapse = False
        self.title = ''
        self.color_rgba = None
        self.chrome_rgba = None
        self.ids = {'control': SimpleNamespace()}


class _Accordion:
    """Kivy's children semantics: children[0] is drawn last, at the bottom;
    add_widget with no index prepends."""

    def __init__(self):
        self.children = []

    def add_widget(self, widget, index=0):
        assert widget.parent is None, 'add_widget refuses an attached widget'
        self.children.insert(index, widget)
        widget.parent = self

    def remove_widget(self, widget):
        if widget in self.children:
            self.children.remove(widget)
            widget.parent = None

    def top_to_bottom(self):
        return [w.layer for w in reversed(self.children)]


def _identity(model):
    return layer_record.resolve_layer_identity(
        board_block=None,
        board_config_read_ok=True,
        motor_model=model,
        configured_model=None,
        models=layer_record.load_scope_models(),
        catalogue=layer_record.release_layers(),
        report=refuse_a_report,
    )


@pytest.fixture
def panel(monkeypatch):
    # a stand-in by design: Kivy is stubbed in the test process; the subject
    # is what show_drawers writes onto each drawer and which it shows
    acquire = []
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=SimpleNamespace(
                set_layer_acquire=lambda layer, v: acquire.append((layer, v)),
                channel_colors=layer_record.release_colors,
            )
        ),
    )
    panel = SimpleNamespace(
        ids={'accordion_id': _Accordion()},
        _drawers={layer: _Drawer(layer) for layer in layer_record.release_catalogue()},
        _shown_layers=frozenset(),
        acquire=acquire,
    )
    panel._resort_accordion = lambda: ImageSettings._resort_accordion(panel)
    return panel


@pytest.mark.parametrize('model', ['LS850', 'LS560', 'Lumi'])
def test_each_layer_the_unit_has_is_shown_from_its_record(panel, model):
    # parametrize-justified: three units carrying different layer sets
    identity = _identity(model)
    assert identity.source == 'scopes'

    ImageSettings.show_drawers(panel, identity)

    assert panel.ids['accordion_id'].top_to_bottom() == [r.key_name for r in identity.layers]
    for record in identity.layers:
        drawer = panel._drawers[record.key_name]
        control = drawer.ids['control']
        assert drawer.title == record.label
        color = layer_record.parse_display_color(record.display_color)
        assert drawer.color_rgba == (*color, 1.0)
        # The chrome is the colour moved toward white, so no plane is darker.
        assert all(c >= w for c, w in zip(drawer.chrome_rgba, (*color, 1.0), strict=True))
        assert control.illumination_support == bool(record.led_channel)
        assert control.exposure_summing_support == (record.modality == 'luminescence')
        assert control.autogain_support == (record.modality != 'luminescence')
    present = {r.key_name for r in identity.layers}
    absent = [layer for layer in layer_record.release_catalogue() if layer not in present]
    assert panel.acquire == [(layer, None) for layer in absent]
    assert all(panel._drawers[layer].collapse for layer in absent)


def test_a_fluorescence_drawer_is_titled_by_its_excitation_and_lumi_sums(panel):
    ImageSettings.show_drawers(panel, _identity('LS850'))
    blue = panel._drawers['Blue']
    assert blue.title == '405 nm'
    assert blue.color_rgba == (0.0, 0.0, 1.0, 1.0)
    assert blue.chrome_rgba == pytest.approx((0.4, 0.4, 1.0, 1.0))
    assert blue.ids['control'].exposure_summing_support is False

    ImageSettings.show_drawers(panel, _identity('Lumi'))
    lumi = panel._drawers['Lumi'].ids['control']
    assert lumi.illumination_support is False
    assert lumi.exposure_summing_support is True
    assert lumi.autogain_support is False


def test_a_model_switch_and_back_restores_the_drawers_in_catalogue_order(panel):
    ImageSettings.show_drawers(panel, _identity('LS850'))
    ImageSettings.show_drawers(panel, _identity('LS560'))
    assert panel.ids['accordion_id'].top_to_bottom() == [
        r.key_name for r in _identity('LS560').layers
    ]

    ImageSettings.show_drawers(panel, _identity('LS850'))

    assert panel.ids['accordion_id'].top_to_bottom() == [
        r.key_name for r in _identity('LS850').layers
    ]

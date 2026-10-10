# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: tile overlap is a persisted system setting.

Tile overlap used to live only in the protocol panel's overlap spinner
and was read straight off the widget at scan time, so it reset to 0%
every launch. It is now promoted to settings['tiling_overlap_percent']:
the spinner is just the editor (writes the setting via
update_tiling_overlap), and scan/apply read the persisted value (a scan
through the sequenced-capture config, Apply through
``ScopeSession.apply_tiling``).

These guard the two halves of that contract:
  - the key ships in the tracked settings.json schema default, so the
    settings_init default-merge backfills it into every user's
    current.json and the bare read never raises KeyError;
  - a scan config carries the setting, not the widget (Apply's read is
    pinned by test_a_protocol_is_tiled_and_stacked_through_the_api).

current.json is gitignored runtime state, not the schema source, so it
is intentionally not asserted here.
"""

from __future__ import annotations

import ast
import json
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(path.read_text(), child)
    raise AssertionError(f'{class_name}.{method_name} not found in {path}')


def test_tiling_overlap_percent_in_settings_schema():
    """The tracked settings.json default carries the key.

    settings_init merges missing settings.json keys into current.json at
    load, so shipping the default here backfills every existing user and
    the bare read in ScopeSession.apply_tiling never raises KeyError.
    """
    data = json.loads((REPO / 'data' / 'settings.json').read_text())
    assert 'tiling_overlap_percent' in data, (
        'data/settings.json must define tiling_overlap_percent so the '
        'default-merge backfills it and the persisted read does not KeyError'
    )


def test_the_modal_populating_the_overlap_is_not_a_pick(monkeypatch, caplog):
    """Opening Advanced Settings writes the stored overlap into the spinner,
    which runs its handler; that is not a person's pick, so nothing is
    recorded or written. A different overlap is a pick: recorded once, and
    stored."""
    import logging
    from types import SimpleNamespace

    import modules.app_context as _app_ctx
    from tests.settings_fixtures import complete_settings, settings_writer
    from ui.advanced_settings import AdvancedSettings

    settings = complete_settings(tiling_overlap_percent=10.0)
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(settings=settings, update_settings=settings_writer(settings)),
    )
    spinner = SimpleNamespace(text='10%')
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(ids={'tiling_overlap_spinner': spinner})

    def _recorded():
        return [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']

    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        AdvancedSettings.update_tiling_overlap(panel)
        assert _recorded() == [], 'the populate was recorded as a pick'

        spinner.text = '20%'
        AdvancedSettings.update_tiling_overlap(panel)

    assert _recorded() == ['SELECT TILING_OVERLAP 20.0']
    assert settings['tiling_overlap_percent'] == 20.0


def test_scan_config_carries_the_persisted_overlap():
    """The overlap reaching a scan config is the persisted setting.

    Pins the VALUE rather than the call shape. This used to assert that the
    UI builder called ProtocolSettings.get_tiling_overlap_percent by name,
    which stopped being true when the builder became a thin adapter over the
    single settings-side builder -- while the contract it existed to protect,
    that overlap comes from the persisted key and never from the spinner,
    held throughout. A source-shape assertion cannot tell those two apart.
    """
    settings = json.loads((REPO / 'data' / 'settings.json').read_text())
    settings['tiling_overlap_percent'] = 25.0

    from modules.config_helpers import get_sequenced_capture_config_from_settings
    from modules.labware_loader import WellPlateLoader
    from modules.objectives_loader import ObjectiveLoader

    config = get_sequenced_capture_config_from_settings(
        settings,
        objective_helper=ObjectiveLoader(),
        wellplate_loader=WellPlateLoader(),
        current_z=0.0,
    )

    assert config['tiling_overlap_percent'] == 25.0

    # The panel is the one GUI caller that states its authoring choices
    # (tiling, z-stacking) to the Session; overlap is not one of them.
    source = _method_source(PROTOCOL_SETTINGS_SRC, 'ProtocolSettings', 'new_protocol')
    assert 'tiling_overlap_spinner' not in source, (
        'overlap must never be read off the spinner widget at scan time'
    )

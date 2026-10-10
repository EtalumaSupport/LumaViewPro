# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The model picker lists the running scope's model catalogue, and parses no file.

The GUI used to open scopes.json itself, from the installation's default
folder, so the picker could offer models the running scope did not know.
It now displays `scope.scope_models`, the catalogue the scope read once
from the folder it was started on.
"""

import types

from ui import advanced_settings
from ui.advanced_settings import AdvancedSettings


def test_the_picker_offers_exactly_the_scopes_models(monkeypatch):
    models = types.MappingProxyType({'LS620': {}, 'LS850T': {}})
    ctx = types.SimpleNamespace(
        session=types.SimpleNamespace(scope=types.SimpleNamespace(scope_models=models))
    )
    monkeypatch.setattr(advanced_settings._app_ctx, 'ctx', ctx)
    spinner = types.SimpleNamespace(values=[])
    picker = types.SimpleNamespace(ids={'scope_spinner': spinner})

    AdvancedSettings.load_scopes(picker)

    assert spinner.values == ['LS620', 'LS850T']

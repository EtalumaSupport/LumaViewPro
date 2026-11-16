# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The model picker lists the running scope's model catalogue, and parses no file.

The GUI used to open scopes.json itself, from the installation's default
folder, so the picker could offer models the running scope did not know.
It now displays `scope.all_scope_models`, the catalogue the scope read
once from the folder it was started on, in the catalogue's order.
"""

import types

from ui import advanced_settings
from ui.advanced_settings import AdvancedSettings


def test_the_picker_offers_exactly_the_scopes_models(monkeypatch):
    # a stand-in by design: the subject is the widget's read of the listing,
    # and a Kivy widget does not run in the test process.
    models = (types.SimpleNamespace(id='LS850T'), types.SimpleNamespace(id='LS620'))
    ctx = types.SimpleNamespace(
        session=types.SimpleNamespace(scope=types.SimpleNamespace(all_scope_models=models))
    )
    monkeypatch.setattr(advanced_settings._app_ctx, 'ctx', ctx)
    spinner = types.SimpleNamespace(values=[])
    picker = types.SimpleNamespace(ids={'scope_spinner': spinner})

    AdvancedSettings.load_scopes(picker)

    assert spinner.values == ['LS850T', 'LS620']

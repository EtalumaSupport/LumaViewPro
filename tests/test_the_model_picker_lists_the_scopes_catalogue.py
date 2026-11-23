# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The model picker lists the running scope's model catalogue, and shows the scope's model.

The GUI used to open scopes.json itself, from the installation's default
folder, so the picker could offer models the running scope did not know.
It now displays `scope.all_scope_models`, the catalogue the scope read
once from the folder it was started on, in the catalogue's order.

On a scope whose board names its model the picker showed the saved model,
the fallback for a scope that does not report its own, so an LS820 was
shown as LS850 and offered a choice it never took (#854). It now shows
the model the board named and is disabled there; elsewhere it shows the
saved model. A pick is an event: the picker's text is display only, so
writing it is never a pick.
"""

import types

from tests.gui_logging_census import load_string_blocks
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


def _identity(model, named):
    return types.SimpleNamespace(model=model, model_named_by_board=named)


def test_the_picker_shows_the_model_the_board_names_and_is_offered_nowhere_else(monkeypatch):
    # a stand-in by design: the subject is the widget's read of the
    # session's two answers, and a Kivy widget does not run in the test
    # process; the answers themselves are the session's, pinned in
    # test_a_model_saved_for_the_next_start_is_the_sessions_answer.py.
    cases = (
        # (the scope's identity, the saved model, the text shown, whether a choice is offered)
        (_identity('LS820', True), 'LS850', 'LS820', False),
        (_identity('LS620', False), 'LS560', 'LS560', True),
    )
    wrong = []
    for identity, saved, shown, offered in cases:
        session = types.SimpleNamespace(
            scope=types.SimpleNamespace(channel_identity=identity), selected_model=saved
        )
        monkeypatch.setattr(
            advanced_settings._app_ctx, 'ctx', types.SimpleNamespace(session=session)
        )
        spinner = types.SimpleNamespace(text='Select')
        picker = types.SimpleNamespace(ids={'scope_spinner': spinner}, model_choice_offered=None)

        AdvancedSettings._show_scope_model(picker)

        if (spinner.text, picker.model_choice_offered) != (shown, offered):
            wrong.append(
                f'{identity}, saved {saved}: {spinner.text!r}, {picker.model_choice_offered!r}'
            )
    assert not wrong, '\n'.join(wrong)


def _scope_spinner_rule():
    """The scope_spinner block's ``name: value`` lines, from the module's kv."""
    (kv,) = load_string_blocks('ui/advanced_settings.py')
    lines = kv.splitlines()
    at = next(i for i, line in enumerate(lines) if line.strip() == 'id: scope_spinner')
    depth = len(lines[at]) - len(lines[at].lstrip())
    rule = {}
    for line in lines[at:]:
        if line.strip() and len(line) - len(line.lstrip()) < depth:
            break
        name, _, value = line.strip().partition(':')
        rule[name] = value.strip()
    return rule


def test_a_pick_not_a_text_write_selects_the_model_and_the_picker_is_disabled_on_the_fact():
    # pin-justified: kv is declarative source with no headless seam; the
    # binding is the contract that a text write is never a pick.
    rule = _scope_spinner_rule()

    assert rule.get('on_pick') == 'root.select_scope(args[1])'
    assert 'on_text' not in rule
    assert 'text_follows_pick' not in rule
    assert 'not root.model_choice_offered' in rule['disabled']

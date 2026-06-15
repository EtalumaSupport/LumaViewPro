# Copyright Etaluma, Inc.
"""Regression guards for the app-wide popup close (X) button.

Kivy's stock Popup ships no close affordance. ui/popup_close.py injects an X
into every popup's title row from a global <Popup> on_open rule. The runtime
behavior (rule fires on open, X injected idempotently, X calls dismiss) was
verified against a real Kivy Popup outside the suite -- the suite mocks kivy
(see tests/conftest.py), so these are source-level guards on the invariants
that would silently regress.

The load-bearing guard is test_does_not_use_crashing_import_form: the global
rule MUST reference the module form (`#:import popup_close ui.popup_close`),
not a dotted attribute path (`#:import x ui.popup_close.add_popup_close`).
The dotted-attribute form hard-crashes the interpreter when resolved during
the module's own import -- no traceback, the process just dies.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POPUP_CLOSE_SRC = (REPO_ROOT / 'ui' / 'popup_close.py').read_text()


class TestInjectorInvariants:
    def test_defines_injector(self):
        assert 'def add_popup_close(' in POPUP_CLOSE_SRC

    def test_idempotency_guard_present(self):
        # on_open fires on every open; the injector must no-op after the first.
        assert '_etaluma_close_added' in POPUP_CLOSE_SRC

    def test_close_button_dismisses(self):
        assert 'on_release' in POPUP_CLOSE_SRC
        assert 'popup.dismiss()' in POPUP_CLOSE_SRC

    def test_uses_ascii_x_glyph(self):
        # Rule 24: ASCII only. The close glyph is a plain capital X.
        assert "text='X'" in POPUP_CLOSE_SRC


class TestGlobalRuleWiring:
    def test_global_popup_rule_registered(self):
        assert '<Popup>:' in POPUP_CLOSE_SRC
        assert 'on_open:' in POPUP_CLOSE_SRC

    def test_does_not_use_crashing_import_form(self):
        # The dotted-attribute #:import form crashes the interpreter when
        # resolved during ui.popup_close's own import. Must use module form.
        assert 'ui.popup_close.add_popup_close' not in POPUP_CLOSE_SRC
        assert '#:import popup_close ui.popup_close' in POPUP_CLOSE_SRC

    def test_imported_at_startup_for_side_effect(self):
        src = (REPO_ROOT / 'lumaviewpro.py').read_text()
        assert 'import ui.popup_close' in src
